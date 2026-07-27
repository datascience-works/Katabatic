from __future__ import annotations

import os
from typing import Any

from katabatic.artifacts import (
    ArtifactStore,
    LocalArtifactStore,
    ModelRef,
    new_train_id,
    write_dataset_artifact,
    write_dataset_artifact_presplit,
)
from katabatic.artifacts.refs import DatasetRef, EvaluationRef
from katabatic.evaluate.tstr.evaluation import TSTREvaluation
from katabatic.models.base_model import Model
from katabatic.pipeline.base_pipeline import Pipeline
from katabatic.utils.split_dataset import split_dataset, split_dataset_presplit

_EVAL_PATH_KEYS = frozenset({"synthetic_dir", "real_test_dir", "real_train_dir"})


def _model_slug(model: Model) -> str:
    return model.__class__.__name__.lower()


def _artifact_state_present(store: ArtifactStore, mr: ModelRef, model: Model) -> bool:
    names = getattr(type(model), "ARTIFACT_STATE_FILES", ())
    if not names:
        return False
    return all(store.exists(f"{mr.state_relpath}/{n}") for n in names)


def _is_tabsyn(model: Model) -> bool:
    """Avoid importing TabSyn stack (heavy optional deps) unless this is a TabSyn instance."""
    return type(model).__name__ == "TabSyn"


def _dataset_dir_for_model(model: Model, store: ArtifactStore, ref: DatasetRef) -> str:
    if _is_tabsyn(model):
        y_npy = f"{ref.extra_relpath}/y_train.npy"
        if store.exists(y_npy):
            return str(store.open_path(ref.extra_relpath))
    return str(store.open_path(ref.train_relpath))


def _default_legacy_synthetic_dir(output_dir: str, model: Model) -> str:
    base = os.path.basename(os.path.normpath(output_dir))
    return os.path.join("synthetic", base, _model_slug(model))


def _per_evaluation_kw(evaluation_kwargs: dict[Any, dict] | None, cls: type) -> dict:
    if not evaluation_kwargs:
        return {}
    return dict(evaluation_kwargs.get(cls) or evaluation_kwargs.get(cls.__name__) or {})


def _run_single_evaluation_artifact(
    evaluation_cls: type,
    store: ArtifactStore,
    mr: ModelRef,
    ds_ref: DatasetRef,
    merged: dict[str, Any],
) -> EvaluationRef | None:
    from_art = getattr(evaluation_cls, "from_artifact", None)
    if callable(from_art):
        passthrough = {k: v for k, v in merged.items() if k not in _EVAL_PATH_KEYS}
        eval_instance, ev_ref = from_art(store, mr, ds_ref, **passthrough)
        eval_instance.evaluate()
        return ev_ref
    eval_instance = evaluation_cls(**merged)
    eval_instance.evaluate()
    return None


class TrainTestSplitPipeline(Pipeline):
    """
    Version 1 of the pipeline.
    Legacy: split_dataset → train(output_dir) → evaluations.
    Artifact: versioned datasets/models/evaluations under an ArtifactStore.

    After ``run()`` completes, ``last_model`` holds the fitted model instance from
    the last successful run (``self.model`` remains the class, factory, or template).
    """

    _evaluations = [TSTREvaluation]

    def __init__(self, model: Model, evaluations=None, override_evaluations=False):
        super().__init__(model)
        self.last_model: Model | None = None

        if evaluations and override_evaluations:
            self._evaluations = evaluations
        elif evaluations:
            self._evaluations = list(type(self)._evaluations) + list(evaluations)
        else:
            self._evaluations = list(type(self)._evaluations)

    def run(self, *args, **kwargs) -> dict[str, Any]:
        kwargs = dict(kwargs)
        artifact_store = kwargs.pop("artifact_store", None)
        artifact_root = kwargs.pop("artifact_root", None)
        if artifact_store is None and artifact_root is not None:
            artifact_store = LocalArtifactStore(artifact_root)

        dataset_name = kwargs.pop("dataset_name", None)
        input_csv = kwargs.pop("input_csv", None)
        train_csv = kwargs.pop("train_csv", None)
        test_csv = kwargs.pop("test_csv", None)
        output_dir = kwargs.pop("output_dir", None)
        extra_source_dir = kwargs.pop("extra_source_dir", None)

        model_kwargs = dict(kwargs.pop("model_kwargs", None) or {})
        eval_kwargs_user = dict(kwargs.pop("eval_kwargs", None) or {})
        evaluation_kwargs = kwargs.pop("evaluation_kwargs", None)
        target_column = kwargs.pop("target_column", None)

        if artifact_store is not None and dataset_name is not None:
            has_single = input_csv is not None
            has_pair = train_csv is not None and test_csv is not None
            if has_single and has_pair:
                raise ValueError(
                    "Provide either input_csv (single file to split) or both train_csv and test_csv, not both."
                )
            if not has_single and not has_pair:
                raise ValueError(
                    "Artifact mode requires input_csv or both train_csv and test_csv."
                )
            return self._run_artifact(
                artifact_store,
                dataset_name,
                *args,
                input_csv=input_csv,
                train_csv=train_csv,
                test_csv=test_csv,
                extra_source_dir=extra_source_dir,
                model_kwargs=model_kwargs,
                eval_kwargs_user=eval_kwargs_user,
                evaluation_kwargs=evaluation_kwargs,
                target_column=target_column,
                **kwargs,
            )

        has_single = input_csv is not None
        has_pair = train_csv is not None and test_csv is not None
        if has_single and has_pair:
            raise ValueError(
                "Provide either input_csv (single file to split) or both train_csv and test_csv, not both."
            )
        if not output_dir:
            raise ValueError(
                "Legacy mode requires output_dir, and input_csv or train_csv+test_csv."
            )
        if not has_single and not has_pair:
            raise ValueError(
                "Legacy mode requires input_csv or both train_csv and test_csv."
            )

        return self._run_legacy(
            input_csv,
            output_dir,
            *args,
            train_csv=train_csv,
            test_csv=test_csv,
            model_kwargs=model_kwargs,
            eval_kwargs_user=eval_kwargs_user,
            evaluation_kwargs=evaluation_kwargs,
            extra_source_dir=extra_source_dir,
            **kwargs,
        )

    def _run_legacy(
        self,
        input_csv: str | None,
        output_dir: str,
        *args,
        train_csv: str | None = None,
        test_csv: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        eval_kwargs_user: dict[str, Any] | None = None,
        evaluation_kwargs: dict[Any, dict] | None = None,
        extra_source_dir: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        model_kwargs = dict(model_kwargs or {})
        eval_kwargs_user = dict(eval_kwargs_user or {})
        current_model = self.instantiate_model()

        if train_csv is not None and test_csv is not None:
            split_dataset_presplit(train_csv, test_csv, output_dir, **kwargs)
        elif input_csv is not None:
            split_dataset(input_csv, output_dir, *args, **kwargs)
        else:
            raise ValueError("internal: expected presplit or input_csv")

        train_kw = dict(kwargs)
        train_kw.update(model_kwargs)
        real_test_dir = train_kw.pop("real_test_dir", output_dir)
        synthetic_dir = train_kw.pop("synthetic_dir", None)
        if not synthetic_dir:
            synthetic_dir = _default_legacy_synthetic_dir(output_dir, current_model)
        os.makedirs(synthetic_dir, exist_ok=True)
        train_kw["synthetic_dir"] = synthetic_dir

        state_dir = os.path.join(output_dir, "_katabatic_model_state")
        os.makedirs(state_dir, exist_ok=True)

        _m = type(current_model)
        if getattr(_m, "ARTIFACT_STATE_FILES", ()) and not _is_tabsyn(current_model):
            train_kw.setdefault("artifact_state_dir", state_dir)
        if _is_tabsyn(current_model):
            train_kw.setdefault("save_dir", state_dir)

        current_model.train(output_dir, *args, **train_kw)

        real_train_dir = output_dir
        eval_merged = {
            **eval_kwargs_user,
            "synthetic_dir": synthetic_dir,
            "real_test_dir": real_test_dir,
            "real_train_dir": real_train_dir,
        }

        tstr_results: list[Any] = []
        for evaluation in self._evaluations:
            per = _per_evaluation_kw(evaluation_kwargs, evaluation)
            merged = {**eval_merged, **per}
            eval_instance = evaluation(**merged)
            tstr_results.append(eval_instance.evaluate())

        self.last_model = current_model

        return {
            "message": "Train test split pipeline executed successfully.",
            "output_dir": output_dir,
            "synthetic_dir": synthetic_dir,
            "real_test_dir": real_test_dir,
            "tstr_results": tstr_results[0] if len(tstr_results) == 1 else tstr_results,
        }

    def _run_artifact(
        self,
        store: ArtifactStore,
        dataset_name: str,
        *args,
        input_csv: str | None = None,
        train_csv: str | None = None,
        test_csv: str | None = None,
        extra_source_dir: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        eval_kwargs_user: dict[str, Any] | None = None,
        evaluation_kwargs: dict[Any, dict] | None = None,
        target_column: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        model_kwargs = dict(model_kwargs or {})
        eval_kwargs_user = dict(eval_kwargs_user or {})
        test_size = kwargs.pop("test_size", 0.2)
        seed = kwargs.pop("seed", 42)
        model_name = kwargs.pop("model_name", None)
        require_registered_dataset = kwargs.pop("require_registered_dataset", False)

        profile_csv = train_csv or input_csv
        assert profile_csv is not None

        from katabatic.datasets.compatibility import check_dataset_for_model
        from katabatic.datasets.registry import DatasetRegistry

        reg = DatasetRegistry(store)
        reg.register_if_absent(dataset_name, profile_csv, target_column=target_column)

        current_model = self.instantiate_model()
        model_name = model_name or _model_slug(current_model)

        if require_registered_dataset:
            entry = reg.get(dataset_name)
            if entry is None:
                raise ValueError(f"Dataset {dataset_name!r} missing from registry after register_if_absent.")
            ok, msg = check_dataset_for_model(entry, model_name)
            if not ok:
                raise ValueError(
                    f"Dataset {dataset_name!r} is incompatible with model {model_name!r}: {msg}"
                )

        if train_csv is not None and test_csv is not None:
            ds_ref = write_dataset_artifact_presplit(
                store,
                train_csv=train_csv,
                test_csv=test_csv,
                dataset_name=dataset_name,
                extra_source_dir=extra_source_dir,
            )
        else:
            ds_ref = write_dataset_artifact(
                store,
                input_csv=input_csv,  # type: ignore[arg-type]
                dataset_name=dataset_name,
                test_size=test_size,
                seed=seed,
                extra_source_dir=extra_source_dir,
            )

        train_run_id = new_train_id()
        mr = ModelRef(
            model_name=model_name,
            dataset_name=dataset_name,
            dataset_version=ds_ref.dataset_version,
            train_run_id=train_run_id,
        )
        store.open_path(mr.state_relpath).mkdir(parents=True, exist_ok=True)
        store.open_path(mr.synthetic_relpath).mkdir(parents=True, exist_ok=True)

        dataset_dir = _dataset_dir_for_model(current_model, store, ds_ref)
        synthetic_dir = str(store.open_path(mr.synthetic_relpath))

        train_kw = dict(kwargs)
        train_kw.update(model_kwargs)
        train_kw["synthetic_dir"] = synthetic_dir

        state_path = str(store.open_path(mr.state_relpath))

        _m = type(current_model)
        if getattr(_m, "ARTIFACT_STATE_FILES", ()) and not _is_tabsyn(current_model):
            train_kw.setdefault("artifact_state_dir", state_path)
        if _is_tabsyn(current_model):
            train_kw.setdefault("save_dir", state_path)

        current_model.train(dataset_dir, *args, **train_kw)

        config = train_kw.get("config")
        if config is not None:
            if hasattr(config, "__dict__") and not isinstance(config, dict):
                cfg_dict = {k: v for k, v in vars(config).items() if not k.startswith("_")}
                store.save_json(f"{mr.root_relpath}/config.json", cfg_dict)
            elif isinstance(config, dict):
                store.save_json(f"{mr.root_relpath}/config.json", config)
            else:
                store.save_json(f"{mr.root_relpath}/config.json", {"repr": repr(config)})

        eval_merged_base = {
            **eval_kwargs_user,
            "synthetic_dir": synthetic_dir,
            "real_test_dir": str(store.open_path(ds_ref.test_relpath)),
            "real_train_dir": str(store.open_path(ds_ref.train_relpath)),
        }

        evaluation_refs: list[EvaluationRef | None] = []
        for evaluation in self._evaluations:
            per = _per_evaluation_kw(evaluation_kwargs, evaluation)
            merged = {**eval_merged_base, **per}
            ref = _run_single_evaluation_artifact(
                evaluation, store, mr, ds_ref, merged
            )
            evaluation_refs.append(ref)

        self.last_model = current_model

        return {
            "message": "Train test split pipeline executed successfully.",
            "dataset_ref": ds_ref,
            "model_ref": mr,
            "evaluation_refs": evaluation_refs,
        }

    def __repr__(self):
        return f"TrainTestSplitPipeline(name={self.model})"
