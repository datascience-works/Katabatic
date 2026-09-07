import json
import logging
import os
import random
import warnings

import fsspec
import numpy as np
import pandas as pd
import torch
from huggingface_hub import resolve_revision
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from katabatic.models.base_model import Model

from .utils import (
    CategoricalStart,
    ContinuousStart,
    GReaTDataCollator,
    GReaTDataset,
    GReaTStart,
    GReaTTrainer,
    RandomStart,
    _array_to_dataframe,
    _convert_text_to_tabular_data,
    _convert_tokens_to_text,
    _get_column_distribution,
    _partial_df_to_promts,
    bcolors,
)


class GReaT(Model):
    """
    GReaT model for synthetic tabular data generation.

    GReaT fine-tunes a causal language model on tabular rows converted into
    text format, then samples new rows by generating text and parsing it back
    into a tabular DataFrame.
    """

    def __init__(
        self,
        llm: str,
        experiment_dir: str = "trainer_great",
        epochs: int = 100,
        batch_size: int = 8,
        efficient_finetuning: str = "",
        float_precision: int | None = None,
        report_to: list[str] | None = None,
        **train_kwargs,
    ):
        super().__init__()
        self.check_dependencies()

        self.efficient_finetuning = efficient_finetuning
        self.llm = llm
        self.experiment_dir = experiment_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.float_precision = float_precision
        self.train_hyperparameters = {
            "report_to": report_to or [],
            **train_kwargs,
        }

        revision = resolve_revision(self.llm)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm,
            revision=revision.resolved,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm,
            revision=revision.resolved,
        )

        if self.efficient_finetuning == "lora":
            try:
                from peft import (
                    LoraConfig,
                    TaskType,
                    get_peft_model,
                    prepare_model_for_int8_training,
                )
            except ImportError as exc:
                raise ImportError(
                    "LoRA fine-tuning requires peft. Install with: pip install peft"
                ) from exc

            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["c_attn"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )

            self.model = prepare_model_for_int8_training(self.model)
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        self.columns = None
        self.num_cols = None
        self.conditional_col = None
        self.conditional_col_dist = None
        self.target_col = None
        self.is_fitted = False

    @classmethod
    def get_required_dependencies(cls) -> list[str]:
        return [
            "transformers",
            "torch",
            "accelerate",
            "datasets",
            "pandas",
            "numpy",
        ]

    def train(
        self,
        data_dir: str,
        *args,
        categorical_cols: list[str] | None = None,
        continuous_cols: list[str] | None = None,
        **kwargs,
    ) -> "GReaT":
        """
        Train GReaT using Katabatic-standard x_train.csv and y_train.csv.

        Parameters
        ----------
        data_dir : str
            Directory containing x_train.csv and y_train.csv.

        Returns
        -------
        GReaT
            Trained model instance.
        """
        x_train_path = os.path.join(data_dir, "x_train.csv")
        y_train_path = os.path.join(data_dir, "y_train.csv")

        if not os.path.exists(x_train_path) or not os.path.exists(y_train_path):
            raise FileNotFoundError(
                f"Expected x_train.csv and y_train.csv inside: {data_dir}"
            )

        x_train = pd.read_csv(x_train_path)
        y_train = pd.read_csv(y_train_path).squeeze()

        self.target_col = y_train.name

        train_df = pd.concat([x_train, y_train], axis=1)

        self.fit(train_df)
        return self

    def fit(
        self,
        data: pd.DataFrame | np.ndarray,
        column_names: list[str] | None = None,
        conditional_col: str | None = None,
        resume_from_checkpoint: bool | str = False,
    ) -> GReaTTrainer:
        """
        Fine-tune GReaT on tabular data.

        Data is converted into text rows using:
        column_name is value, column_name is value, ...
        """
        df = _array_to_dataframe(data, columns=column_names)

        self._update_column_information(df)
        self._update_conditional_information(df, conditional_col)

        logging.info("Converting dataframe into HuggingFace dataset...")
        great_ds = GReaTDataset.from_pandas(df)
        great_ds.set_tokenizer(self.tokenizer, self.float_precision)

        logging.info("Creating GReaT trainer...")
        training_args = TrainingArguments(
            output_dir=self.experiment_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            **self.train_hyperparameters,
        )

        # Important:
        # New Transformers versions do not accept tokenizer= in Trainer.
        great_trainer = GReaTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=great_ds,
            data_collator=GReaTDataCollator(self.tokenizer),
        )

        logging.info("Starting GReaT training...")
        great_trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        self.is_fitted = True
        return great_trainer

    def sample(
        self,
        n_samples: int,
        start_col: str | None = "",
        start_col_dist: dict | list | None = None,
        temperature: float = 0.7,
        k: int = 100,
        max_length: int = 100,
        drop_nan: bool = False,
        device: str = "cpu",
        guided_sampling: bool = False,
        random_feature_order: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate synthetic tabular samples.

        Parameters
        ----------
        n_samples : int
            Number of synthetic rows to generate.
        seed : int, optional
            Runtime seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Synthetic dataframe.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before sampling.")

        seed = kwargs.pop("seed", None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if guided_sampling:
            return self._guided_sample(
                n_samples=n_samples,
                temperature=temperature,
                max_length=max_length,
                device=device,
                random_feature_order=random_feature_order,
            )

        return self._legacy_sample(
            n_samples=n_samples,
            start_col=start_col,
            start_col_dist=start_col_dist,
            temperature=temperature,
            k=k,
            max_length=max_length,
            drop_nan=drop_nan,
            device=device,
        )

    def _legacy_sample(
        self,
        n_samples: int,
        start_col: str | None = "",
        start_col_dist: dict | list | None = None,
        temperature: float = 0.7,
        k: int = 100,
        max_length: int = 100,
        drop_nan: bool = False,
        device: str = "cpu",
    ) -> pd.DataFrame:
        """
        Legacy GReaT batch sampling method.
        """
        great_start = self._get_start_sampler(start_col, start_col_dist)

        self.model.to(device)
        self.model.eval()

        dfs = []

        with tqdm(total=n_samples) as pbar:
            already_generated = 0
            attempts = 0

            while n_samples > already_generated:
                start_tokens = great_start.get_start_tokens(k)
                start_tokens = torch.tensor(start_tokens).to(device)

                attention_mask = (start_tokens != self.tokenizer.pad_token_id).long()

                try:
                    tokens = self.model.generate(
                        input_ids=start_tokens,
                        attention_mask=attention_mask,
                        max_length=max_length,
                        do_sample=True,
                        temperature=temperature,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                    text_data = _convert_tokens_to_text(tokens, self.tokenizer)
                    df_gen = _convert_text_to_tabular_data(text_data, self.columns)

                    if df_gen.empty:
                        attempts += 1
                        if attempts > 13 and already_generated == 0:
                            raise RuntimeError(
                                "Unable to generate samples after multiple attempts."
                            )
                        continue

                    df_gen = df_gen[~(df_gen == "placeholder").any(axis=1)]
                    df_gen = df_gen.dropna(how="all")

                    if drop_nan:
                        df_gen = df_gen.dropna()

                    for col in self.num_cols:
                        if col in df_gen.columns:
                            df_gen[col] = pd.to_numeric(df_gen[col], errors="coerce")
                            df_gen = df_gen[df_gen[col].notnull()]

                    if self.num_cols:
                        valid_num_cols = [
                            c for c in self.num_cols if c in df_gen.columns
                        ]
                        df_gen[valid_num_cols] = df_gen[valid_num_cols].astype(float)

                    if not df_gen.empty:
                        dfs.append(df_gen)
                        already_generated += len(df_gen)
                        pbar.update(len(df_gen))

                    attempts += 1

                    if attempts > 13 and already_generated == 0:
                        raise RuntimeError(
                            "No valid samples generated. Try guided_sampling=True, "
                            "increase epochs, or increase max_length."
                        )

                except Exception as exc:
                    print(f"{bcolors.FAIL}Sampling error: {exc}{bcolors.ENDC}")
                    print(
                        f"{bcolors.WARNING}Try guided_sampling=True, higher epochs, "
                        f"or larger max_length.{bcolors.ENDC}"
                    )
                    break

        if dfs:
            df_out = pd.concat(dfs, ignore_index=True)
            return df_out.head(n_samples)

        return pd.DataFrame(columns=self.columns)

    def _guided_sample(
        self,
        n_samples: int = 10,
        temperature: float = 0.7,
        max_length: int = 100,
        device: str = "cpu",
        random_feature_order: bool = True,
    ) -> pd.DataFrame:
        """
        Guided feature-by-feature generation.
        Slower than legacy sampling but sometimes more reliable.
        """
        if self.columns is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        self.model.to(device)
        self.model.eval()
        torch_device = torch.device(device)

        synthetic_data = []

        with tqdm(total=n_samples) as pbar:
            for i in range(n_samples):
                try:
                    feature_names = self.columns.copy()

                    if random_feature_order:
                        random.shuffle(feature_names)

                    sample_text = ""
                    sample_values = {}

                    for feature in feature_names:
                        prompt = f"{sample_text}{feature} is"
                        inputs = self.tokenizer(prompt, return_tensors="pt").to(
                            torch_device
                        )

                        output = self.model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs.get("attention_mask"),
                            max_length=len(inputs["input_ids"][0]) + 30,
                            temperature=temperature,
                            pad_token_id=self.tokenizer.eos_token_id,
                            do_sample=True,
                        )

                        generated_text = self.tokenizer.decode(
                            output[0],
                            skip_special_tokens=True,
                        )

                        raw_value = generated_text[len(prompt) :].strip()

                        if ";" in raw_value:
                            value = raw_value.split(";")[0].strip()
                        else:
                            for delimiter in [",", "\n"]:
                                if delimiter in raw_value:
                                    value = raw_value.split(delimiter)[0].strip()
                                    break
                            else:
                                value = raw_value[:30].strip()

                        while value and not (
                            value[-1].isalnum() or value[-1] in [".", "-"]
                        ):
                            value = value[:-1]

                        sample_values[feature] = value
                        sample_text += f"{feature} is {value}; "

                    ordered_sample = {
                        feature: sample_values.get(feature, "")
                        for feature in self.columns
                    }

                    synthetic_data.append(ordered_sample)
                    pbar.update(1)

                except Exception as exc:
                    print(f"Error generating sample {i + 1}: {exc}")
                    continue

        if synthetic_data:
            df = pd.DataFrame(synthetic_data)

            for col in self.num_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            return df.head(n_samples)

        return pd.DataFrame(columns=self.columns)

    def great_sample(
        self,
        starting_prompts: str | list[str],
        temperature: float = 0.7,
        max_length: int = 100,
        device: str = "cpu",
    ) -> pd.DataFrame:
        """
        Generate samples conditioned on custom text prompts.
        """
        self.model.to(device)
        self.model.eval()

        if isinstance(starting_prompts, str):
            starting_prompts = [starting_prompts]

        generated_data = []

        for prompt in tqdm(starting_prompts):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

            gen = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            generated_data.append(torch.squeeze(gen))

        decoded_data = _convert_tokens_to_text(generated_data, self.tokenizer)
        return _convert_text_to_tabular_data(decoded_data, self.columns)

    def impute(
        self,
        df_miss: pd.DataFrame,
        temperature: float = 0.7,
        max_length: int = 100,
        max_retries: int = 15,
        device: str = "cpu",
    ) -> pd.DataFrame:
        """
        Impute missing values using GReaT.
        """
        if set(df_miss.columns) != set(self.columns):
            raise ValueError(
                "The columns of df_miss must match the columns used for training."
            )

        self.model.to(device)
        df_list = []

        with tqdm(total=len(df_miss)) as pbar:
            for index in range(len(df_miss)):
                retries = 0
                df_curr = df_miss.iloc[[index]]
                original_index = df_curr.index

                while retries < max_retries:
                    starting_prompts = _partial_df_to_promts(
                        df_curr,
                        self.float_precision,
                    )

                    df_curr = self.great_sample(
                        starting_prompts,
                        temperature=temperature,
                        max_length=max_length,
                        device=device,
                    )

                    for col in self.num_cols:
                        if col in df_curr.columns:
                            df_curr[col] = pd.to_numeric(
                                df_curr[col],
                                errors="coerce",
                            )

                    if not df_curr.isna().any().any():
                        df_list.append(df_curr.set_index(original_index))
                        break

                    retries += 1

                if retries >= max_retries:
                    warnings.warn("Max retries reached during imputation.")

                pbar.update(1)

        if df_list:
            return pd.concat(df_list, axis=0)

        return pd.DataFrame(columns=self.columns)

    def evaluate(self, *args, **kwargs):
        """
        Standalone evaluation is not implemented here.

        Use Katabatic's SyntheticEvaluationPipeline or a TSTR/TRTR runner.
        """
        raise NotImplementedError(
            "Use SyntheticEvaluationPipeline or an external TSTR/TRTR runner."
        )

    def save(self, path: str):
        """
        Save GReaT model weights and configuration.
        """
        fs = fsspec.filesystem(fsspec.utils.get_protocol(path))

        if fs.exists(path):
            warnings.warn(f"Directory {path} already exists and will be overwritten.")
        else:
            fs.mkdir(path)

        with fs.open(path + "/config.json", "w") as f:
            attributes = self.__dict__.copy()
            attributes.pop("tokenizer", None)
            attributes.pop("model", None)

            if isinstance(attributes.get("conditional_col_dist"), np.ndarray):
                attributes["conditional_col_dist"] = list(
                    attributes["conditional_col_dist"]
                )

            json.dump(attributes, f)

        torch.save(self.model.state_dict(), fs.open(path + "/model.pt", "wb"))

    def load_finetuned_model(self, path: str):
        """
        Load fine-tuned model weights.
        """
        self.model.load_state_dict(
            torch.load(
                fsspec.open(path, "rb"),
                weights_only=True,
            )
        )

    @classmethod
    def load_from_dir(cls, path: str):
        """
        Load a saved GReaT model from directory.
        """
        fs = fsspec.filesystem(fsspec.utils.get_protocol(path))

        if not fs.exists(path):
            raise FileNotFoundError(f"Directory does not exist: {path}")

        with fs.open(path + "/config.json", "r") as f:
            attributes = json.load(f)

        great = cls(attributes["llm"])

        for key, value in attributes.items():
            setattr(great, key, value)

        great.model.load_state_dict(
            torch.load(
                fs.open(path + "/model.pt", "rb"),
                map_location="cpu",
                weights_only=True,
            )
        )

        return great

    def _update_column_information(self, df: pd.DataFrame):
        self.columns = df.columns.to_list()
        self.num_cols = df.select_dtypes(include=np.number).columns.to_list()

    def _update_conditional_information(
        self,
        df: pd.DataFrame,
        conditional_col: str | None = None,
    ):
        if conditional_col is not None and not isinstance(conditional_col, str):
            raise TypeError("conditional_col must be a string or None.")

        if conditional_col is not None and conditional_col not in df.columns:
            raise ValueError(f"Column {conditional_col} is not in dataframe.")

        self.conditional_col = conditional_col if conditional_col else df.columns[-1]
        self.conditional_col_dist = _get_column_distribution(
            df,
            self.conditional_col,
        )
        self.target_col = self.conditional_col

    def _get_start_sampler(
        self,
        start_col: str | None,
        start_col_dist: dict | list | None,
    ) -> GReaTStart:
        if start_col and start_col_dist is None:
            raise ValueError(
                f"Start column {start_col} was given without a distribution."
            )

        if start_col_dist is not None and not start_col:
            raise ValueError("start_col_dist was given, but start_col is missing.")

        start_col = start_col if start_col else self.conditional_col
        start_col_dist = (
            start_col_dist if start_col_dist is not None else self.conditional_col_dist
        )

        if isinstance(start_col_dist, dict):
            return CategoricalStart(self.tokenizer, start_col, start_col_dist)

        if isinstance(start_col_dist, list):
            return ContinuousStart(self.tokenizer, start_col, start_col_dist)

        return RandomStart(self.tokenizer, self.columns)
