import json
import logging
import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

from katabatic.models.base_model import Model

from .utils import (
    CategoricalStart,
    ContinuousStart,
    RandomStart,
    TabulaDataCollator,
    TabulaDataset,
    TabulaStart,
    TabulaTrainer,
    _array_to_dataframe,
    _convert_text_to_tabular_data,
    _convert_tokens_to_text,
    _get_column_distribution,
)

logger = logging.getLogger("katabatic.models.tabula")


# ============================================================
# ORIGINAL TABULA IMPLEMENTATION
# ============================================================


class Tabula:
    """Tabula Class

    The Tabula class handles the whole generation flow. It is used
    to fine-tune a large language model for tabular data, and to
    sample synthetic tabular data.
    """

    def __init__(
        self,
        llm: str,
        experiment_dir: str = "trainer_tabula",
        epochs: int = 100,
        batch_size: int = 8,
        categorical_columns: list | None = None,
        **train_kwargs,
    ):
        if categorical_columns is None:
            categorical_columns = []

        self.llm = llm

        self.tokenizer = AutoTokenizer.from_pretrained(self.llm)

        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.config = AutoConfig.from_pretrained(self.llm)

        self.model = AutoModelForCausalLM.from_config(self.config)

        self.experiment_dir = experiment_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.categorical_columns = categorical_columns

        self.train_hyperparameters = train_kwargs

        self.columns = None
        self.num_cols = None
        self.conditional_col = None
        self.conditional_col_dist = None

    # ========================================================
    # CATEGORICAL ENCODING
    # ========================================================

    def encode_categorical_column(
        self,
        data: pd.DataFrame,
    ):
        self.label_encoder_list = []

        for column_index, column in enumerate(data.columns):
            if column in self.categorical_columns:
                label_encoder = preprocessing.LabelEncoder()

                data[column] = data[column].astype(str)

                label_encoder.fit(data[column])

                current_label_encoder = {}

                current_label_encoder["column"] = column
                current_label_encoder["label_encoder"] = label_encoder

                transformed_column = label_encoder.transform(data[column])

                data[column] = transformed_column

                self.label_encoder_list.append(current_label_encoder)

        return data

    def decode_categorical_column(
        self,
        data: pd.DataFrame,
    ):

        for i in range(len(self.label_encoder_list)):
            le = self.label_encoder_list[i]["label_encoder"]

            allowed_values = list(range(len(le.classes_)))

            data[self.label_encoder_list[i]["column"]] = pd.to_numeric(
                data[self.label_encoder_list[i]["column"]],
                errors="coerce",
            )

            data = data.dropna(subset=[self.label_encoder_list[i]["column"]])

            data[self.label_encoder_list[i]["column"]] = data[
                self.label_encoder_list[i]["column"]
            ].astype(float)

            data = data[data[self.label_encoder_list[i]["column"]].isin(allowed_values)]

        for i in range(len(self.label_encoder_list)):
            le = self.label_encoder_list[i]["label_encoder"]

            data[self.label_encoder_list[i]["column"]] = data[
                self.label_encoder_list[i]["column"]
            ].astype(int)

            data[self.label_encoder_list[i]["column"]] = le.inverse_transform(
                data[self.label_encoder_list[i]["column"]]
            )

        return data

    # ========================================================
    # TRAINING
    # ========================================================

    def fit(
        self,
        data: pd.DataFrame | np.ndarray,
        column_names: list[str] | None = None,
        conditional_col: str | None = None,
        resume_from_checkpoint: bool | str = False,
    ) -> TabulaTrainer:

        df = _array_to_dataframe(
            data,
            columns=column_names,
        )

        self._update_column_information(df)

        self._update_conditional_information(
            df,
            conditional_col,
        )

        if self.categorical_columns != []:
            df = self.encode_categorical_column(df)

        logging.info("Convert data into HuggingFace dataset object...")

        tabula_ds = TabulaDataset.from_pandas(df)

        tabula_ds.set_tokenizer(self.tokenizer)

        logging.info("Create Tabula Trainer...")

        training_args = TrainingArguments(
            self.experiment_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            save_strategy="no",
            **self.train_hyperparameters,
        )

        tabula_trainer = TabulaTrainer(
            self.model,
            training_args,
            train_dataset=tabula_ds,
            tokenizer=self.tokenizer,
            data_collator=TabulaDataCollator(self.tokenizer),
        )

        logging.info("Start training...")

        tabula_trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        return tabula_trainer

    # ========================================================
    # SAMPLING
    # ========================================================

    def sample(
        self,
        n_samples: int,
        start_col: str | None = "",
        start_col_dist: dict | list | None = None,
        temperature: float = 0.7,
        k: int = 100,
        max_length: int = 100,
        device: str = "cuda",
    ) -> pd.DataFrame:

        tabula_start = self._get_start_sampler(
            start_col,
            start_col_dist,
        )

        self.model.to(device)

        df_gen = pd.DataFrame(columns=self.columns)

        with tqdm(total=n_samples) as pbar:
            already_generated = 0

            while n_samples > df_gen.shape[0]:
                start_tokens = tabula_start.get_start_tokens(k)

                start_tokens = torch.tensor(start_tokens).to(device)

                tokens = self.model.generate(
                    input_ids=start_tokens,
                    max_length=max_length,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=int(self.tokenizer.eos_token_id),
                )

                text_data = _convert_tokens_to_text(
                    tokens,
                    self.tokenizer,
                )

                df_gen = _convert_text_to_tabular_data(
                    text_data,
                    df_gen,
                )

                for i_num_cols in self.num_cols:
                    df_gen = df_gen[
                        pd.to_numeric(
                            df_gen[i_num_cols],
                            errors="coerce",
                        ).notnull()
                    ]

                df_gen[self.num_cols] = df_gen[self.num_cols].astype(float)

                df_gen = df_gen.drop(df_gen[df_gen.isna().any(axis=1)].index)

                pbar.update(df_gen.shape[0] - already_generated)

                already_generated = df_gen.shape[0]

        df_gen = df_gen.reset_index(drop=True)

        if self.categorical_columns == []:
            return df_gen.head(n_samples)

        return self.decode_categorical_column(df_gen.head(n_samples))

    # ========================================================
    # CONDITIONAL SAMPLING
    # ========================================================

    def tabula_sample(
        self,
        starting_prompts: str | list[str],
        temperature: float = 0.7,
        max_length: int = 100,
        device: str = "cuda",
    ) -> pd.DataFrame:

        self.model.to(device)

        starting_prompts = (
            [starting_prompts]
            if isinstance(starting_prompts, str)
            else starting_prompts
        )

        generated_data = []

        for prompt in tqdm(starting_prompts):
            start_token = torch.tensor(self.tokenizer(prompt)["input_ids"]).to(device)

            gen = self.model.generate(
                input_ids=torch.unsqueeze(
                    start_token,
                    0,
                ),
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                pad_token_id=int(self.tokenizer.eos_token_id),
            )

            generated_data.append(torch.squeeze(gen))

        decoded_data = _convert_tokens_to_text(
            generated_data,
            self.tokenizer,
        )

        df_gen = _convert_text_to_tabular_data(
            decoded_data,
            pd.DataFrame(columns=self.columns),
        )

        return df_gen

    # ========================================================
    # SAVE / LOAD
    # ========================================================

    def save(self, path: str):

        if os.path.isdir(path):
            warnings.warn(f"Directory {path} already exists and is overwritten now.")
        else:
            os.mkdir(path)

        with open(
            path + "/config.json",
            "w",
        ) as f:
            attributes = self.__dict__.copy()

            attributes.pop("tokenizer")
            attributes.pop("model")

            if isinstance(
                attributes["conditional_col_dist"],
                np.ndarray,
            ):
                attributes["conditional_col_dist"] = list(
                    attributes["conditional_col_dist"]
                )

            json.dump(
                attributes,
                f,
            )

        torch.save(
            self.model.state_dict(),
            path + "/model.pt",
        )

    def load_finetuned_model(
        self,
        path: str,
    ):
        self.model.load_state_dict(torch.load(path))

    @classmethod
    def load_from_dir(
        cls,
        path: str,
    ):

        assert os.path.isdir(path), f"Directory {path} does not exist."

        with open(
            path + "/config.json",
        ) as f:
            attributes = json.load(f)

        tabula = cls(attributes["llm"])

        for k, v in attributes.items():
            setattr(
                tabula,
                k,
                v,
            )

        tabula.model.load_state_dict(
            torch.load(
                path + "/model.pt",
                map_location="cpu",
            )
        )

        return tabula

    # ========================================================
    # COLUMN INFORMATION
    # ========================================================

    def _update_column_information(
        self,
        df: pd.DataFrame,
    ):
        self.columns = df.columns.to_list()

        self.num_cols = df.select_dtypes(include=np.number).columns.to_list()

    def _update_conditional_information(
        self,
        df: pd.DataFrame,
        conditional_col: str | None = None,
    ):

        assert conditional_col is None or isinstance(
            conditional_col,
            str,
        ), f"The column name has to be a string and not {type(conditional_col)}"

        assert conditional_col is None or conditional_col in df.columns, (
            f"The column name {conditional_col} "
            f"is not in the feature names of the given dataset"
        )

        self.conditional_col = conditional_col if conditional_col else df.columns[-1]

        self.conditional_col_dist = _get_column_distribution(
            df,
            self.conditional_col,
        )

    def _get_start_sampler(
        self,
        start_col: str | None,
        start_col_dist: dict | list | None,
    ) -> TabulaStart:

        if start_col and start_col_dist is None:
            raise ValueError(
                f"Start column {start_col} was given, "
                "but no corresponding distribution."
            )

        if start_col_dist is not None and not start_col:
            raise ValueError(
                f"Start column distribution {start_col_dist} "
                f"was given, the column name is missing."
            )

        assert start_col is None or isinstance(
            start_col,
            str,
        )

        assert (
            start_col_dist is None
            or isinstance(
                start_col_dist,
                dict,
            )
            or isinstance(
                start_col_dist,
                list,
            )
        )

        start_col = start_col if start_col else self.conditional_col

        start_col_dist = start_col_dist if start_col_dist else self.conditional_col_dist

        if isinstance(
            start_col_dist,
            dict,
        ):
            return CategoricalStart(
                self.tokenizer,
                start_col,
                start_col_dist,
            )

        elif isinstance(
            start_col_dist,
            list,
        ):
            return ContinuousStart(
                self.tokenizer,
                start_col,
                start_col_dist,
            )

        else:
            return RandomStart(
                self.tokenizer,
                self.columns,
            )


# ============================================================
# KATABATIC MODEL WRAPPER
# ============================================================


class TABULA(Model):
    """Katabatic wrapper for the TabuLa model."""

    def __init__(
        self,
        llm="distilgpt2",
        experiment_dir="trainer_tabula",
        epochs=5,
        batch_size=8,
        categorical_columns=None,
        **train_kwargs,
    ):
        super().__init__()

        self.check_dependencies()

        if categorical_columns is None:
            categorical_columns = []

        self.core = Tabula(
            llm=llm,
            experiment_dir=experiment_dir,
            epochs=epochs,
            batch_size=batch_size,
            categorical_columns=categorical_columns,
            **train_kwargs,
        )

    # ========================================================
    # DEPENDENCIES
    # ========================================================

    @classmethod
    def get_required_dependencies(cls):
        return [
            "torch",
            "transformers",
            "datasets",
            "tqdm",
        ]

    # ========================================================
    # KATABATIC TRAIN
    # ========================================================

    def train(
        self,
        dataset_dir,
        synthetic_dir=None,
        device=None,
        **kwargs,
    ):

        logger.info("Training TabuLa model")

        x_train_path = os.path.join(
            dataset_dir,
            "x_train.csv",
        )

        y_train_path = os.path.join(
            dataset_dir,
            "y_train.csv",
        )

        X_train = pd.read_csv(x_train_path)

        y_train_df = pd.read_csv(y_train_path)

        if y_train_df.shape[1] == 1:
            y_train = y_train_df.iloc[:, 0]
        else:
            y_train = y_train_df

        df_train = pd.concat(
            [
                X_train,
                y_train,
            ],
            axis=1,
        )

        original_x_columns = X_train.columns.tolist()

        df_train = self._sanitize_columns(df_train)

        # Allow small smoke tests to override
        # the default number of epochs.
        if "epochs" in kwargs:
            self.core.epochs = int(kwargs["epochs"])

        self.core.fit(df_train)

        n_samples = int(
            kwargs.get(
                "n_samples",
                min(
                    len(df_train),
                    10000,
                ),
            )
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        k = int(
            kwargs.get(
                "k",
                min(
                    16,
                    max(
                        1,
                        n_samples,
                    ),
                ),
            )
        )

        temperature = float(
            kwargs.get(
                "temperature",
                0.8,
            )
        )

        max_length = int(
            kwargs.get(
                "max_length",
                self._estimate_max_length(df_train),
            )
        )

        max_rounds = int(
            kwargs.get(
                "max_rounds",
                100,
            )
        )

        df_synth = self._safe_sample(
            n_samples=n_samples,
            k=k,
            temperature=temperature,
            max_length=max_length,
            device=device,
            max_rounds=max_rounds,
        )

        if df_synth.empty:
            raise RuntimeError("TabuLa did not generate any valid synthetic rows.")

        x_synth = df_synth.iloc[:, :-1].copy()

        y_synth = df_synth.iloc[:, -1].copy()

        x_synth.columns = original_x_columns

        if synthetic_dir is None:
            synthetic_dir = os.path.join(
                "synthetic",
                os.path.basename(os.path.normpath(dataset_dir)),
                "tabula",
            )

        os.makedirs(
            synthetic_dir,
            exist_ok=True,
        )

        x_synth.to_csv(
            os.path.join(
                synthetic_dir,
                "x_synth.csv",
            ),
            index=False,
        )

        y_synth.to_csv(
            os.path.join(
                synthetic_dir,
                "y_synth.csv",
            ),
            index=False,
        )

        self.is_fitted = True

        logger.info(
            "Generated %d synthetic rows",
            len(df_synth),
        )

        return self

    # ========================================================
    # SAMPLE
    # ========================================================

    def sample(
        self,
        *args,
        **kwargs,
    ):
        return self._safe_sample(
            *args,
            **kwargs,
        )

    # ========================================================
    # EVALUATE
    # ========================================================

    def evaluate(
        self,
        *args,
        **kwargs,
    ):
        return 0.0

    # ========================================================
    # SAFE SAMPLING
    # ========================================================

    def _safe_sample(
        self,
        n_samples,
        k=16,
        temperature=0.8,
        max_length=256,
        device=None,
        max_rounds=100,
    ):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        tabula_start = self.core._get_start_sampler(
            None,
            None,
        )

        self.core.model.to(device)

        self.core.model.eval()

        df_gen = pd.DataFrame(columns=self.core.columns)

        pad_id = int(self.core.tokenizer.eos_token_id)

        rounds = 0

        while len(df_gen) < n_samples and rounds < max_rounds:
            rounds += 1

            current_n = min(
                k,
                n_samples - len(df_gen),
            )

            start_tokens = tabula_start.get_start_tokens(current_n)

            input_ids = torch.tensor(
                start_tokens,
                dtype=torch.long,
                device=device,
            )

            # Fix the GPT-2 attention mask warning.
            attention_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                generated = self.core.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=pad_id,
                )

            # ------------------------------------------------
            # DEBUG OUTPUT
            # ------------------------------------------------

            print("\n" + "=" * 60)
            print("RAW TABULA GENERATION")
            print("=" * 60)

            raw_text = self.core.tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )

            for i, row in enumerate(raw_text):
                print(f"\nGenerated row {i + 1}:")
                print(row)

            # ------------------------------------------------
            # Convert tokens to text
            # ------------------------------------------------

            text = _convert_tokens_to_text(
                generated,
                self.core.tokenizer,
            )

            # ------------------------------------------------
            # Convert text into DataFrame
            # ------------------------------------------------

            df_gen = _convert_text_to_tabular_data(
                text,
                df_gen,
            )

            # ------------------------------------------------
            # Clean numeric columns
            # ------------------------------------------------

            for col in self.core.num_cols:
                if col in df_gen.columns:
                    df_gen[col] = pd.to_numeric(
                        df_gen[col],
                        errors="coerce",
                    )

            # ------------------------------------------------
            # Remove invalid rows
            # ------------------------------------------------

            df_gen = df_gen.dropna(how="any")

            if len(df_gen) > n_samples:
                df_gen = df_gen.head(n_samples)

        if len(df_gen) < n_samples:
            logger.warning(
                "Only generated %d/%d valid rows",
                len(df_gen),
                n_samples,
            )

        if self.core.categorical_columns:
            df_gen = self.core.decode_categorical_column(df_gen)

        return df_gen.head(n_samples)

    # ========================================================
    # COLUMN SANITISATION
    # ========================================================

    @staticmethod
    def _sanitize_columns(
        df,
    ):

        df = df.copy()

        df.columns = [
            str(col).replace(
                " ",
                "_",
            )
            for col in df.columns
        ]

        return df

    # ========================================================
    # MAXIMUM TOKEN LENGTH ESTIMATION
    # ========================================================

    def _estimate_max_length(
        self,
        df,
    ):

        sample_df = df.sample(
            min(
                len(df),
                256,
            ),
            random_state=42,
        )

        lengths = []

        for _, row in sample_df.iterrows():
            text = ", ".join(
                [f"{col} {str(value).strip()}" for col, value in row.items()]
            )

            tokens = self.core.tokenizer(text)["input_ids"]

            lengths.append(len(tokens))

        if not lengths:
            return 128

        estimated = int(
            np.percentile(
                lengths,
                95,
            )
            + 8
        )

        return max(
            128,
            min(
                estimated,
                512,
            ),
        )
