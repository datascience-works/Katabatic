import numpy as np
import torch
from tabpfn import TabPFNClassifier, TabPFNRegressor
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="sklearn.preprocessing._encoders"
)


class TabPFGen:
    """
    TabPFGen: SGLD-based tabular synthetic data generator guided by TabPFN.
    Supports classification + regression generation.

    Note:
    - This is intended to be used by Katabatic's wrapper model (model.py).
    - Keep this file ONLY for the generator core logic.
    """

    def __init__(
        self,
        n_sgld_steps: int = 1000,
        sgld_step_size: float = 0.01,
        sgld_noise_scale: float = 0.01,
        device: str = "auto",
    ):
        self.n_sgld_steps = n_sgld_steps
        self.sgld_step_size = sgld_step_size
        self.sgld_noise_scale = sgld_noise_scale
        self.scaler = StandardScaler()
        self.device = self._infer_device(device)

    def _infer_device(self, device: str | torch.device | None) -> torch.device:
        if (device is None) or (isinstance(device, str) and device == "auto"):
            device_type_ = "cuda" if torch.cuda.is_available() else "cpu"
            return torch.device(device_type_)
        if isinstance(device, str):
            return torch.device(device)
        if isinstance(device, torch.device):
            return device
        raise ValueError(f"Invalid device: {device}")

    def _compute_energy(
        self,
        x_synth: torch.Tensor,
        y_synth: torch.Tensor,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
    ) -> torch.Tensor:
        """
        Differentiable surrogate "energy" based on distance to training points,
        with an additional class-conditional distance term.
        """
        distances = torch.cdist(x_synth, x_train)
        min_distances, _ = distances.min(dim=1)

        class_mask = y_synth.unsqueeze(1) == y_train.unsqueeze(0)
        class_distances = distances * class_mask.float()
        class_distances = class_distances.sum(dim=1) / (class_mask.float().sum(dim=1) + 1e-6)

        energy = min_distances + class_distances
        return energy

    def _sgld_step(
        self,
        x_synth: torch.Tensor,
        y_synth: torch.Tensor,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
    ) -> torch.Tensor:
        """
        One SGLD update step: x <- x - step_size * grad(E) + noise
        """
        x_synth = x_synth.clone().detach().requires_grad_(True)

        energy = self._compute_energy(x_synth, y_synth, x_train, y_train)
        energy_sum = energy.sum()

        grad = torch.autograd.grad(
            energy_sum,
            x_synth,
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )[0]

        if grad is None:
            grad = torch.zeros_like(x_synth)

        noise = torch.randn_like(x_synth) * np.sqrt(2 * self.sgld_step_size)
        x_synth_new = x_synth - self.sgld_step_size * grad + self.sgld_noise_scale * noise
        return x_synth_new

    def _generate_samples_for_class(
        self,
        class_label: int,
        n_samples: int,
        x_train_scaled: torch.Tensor,
        y_train: torch.Tensor,
        X_train_scaled: np.ndarray,
    ) -> torch.Tensor:
        class_indices = torch.where(y_train == class_label)[0]

        sample_indices = torch.randint(0, len(class_indices), (n_samples,))
        selected_indices = class_indices[sample_indices]

        x_synth = (
            x_train_scaled[selected_indices]
            + torch.randn(n_samples, X_train_scaled.shape[1], device=self.device) * 0.01
        )
        y_synth = torch.full((n_samples,), class_label, device=self.device)

        for step in range(self.n_sgld_steps):
            x_synth = self._sgld_step(x_synth, y_synth, x_train_scaled, y_train)
            if step % 200 == 0:
                print(f"  Class {class_label}: Step {step}/{self.n_sgld_steps}")

        return x_synth

    def balance_dataset(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        target_per_class: Optional[int] = None,
        min_class_size: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same number of samples")

        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        class_distribution = dict(zip(unique_classes, class_counts))
        max_class_size = int(max(class_counts))

        if target_per_class is None:
            target_size = max_class_size
        else:
            if target_per_class < max_class_size:
                raise ValueError(
                    f"target_per_class ({target_per_class}) must be >= largest class size ({max_class_size})"
                )
            target_size = int(target_per_class)

        print("=== Dataset Balancing Statistics ===")
        print("Original class distribution:")
        for cls, count in class_distribution.items():
            print(f"  Class {cls}: {count} samples")
        print(f"Target samples per class: {target_size}")
        print(f"Minimum class size threshold: {min_class_size}")

        valid_classes = []
        skipped_classes = []
        synthetic_needed = {}

        for cls, count in class_distribution.items():
            if count < min_class_size:
                skipped_classes.append((cls, count))
                print(f"Warning: Skipping class {cls} (only {count} samples, below threshold {min_class_size})")
            else:
                valid_classes.append(cls)
                synthetic_needed[cls] = max(0, target_size - count)

        if not valid_classes:
            raise ValueError("No classes meet the minimum size requirement")

        total_synthetic = sum(synthetic_needed.values())
        print("\nSynthetic samples to generate:")
        for cls in valid_classes:
            print(f"  Class {cls}: {synthetic_needed[cls]} synthetic samples")

        if total_synthetic == 0:
            print("No synthetic samples needed - dataset is already balanced!")
            return (
                np.array([]).reshape(0, X_train.shape[1]),
                np.array([]),
                X_train.copy(),
                y_train.copy(),
            )

        X_scaled = self.scaler.fit_transform(X_train)

        x_train = torch.tensor(X_scaled, device=self.device, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, device=self.device)

        print(f"\nGenerating {total_synthetic} synthetic samples...")
        x_synthetic_list = []
        y_synthetic_list = []

        for cls in valid_classes:
            n_needed = synthetic_needed[cls]
            if n_needed > 0:
                print(f"\nGenerating {n_needed} samples for class {cls}...")
                x_synth_class = self._generate_samples_for_class(cls, n_needed, x_train, y_train_tensor, X_scaled)
                y_synth_class = torch.full((n_needed,), cls, device=self.device)
                x_synthetic_list.append(x_synth_class)
                y_synthetic_list.append(y_synth_class)

        if x_synthetic_list:
            x_synthetic_combined = torch.cat(x_synthetic_list, dim=0)

            print("Refining synthetic sample labels with TabPFN...")
            x_synthetic_np = x_synthetic_combined.detach().cpu().numpy()

            valid_mask = np.isin(y_train, valid_classes)
            X_train_valid = X_scaled[valid_mask]
            y_train_valid = y_train[valid_mask]

            # Use device.type for compatibility ("cpu"/"cuda")
            clf = TabPFNClassifier(device=self.device.type)
            clf.fit(X_train_valid, y_train_valid)

            probs = clf.predict_proba(x_synthetic_np)
            y_synthetic_refined = probs.argmax(axis=1)

            X_synthetic = self.scaler.inverse_transform(x_synthetic_np)

            unique_valid_classes = np.unique(y_train_valid)
            y_synthetic_mapped = unique_valid_classes[y_synthetic_refined]
        else:
            X_synthetic = np.array([]).reshape(0, X_train.shape[1])
            y_synthetic_mapped = np.array([])

        X_combined = np.vstack([X_train, X_synthetic]) if len(X_synthetic) > 0 else X_train.copy()
        y_combined = np.concatenate([y_train, y_synthetic_mapped]) if len(y_synthetic_mapped) > 0 else y_train.copy()

        print("\n=== Final Statistics ===")
        final_unique, final_counts = np.unique(y_combined, return_counts=True)
        final_distribution = dict(zip(final_unique, final_counts))

        print("Final combined class distribution:")
        for cls in sorted(final_distribution.keys()):
            count = final_distribution[cls]
            original_count = class_distribution.get(cls, 0)
            synthetic_count = count - original_count
            print(f"  Class {cls}: {count} total ({original_count} original + {synthetic_count} synthetic)")

        if skipped_classes:
            print(f"\nSkipped classes: {[cls for cls, _ in skipped_classes]}")

        print("Dataset balancing completed!\n")
        print("Note: approximate balance that preserves data quality.\n")

        return X_synthetic, y_synthetic_mapped, X_combined, y_combined

    def generate_classification(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_samples: int,
        balance_classes: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_scaled = self.scaler.fit_transform(X_train)

        x_train = torch.tensor(X_scaled, device=self.device, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, device=self.device)

        if balance_classes:
            classes = np.unique(y_train)
            n_per_class = n_samples // len(classes)

            x_synth_list = []
            y_synth_list = []
            for cls in classes:
                idx = np.where(y_train == cls)[0]
                sample_idx = np.random.choice(idx, size=n_per_class, replace=True)
                x_init = x_train[sample_idx] + torch.randn(n_per_class, X_train.shape[1], device=self.device) * 0.01
                y_init = torch.full((n_per_class,), int(cls), device=self.device)
                x_synth_list.append(x_init)
                y_synth_list.append(y_init)

            x_synth = torch.cat(x_synth_list, dim=0)
            y_synth = torch.cat(y_synth_list, dim=0)
        else:
            x_synth = torch.randn(n_samples, X_train.shape[1], device=self.device) * 0.01
            y_synth = torch.randint(0, len(np.unique(y_train)), (n_samples,), device=self.device)

        for step in range(self.n_sgld_steps):
            x_synth = self._sgld_step(x_synth, y_synth, x_train, y_train_t)
            if step % 100 == 0:
                print(f"Step {step}/{self.n_sgld_steps}")

        x_synth_np = x_synth.detach().cpu().numpy()
        x_train_np = x_train.detach().cpu().numpy()
        y_train_np = y_train_t.detach().cpu().numpy()

        clf = TabPFNClassifier(device=self.device.type)
        clf.fit(x_train_np, y_train_np)
        probs = clf.predict_proba(x_synth_np)

        y_synth_np = probs.argmax(axis=1)
        X_synth = self.scaler.inverse_transform(x_synth_np)

        return X_synth, y_synth_np

    def generate_regression(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_samples: int,
        use_quantiles: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        regressor = TabPFNRegressor(device=self.device.type)

        X_scaled = self.scaler.fit_transform(X_train)
        y_mean, y_std = y_train.mean(), y_train.std()
        y_scaled = (y_train - y_mean) / (y_std + 1e-12)

        x_train = torch.tensor(X_scaled, device=self.device, dtype=torch.float32)

        n_strata = 10
        y_strata = np.quantile(y_train, np.linspace(0, 1, n_strata + 1))
        x_synth_list = []
        samples_per_stratum = max(1, n_samples // n_strata)

        for i in range(n_strata):
            mask = (y_train >= y_strata[i]) & (y_train <= y_strata[i + 1])
            stratum_indices = np.where(mask)[0]

            if len(stratum_indices) > 0:
                sampled_indices = np.random.choice(stratum_indices, size=samples_per_stratum, replace=True)
                x_stratum = X_scaled[sampled_indices]
                stratum_std = np.std(x_stratum, axis=0)
                noise = np.random.normal(0, stratum_std * 0.1, (samples_per_stratum, X_train.shape[1]))
                x_synth_list.append(x_stratum + noise)

        x_synth_init = np.vstack(x_synth_list)[:n_samples]
        x_synth = torch.tensor(x_synth_init, device=self.device, dtype=torch.float32)

        adaptive_step_size = self.sgld_step_size
        for step in range(self.n_sgld_steps):
            if step % 100 == 0 and step > 0:
                adaptive_step_size *= 0.9

            # Use same step logic; SGLD uses gradient of energy proxy
            self.sgld_step_size = adaptive_step_size
            x_synth = self._sgld_step(
                x_synth,
                torch.zeros(len(x_synth), device=self.device),
                x_train,
                torch.zeros(len(x_train), device=self.device),
            )

            if step % 100 == 0:
                print(f"Step {step}/{self.n_sgld_steps}")

        x_synth_np = x_synth.detach().cpu().numpy()

        try:
            regressor.fit(X_scaled, y_scaled)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                preds = regressor.predict(x_synth_np, output_type="full")

            if use_quantiles and "quantiles" in preds:
                quantiles = preds["quantiles"]
                if not isinstance(quantiles, list):
                    quantiles = [quantiles]
                n_q = len(quantiles)
                probs = np.abs(np.random.normal(0, 0.5, size=len(x_synth_np)))
                q_idx = (probs * n_q).astype(int).clip(0, n_q - 1)
                y_synth = np.array([quantiles[i][j] for j, i in enumerate(q_idx)])
            else:
                y_synth = np.array(preds.get("median", preds))

            y_synth += np.random.normal(0, 0.01, size=len(y_synth))
        except Exception as e:
            print(f"Warning: regression prediction failed: {e}")
            y_synth = np.random.normal(y_mean, y_std, size=len(x_synth_np))

        X_synth = self.scaler.inverse_transform(x_synth_np)
        y_synth = y_synth * y_std + y_mean

        return X_synth, y_synth
