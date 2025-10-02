# driver.py
from pathlib import Path
import numpy as np
import tempfile, shutil, os, sys, runpy

class MedganCLI:
    def __init__(self, repo_root=".", python_exec="python"):
        self.repo_root = Path(repo_root)
        self.python_exec = python_exec
        self.medgan_py = Path("medgan.py")

    def sample(
        self,
        matrix_path: str,
        out_path: str,
        model_file: str,          # checkpoint *prefix* (e.g., runs/adult_medgan-19)
        n_samples: int = None,    # optional; None = keep input size
        data_type: str = "binary",
    ) -> str:
        """Call medgan.py's argparse main in-process (no subprocess)."""
        # Resolve absolute paths
        repo = self.repo_root.resolve()
        medgan_py = (repo / self.medgan_py).resolve()
        in_mat = Path(matrix_path).resolve()
        out_p = Path(out_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)
        ckpt = Path(model_file).resolve()

        # Optional slice/repeat to exactly n_samples
        use_mat = in_mat
        temp_dir = None
        if n_samples is not None:
            X = np.load(in_mat, allow_pickle=True)
            if n_samples != X.shape[0]:
                temp_dir = Path(tempfile.mkdtemp(prefix="medgan_sample_"))
                tmp_mat = temp_dir / "matrix.npy"
                if n_samples <= X.shape[0]:
                    X_out = X[:n_samples]
                else:
                    reps = int(np.ceil(n_samples / X.shape[0]))
                    X_out = np.vstack([X] * reps)[:n_samples]
                np.save(tmp_mat, X_out, allow_pickle=True)
                use_mat = tmp_mat.resolve()

        # Build argv exactly as medgan.py expects
        argv = [
            str(medgan_py),
            str(use_mat),
            str(out_p),
            "--model_file", str(ckpt),          # NOTE: prefix, no extension
            "--generate_data", "True",
            "--data_type", data_type,
        ]

        # Run medgan.py __main__ with those args and correct cwd
        old_argv, old_cwd = sys.argv[:], os.getcwd()
        try:
            sys.argv = argv
            os.chdir(str(repo))
            runpy.run_path(str(medgan_py), run_name="__main__")
        finally:
            sys.argv = old_argv
            os.chdir(old_cwd)
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)

        # If medgan.py didn’t raise, the file should now exist
        if not out_p.exists():
            raise RuntimeError(f"Sampling finished but output not found: {out_p}")
        return str(out_p)
