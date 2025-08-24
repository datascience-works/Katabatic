import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
import time
import torch
import lib
from typing import Optional, Dict, Any

from .train import train
from .sample import sample
from .eval_catboost import train_catboost
from .eval_mlp import train_mlp
from .eval_simple import train_simple


# ---------- helpers ----------
def _pick_device(cfg):
    if "device" in cfg:
        return torch.device(cfg["device"])
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        shutil.copyfile(os.path.abspath(src), dst)
    except shutil.SameFileError:
        pass

def load_cfg(config_path: str):
    """Load TOML config via your lib.load_config."""
    return lib.load_config(config_path)

def _maybe_copy_side_files(cfg, config_path):
    # Save an immutable copy of the config next to outputs
    _copy(config_path, os.path.join(cfg["parent_dir"], "config.toml"))
    # Save info.json alongside outputs (matches your pipeline.py behavior)
    _copy(os.path.join(cfg["real_data_path"], "info.json"),
          os.path.join(cfg["parent_dir"], "info.json"))


# ---------- runners ----------
def run_train(cfg: dict, change_val: bool = False, device=None, main_overrides: Optional[Dict[str, Any]] = None):
    """Run training and return whatever train() returns + timing."""
    
    main_args = dict(cfg["train"]["main"])  
    if main_overrides:
        main_args.update(main_overrides)
        
    device = device or _pick_device(cfg)
    t0 = time.perf_counter()
    out = train(
        **main_args,
        **cfg["diffusion_params"],
        parent_dir=cfg["parent_dir"],
        real_data_path=cfg["real_data_path"],
        model_type=cfg["model_type"],
        model_params=cfg["model_params"],
        T_dict=cfg["train"]["T"],
        num_numerical_features=cfg["num_numerical_features"],
        device=device,
        change_val=change_val,
    )
    return {"result": out, "elapsed_sec": time.perf_counter() - t0, "device": str(device)}

def run_sample(
    cfg: Dict[str, Any],
    change_val: bool = False,
    device=None,
    sample_overrides: Optional[Dict[str, Any]] = None,
    diffusion_overrides: Optional[Dict[str, Any]] = None,
    model_overrides: Optional[Dict[str, Any]] = None,
    T_overrides: Optional[Dict[str, Any]] = None,
):

    # ---- merge args (non-destructive) ----
    sample_args = dict(cfg["sample"])
    if sample_overrides:
        sample_args.update(sample_overrides)
        
    diffusion_args = dict(cfg["diffusion_params"])
    if diffusion_overrides:
        diffusion_args.update(diffusion_overrides)

    model_params = dict(cfg["model_params"])
    if model_overrides:
        model_params.update(model_overrides)

    T_dict = dict(cfg["train"]["T"])
    if T_overrides:
        T_dict.update(T_overrides)

    device = device or _pick_device(cfg)
    t0 = time.perf_counter()
    out = sample(
        num_samples=sample_args["num_samples"],
        batch_size=sample_args["batch_size"],
        disbalance=sample_args.get("disbalance", None),
        **diffusion_args,
        parent_dir=cfg["parent_dir"],
        real_data_path=cfg["real_data_path"],
        model_path=os.path.join(cfg["parent_dir"], "model.pt"),
        model_type=cfg["model_type"],
        model_params=model_params,
        T_dict=T_dict,
        num_numerical_features=cfg["num_numerical_features"],
        device=device,
        seed=sample_args.get("seed", 0),
        change_val=change_val,
    )
    return {"result": out, "elapsed_sec": time.perf_counter() - t0, "device": str(device)}

def run_eval(cfg: dict, change_val: bool = False, device=None):
    """Run eval (catboost/mlp/simple) and return its result + timing."""
    device = device or _pick_device(cfg)
    t0 = time.perf_counter()
    eval_model = cfg["eval"]["type"]["eval_model"]

    if eval_model == "catboost":
        out = train_catboost(
            parent_dir=cfg["parent_dir"],
            real_data_path=cfg["real_data_path"],
            eval_type=cfg["eval"]["type"]["eval_type"],
            T_dict=cfg["eval"]["T"],
            seed=cfg["seed"],
            change_val=change_val,
        )
    elif eval_model == "mlp":
        out = train_mlp(
            parent_dir=cfg["parent_dir"],
            real_data_path=cfg["real_data_path"],
            eval_type=cfg["eval"]["type"]["eval_type"],
            T_dict=cfg["eval"]["T"],
            seed=cfg["seed"],
            change_val=change_val,
            device=device,
        )
    elif eval_model == "simple":
        out = train_simple(
            parent_dir=cfg["parent_dir"],
            real_data_path=cfg["real_data_path"],
            eval_type=cfg["eval"]["type"]["eval_type"],
            T_dict=cfg["eval"]["T"],
            seed=cfg["seed"],
            change_val=change_val,
        )
    else:
        raise ValueError(f"Unknown eval model: {eval_model}")

    return {"result": out, "elapsed_sec": time.perf_counter() - t0, "device": str(device)}

def run_all(config_path: str, do_train=True, do_sample=True, do_eval=True, change_val=False):
    cfg = load_cfg(config_path)
    _maybe_copy_side_files(cfg, config_path)
    device = _pick_device(cfg)

    results = {}
    if do_train:
        results["train"] = run_train(cfg, change_val=change_val, device=device)
    if do_sample:
        results["sample"] = run_sample(cfg, change_val=change_val, device=device)
    if do_eval:
        results["eval"] = run_eval(cfg, change_val=change_val, device=device)
    return results
