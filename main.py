from __future__ import annotations

import argparse
import copy
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

from models import get_model
from trainer import train_model
from utils.dataloader import get_dataloaders

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent


PRESET_REGISTRY = {
    "ablation_t1": {
        "runtime": {"anchor": "primary", "steps": 1, "update_mode": "guided", "protocol": "default"},
        "trainer": {"epochs": 50, "lr": 1e-4, "weight_decay": 1e-2, "loss": "dice", "optimizer": "adamw", "scheduler": "cosine", "use_amp": False},
        "requires_host_checkpoint": True,
    },
    "ablation_t2": {
        "runtime": {"anchor": "primary", "steps": 2, "update_mode": "guided", "protocol": "default"},
        "trainer": {"epochs": 50, "lr": 1e-4, "weight_decay": 1e-2, "loss": "dice", "optimizer": "adamw", "scheduler": "cosine", "use_amp": False},
        "requires_host_checkpoint": True,
    },
    "paper_main": {
        "runtime": {"anchor": "primary", "steps": 3, "update_mode": "guided", "protocol": "default"},
        "trainer": {"epochs": 50, "lr": 1e-4, "weight_decay": 1e-2, "loss": "dice", "optimizer": "adamw", "scheduler": "cosine", "use_amp": False},
        "requires_host_checkpoint": True,
    },
    "ablation_no_risk_update": {
        "runtime": {"anchor": "primary", "steps": 3, "update_mode": "no_risk", "protocol": "default"},
        "trainer": {"epochs": 50, "lr": 1e-4, "weight_decay": 1e-2, "loss": "dice", "optimizer": "adamw", "scheduler": "cosine", "use_amp": False},
        "requires_host_checkpoint": True,
    },
    "paper_alt_protocol": {
        "runtime": {"anchor": "secondary", "steps": 3, "update_mode": "guided", "protocol": "alt"},
        "trainer": {"epochs": 50, "lr": 1e-4, "weight_decay": 1e-2, "loss": "dice", "optimizer": "adamw", "scheduler": "cosine", "use_amp": False},
        "requires_host_checkpoint": True,
    },
}


def _add_bool_optional_flag(parser: argparse.ArgumentParser, name: str, default=None):
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(name, action=argparse.BooleanOptionalAction, default=default)
        return
    dest = str(name).lstrip("-").replace("-", "_")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(name, dest=dest, action="store_true")
    group.add_argument(f"--no-{str(name).lstrip('-')}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def _expand_model_entry(runtime_cfg: dict) -> dict:
    from models import build_refiner_params

    return {
        "name": "paper_refiner",
        "params": build_refiner_params(
            anchor=str(runtime_cfg.get("anchor", "primary")),
            recursion_steps=int(runtime_cfg.get("steps", 3)),
            update_mode=str(runtime_cfg.get("update_mode", "guided")),
            protocol=str(runtime_cfg.get("protocol", "default")),
        ),
    }


def _extract_state_dict(payload):
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict"):
            state_dict = payload.get(key)
            if isinstance(state_dict, dict):
                return state_dict
        if all(torch.is_tensor(v) for v in payload.values()):
            return payload
    raise RuntimeError(f"Unsupported checkpoint payload type: {type(payload)!r}")



def _resolve_module_by_name(model, module_name: str):
    module = model
    for part in str(module_name).split("."):
        if not part:
            continue
        if not hasattr(module, part):
            raise AttributeError(f"Module '{type(model).__name__}' has no submodule '{module_name}'")
        module = getattr(module, part)
    return module



def _load_submodule_from_checkpoint(model, checkpoint_path: str, module_name: str = "host", min_matches: int = 1):
    target_module = _resolve_module_by_name(model, module_name)
    target_state = target_module.state_dict()
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(payload)

    candidate_maps = [("", state_dict)]
    prefix = f"{module_name}."
    prefixed = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    if prefixed:
        candidate_maps.append((prefix, prefixed))
    if any(k.startswith("module.") for k in state_dict):
        no_dp = {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}
        candidate_maps.append(("module.", no_dp))
        prefixed_no_dp = {
            k[len(f"module.{prefix}"):]: v
            for k, v in state_dict.items()
            if k.startswith(f"module.{prefix}")
        }
        if prefixed_no_dp:
            candidate_maps.append((f"module.{prefix}", prefixed_no_dp))

    best_prefix = ""
    best_state = {}
    best_matches = -1
    for used_prefix, candidate in candidate_maps:
        matched = {
            key: value
            for key, value in candidate.items()
            if key in target_state and target_state[key].shape == value.shape
        }
        if len(matched) > best_matches:
            best_matches = len(matched)
            best_prefix = used_prefix
            best_state = matched

    if best_matches < int(min_matches):
        raise RuntimeError(
            f"Checkpoint match count too low for module '{module_name}': matched={best_matches}, min_matches={min_matches}"
        )

    load_result = target_module.load_state_dict(best_state, strict=False)
    report = {
        "prefix": best_prefix,
        "matched": int(best_matches),
        "missing": len(load_result.missing_keys),
        "unexpected": len(load_result.unexpected_keys),
        "checkpoint": str(checkpoint_path),
    }
    LOGGER.info(
        "Loaded checkpoint into submodule '%s'. prefix='%s' matched=%d missing=%d unexpected=%d checkpoint=%s",
        module_name,
        best_prefix,
        report["matched"],
        report["missing"],
        report["unexpected"],
        checkpoint_path,
    )
    return report



def _freeze_module_by_name(model, module_name: str):
    target_module = _resolve_module_by_name(model, module_name)
    for param in target_module.parameters():
        param.requires_grad = False
    target_module.eval()
    LOGGER.info("Froze module '%s'", module_name)



def _set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



def _resolve_device_string(args) -> str:
    if args.force_cpu:
        return "cpu"
    if args.device:
        return str(args.device)
    return "cuda" if torch.cuda.is_available() else "cpu"



def _build_runtime_config(args) -> dict:
    preset = copy.deepcopy(PRESET_REGISTRY[args.preset])
    seed = int(args.seed)
    split_seed = int(args.split_seed if args.split_seed is not None else seed)
    image_size = int(args.image_size)
    output_root = Path(args.output_dir) if args.output_dir else Path("runs")
    run_name = str(args.run_name or f"{args.preset}_seed{seed}")
    log_dir = output_root / "logs" / run_name
    ckpt_dir = output_root / "checkpoints"

    data_cfg = {
        "root": str(Path(args.data_root)),
        "dataset": str(args.dataset).lower(),
        "batch_size": int(args.batch_size),
        "val_batch_size": int(args.val_batch_size or args.batch_size),
        "num_workers": int(args.num_workers),
        "split_seed": split_seed,
        "loader_seed": seed,
        "target_size": (image_size, image_size),
        "image_size": image_size,
        "direct_raw_rgb_loading": True,
        "force_rgb_input": True,
        "mask_threshold": float(args.mask_threshold),
        "aug_profile": str(args.aug_profile),
        "normalize_mode": str(args.normalize_mode),
        "light_mask_prebinarize": True,
    }

    trainer_cfg = preset["trainer"]
    trainer_cfg.update(
        {
            "epochs": int(args.epochs or trainer_cfg.get("epochs", 1)),
            "lr": float(args.lr or trainer_cfg.get("lr", 1e-4)),
            "optimizer": str(args.optimizer or trainer_cfg.get("optimizer", "adamw")),
            "scheduler": str(args.scheduler or trainer_cfg.get("scheduler", "none")),
            "weight_decay": float(args.weight_decay if args.weight_decay is not None else trainer_cfg.get("weight_decay", 1e-2)),
            "loss": str(args.loss or trainer_cfg.get("loss", "dice")),
            "device": _resolve_device_string(args),
            "run_name": run_name,
            "log_dir": str(log_dir),
            "best_path": str(ckpt_dir / f"{run_name}.pth"),
            "last_path": str(ckpt_dir / f"{run_name}.last.pth"),
            "disable_progress": bool(args.no_progress),
            "metric_threshold": float(args.metric_threshold),
            "checkpoint_metric": "val_dice",
            "checkpoint_mode": "max",
            "use_amp": trainer_cfg.get("use_amp", False) if args.use_amp is None else bool(args.use_amp),
            "seed": seed,
        }
    )

    resolved = {
        "preset": args.preset,
        "data": data_cfg,
        "model": _expand_model_entry(preset["runtime"]),
        "trainer": trainer_cfg,
        "requires_host_checkpoint": bool(preset.get("requires_host_checkpoint", False)),
    }
    return resolved



def _instantiate_model(resolved_cfg: dict, args):
    model_cfg = resolved_cfg["model"]
    model = get_model(model_cfg["name"], **copy.deepcopy(model_cfg.get("params", {})))

    if resolved_cfg.get("requires_host_checkpoint"):
        if not args.host_checkpoint:
            raise RuntimeError(f"Preset '{args.preset}' requires --host-checkpoint.")
        _load_submodule_from_checkpoint(
            model,
            checkpoint_path=str(Path(args.host_checkpoint).resolve()),
            module_name="host",
            min_matches=1,
        )
        if bool(model_cfg.get("params", {}).get("freeze_host", True)):
            _freeze_module_by_name(model, "host")
    return model



def _write_resolved_run_spec(resolved_cfg: dict):
    log_dir = Path(resolved_cfg["trainer"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    spec_path = log_dir / "run_spec.json"
    spec_path.write_text(json.dumps(resolved_cfg, indent=2), encoding="utf-8")



def main():
    parser = argparse.ArgumentParser(description="APSIPA RIGS-Refiner minimal training entry")
    parser.add_argument("--preset", required=True, choices=sorted(PRESET_REGISTRY.keys()))
    parser.add_argument("--data-root", required=True, help="Dataset root. Expected layout: train|val/{images,masks}.")
    parser.add_argument("--dataset", default="kvasirseg")
    parser.add_argument("--host-checkpoint", help="Baseline host checkpoint path. Required for refiner presets.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "runs"))
    parser.add_argument("--run-name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--image-size", type=int, default=352)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--optimizer")
    parser.add_argument("--loss")
    parser.add_argument("--scheduler")
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--device")
    parser.add_argument("--force-cpu", action="store_true")
    _add_bool_optional_flag(parser, "--use-amp", default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--metric-threshold", type=float, default=0.1)
    parser.add_argument("--mask-threshold", type=float, default=0.1)
    parser.add_argument("--aug-profile", default="superlight_default")
    parser.add_argument("--normalize-mode", default="instance_norm")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    )

    resolved_cfg = _build_runtime_config(args)
    _set_global_seed(int(resolved_cfg["trainer"]["seed"]))
    if args.dry_run:
        print(json.dumps(resolved_cfg, indent=2))
        return

    _write_resolved_run_spec(resolved_cfg)

    dataloaders = get_dataloaders(resolved_cfg["data"])
    model = _instantiate_model(resolved_cfg, args)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info("Model initialized: %s", resolved_cfg["model"]["name"])
    LOGGER.info("Total Parameters: %s", f"{total_params:,}")
    LOGGER.info("Trainable Parameters: %s", f"{trainable_params:,}")
    LOGGER.info("Run preset: %s", args.preset)

    train_model(model, dataloaders, resolved_cfg["trainer"], task="seg")


if __name__ == "__main__":
    main()
