from __future__ import annotations

from typing import Callable

import torch.nn as nn

from model.post_rigs_refiner import HostWithRIGSRefiner
from model.internal_hosts import InternalAnchorA, InternalAnchorB


def _build_anchor_primary(**kwargs) -> nn.Module:
    params = dict(kwargs)
    params.pop("in_channels", None)
    return InternalAnchorA(**params)


def _build_anchor_secondary(**kwargs) -> nn.Module:
    return InternalAnchorB(**dict(kwargs))


def _build_refiner_runtime(**kwargs) -> nn.Module:
    params = dict(kwargs)
    anchor_name = str(params.pop("host_model_name"))
    anchor_kwargs = dict(params.pop("host_model_kwargs", {}) or {})
    host = get_model(anchor_name, **anchor_kwargs)
    return HostWithRIGSRefiner(host=host, host_model_name=anchor_name, **params)


def build_refiner_params(anchor: str, recursion_steps: int, update_mode: str = "guided", protocol: str = "default") -> dict:
    params = {
        "host_model_name": "anchor_primary" if str(anchor) == "primary" else "anchor_secondary",
        "freeze_host": True,
        "joint_finetune": False,
        "method_variant": "M3",
        "prior_family": "P0",
        "mixer_family": "C1",
        "state_form": "N0",
        "update_rule": "U2",
        "sparse_policy": "S0",
        "budget_policy": "B0",
        "plugin_channels": 4,
        "hidden_channels": 8,
        "recursion_steps": int(recursion_steps),
        "recurrence_share_weights": True,
    }
    if str(update_mode) == "no_risk":
        params["update_rule"] = "U1"
    if str(protocol) == "alt":
        params["host_model_name"] = "anchor_secondary"
        params["mixer_family"] = "C2"
    return params


_PUBLIC_MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "paper_refiner": _build_refiner_runtime,
}

_INTERNAL_HOST_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "anchor_primary": _build_anchor_primary,
    "anchor_secondary": _build_anchor_secondary,
}


def available_models() -> list[str]:
    return sorted(_PUBLIC_MODEL_REGISTRY)


def get_model(name: str, **kwargs) -> nn.Module:
    key = str(name)
    builder = _PUBLIC_MODEL_REGISTRY.get(key)
    if builder is None:
        builder = _INTERNAL_HOST_REGISTRY.get(key)
    if builder is None:
        raise ValueError(f"Unsupported model '{name}'. This release exposes only: {', '.join(available_models())}")
    return builder(**kwargs)
