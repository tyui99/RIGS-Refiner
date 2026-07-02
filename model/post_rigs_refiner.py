from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _safe_logit(prob: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    prob = prob.clamp(eps, 1.0 - eps)
    return torch.log(prob) - torch.log1p(-prob)


def foreground_logit(output: torch.Tensor, kind: str) -> torch.Tensor:
    if output.ndim != 4 or output.shape[1] not in (1, 2):
        raise ValueError(f"Expected host output shape (B,1,H,W) or (B,2,H,W), got {tuple(output.shape)}")
    if output.shape[1] == 1:
        return _safe_logit(output) if kind == "prob" else output
    if kind == "prob":
        return _safe_logit(output[:, 1:2])
    return output[:, 1:2] - output[:, 0:1]


def binary_logits_from_foreground_logit(fg_logit: torch.Tensor) -> torch.Tensor:
    if fg_logit.ndim != 4 or fg_logit.shape[1] != 1:
        raise ValueError(f"Expected foreground logit shape (B,1,H,W), got {tuple(fg_logit.shape)}")
    return torch.cat([-0.5 * fg_logit, 0.5 * fg_logit], dim=1)


def tensor_stat(value: torch.Tensor, mode: str = "mean") -> float:
    detached = value.detach()
    if detached.numel() == 0:
        return 0.0
    if mode == "mean":
        return float(detached.mean().item())
    if mode == "std":
        return float(detached.std().item())
    if mode == "max":
        return float(detached.max().item())
    if mode == "min":
        return float(detached.min().item())
    raise ValueError(f"Unsupported tensor_stat mode: {mode}")


def _freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _set_module_trainable(module: nn.Module, trainable: bool) -> None:
    for param in module.parameters():
        param.requires_grad = bool(trainable)


class FrozenHostPostRefinerBase(nn.Module):
    def __init__(
        self,
        host: nn.Module,
        host_output_kind: str = "auto",
        freeze_host: bool = True,
        joint_finetune: bool = False,
    ):
        super().__init__()
        self.host = host
        self.host_output_kind = str(host_output_kind).strip().lower()
        self.freeze_host = bool(freeze_host)
        self.joint_finetune = bool(joint_finetune)
        self.output_kind = "logits"
        self.last_aux: dict[str, Any] = {}
        self.par_loss: torch.Tensor | None = None
        if self.freeze_host and not self.joint_finetune:
            _freeze_module(self.host)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_host and not self.joint_finetune:
            self.host.eval()
            _freeze_module(self.host)
        else:
            self.host.train(mode)
            _set_module_trainable(self.host, True)
        return self

    def _host_forward(self, x: torch.Tensor):
        if self.freeze_host and not self.joint_finetune:
            with torch.no_grad():
                return self.host(x)
        return self.host(x)


def _load_package_module(module_name: str, module_path: Path, package_root: Path):
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(module_path),
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load package module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


_RIGS_ROOT = Path(__file__).resolve().parent / "RIGS-Refiner"
_RIGS_PKG = _load_package_module(
    "model.rigs_refiner_pkg",
    _RIGS_ROOT / "__init__.py",
    _RIGS_ROOT,
)

ExperimentConfig = _RIGS_PKG.ExperimentConfig
HostConfig = _RIGS_PKG.HostConfig
PluginConfig = _RIGS_PKG.PluginConfig
SparseConfig = _RIGS_PKG.SparseConfig
StateAlignmentConfig = _RIGS_PKG.StateAlignmentConfig
TrainConfig = _RIGS_PKG.TrainConfig
UpdateRuleConfig = _RIGS_PKG.UpdateRuleConfig
EvidenceLevelConfig = _RIGS_PKG.EvidenceLevelConfig
BudgetPolicyConfig = _RIGS_PKG.BudgetPolicyConfig
build_host_adapter = _RIGS_PKG.build_host_adapter
RefinementPlugin = _RIGS_PKG.RefinementPlugin


class HostWithRIGSRefiner(FrozenHostPostRefinerBase):
    @staticmethod
    def _legacy_evidence_level(method_variant: str) -> str:
        norm = str(method_variant or "M3").upper()
        if norm == "M1":
            return "E0"
        if norm == "M2":
            return "E1"
        if norm == "M6GEOM":
            return "E3"
        return "E2"

    @staticmethod
    def _budget_policy_from_sparse(sparse_policy: str) -> str:
        mapping = {
            "S0": "B0",
            "S1": "B2",
            "S2": "B4",
            "S3": "B5",
        }
        return mapping.get(str(sparse_policy or "S0").upper(), str(sparse_policy or "B0").upper())

    @staticmethod
    def _recursion_steps_from_tier(recursion_tier: str, recursion_steps: int | None = None) -> int:
        if recursion_steps is not None and int(recursion_steps) > 0:
            return int(recursion_steps)
        return {
            "K0": 1,
            "K1": 2,
            "K2": 3,
        }.get(str(recursion_tier or "K0").upper(), 1)

    def __init__(
        self,
        host: nn.Module,
        host_model_name: str = "unknown",
        host_output_kind: str = "auto",
        freeze_host: bool = True,
        joint_finetune: bool = False,
        method_variant: str = "M3",
        prior_family: str = "P0",
        mixer_family: str = "C0",
        capacity_tier: str = "R0",
        sparse_policy: str = "S0",
        sparse_topk_ratio: float = 0.10,
        plugin_channels: int = 4,
        hidden_channels: int = 8,
        recursion_tier: str = "K0",
        recursion_steps: int | None = None,
        recurrence_share_weights: bool = True,
        update_alpha: float = 1.0,
        use_geometry: bool = False,
        state_form: str = "N0",
        update_rule: str = "U2",
        evidence_level: str | None = None,
        budget_policy: str | None = None,
        temperature_mode: str = "fixed",
        temperature_init: float = 1.0,
        temperature_min: float = 0.25,
        temperature_max: float = 4.0,
        standardize_scope: str = "per_image",
        standardize_eps: float = 1.0e-6,
        dynamic_alpha_hidden: int = 4,
        alpha_min: float = 0.0,
        alpha_max: float = 2.0,
        sparsity_weight: float = 0.0,
        delta_weight: float = 0.0,
        risk_bias: float = 0.0,
        strict_evidence_contract: bool = True,
        disable_image_priors_for_correction: bool = False,
        edge_operator: str = "sobel",
        lr: float = 1e-4,
        epochs: int = 50,
        loss_name: str = "dice",
    ):
        super().__init__(
            host=host,
            host_output_kind=host_output_kind,
            freeze_host=freeze_host,
            joint_finetune=joint_finetune,
        )
        self.host_model_name = str(host_model_name)
        self.host_adapter = build_host_adapter(host, host_output_kind=host_output_kind)
        self.state_form = str(state_form or "N0").upper()
        self.update_rule = str(update_rule or "U2").upper()
        self.evidence_level = str(evidence_level or self._legacy_evidence_level(method_variant)).upper()
        self.budget_policy = str(budget_policy or self._budget_policy_from_sparse(sparse_policy)).upper()
        self.capacity_tier = str(capacity_tier or "R0").upper()
        self.recursion_tier = str(recursion_tier or "K0").upper()
        self.recursion_steps = self._recursion_steps_from_tier(self.recursion_tier, recursion_steps)
        self.temperature_mode = str(temperature_mode or "fixed").lower()
        self.temperature_min = float(temperature_min)
        self.temperature_max = float(temperature_max)
        self.standardize_scope = str(standardize_scope or "per_image").lower()
        self.standardize_eps = float(standardize_eps)
        self.host_cfg = HostConfig(
            host_model_name=self.host_model_name,
            host_output_kind=host_output_kind,
            freeze_host=freeze_host,
            joint_finetune=joint_finetune,
        )
        self.plugin_cfg = PluginConfig(
            method_variant=method_variant,
            prior_family=prior_family,
            mixer_family=mixer_family,
            capacity_tier=self.capacity_tier,
            state_form=self.state_form,
            update_rule=self.update_rule,
            evidence_level=self.evidence_level,
            recursion_tier=self.recursion_tier,
            recursion_steps=int(self.recursion_steps),
            recurrence_share_weights=bool(recurrence_share_weights),
            plugin_channels=plugin_channels,
            hidden_channels=hidden_channels,
            use_geometry=bool(use_geometry or str(method_variant).upper() == "M6GEOM"),
            update_alpha=update_alpha,
            temperature_mode=self.temperature_mode,
            temperature_init=float(temperature_init),
            temperature_min=self.temperature_min,
            temperature_max=self.temperature_max,
            standardize_scope=self.standardize_scope,
            standardize_eps=self.standardize_eps,
            dynamic_alpha_hidden=int(dynamic_alpha_hidden),
            alpha_min=float(alpha_min),
            alpha_max=float(alpha_max),
            sparsity_weight=sparsity_weight,
            delta_weight=delta_weight,
            risk_bias=float(risk_bias),
            strict_evidence_contract=bool(strict_evidence_contract),
            disable_image_priors_for_correction=bool(disable_image_priors_for_correction),
            edge_operator=str(edge_operator or "sobel"),
        )
        self.sparse_cfg = SparseConfig(policy=sparse_policy, budget_policy=self.budget_policy, topk_ratio=sparse_topk_ratio)
        self.train_cfg = TrainConfig(lr=lr, epochs=epochs, loss_name=loss_name)
        self.state_alignment_cfg = StateAlignmentConfig(
            state_form=self.state_form,
            standardize_scope=self.standardize_scope,
            standardize_eps=self.standardize_eps,
            temperature_mode=self.temperature_mode,
            temperature_init=float(temperature_init),
            temperature_min=self.temperature_min,
            temperature_max=self.temperature_max,
        )
        self.update_rule_cfg = UpdateRuleConfig(
            update_rule=self.update_rule,
            alpha_init=float(update_alpha),
            alpha_min=float(alpha_min),
            alpha_max=float(alpha_max),
            dynamic_alpha_hidden=int(dynamic_alpha_hidden),
        )
        self.evidence_cfg = EvidenceLevelConfig(
            evidence_level=self.evidence_level,
            use_low_high=self.evidence_level == "E2",
            use_structure_lite=self.evidence_level == "E3",
        )
        self.budget_cfg = BudgetPolicyConfig(
            budget_policy=self.budget_policy,
            topk_ratio=float(sparse_topk_ratio),
            boundary_band_width=self.sparse_cfg.hard_boundary_width,
            random_seed_offset=self.sparse_cfg.random_seed_offset,
        )
        self.experiment_cfg = ExperimentConfig(
            host=self.host_cfg,
            plugin=self.plugin_cfg,
            sparse=self.sparse_cfg,
            train=self.train_cfg,
            state_alignment=self.state_alignment_cfg,
            update_rule=self.update_rule_cfg,
            evidence=self.evidence_cfg,
            budget=self.budget_cfg,
        )
        self.refiner = RefinementPlugin(self.plugin_cfg, self.sparse_cfg)
        self.reproduction_level = "risk-guided-image-aware-plugin"
        self.host_feature_dependency = "final-output-only"
        self.last_aux_visuals: dict[str, torch.Tensor] | None = None
        self.last_plugin_output: dict[str, object] | None = None

    def _current_temperature(self) -> torch.Tensor:
        return self.refiner.state_aligner.current_temperature(
            device=next(self.refiner.parameters()).device,
            dtype=next(self.refiner.parameters()).dtype,
        )

    def forward(self, x: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        self.last_aux_visuals = None
        self.last_plugin_output = None

        raw = self._host_forward(x)
        host_bundle = self.host_adapter.normalize_output(raw, x)
        # The host is frozen in the current RIGS screening protocol, so feed the
        # plugin stable copies to avoid retaining or aliasing host-side graphs.
        host_logits = host_bundle.logits.detach().contiguous()
        host_prob = host_bundle.prob.detach().contiguous()
        host_fg_logit = foreground_logit(host_logits, host_bundle.kind).detach().contiguous()
        plugin_out = self.refiner(
            image=x,
            logits=host_fg_logit,
            prob=host_prob,
            target=target,
        )
        refined_logits = plugin_out["refined_logits"]
        output = binary_logits_from_foreground_logit(refined_logits)
        self.last_plugin_output = plugin_out

        stats = dict(plugin_out.get("stats", {}) or {})
        self.last_aux = {
            "freeze_host": float(self.freeze_host),
            "joint_finetune": float(self.joint_finetune),
            "rigs_refiner": 1.0,
            "host_adapter_pranet": 1.0 if self.host_adapter.adapter_name == "pranet" else 0.0,
            "host_adapter_segformer_b0": 1.0 if self.host_adapter.adapter_name == "segformer_b0" else 0.0,
            "host_adapter_unet": 1.0 if self.host_adapter.adapter_name == "unet" else 0.0,
            "host_adapter_i2unet": 1.0 if self.host_adapter.adapter_name == "i2unet" else 0.0,
            "state_form_n0": 1.0 if self.state_form == "N0" else 0.0,
            "state_form_n1": 1.0 if self.state_form == "N1" else 0.0,
            "state_form_n2": 1.0 if self.state_form == "N2" else 0.0,
            "state_form_n3": 1.0 if self.state_form == "N3" else 0.0,
            "update_rule_u1": 1.0 if self.update_rule == "U1" else 0.0,
            "update_rule_u2": 1.0 if self.update_rule == "U2" else 0.0,
            "update_rule_u3": 1.0 if self.update_rule == "U3" else 0.0,
            "update_rule_u4": 1.0 if self.update_rule == "U4" else 0.0,
            "evidence_level_e0": 1.0 if self.evidence_level == "E0" else 0.0,
            "evidence_level_e1": 1.0 if self.evidence_level == "E1" else 0.0,
            "evidence_level_e2": 1.0 if self.evidence_level == "E2" else 0.0,
            "evidence_level_e3": 1.0 if self.evidence_level == "E3" else 0.0,
            "evidence_level_e4": 1.0 if self.evidence_level == "E4" else 0.0,
            "budget_policy_b0": 1.0 if self.budget_policy == "B0" else 0.0,
            "budget_policy_b1": 1.0 if self.budget_policy == "B1" else 0.0,
            "budget_policy_b2": 1.0 if self.budget_policy == "B2" else 0.0,
            "budget_policy_b3": 1.0 if self.budget_policy == "B3" else 0.0,
            "budget_policy_b4": 1.0 if self.budget_policy == "B4" else 0.0,
            "budget_policy_b5": 1.0 if self.budget_policy == "B5" else 0.0,
            "method_variant_m1": 1.0 if self.plugin_cfg.method_variant.upper() == "M1" else 0.0,
            "method_variant_m2": 1.0 if self.plugin_cfg.method_variant.upper() == "M2" else 0.0,
            "method_variant_m3": 1.0 if self.plugin_cfg.method_variant.upper() == "M3" else 0.0,
            "method_variant_m6geom": 1.0 if self.plugin_cfg.method_variant.upper() == "M6GEOM" else 0.0,
            "disable_image_priors_for_correction": 1.0 if self.plugin_cfg.disable_image_priors_for_correction else 0.0,
            "temperature_value": float(self._current_temperature().detach().item()),
            "plugin_params": float(sum(param.numel() for param in self.refiner.parameters())),
            "host_fg_prob_mean": tensor_stat(host_prob, "mean"),
            "host_fg_logit_mean": tensor_stat(host_fg_logit, "mean"),
            "host_fg_logit_std": tensor_stat(host_fg_logit, "std"),
            "state_prob_mean": tensor_stat(plugin_out["aligned_state"]["state_prob"], "mean"),
            "state_logit_mean": tensor_stat(plugin_out["aligned_state"]["state_logit"], "mean"),
            "state_logit_std": tensor_stat(plugin_out["aligned_state"]["state_logit"], "std"),
            "refined_fg_prob_mean": tensor_stat(plugin_out["refined_prob"], "mean"),
            "delta_abs_mean": tensor_stat(plugin_out["delta_logits"].abs(), "mean"),
            "gate_mean": tensor_stat(plugin_out["gate"], "mean"),
            "updated_ratio": float(stats.get("updated_ratio", 0.0)),
            "risk_mean": float(stats.get("risk_mean", 0.0)),
            "uncertainty_mean": float(stats.get("uncertainty_mean", 0.0)),
            "alpha_mean": float(stats.get("alpha_mean", 0.0)),
            "alpha_std": float(stats.get("alpha_std", 0.0)),
            "boundary_iou": float(stats.get("boundary_iou", 0.0)),
            "latency_proxy_ms": float(stats.get("latency_proxy_ms", 0.0)),
            "recursion_steps": float(self.recursion_steps),
        }
        alpha_visual = plugin_out["step_records"][-1]["alpha_tensor"].detach().expand_as(plugin_out["refined_prob"])
        self.last_aux_visuals = {
            "host_prob": host_prob.detach(),
            "state_prob": plugin_out["aligned_state"]["state_prob"].detach(),
            "state_logit": plugin_out["aligned_state"]["state_logit"].detach(),
            "refined_prob": plugin_out["refined_prob"].detach(),
            "final_pred": plugin_out["refined_prob"].detach(),
            "gate": plugin_out["gate"].detach(),
            "mask": plugin_out["update_mask"].detach(),
            "alpha": alpha_visual,
            "delta_logits": plugin_out["delta_logits"].detach(),
            "risk_score": torch.sigmoid(plugin_out["risk_score"]).detach(),
        }
        if target is not None:
            target_vis = target if target.ndim == 4 else target.unsqueeze(1)
            self.last_aux_visuals["mask_gt"] = target_vis.float().detach()
        return output
