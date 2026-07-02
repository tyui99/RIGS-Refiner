from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class HostConfig:
    host_model_name: str
    host_output_kind: str = "auto"
    freeze_host: bool = True
    joint_finetune: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PluginConfig:
    method_variant: str = "M3"
    prior_family: str = "P0"
    mixer_family: str = "C0"
    capacity_tier: str = "R0"
    state_form: str = "N0"
    update_rule: str = "U2"
    evidence_level: str = "E2"
    recursion_tier: str = "K0"
    recursion_steps: int = 1
    recurrence_share_weights: bool = True
    edge_operator: str = "sobel"
    image_channels: int = 3
    plugin_channels: int = 4
    hidden_channels: int = 8
    use_geometry: bool = False
    strict_evidence_contract: bool = True
    disable_image_priors_for_correction: bool = False
    update_alpha: float = 1.0
    temperature_mode: str = "fixed"
    temperature_init: float = 1.0
    temperature_min: float = 0.25
    temperature_max: float = 4.0
    standardize_scope: str = "per_image"
    standardize_eps: float = 1.0e-6
    dynamic_alpha_hidden: int = 4
    alpha_min: float = 0.0
    alpha_max: float = 2.0
    sparsity_weight: float = 0.0
    delta_weight: float = 0.0
    risk_bias: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SparseConfig:
    policy: str = "S0"
    budget_policy: str = "B0"
    topk_ratio: float = 0.10
    hard_boundary_width: int = 3
    random_seed_offset: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-4
    epochs: int = 50
    loss_name: str = "dice"
    align_weight: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StateAlignmentConfig:
    state_form: str = "N0"
    standardize_scope: str = "per_image"
    standardize_eps: float = 1.0e-6
    temperature_mode: str = "fixed"
    temperature_init: float = 1.0
    temperature_min: float = 0.25
    temperature_max: float = 4.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class UpdateRuleConfig:
    update_rule: str = "U2"
    alpha_init: float = 1.0
    alpha_min: float = 0.0
    alpha_max: float = 2.0
    dynamic_alpha_hidden: int = 4

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceLevelConfig:
    evidence_level: str = "E2"
    edge_operator: str = "sobel"
    use_low_high: bool = True
    use_structure_lite: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BudgetPolicyConfig:
    budget_policy: str = "B0"
    topk_ratio: float = 0.10
    boundary_band_width: int = 3
    random_seed_offset: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentConfig:
    host: HostConfig
    plugin: PluginConfig
    sparse: SparseConfig
    train: TrainConfig
    state_alignment: StateAlignmentConfig | None = None
    update_rule: UpdateRuleConfig | None = None
    evidence: EvidenceLevelConfig | None = None
    budget: BudgetPolicyConfig | None = None

    def to_dict(self) -> dict[str, object]:
        payload = {
            "host": self.host.to_dict(),
            "plugin": self.plugin.to_dict(),
            "sparse": self.sparse.to_dict(),
            "train": self.train.to_dict(),
        }
        if self.state_alignment is not None:
            payload["state_alignment"] = self.state_alignment.to_dict()
        if self.update_rule is not None:
            payload["update_rule"] = self.update_rule.to_dict()
        if self.evidence is not None:
            payload["evidence"] = self.evidence.to_dict()
        if self.budget is not None:
            payload["budget"] = self.budget.to_dict()
        return payload

def select_main_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for key in ("logits", "pred", "output", "outputs", "preds"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
            if isinstance(value, (list, tuple)):
                tensors = [item for item in value if isinstance(item, torch.Tensor)]
                if tensors:
                    return tensors[-1]
    if isinstance(output, (list, tuple)):
        tensors = [item for item in output if isinstance(item, torch.Tensor)]
        if tensors:
            return tensors[-1]
    raise TypeError(f"Unsupported host output type: {type(output)!r}")


def resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


def infer_output_kind(output: torch.Tensor, declared_kind: str = "auto") -> str:
    kind = str(declared_kind or "auto").strip().lower()
    if kind in {"logits", "prob"}:
        return kind
    detached = output.detach()
    if float(detached.min().item()) >= -1e-5 and float(detached.max().item()) <= 1.0 + 1e-5:
        return "prob"
    return "logits"


def foreground_prob(output: torch.Tensor, kind: str) -> torch.Tensor:
    if output.ndim != 4 or output.shape[1] not in (1, 2):
        raise ValueError(f"Expected host output shape (B,1,H,W) or (B,2,H,W), got {tuple(output.shape)}")
    if output.shape[1] == 1:
        return output.clamp(0.0, 1.0) if kind == "prob" else torch.sigmoid(output)
    if kind == "prob":
        return output[:, 1:2].clamp(0.0, 1.0)
    return torch.softmax(output, dim=1)[:, 1:2]


def edge_map(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"edge_map expects BCHW tensor, got {tuple(x.shape)}")
    if x.shape[1] != 1:
        x = x.mean(dim=1, keepdim=True)
    sobel_x = x.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    sobel_y = x.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    return torch.sqrt(gx.square() + gy.square() + 1e-6)


def normalize_target_mask(target: torch.Tensor) -> torch.Tensor:
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    return (target.float() > 0.5).float()


@dataclass
class HostForwardBundle:
    logits: torch.Tensor
    prob: torch.Tensor
    mask: torch.Tensor
    kind: str


class HostAdapter:
    adapter_name = "generic"

    def __init__(self, host: nn.Module, host_output_kind: str = "auto"):
        self.host = host
        self.host_output_kind = str(host_output_kind or "auto").strip().lower()

    def _resolve_kind(self, logits: torch.Tensor) -> str:
        if self.host_output_kind in {"logits", "prob"}:
            return self.host_output_kind
        host_kind = str(getattr(self.host, "output_kind", "") or "").strip().lower()
        declared = host_kind if host_kind in {"logits", "prob"} else "auto"
        return infer_output_kind(logits, declared_kind=declared)

    def normalize_output(self, raw_output: Any, ref_input: torch.Tensor) -> HostForwardBundle:
        main = resize_like(select_main_tensor(raw_output), ref_input)
        kind = self._resolve_kind(main)
        prob = foreground_prob(main, kind)
        mask = (prob > 0.5).float()
        return HostForwardBundle(logits=main, prob=prob, mask=mask, kind=kind)


class AnchorAdapterA(HostAdapter):
    adapter_name = "anchor_primary"


class AnchorAdapterB(HostAdapter):
    adapter_name = "anchor_secondary"


class UNetAdapter(HostAdapter):
    adapter_name = "unet"


class I2UNetAdapter(HostAdapter):
    adapter_name = "i2unet"


def build_host_adapter(host: nn.Module, host_output_kind: str = "auto") -> HostAdapter:
    adapter_name = str(getattr(host, "adapter_name", "") or "").strip().lower()
    name = type(host).__name__.lower()
    if adapter_name == "anchor_primary" or "pranet" in name:
        return AnchorAdapterA(host, host_output_kind=host_output_kind)
    if adapter_name == "anchor_secondary" or "segformer" in name:
        return AnchorAdapterB(host, host_output_kind=host_output_kind)
    if "i2u" in name:
        return I2UNetAdapter(host, host_output_kind=host_output_kind)
    if "unet" in name:
        return UNetAdapter(host, host_output_kind=host_output_kind)
    return HostAdapter(host, host_output_kind=host_output_kind)





def _to_three_channels(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 3:
        return x
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    if x.shape[1] > 3:
        return x[:, :3, :, :]
    pad_channels = 3 - x.shape[1]
    pad = x[:, -1:, :, :].repeat(1, pad_channels, 1, 1)
    return torch.cat([x, pad], dim=1)


def _gray_image(x: torch.Tensor) -> torch.Tensor:
    x = _to_three_channels(x)
    return x.mean(dim=1, keepdim=True)


def _norm01(x: torch.Tensor) -> torch.Tensor:
    min_val = x.amin(dim=(-2, -1), keepdim=True)
    max_val = x.amax(dim=(-2, -1), keepdim=True)
    return (x - min_val) / (max_val - min_val + 1e-6)


def _avg_blur(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    padding = kernel_size // 2
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)


def _edge_feature(x: torch.Tensor, operator: str = "sobel") -> torch.Tensor:
    x = x if x.shape[1] == 1 else x.mean(dim=1, keepdim=True)
    op = str(operator or "sobel").strip().lower()
    if op == "sobel":
        return edge_map(x)
    if op != "scharr":
        raise ValueError(f"Unsupported edge operator: {operator}")
    kernel_x = x.new_tensor(
        [[3.0, 0.0, -3.0], [10.0, 0.0, -10.0], [3.0, 0.0, -3.0]],
    ).view(1, 1, 3, 3)
    kernel_y = x.new_tensor(
        [[3.0, 10.0, 3.0], [0.0, 0.0, 0.0], [-3.0, -10.0, -3.0]],
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(x, kernel_x, padding=1)
    grad_y = F.conv2d(x, kernel_y, padding=1)
    return torch.sqrt(grad_x.square() + grad_y.square() + 1.0e-6)


def _laplacian_of_gaussian(x: torch.Tensor) -> torch.Tensor:
    blurred = _avg_blur(x, kernel_size=5)
    kernel = x.new_tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
    ).view(1, 1, 3, 3)
    return F.conv2d(blurred, kernel, padding=1)


def _fft_lite_prior(x: torch.Tensor) -> torch.Tensor:
    fft = torch.fft.rfft2(x, norm="ortho")
    _, _, h, w = x.shape
    fy = torch.linspace(0.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1)
    fx = torch.linspace(0.0, 1.0, fft.shape[-1], device=x.device, dtype=x.dtype).view(1, 1, 1, fft.shape[-1])
    mask = ((fy.square() + fx.square()) > 0.12).to(x.dtype)
    filtered = fft * mask
    return torch.fft.irfft2(filtered, s=(h, w), norm="ortho")


def _signed_geometry_proxy(prob: torch.Tensor) -> torch.Tensor:
    soft_mask = (prob > 0.5).float()
    inside = _avg_blur(soft_mask, kernel_size=9)
    outside = _avg_blur(1.0 - soft_mask, kernel_size=9)
    return inside - outside


def _component_anomaly_proxy(prob: torch.Tensor) -> torch.Tensor:
    local = _avg_blur(prob, kernel_size=11)
    return (prob - local).abs()


def _entropy_binary(prob: torch.Tensor) -> torch.Tensor:
    prob = prob.clamp(1e-6, 1.0 - 1e-6)
    return -(prob * prob.log() + (1.0 - prob) * (1.0 - prob).log())


def _topk_mask(score: torch.Tensor, ratio: float) -> torch.Tensor:
    b, _, h, w = score.shape
    flat = score.flatten(start_dim=1)
    k = max(1, min(flat.shape[1], int(round(flat.shape[1] * float(ratio)))))
    values = torch.topk(flat, k=k, dim=1, largest=True, sorted=False).values
    thresh = values.amin(dim=1, keepdim=True)
    return (flat >= thresh).view(b, 1, h, w).float()


def _graph_safe_clone(x: torch.Tensor) -> torch.Tensor:
    # Preserve gradient flow while forcing a fresh contiguous tensor boundary.
    return x.contiguous().clone()


def _canonical_budget_policy(policy: str) -> str:
    norm = str(policy or "S0").strip().upper()
    alias = {
        "S0": "B0",
        "S1": "B2",
        "S2": "B4",
        "S3": "B5",
        "SOFT_FULL": "B0",
        "RISK_TOP20": "B1",
        "RISK_TOP10": "B2",
        "RISK_TOP5": "B3",
        "RANDOM_TOP10": "B4",
        "BOUNDARY_TOP10": "B5",
    }
    return alias.get(norm, norm)


def _canonical_prior_family(prior_family: str) -> str:
    norm = str(prior_family or "D0").strip().upper()
    alias = {
        "P0": "D0",
        "P1": "D1",
        "P2": "D2",
        "P3": "D3",
    }
    return alias.get(norm, norm)


def _evidence_flags(evidence_level: str) -> dict[str, bool]:
    level = str(evidence_level or "E1").strip().upper()
    return {
        "use_edge": level in {"E1", "E2", "E3", "E4"},
        "use_low_high": level in {"E2", "E4"},
        "use_structure": level in {"E3", "E4"},
        "use_complexity_artifact": level == "E4",
    }


def _resolve_capacity_hidden(hidden_channels: int, capacity_tier: str) -> int:
    base_hidden = max(1, int(hidden_channels))
    tier = str(capacity_tier or "R0").strip().upper()
    scale = {
        "R0": 1.0,
        "R1": 1.5,
        "R2": 1.75,
        "R3": 2.0,
    }.get(tier, 1.0)
    return max(1, int(round(base_hidden * scale)))


class StateAligner(nn.Module):
    def __init__(
        self,
        state_form: str = "N0",
        standardize_scope: str = "per_image",
        standardize_eps: float = 1.0e-6,
        temperature_mode: str = "fixed",
        temperature_init: float = 1.0,
        temperature_min: float = 0.25,
        temperature_max: float = 4.0,
    ):
        super().__init__()
        self.state_form = str(state_form or "N0").upper()
        self.standardize_scope = str(standardize_scope or "per_image").lower()
        self.standardize_eps = float(standardize_eps)
        self.temperature_mode = str(temperature_mode or "fixed").lower()
        self.temperature_min = float(temperature_min)
        self.temperature_max = float(temperature_max)
        if self.state_form == "N3" and self.temperature_mode == "learnable":
            self.temperature_param = nn.Parameter(torch.tensor(float(temperature_init), dtype=torch.float32))
        else:
            self.register_buffer(
                "temperature_param",
                torch.tensor(float(temperature_init), dtype=torch.float32),
                persistent=False,
            )

    def current_temperature(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        temp = self.temperature_param.to(device=device, dtype=dtype)
        return temp.clamp(min=self.temperature_min, max=self.temperature_max)

    def forward(self, fg_logit: torch.Tensor, fg_prob: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.state_form == "N0":
            state_logit = fg_logit
            state_prob = fg_prob
        elif self.state_form == "N1":
            state_prob = fg_prob.clamp(1.0e-4, 1.0 - 1.0e-4)
            state_logit = torch.log(state_prob / (1.0 - state_prob))
        elif self.state_form == "N2":
            if self.standardize_scope != "per_image":
                raise ValueError(f"Unsupported standardize_scope: {self.standardize_scope}")
            mean = fg_logit.mean(dim=(-2, -1), keepdim=True)
            std = fg_logit.std(dim=(-2, -1), keepdim=True, unbiased=False)
            state_logit = (fg_logit - mean) / (std + self.standardize_eps)
            state_prob = torch.sigmoid(state_logit)
        elif self.state_form == "N3":
            temperature = self.current_temperature(device=fg_logit.device, dtype=fg_logit.dtype)
            state_logit = fg_logit / temperature.view(1, 1, 1, 1)
            state_prob = torch.sigmoid(state_logit)
        else:
            raise ValueError(f"Unsupported state_form: {self.state_form}")
        return {
            "state_logit": state_logit,
            "state_prob": state_prob,
            "temperature": self.current_temperature(device=fg_logit.device, dtype=fg_logit.dtype),
        }


class PriorExtractor(nn.Module):
    def __init__(self, prior_family: str = "P0", out_channels: int = 4, edge_operator: str = "sobel"):
        super().__init__()
        self.prior_family = _canonical_prior_family(prior_family)
        self.out_channels = int(out_channels)
        self.edge_operator = "scharr" if self.prior_family == "D1" else str(edge_operator or "sobel").lower()
        # Fixed 7-channel stack: edge / low / high / log / fft / structure-lite(2).
        self.proj = nn.Conv2d(7, self.out_channels, kernel_size=1, bias=True)

    def project_stack(self, raw_img_stack: torch.Tensor) -> torch.Tensor:
        return self.proj(raw_img_stack)

    def forward(self, image: torch.Tensor, geometry_map: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        gray = _gray_image(image)
        edge = _norm01(_edge_feature(gray, self.edge_operator))
        low = _avg_blur(gray, kernel_size=7)
        high = gray - low

        log_like = torch.zeros_like(edge)
        fft_like = torch.zeros_like(edge)
        if self.prior_family == "D2":
            log_like = _norm01(_laplacian_of_gaussian(gray).abs())
        elif self.prior_family == "D3":
            fft_like = _norm01(_fft_lite_prior(gray).abs())
        if geometry_map is None:
            structure_lite = torch.zeros(
                gray.shape[0],
                2,
                gray.shape[-2],
                gray.shape[-1],
                device=gray.device,
                dtype=gray.dtype,
            )
        else:
            structure_lite = geometry_map if geometry_map.shape[1] == 2 else geometry_map[:, :1].repeat(1, 2, 1, 1)

        raw_img_stack = torch.cat([edge, low, high, log_like, fft_like, structure_lite], dim=1)
        img_feat = self.project_stack(raw_img_stack)
        return {
            "edge_map": edge,
            "low_freq": low,
            "high_freq": high,
            "geometry_map": structure_lite,
            "structure_lite_map": structure_lite,
            "img_feat": img_feat,
            "log_like": log_like,
            "fft_lite": fft_like,
            "raw_img_stack": raw_img_stack,
            "edge_operator": self.edge_operator,
        }


class RiskBuilder(nn.Module):
    def __init__(
        self,
        pred_channels: int = 4,
        edge_operator: str = "sobel",
        risk_bias: float = 0.0,
        evidence_level: str = "E2",
    ):
        super().__init__()
        self.pred_channels = int(pred_channels)
        self.edge_operator = str(edge_operator or "sobel").lower()
        self.evidence_level = str(evidence_level or "E2").strip().upper()
        self.proj = nn.Conv2d(14, self.pred_channels, kernel_size=1, bias=True)
        self.score_head = nn.Conv2d(self.pred_channels, 1, kernel_size=1, bias=True)
        if self.score_head.bias is not None:
            with torch.no_grad():
                self.score_head.bias.fill_(float(risk_bias))

    def forward(
        self,
        logits: torch.Tensor,
        prob: torch.Tensor,
        edge_map_tensor: torch.Tensor,
        geometry_map: torch.Tensor | None = None,
        prior_maps: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        prior_maps = dict(prior_maps or {})
        flags = _evidence_flags(self.evidence_level)
        state_logits = logits if logits.shape[1] == 1 else logits[:, 1:2]
        uncertainty = _entropy_binary(prob)
        boundary = _norm01(_edge_feature(prob, self.edge_operator))
        if flags["use_edge"]:
            img_conflict = (_norm01(edge_map_tensor) - boundary).abs()
        else:
            img_conflict = torch.zeros_like(boundary)
        if geometry_map is not None:
            geo_edge = _norm01(_edge_feature(geometry_map, self.edge_operator))
            geo_conflict = (geo_edge - boundary).abs()
        else:
            geo_conflict = torch.zeros_like(boundary)
        if not flags["use_structure"]:
            geo_conflict = torch.zeros_like(boundary)
        state_abs = _norm01(state_logits.abs())
        state_mean = state_logits.mean(dim=(-2, -1), keepdim=True)
        state_std = state_logits.std(dim=(-2, -1), keepdim=True, unbiased=False)
        state_margin = _norm01(((state_logits - state_mean) / (state_std + 1.0e-6)).abs())
        state_boundary = _norm01(_edge_feature(state_logits, self.edge_operator))
        state_local_variation = _norm01((state_logits - _avg_blur(state_logits, kernel_size=7)).abs())
        prob_saturation = (prob - 0.5).abs() * 2.0
        if geometry_map is not None and geometry_map.shape[1] >= 2:
            structure_signed = _norm01(geometry_map[:, 0:1].abs())
            structure_anomaly = _norm01(geometry_map[:, 1:2].abs())
        elif geometry_map is not None:
            structure_signed = _norm01(geometry_map[:, :1].abs())
            structure_anomaly = torch.zeros_like(structure_signed)
        else:
            structure_signed = torch.zeros_like(boundary)
            structure_anomaly = torch.zeros_like(boundary)
        if not flags["use_structure"]:
            structure_signed = torch.zeros_like(boundary)
            structure_anomaly = torch.zeros_like(boundary)

        complexity_sources = []
        for key in ("high_freq", "log_like", "fft_lite"):
            value = prior_maps.get(key)
            if torch.is_tensor(value):
                complexity_sources.append(_norm01(value.abs()))
        if complexity_sources:
            complexity_proxy = torch.stack(complexity_sources, dim=0).mean(dim=0)
        else:
            complexity_proxy = torch.zeros_like(boundary)
        artifact_proxy = _norm01((complexity_proxy - state_local_variation).abs() + (prob - _avg_blur(prob, kernel_size=11)).abs())
        if not flags["use_complexity_artifact"]:
            complexity_proxy = torch.zeros_like(boundary)
            artifact_proxy = torch.zeros_like(boundary)
        pred_stack = torch.cat(
            [
                prob,
                uncertainty,
                boundary,
                img_conflict,
                geo_conflict,
                prob_saturation,
                state_abs,
                state_margin,
                state_boundary,
                state_local_variation,
                structure_signed,
                structure_anomaly,
                complexity_proxy,
                artifact_proxy,
            ],
            dim=1,
        )
        pred_feat = self.proj(pred_stack)
        risk_score = self.score_head(pred_feat)
        return {
            "prob": prob,
            "uncertainty": uncertainty,
            "boundary_band": boundary,
            "img_conflict": img_conflict,
            "geo_conflict": geo_conflict,
            "prob_saturation": prob_saturation,
            "state_abs": state_abs,
            "state_margin": state_margin,
            "state_boundary": state_boundary,
            "state_local_variation": state_local_variation,
            "structure_signed": structure_signed,
            "structure_anomaly": structure_anomaly,
            "complexity_proxy": complexity_proxy,
            "artifact_proxy": artifact_proxy,
            "pred_feat": pred_feat,
            "risk_score": risk_score,
        }


class UpdateSelector(nn.Module):
    def __init__(self, policy: str = "S0", topk_ratio: float = 0.10, boundary_width: int = 3, random_seed_offset: int = 0):
        super().__init__()
        self.policy = _canonical_budget_policy(policy)
        self.topk_ratio = float(topk_ratio)
        self.boundary_width = int(boundary_width)
        self.random_seed_offset = int(random_seed_offset)

    def forward(
        self,
        risk_score: torch.Tensor,
        boundary_band: torch.Tensor,
        prob: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del prob
        if self.policy == "B0":
            gate = torch.sigmoid(risk_score)
            update_mask = gate
        elif self.policy == "B1":
            update_mask = _topk_mask(risk_score, 0.20)
            gate = update_mask
        elif self.policy == "B2":
            update_mask = _topk_mask(risk_score, self.topk_ratio)
            gate = update_mask
        elif self.policy == "B3":
            update_mask = _topk_mask(risk_score, 0.05)
            gate = update_mask
        elif self.policy == "B4":
            generator = torch.Generator(device=risk_score.device)
            generator.manual_seed(1234 + self.random_seed_offset)
            random_score = torch.rand(risk_score.shape, generator=generator, device=risk_score.device, dtype=risk_score.dtype)
            update_mask = _topk_mask(random_score, self.topk_ratio)
            gate = update_mask
        elif self.policy == "B5":
            band = (_avg_blur(boundary_band, kernel_size=max(3, 2 * self.boundary_width + 1)) > 0.05).float()
            weighted = risk_score * band + (band - 1.0) * 1e6
            update_mask = _topk_mask(weighted, self.topk_ratio)
            gate = update_mask
        else:
            raise ValueError(f"Unsupported sparse policy: {self.policy}")
        return {"gate": gate, "update_mask": update_mask}


class CorrectionCore(nn.Module):
    def __init__(
        self,
        mixer_family: str = "C0",
        img_channels: int = 4,
        pred_channels: int = 4,
        hidden_channels: int = 8,
        capacity_tier: str = "R0",
    ):
        super().__init__()
        self.mixer_family = str(mixer_family).upper()
        self.capacity_tier = str(capacity_tier or "R0").upper()
        in_channels = int(img_channels) + int(pred_channels) + 1
        hidden_channels = _resolve_capacity_hidden(hidden_channels=int(hidden_channels), capacity_tier=self.capacity_tier)
        self.hidden_channels = int(hidden_channels)
        self.pre = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True)
        self.depthwise = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels, bias=True)
        self.depthwise_d2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=2, dilation=2, groups=hidden_channels, bias=True)
        self.depthwise_d3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=3, dilation=3, groups=hidden_channels, bias=True)
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.post = nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True)

    def forward(self, img_feat: torch.Tensor, pred_feat: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([img_feat, pred_feat, gate], dim=1).contiguous()
        pre_activated = self.pre(fused)
        hidden = F.relu(pre_activated, inplace=False)
        base = _graph_safe_clone(hidden)
        if self.mixer_family == "C0":
            # C0 is the only pure identity mixer. Keep an explicit graph barrier so
            # the primary loss and auxiliary regularizer do not share the same
            # activation buffer through aliasing.
            mixed = _graph_safe_clone(base)
        elif self.mixer_family == "C1":
            mixed = base + self.depthwise(base)
            mixed = mixed * self.global_gate(mixed)
        elif self.mixer_family == "C2":
            mixed = base + self.depthwise(base) + self.depthwise_d2(base) + self.depthwise_d3(base)
        else:
            raise ValueError(f"Unsupported mixer family: {self.mixer_family}")
        mixed = _graph_safe_clone(mixed)
        activated = F.relu(mixed, inplace=False)
        return self.post(_graph_safe_clone(activated))


class RecursiveCorrectionCell(nn.Module):
    def __init__(
        self,
        correction_core: CorrectionCore,
        update_rule: str = "U2",
        update_alpha: float = 1.0,
        alpha_min: float = 0.0,
        alpha_max: float = 2.0,
        dynamic_alpha_hidden: int = 4,
    ):
        super().__init__()
        self.correction_core = correction_core
        self.update_rule = str(update_rule or "U2").upper()
        self.update_alpha = float(update_alpha)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        alpha_hidden = max(1, int(dynamic_alpha_hidden))
        self.dynamic_alpha = nn.Sequential(
            nn.Linear(3, alpha_hidden, bias=True),
            nn.ReLU(inplace=False),
            nn.Linear(alpha_hidden, 1, bias=True),
            nn.Sigmoid(),
        )

    def _resolve_alpha(self, base_logit: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        b = base_logit.shape[0]
        if self.update_rule in {"U1", "U2"}:
            return torch.ones((b, 1, 1, 1), device=base_logit.device, dtype=base_logit.dtype)
        if self.update_rule == "U3":
            return torch.full(
                (b, 1, 1, 1),
                float(self.update_alpha),
                device=base_logit.device,
                dtype=base_logit.dtype,
            )
        if self.update_rule == "U4":
            features = torch.stack(
                [
                    base_logit.mean(dim=(-2, -1)).squeeze(1),
                    base_logit.std(dim=(-2, -1), unbiased=False).squeeze(1),
                    uncertainty.mean(dim=(-2, -1)).squeeze(1),
                ],
                dim=1,
            )
            alpha01 = self.dynamic_alpha(features).view(b, 1, 1, 1)
            return self.alpha_min + (self.alpha_max - self.alpha_min) * alpha01
        raise ValueError(f"Unsupported update_rule: {self.update_rule}")

    def forward(
        self,
        img_feat: torch.Tensor,
        pred_feat: torch.Tensor,
        gate: torch.Tensor,
        update_mask: torch.Tensor,
        current_logit: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        delta_logits = self.correction_core(img_feat=img_feat, pred_feat=pred_feat, gate=gate)
        alpha_tensor = self._resolve_alpha(base_logit=current_logit, uncertainty=uncertainty)
        if self.update_rule == "U1":
            refined_logits = current_logit + delta_logits
        elif self.update_rule == "U2":
            refined_logits = current_logit + update_mask * delta_logits
        else:
            refined_logits = current_logit + alpha_tensor * update_mask * delta_logits
        refined_prob = torch.sigmoid(refined_logits)
        return {
            "delta_logits": delta_logits,
            "alpha_tensor": alpha_tensor,
            "refined_logits": refined_logits,
            "refined_prob": refined_prob,
        }


class RecurrenceController(nn.Module):
    def __init__(
        self,
        risk_builder: RiskBuilder,
        update_selector: UpdateSelector,
        correction_cell: RecursiveCorrectionCell,
        recursion_steps: int = 1,
    ):
        super().__init__()
        self.risk_builder = risk_builder
        self.update_selector = update_selector
        self.correction_cell = correction_cell
        self.recursion_steps = max(1, int(recursion_steps))

    def forward(
        self,
        img_feat: torch.Tensor,
        state_logit: torch.Tensor,
        state_prob: torch.Tensor,
        edge_map_tensor: torch.Tensor,
        geometry_map: torch.Tensor | None = None,
        prior_maps: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, object]:
        current_logit = state_logit
        current_prob = state_prob
        step_records: list[dict[str, torch.Tensor]] = []
        pred_seq: list[torch.Tensor] = []
        alpha_seq: list[torch.Tensor] = []
        for _ in range(self.recursion_steps):
            risk_outputs = self.risk_builder(
                logits=current_logit,
                prob=current_prob,
                edge_map_tensor=edge_map_tensor,
                geometry_map=geometry_map,
                prior_maps=prior_maps,
            )
            selector_outputs = self.update_selector(
                risk_score=risk_outputs["risk_score"],
                boundary_band=risk_outputs["boundary_band"],
                prob=current_prob,
            )
            cell_outputs = self.correction_cell(
                img_feat=img_feat,
                pred_feat=risk_outputs["pred_feat"],
                gate=selector_outputs["gate"],
                update_mask=selector_outputs["update_mask"],
                current_logit=current_logit,
                uncertainty=risk_outputs["uncertainty"],
            )
            step_records.append(
                {
                    "risk_score": risk_outputs["risk_score"],
                    "pred_feat": risk_outputs["pred_feat"],
                    "uncertainty": risk_outputs["uncertainty"],
                    "boundary_band": risk_outputs["boundary_band"],
                    "img_conflict": risk_outputs["img_conflict"],
                    "geo_conflict": risk_outputs["geo_conflict"],
                    "prob_saturation": risk_outputs["prob_saturation"],
                    "state_abs": risk_outputs["state_abs"],
                    "state_margin": risk_outputs["state_margin"],
                    "state_boundary": risk_outputs["state_boundary"],
                    "state_local_variation": risk_outputs["state_local_variation"],
                    "structure_signed": risk_outputs["structure_signed"],
                    "structure_anomaly": risk_outputs["structure_anomaly"],
                    "complexity_proxy": risk_outputs["complexity_proxy"],
                    "artifact_proxy": risk_outputs["artifact_proxy"],
                    "gate": selector_outputs["gate"],
                    "update_mask": selector_outputs["update_mask"],
                    "delta_logits": cell_outputs["delta_logits"],
                    "alpha_tensor": cell_outputs["alpha_tensor"],
                    "refined_logits": cell_outputs["refined_logits"],
                    "refined_prob": cell_outputs["refined_prob"],
                }
            )
            pred_seq.append(cell_outputs["refined_logits"])
            alpha_seq.append(cell_outputs["alpha_tensor"])
            current_logit = cell_outputs["refined_logits"]
            current_prob = cell_outputs["refined_prob"]
        return {
            "final_logits": current_logit,
            "final_prob": current_prob,
            "pred_seq": pred_seq,
            "alpha_seq": alpha_seq,
            "step_records": step_records,
        }


def build_geometry_map(method_variant: str, prob: torch.Tensor) -> torch.Tensor | None:
    if str(method_variant).upper() != "M6GEOM":
        return None
    signed_proxy = _signed_geometry_proxy(prob)
    anomaly = _component_anomaly_proxy(prob)
    return torch.cat([signed_proxy, anomaly], dim=1)


def build_structure_lite_map(prob: torch.Tensor) -> torch.Tensor:
    signed_proxy = _signed_geometry_proxy(prob)
    anomaly = _component_anomaly_proxy(prob)
    return torch.cat([signed_proxy, anomaly], dim=1)


def parameter_count(module: nn.Module) -> int:
    return int(sum(param.numel() for param in module.parameters()))


def boundary_iou(pred_mask: torch.Tensor, target_mask: torch.Tensor, width: int = 3) -> torch.Tensor:
    kernel = 2 * int(width) + 1
    pred_band = (_avg_blur(pred_mask, kernel_size=kernel) - _avg_blur(pred_mask, kernel_size=max(3, kernel - 2))).abs() > 1e-3
    tgt_band = (_avg_blur(target_mask, kernel_size=kernel) - _avg_blur(target_mask, kernel_size=max(3, kernel - 2))).abs() > 1e-3
    inter = (pred_band & tgt_band).float().sum(dim=(-2, -1))
    union = (pred_band | tgt_band).float().sum(dim=(-2, -1)).clamp_min(1.0)
    return inter / union


def latency_proxy_ms(feature_hw: tuple[int, int], updated_ratio: float, params: int) -> float:
    h, w = feature_hw
    pixels = float(h * w)
    return float((pixels * max(updated_ratio, 1e-4) * 1e-5) + params * 1e-4 + math.log2(pixels + 1.0))



class RefinementPlugin(nn.Module):
    def __init__(self, plugin_cfg: PluginConfig, sparse_cfg: SparseConfig):
        super().__init__()
        self.plugin_cfg = plugin_cfg
        self.sparse_cfg = sparse_cfg
        self.method_variant = str(plugin_cfg.method_variant).upper()
        self.prior_family = str(plugin_cfg.prior_family).upper()
        self.mixer_family = str(plugin_cfg.mixer_family).upper()
        self.update_rule = str(plugin_cfg.update_rule).upper()
        self.evidence_level = str(plugin_cfg.evidence_level).upper()
        self.capacity_tier = str(getattr(plugin_cfg, "capacity_tier", "R0") or "R0").upper()
        self.recursion_tier = str(getattr(plugin_cfg, "recursion_tier", "K0") or "K0").upper()
        self.recursion_steps = max(1, int(getattr(plugin_cfg, "recursion_steps", 1)))
        self.edge_operator = str(plugin_cfg.edge_operator or "sobel").lower()
        self.strict_evidence_contract = bool(plugin_cfg.strict_evidence_contract)
        self.disable_image_priors_for_correction = bool(plugin_cfg.disable_image_priors_for_correction)
        self.state_aligner = StateAligner(
            state_form=str(plugin_cfg.state_form or "N0").upper(),
            standardize_scope=str(plugin_cfg.standardize_scope or "per_image"),
            standardize_eps=float(plugin_cfg.standardize_eps),
            temperature_mode=str(plugin_cfg.temperature_mode or "fixed"),
            temperature_init=float(plugin_cfg.temperature_init),
            temperature_min=float(plugin_cfg.temperature_min),
            temperature_max=float(plugin_cfg.temperature_max),
        )
        self.prior_extractor = PriorExtractor(
            prior_family=self.prior_family,
            out_channels=int(plugin_cfg.plugin_channels),
            edge_operator=self.edge_operator,
        )
        self.risk_builder = RiskBuilder(
            pred_channels=int(plugin_cfg.plugin_channels),
            edge_operator=self.edge_operator,
            risk_bias=float(plugin_cfg.risk_bias),
            evidence_level=self.evidence_level,
        )
        self.update_selector = UpdateSelector(
            policy=sparse_cfg.policy,
            topk_ratio=sparse_cfg.topk_ratio,
            boundary_width=sparse_cfg.hard_boundary_width,
            random_seed_offset=sparse_cfg.random_seed_offset,
        )
        self.correction_core = CorrectionCore(
            mixer_family=self.mixer_family,
            img_channels=int(plugin_cfg.plugin_channels),
            pred_channels=int(plugin_cfg.plugin_channels),
            hidden_channels=int(plugin_cfg.hidden_channels),
            capacity_tier=self.capacity_tier,
        )
        self.recursive_cell = RecursiveCorrectionCell(
            correction_core=self.correction_core,
            update_rule=self.update_rule,
            update_alpha=float(plugin_cfg.update_alpha),
            alpha_min=float(plugin_cfg.alpha_min),
            alpha_max=float(plugin_cfg.alpha_max),
            dynamic_alpha_hidden=int(plugin_cfg.dynamic_alpha_hidden),
        )
        self.recurrence_controller = RecurrenceController(
            risk_builder=self.risk_builder,
            update_selector=self.update_selector,
            correction_cell=self.recursive_cell,
            recursion_steps=self.recursion_steps,
        )

    def _method_expected_evidence(self) -> str:
        return {
            "M1": "E0",
            "M2": "E1",
            "M3": "E2",
            "M6GEOM": "E3",
        }.get(self.method_variant, self.evidence_level)

    def _evidence_rank(self, evidence_level: str) -> int:
        return {"E0": 0, "E1": 1, "E2": 2, "E3": 3, "E4": 4}.get(str(evidence_level).upper(), -1)

    def _resolve_evidence_contract(self) -> dict[str, object]:
        use_edge = self.evidence_level in {"E1", "E2", "E3", "E4"}
        use_low_high = self.evidence_level in {"E2", "E4"}
        use_spectral = self.evidence_level in {"E2", "E4"}
        use_structure_lite = self.evidence_level in {"E3", "E4"}
        use_complexity_artifact = self.evidence_level == "E4"
        method_floor = self._method_expected_evidence()
        return {
            "level": self.evidence_level,
            "method_variant": self.method_variant,
            "method_floor": method_floor,
            "floor_satisfied": self._evidence_rank(self.evidence_level) >= self._evidence_rank(method_floor),
            "strict_contract": self.strict_evidence_contract,
            "use_edge": use_edge,
            "use_low_high": use_low_high,
            "use_spectral": use_spectral,
            "use_structure_lite": use_structure_lite,
            "use_complexity_artifact": use_complexity_artifact,
        }

    def _zeros_like_prior(self, prob: torch.Tensor) -> dict[str, torch.Tensor]:
        channels = int(self.plugin_cfg.plugin_channels)
        zeros = torch.zeros_like(prob)
        raw_channels = 7
        raw_stack = torch.zeros(
            prob.shape[0],
            raw_channels,
            prob.shape[-2],
            prob.shape[-1],
            device=prob.device,
            dtype=prob.dtype,
        )
        contract = self._resolve_evidence_contract()
        return {
            "edge_map": zeros,
            "low_freq": zeros,
            "high_freq": zeros,
            "geometry_map": torch.zeros(
                prob.shape[0],
                2,
                prob.shape[-2],
                prob.shape[-1],
                device=prob.device,
                dtype=prob.dtype,
            ),
            "structure_lite_map": torch.zeros(
                prob.shape[0],
                2,
                prob.shape[-2],
                prob.shape[-1],
                device=prob.device,
                dtype=prob.dtype,
            ),
            "img_feat": torch.zeros(
                prob.shape[0],
                channels,
                prob.shape[-2],
                prob.shape[-1],
                device=prob.device,
                dtype=prob.dtype,
            ),
            "log_like": zeros,
            "fft_lite": zeros,
            "raw_img_stack": raw_stack,
            "selected_img_stack": raw_stack,
            "evidence_contract": contract,
            "selected_channel_count": 0.0,
            "edge_operator": self.edge_operator,
        }

    def _expand_map(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat(1, int(self.plugin_cfg.plugin_channels), 1, 1)

    def _apply_evidence_contract(self, prior_outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        contract = self._resolve_evidence_contract()
        selected_stack = prior_outputs["raw_img_stack"]
        mask = selected_stack.new_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).view(1, 7, 1, 1)
        if contract["use_edge"]:
            mask[:, 0:1] = 1.0
        if contract["use_low_high"]:
            mask[:, 1:3] = 1.0
        if contract["use_spectral"]:
            mask[:, 3:5] = 1.0
        if contract["use_structure_lite"]:
            mask[:, 5:7] = 1.0

        selected_stack = selected_stack * mask
        prior_outputs["edge_map"] = prior_outputs["edge_map"] if contract["use_edge"] else torch.zeros_like(prior_outputs["edge_map"])
        prior_outputs["low_freq"] = (
            prior_outputs["low_freq"] if contract["use_low_high"] else torch.zeros_like(prior_outputs["low_freq"])
        )
        prior_outputs["high_freq"] = (
            prior_outputs["high_freq"] if contract["use_low_high"] else torch.zeros_like(prior_outputs["high_freq"])
        )
        prior_outputs["log_like"] = (
            prior_outputs["log_like"] if contract["use_spectral"] else torch.zeros_like(prior_outputs["log_like"])
        )
        prior_outputs["fft_lite"] = (
            prior_outputs["fft_lite"] if contract["use_spectral"] else torch.zeros_like(prior_outputs["fft_lite"])
        )
        if not contract["use_structure_lite"]:
            zeros_geom = torch.zeros_like(prior_outputs["geometry_map"])
            prior_outputs["geometry_map"] = zeros_geom
            prior_outputs["structure_lite_map"] = zeros_geom
        prior_outputs["selected_img_stack"] = selected_stack
        prior_outputs["img_feat"] = self.prior_extractor.project_stack(selected_stack)
        prior_outputs["evidence_contract"] = contract
        prior_outputs["selected_channel_count"] = float(mask.sum().item())
        return prior_outputs

    def _build_prior_outputs(self, image: torch.Tensor, prob: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.evidence_level == "E0":
            return self._zeros_like_prior(prob)

        geometry_map = None
        if self.evidence_level in {"E3", "E4"} or self.method_variant == "M6GEOM":
            geometry_map = build_structure_lite_map(prob)
        elif self.plugin_cfg.use_geometry:
            geometry_map = build_geometry_map(self.method_variant, prob)

        prior_outputs = self.prior_extractor(image, geometry_map=geometry_map)
        return self._apply_evidence_contract(prior_outputs)

    def _prepare_img_feat_for_correction(self, img_feat: torch.Tensor) -> torch.Tensor:
        if not self.disable_image_priors_for_correction:
            return img_feat
        return torch.zeros_like(img_feat)

    def forward(
        self,
        image: torch.Tensor,
        logits: torch.Tensor,
        prob: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, float]]:
        prior_outputs = self._build_prior_outputs(image, prob)
        prior_outputs["img_feat_for_correction"] = self._prepare_img_feat_for_correction(prior_outputs["img_feat"])
        aligned = self.state_aligner(logits, prob)
        base_logit = aligned["state_logit"] if aligned["state_logit"].shape[1] == 1 else aligned["state_logit"][:, 1:2]
        recurrence_outputs = self.recurrence_controller(
            img_feat=prior_outputs["img_feat_for_correction"],
            state_logit=base_logit,
            state_prob=aligned["state_prob"],
            edge_map_tensor=prior_outputs["edge_map"],
            geometry_map=prior_outputs["geometry_map"],
            prior_maps=prior_outputs,
        )
        step_records = recurrence_outputs["step_records"]
        final_step = step_records[-1]
        risk_outputs = {
            "pred_feat": final_step["pred_feat"],
            "risk_score": final_step["risk_score"],
            "uncertainty": final_step["uncertainty"],
            "boundary_band": final_step["boundary_band"],
            "img_conflict": final_step["img_conflict"],
            "geo_conflict": final_step["geo_conflict"],
            "prob_saturation": final_step["prob_saturation"],
            "state_abs": final_step["state_abs"],
            "state_margin": final_step["state_margin"],
            "state_boundary": final_step["state_boundary"],
            "state_local_variation": final_step["state_local_variation"],
            "structure_signed": final_step["structure_signed"],
            "structure_anomaly": final_step["structure_anomaly"],
            "complexity_proxy": final_step["complexity_proxy"],
            "artifact_proxy": final_step["artifact_proxy"],
        }
        selector_outputs = {
            "gate": final_step["gate"],
            "update_mask": final_step["update_mask"],
        }
        delta_logits = final_step["delta_logits"]
        alpha_tensor = final_step["alpha_tensor"]
        refined_logits = final_step["refined_logits"]
        refined_prob = final_step["refined_prob"]
        refined_mask = (refined_prob > 0.5).float()

        updated_ratio = float(selector_outputs["update_mask"].detach().mean().item())
        gate_det = selector_outputs["gate"].detach()
        gate_prob = gate_det.clamp(1.0e-6, 1.0 - 1.0e-6)
        delta_det = delta_logits.detach()
        update_mask_det = selector_outputs["update_mask"].detach()
        risk_prob = torch.sigmoid(risk_outputs["risk_score"]).detach()
        selected_pixels = update_mask_det.sum().clamp_min(1.0)
        selected_risk_mean = float((risk_prob * update_mask_det).sum().item() / selected_pixels.item())
        selected_boundary_overlap = float(
            (risk_outputs["boundary_band"].detach() * update_mask_det).sum().item() / selected_pixels.item()
        )
        evidence_contract = prior_outputs["evidence_contract"]
        alpha_stack = torch.stack([alpha.detach().mean() for alpha in recurrence_outputs["alpha_seq"]], dim=0)
        stats = {
            "updated_ratio": updated_ratio,
            "risk_mean": float(torch.sigmoid(risk_outputs["risk_score"]).detach().mean().item()),
            "risk_std": float(torch.sigmoid(risk_outputs["risk_score"]).detach().std(unbiased=False).item()),
            "selected_risk_mean": selected_risk_mean,
            "uncertainty_mean": float(risk_outputs["uncertainty"].detach().mean().item()),
            "prob_saturation_mean": float(risk_outputs["prob_saturation"].detach().mean().item()),
            "state_abs_mean": float(risk_outputs["state_abs"].detach().mean().item()),
            "state_margin_mean": float(risk_outputs["state_margin"].detach().mean().item()),
            "state_boundary_mean": float(risk_outputs["state_boundary"].detach().mean().item()),
            "state_local_variation_mean": float(risk_outputs["state_local_variation"].detach().mean().item()),
            "delta_abs_mean": float(delta_logits.detach().abs().mean().item()),
            "delta_signed_mean": float(delta_det.mean().item()),
            "delta_positive_ratio": float((delta_det > 0.0).float().mean().item()),
            "delta_negative_ratio": float((delta_det < 0.0).float().mean().item()),
            "gate_mean": float(selector_outputs["gate"].detach().mean().item()),
            "gate_std": float(gate_det.std(unbiased=False).item()),
            "gate_entropy": float(
                (-(gate_prob * gate_prob.log() + (1.0 - gate_prob) * (1.0 - gate_prob).log())).mean().item()
            ),
            "alpha_mean": float(alpha_tensor.detach().mean().item()),
            "alpha_std": float(alpha_tensor.detach().std(unbiased=False).item()),
            "alpha_seq_mean": float(alpha_stack.mean().item()),
            "alpha_seq_last": float(alpha_stack[-1].item()),
            "state_mean": float(base_logit.detach().mean().item()),
            "state_std": float(base_logit.detach().std(unbiased=False).item()),
            "state_min": float(base_logit.detach().amin().item()),
            "state_max": float(base_logit.detach().amax().item()),
            "refined_prob_mean": float(refined_prob.detach().mean().item()),
            "selected_boundary_overlap": selected_boundary_overlap,
            "img_feat_abs_mean": float(prior_outputs["img_feat"].detach().abs().mean().item()),
            "img_feat_for_correction_abs_mean": float(
                prior_outputs["img_feat_for_correction"].detach().abs().mean().item()
            ),
            "edge_mean": float(prior_outputs["edge_map"].detach().mean().item()),
            "low_freq_mean": float(prior_outputs["low_freq"].detach().mean().item()),
            "high_freq_mean": float(prior_outputs["high_freq"].detach().mean().item()),
            "structure_lite_mean": float(prior_outputs["structure_lite_map"].detach().abs().mean().item()),
            "complexity_mean": float(risk_outputs["complexity_proxy"].detach().mean().item()),
            "artifact_mean": float(risk_outputs["artifact_proxy"].detach().mean().item()),
            "selected_channel_count": float(prior_outputs["selected_channel_count"]),
            "evidence_level_e0": 1.0 if self.evidence_level == "E0" else 0.0,
            "evidence_level_e1": 1.0 if self.evidence_level == "E1" else 0.0,
            "evidence_level_e2": 1.0 if self.evidence_level == "E2" else 0.0,
            "evidence_level_e3": 1.0 if self.evidence_level == "E3" else 0.0,
            "evidence_level_e4": 1.0 if self.evidence_level == "E4" else 0.0,
            "evidence_use_edge": 1.0 if evidence_contract["use_edge"] else 0.0,
            "evidence_use_low_high": 1.0 if evidence_contract["use_low_high"] else 0.0,
            "evidence_use_spectral": 1.0 if evidence_contract["use_spectral"] else 0.0,
            "evidence_use_structure_lite": 1.0 if evidence_contract["use_structure_lite"] else 0.0,
            "evidence_use_complexity_artifact": 1.0 if evidence_contract["use_complexity_artifact"] else 0.0,
            "method_evidence_floor_satisfied": 1.0 if evidence_contract["floor_satisfied"] else 0.0,
            "recursion_steps": float(self.recursion_steps),
            "recursion_tier_k0": 1.0 if self.recursion_tier == "K0" else 0.0,
            "recursion_tier_k1": 1.0 if self.recursion_tier == "K1" else 0.0,
            "recursion_tier_k2": 1.0 if self.recursion_tier == "K2" else 0.0,
            "capacity_tier_r0": 1.0 if self.capacity_tier == "R0" else 0.0,
            "capacity_tier_r1": 1.0 if self.capacity_tier == "R1" else 0.0,
            "capacity_tier_r3": 1.0 if self.capacity_tier == "R3" else 0.0,
            "disable_image_priors_for_correction": 1.0 if self.disable_image_priors_for_correction else 0.0,
            "plugin_params": float(parameter_count(self)),
            "latency_proxy_ms": float(
                latency_proxy_ms(
                    feature_hw=(int(logits.shape[-2]), int(logits.shape[-1])),
                    updated_ratio=updated_ratio,
                    params=parameter_count(self),
                )
            ),
        }
        if target is not None:
            target_bin = normalize_target_mask(target).unsqueeze(1)
            stats["boundary_iou"] = float(boundary_iou(refined_mask, target_bin).mean().detach().item())

        return {
            "refined_logits": refined_logits,
            "refined_prob": refined_prob,
            "refined_mask": refined_mask,
            "delta_logits": delta_logits,
            "gate": selector_outputs["gate"],
            "update_mask": selector_outputs["update_mask"],
            "risk_score": risk_outputs["risk_score"],
            "prior_outputs": prior_outputs,
            "risk_outputs": risk_outputs,
            "aligned_state": aligned,
            "pred_seq": recurrence_outputs["pred_seq"],
            "alpha_seq": recurrence_outputs["alpha_seq"],
            "step_records": step_records,
            "stats": stats,
        }
__all__ = [
    "HostAdapter",
    "HostForwardBundle",
    "AnchorAdapterA",
    "AnchorAdapterB",
    "UNetAdapter",
    "I2UNetAdapter",
    "build_host_adapter",
    "HostConfig",
    "PluginConfig",
    "SparseConfig",
    "TrainConfig",
    "StateAlignmentConfig",
    "UpdateRuleConfig",
    "EvidenceLevelConfig",
    "BudgetPolicyConfig",
    "ExperimentConfig",
    "PriorExtractor",
    "RiskBuilder",
    "UpdateSelector",
    "StateAligner",
    "CorrectionCore",
    "RecursiveCorrectionCell",
    "RecurrenceController",
    "RefinementPlugin",
]
