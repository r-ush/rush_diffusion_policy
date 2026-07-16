"""Modality attribution core library.

롤아웃 중 policy의 action이 어느 modality(vision / wrench(force) / low_dim)에
더 좌우되는지 진단하기 위한 핵심 도구 모음.

지원하는 방법:
  1) Counterfactual ablation  : 특정 modality만 baseline으로 교체 후 action 변화(Δ) 측정 (가장 권장)
  2) Gradient saliency        : 최종 action을 global_cond로 backprop 해서 구간별 gradient norm 측정
  3) Feature-slice 유틸        : concat fuse_mode에서 global_cond 안의 modality별 구간(offset) 계산

설계 원칙:
  - policy 코드(predict_action)를 그대로 재사용한다. 재구현하지 않는다.
  - diffusion sampling의 randomness는 generator seed로 고정한다. 안 그러면
    Δ가 "modality 차이"가 아니라 "noise 차이"를 재게 된다.
  - action 비교는 normalized action 공간(기본)에서 하고, 위치(xyz)는 물리 단위(m)로도 함께 잰다.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply


# ---------------------------------------------------------------------------
# obs_dict helpers
# ---------------------------------------------------------------------------

def obs_np_to_tensor(obs_dict_np: Dict[str, np.ndarray], device) -> Dict[str, torch.Tensor]:
    """recorder가 저장한 배치 없는 numpy obs를 predict_action이 먹는 (B=1, ...) 텐서로 변환.

    eval 스크립트의 `dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))`
    와 동일하게 맞춘다.
    """
    return dict_apply(
        obs_dict_np,
        lambda x: torch.from_numpy(np.asarray(x)).unsqueeze(0).to(device),
    )


def clone_obs(obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in obs_dict.items()}


# ---------------------------------------------------------------------------
# feature layout (concat fuse_mode 전용)
# ---------------------------------------------------------------------------

@dataclass
class FeatureLayout:
    """concat fuse_mode에서 global_cond(=obs feature) 안의 modality별 구간.

    global_cond = cat(vision_features + [low_dim] + force_features)  (predict_action 참고)
    """
    vision: slice
    low_dim: slice
    force: slice
    total: int

    def as_dict(self) -> Dict[str, slice]:
        return {"vision": self.vision, "low_dim": self.low_dim, "force": self.force}


def compute_feature_layout(policy) -> FeatureLayout:
    """policy 속성으로부터 concat global_cond 구간을 계산한다."""
    To = int(policy.n_obs_steps)
    vision_dim = int(getattr(policy, "vision_feature_dim", 0)) * len(policy.rgb_keys) * To
    low_dim_dim = int(getattr(policy, "num_low_dim_component", 0)) * To
    force_dim = int(getattr(policy, "force_feature_dim", 0)) * int(getattr(policy, "force_obs_steps", 0))

    v0, v1 = 0, vision_dim
    l0, l1 = v1, v1 + low_dim_dim
    f0, f1 = l1, l1 + force_dim
    return FeatureLayout(
        vision=slice(v0, v1),
        low_dim=slice(l0, l1),
        force=slice(f0, f1),
        total=f1,
    )


# ---------------------------------------------------------------------------
# deterministic prediction
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _fixed_generator(policy, seed: int, device):
    """predict_action이 내부에서 쓰는 sampling generator를 seed로 고정.

    conditional_sample(..., **self.kwargs) 경로로 generator가 전달되므로
    policy.kwargs['generator']에 넣어주면 초기 noise와 scheduler.step이 모두 결정적이 된다.
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    prev = policy.kwargs.get("generator", None) if isinstance(policy.kwargs, dict) else None
    had = isinstance(policy.kwargs, dict) and ("generator" in policy.kwargs)
    policy.kwargs["generator"] = gen
    try:
        yield gen
    finally:
        if had:
            policy.kwargs["generator"] = prev
        else:
            policy.kwargs.pop("generator", None)


@torch.no_grad()
def predict_action(policy, obs_dict: Dict[str, torch.Tensor], seed: int) -> Dict[str, torch.Tensor]:
    """seed 고정된 결정적 action 예측. result['action'] = (B, n_action_steps, Da) (unnormalized)."""
    device = policy.device
    with _fixed_generator(policy, seed, device):
        return policy.predict_action(obs_dict)


# ---------------------------------------------------------------------------
# action distance metrics
# ---------------------------------------------------------------------------

@dataclass
class ActionDelta:
    total: float          # normalized action 공간 전체 L2 (step 평균)
    pos: float            # xyz 위치 차이 (m, step 평균)  -- 물리적으로 해석 가능
    rot: float            # 회전 파트(rot6d 등) normalized L2 (step 평균)

    def as_dict(self) -> Dict[str, float]:
        return {"total": self.total, "pos": self.pos, "rot": self.rot}


def action_delta(policy, action_a: torch.Tensor, action_b: torch.Tensor,
                 pos_dims: int = 3, arm_dim: int = 9) -> ActionDelta:
    """두 action( (B, S, Da) unnormalized )의 차이를 여러 관점으로 잰다.

    - total: normalizer로 정규화 후 전 차원 L2 (스케일 편향 제거)
    - pos  : 앞 3차원(xyz) 유클리드 거리 (m)
    - rot  : arm 회전 파트(3:arm_dim) normalized L2
    """
    a = action_a.detach().float()
    b = action_b.detach().float()

    na = policy.normalizer["action"].normalize(a)
    nb = policy.normalizer["action"].normalize(b)
    total = torch.linalg.norm(na - nb, dim=-1).mean().item()

    pos = torch.linalg.norm(a[..., :pos_dims] - b[..., :pos_dims], dim=-1).mean().item()

    if na.shape[-1] >= arm_dim:
        rot = torch.linalg.norm(na[..., pos_dims:arm_dim] - nb[..., pos_dims:arm_dim], dim=-1).mean().item()
    else:
        rot = float("nan")
    return ActionDelta(total=total, pos=pos, rot=rot)


# ---------------------------------------------------------------------------
# baseline builders (ablation용 반사실 obs 생성기)
# ---------------------------------------------------------------------------

def make_zero_wrench(policy) -> Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """wrench 채널을 0(=무접촉 raw 값)으로 지우는 반사실. 'force가 없었다면?'"""
    def fn(obs_dict):
        out = clone_obs(obs_dict)
        for key in policy.wrench_keys:
            if key in out:
                out[key] = torch.zeros_like(out[key])
        return out
    return fn


def make_freeze_vision(policy, frozen_obs: Dict[str, torch.Tensor]) -> Callable:
    """모든 rgb 스텝을 기준 프레임(frozen_obs의 마지막 스텝)으로 고정. '화면이 안 바뀌었다면?'"""
    frozen_frame = {}
    for key in policy.rgb_keys:
        if key in frozen_obs:
            # (B, To, C, H, W) -> 마지막 스텝 (B, 1, C, H, W)
            frozen_frame[key] = frozen_obs[key][:, -1:, ...].clone()

    def fn(obs_dict):
        out = clone_obs(obs_dict)
        for key in policy.rgb_keys:
            if key in out and key in frozen_frame:
                To = out[key].shape[1]
                out[key] = frozen_frame[key].expand(-1, To, *frozen_frame[key].shape[2:]).clone()
        return out
    return fn


def make_replace_vision(policy, value_obs: Dict[str, torch.Tensor]) -> Callable:
    """rgb를 임의의 기준 obs 값으로 통째 교체(예: 데이터셋 평균 이미지)."""
    ref = {k: value_obs[k].clone() for k in policy.rgb_keys if k in value_obs}

    def fn(obs_dict):
        out = clone_obs(obs_dict)
        for key in policy.rgb_keys:
            if key in out and key in ref:
                out[key] = ref[key].expand_as(out[key]).clone()
        return out
    return fn


def make_blank_vision(policy) -> Callable:
    """rgb를 (그 프레임의) 채널 평균색으로 통째 교체 = '시각 정보 제거'.

    zero-wrench('힘이 없었다면')의 vision 대응('안 보였다면'). freeze-to-start와 달리
    로봇 이동량에 의존하지 않아 vision vs wrench 를 공정하게 비교할 수 있다.
    """
    def fn(obs_dict):
        out = clone_obs(obs_dict)
        for key in policy.rgb_keys:
            if key in out:
                x = out[key]                       # (B, T, C, H, W)
                mean = x.mean(dim=(1, 3, 4), keepdim=True)   # (B,1,C,1,1) 채널 평균
                out[key] = mean.expand_as(x).clone()
        return out
    return fn


def make_replace_low_dim(policy, value_obs: Dict[str, torch.Tensor]) -> Callable:
    """low_dim(pose 등)을 기준 obs로 교체."""
    ref = {k: value_obs[k].clone() for k in policy.low_dim_keys if k in value_obs}

    def fn(obs_dict):
        out = clone_obs(obs_dict)
        for key in policy.low_dim_keys:
            if key in out and key in ref:
                out[key] = ref[key].expand_as(out[key]).clone()
        return out
    return fn


# ---------------------------------------------------------------------------
# 1) counterfactual ablation
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    base_action: torch.Tensor                     # (B, S, Da)
    deltas: Dict[str, ActionDelta] = field(default_factory=dict)
    seeds: Sequence[int] = ()


def ablation_deltas(
    policy,
    obs_dict: Dict[str, torch.Tensor],
    baselines: Dict[str, Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]],
    seeds: Sequence[int] = (0, 1, 2),
) -> AblationResult:
    """각 baseline(modality 교체기)에 대해 base action과의 Δ를 여러 seed 평균으로 계산.

    Δ가 큰 modality가 현재 action을 더 지배한다.

    baselines 예:
        {"wrench": make_zero_wrench(policy),
         "vision": make_freeze_vision(policy, obs_dict)}
    """
    seeds = list(seeds)

    # base action: seed별로 뽑아 두고, 같은 seed끼리 비교(순수 conditioning 차이만 측정)
    base_actions = {s: predict_action(policy, obs_dict, s)["action"] for s in seeds}

    deltas: Dict[str, List[ActionDelta]] = {name: [] for name in baselines}
    for name, builder in baselines.items():
        ablated = builder(obs_dict)
        for s in seeds:
            act = predict_action(policy, ablated, s)["action"]
            deltas[name].append(action_delta(policy, base_actions[s], act))

    mean_deltas: Dict[str, ActionDelta] = {}
    for name, lst in deltas.items():
        mean_deltas[name] = ActionDelta(
            total=float(np.mean([d.total for d in lst])),
            pos=float(np.mean([d.pos for d in lst])),
            rot=float(np.mean([d.rot for d in lst])),
        )

    return AblationResult(
        base_action=base_actions[seeds[0]],
        deltas=mean_deltas,
        seeds=seeds,
    )


# ---------------------------------------------------------------------------
# 1b) joint / conditional ablation  (중복 vs 진짜-미사용 판정)
# ---------------------------------------------------------------------------

def compose_baselines(*builders: Callable) -> Callable:
    """여러 baseline 빌더를 순차 적용(= 여러 modality 동시 제거).

    각 빌더는 서로 다른 키만 건드리므로(vision↔wrench↔low_dim) 적용 순서는 무관.
    """
    def fn(obs_dict):
        out = obs_dict
        for b in builders:
            out = b(out)
        return out
    return fn


@dataclass
class InteractionDeltas:
    """단일/동시/조건부 ablation Δ (모두 action_delta.total, seed 평균).

    핵심은 conditional: 한 modality를 이미 지운 상태에서 '다른 하나'를 더 지웠을 때의 Δ.
    - wrench_given_no_vision 이 크면 → vision이 빠지면 wrench가 대응 = '중복에 가려짐(보완 가능)'.
    - wrench_given_no_vision 이 ~0 이면 → vision이 없어도 wrench 반응 없음 = '진짜 미사용'.
    """
    vision: float                    # base 대비 vision 제거 Δ
    wrench: float                    # base 대비 wrench 제거 Δ
    both: float                      # base 대비 vision+wrench 동시 제거 Δ
    wrench_given_no_vision: float    # d(no_vision, no_vision_no_wrench)
    vision_given_no_wrench: float    # d(no_wrench, no_vision_no_wrench)

    @property
    def redundancy(self) -> float:
        """(Δvision+Δwrench − Δboth). >0 = 정보 중복(동시 제거가 합보다 덜 바뀜)."""
        return self.vision + self.wrench - self.both

    def as_dict(self) -> Dict[str, float]:
        return {
            "vision": self.vision, "wrench": self.wrench, "both": self.both,
            "wrench_given_no_vision": self.wrench_given_no_vision,
            "vision_given_no_wrench": self.vision_given_no_wrench,
            "redundancy": self.redundancy,
        }


def interaction_deltas(
    policy,
    obs_dict: Dict[str, torch.Tensor],
    vision_baseline: Callable,
    wrench_baseline: Callable,
    seeds: Sequence[int] = (0, 1),
) -> InteractionDeltas:
    """vision/wrench 의 단일·동시·조건부 ablation Δ를 한 번에 계산한다.

    같은 seed 안에서 base / no_vision / no_wrench / no_both 를 모두 뽑아 짝지어 비교한다.
    """
    seeds = list(seeds)

    def d(a, b):
        return action_delta(policy, a, b).total

    keys = ["vision", "wrench", "both", "wrench_given_no_vision", "vision_given_no_wrench"]
    accum: Dict[str, List[float]] = {k: [] for k in keys}

    for s in seeds:
        obs_noV = vision_baseline(obs_dict)
        obs_noW = wrench_baseline(obs_dict)
        obs_noVW = wrench_baseline(obs_noV)   # vision 지운 obs 위에 wrench까지 제거

        base = predict_action(policy, obs_dict, s)["action"]
        a_noV = predict_action(policy, obs_noV, s)["action"]
        a_noW = predict_action(policy, obs_noW, s)["action"]
        a_noVW = predict_action(policy, obs_noVW, s)["action"]

        accum["vision"].append(d(base, a_noV))
        accum["wrench"].append(d(base, a_noW))
        accum["both"].append(d(base, a_noVW))
        accum["wrench_given_no_vision"].append(d(a_noV, a_noVW))
        accum["vision_given_no_wrench"].append(d(a_noW, a_noVW))

    return InteractionDeltas(**{k: float(np.mean(v)) for k, v in accum.items()})


# ---------------------------------------------------------------------------
# 2) gradient saliency (global_cond 구간별)  -- 실험적
# ---------------------------------------------------------------------------

@dataclass
class GradSaliency:
    per_group: Dict[str, float]      # 구간별 gradient norm (차원 정규화)
    raw_grad_norm: float             # 전체 global_cond grad norm


def gradient_saliency(
    policy,
    obs_dict: Dict[str, torch.Tensor],
    seed: int = 0,
    normalize_by_dim: bool = True,
) -> Optional[GradSaliency]:
    """최종 action의 크기를 global_cond로 backprop 해서 vision/low_dim/force 구간별 민감도 측정.

    concat fuse_mode에서만 구간 분해가 의미 있다. modality-attention에서는 None을 돌려준다.
    16-step diffusion loop 전체에 그래프가 쌓이므로 B=1에서만 쓰는 걸 권장.
    """
    if getattr(policy, "fuse_mode", "concat") != "concat":
        return None

    layout = compute_feature_layout(policy)
    captured: Dict[str, torch.Tensor] = {}

    orig = policy.conditional_sample

    def wrapper(condition_data, condition_mask, local_cond=None, global_cond=None,
                generator=None, **kw):
        gc = global_cond.detach().clone().requires_grad_(True)
        captured["gc"] = gc
        return orig(condition_data, condition_mask,
                    local_cond=local_cond, global_cond=gc, generator=generator, **kw)

    device = policy.device
    policy.conditional_sample = wrapper
    try:
        with _fixed_generator(policy, seed, device):
            with torch.enable_grad():
                result = policy.predict_action(obs_dict)
                action = result["action"]
                # 실행 구간 action 크기를 스칼라로
                scalar = action.pow(2).sum()
                scalar.backward()
    finally:
        policy.conditional_sample = orig

    gc = captured.get("gc", None)
    if gc is None or gc.grad is None:
        return None
    grad = gc.grad[0].detach().float().cpu()  # (obs_feature_dim,)

    per_group = {}
    for name, sl in layout.as_dict().items():
        g = grad[sl]
        if g.numel() == 0:
            per_group[name] = float("nan")
            continue
        val = torch.linalg.norm(g).item()
        if normalize_by_dim:
            val = val / (g.numel() ** 0.5)
        per_group[name] = val

    return GradSaliency(
        per_group=per_group,
        raw_grad_norm=torch.linalg.norm(grad).item(),
    )


# ---------------------------------------------------------------------------
# 4) modality-attention weight capture  -- modality-attention fuse_mode 전용, 실험적
# ---------------------------------------------------------------------------

def capture_modality_attention(policy, obs_dict: Dict[str, torch.Tensor], seed: int = 0):
    """transformer_encoder self-attention의 토큰별 attention weight를 뽑는다.

    token 순서 = [vision tokens (rgb_keys x n_obs_steps), wrench token 1개].
    반환: (attn_weights (n_tokens, n_tokens) 평균, token_labels) 또는 None.

    주의: attention은 attribution의 근사일 뿐이다. 1)/2)와 교차검증용으로만 쓸 것.
    """
    if getattr(policy, "fuse_mode", None) != "modality-attention":
        return None
    enc = getattr(policy, "transformer_encoder", None)
    if enc is None:
        return None

    captured = {}
    self_attn = enc.self_attn
    orig_forward = self_attn.forward

    def patched(query, key, value, **kw):
        kw["need_weights"] = True
        kw["average_attn_weights"] = True
        out, w = orig_forward(query, key, value, **kw)
        captured["w"] = w.detach().float().cpu()
        return out, w

    self_attn.forward = patched
    try:
        predict_action(policy, obs_dict, seed)
    finally:
        self_attn.forward = orig_forward

    w = captured.get("w", None)
    if w is None:
        return None

    labels = []
    for key in policy.rgb_keys:
        for t in range(policy.n_obs_steps):
            labels.append(f"{key}#t{t}")
    for _ in range(int(getattr(policy, "force_obs_steps", 0))):
        labels.append("wrench")
    return w[0], labels
