from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from model.blocks import polar_reduce, polar_temp_null
from train.model import (
    CausalSelfAttention,
    LinearNoBias,
    PolarAttention,
    Rotary,
    causal_conv1d_fn,
)

try:
    from kernel.polar_triton import HAS_TRITON as HAS_POLAR_TRITON
    from kernel.polar_triton import polar_attention_sparse
except Exception:  # pragma: no cover - CPU-only environments do not ship Triton.
    HAS_POLAR_TRITON = False
    polar_attention_sparse = None

try:
    from torch.nn.attention.flex_attention import BlockMask, flex_attention

    try:
        from torch.nn.attention.flex_attention import AuxRequest
    except ImportError:  # PyTorch < 2.9
        AuxRequest = None

    HAS_FLEX_ATTENTION = True
except Exception:  # pragma: no cover - depends on the training PyTorch build.
    BlockMask = None
    flex_attention = None
    AuxRequest = None
    HAS_FLEX_ATTENTION = False


@dataclass
class Route:
    page_indices: Tensor  # (B, Q_blocks, remote_capacity)
    page_counts: Tensor  # (B, Q_blocks)
    local_indices: Tensor  # (B, Q_blocks, max_local_blocks)
    local_counts: Tensor  # (B, Q_blocks)
    page_scores: Tensor  # differentiable (B, Q_blocks, pages)
    cap_rate: Tensor
    mean_remote_pages: Tensor
    local_only_rate: Tensor


def _masked_softmax(scores: Tensor, mask: Tensor) -> Tensor:
    """Softmax that returns all zeros for rows without a valid entry."""

    safe = scores.float().masked_fill(~mask, -torch.inf)
    row_max = safe.amax(dim=-1, keepdim=True)
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    numer = torch.exp(safe - row_max) * mask
    return numer / numer.sum(dim=-1, keepdim=True).clamp_min(1e-20)


def select_pages(
    page_scores: Tensor,
    *,
    page_size: int,
    local_window: int,
    top_p: float,
    min_remote_pages: int,
    max_remote_pages: int,
    remote_capacity: int,
) -> Route:
    """Select causal remote pages for each query block.

    The route for a block is computed from its first query token. Completed pages in
    the local window contribute to the predicted mass but are not returned as remote
    pages. This prevents future tokens in the query block from influencing an earlier
    query's sparse support.
    """

    batch, query_blocks, pages = page_scores.shape
    if query_blocks != pages:
        raise ValueError("the pilot requires one query block per KV page")
    if local_window % page_size:
        raise ValueError("local_window must be divisible by page_size")
    if max_remote_pages > remote_capacity:
        raise ValueError("max_remote_pages exceeds the allocated route capacity")

    device = page_scores.device
    qb = torch.arange(query_blocks, device=device)
    pb = torch.arange(pages, device=device)
    local_page_span = local_window // page_size

    # At the first token of query block q, pages [0, q) are complete. The
    # page q itself contains the query and future tokens and is never indexed.
    complete = pb.view(1, pages) < qb.view(query_blocks, 1)
    first_local = (qb - local_page_span).clamp_min(0)
    local_complete = complete & (pb.view(1, pages) >= first_local[:, None])
    remote = complete & (pb.view(1, pages) < first_local[:, None])

    probs = _masked_softmax(page_scores, complete[None].expand(batch, -1, -1))
    local_mass = (probs * local_complete[None]).sum(dim=-1)
    remote_probs = probs.masked_fill(~remote[None], -1.0)
    sorted_prob, sorted_idx = remote_probs.sort(dim=-1, descending=True)
    sorted_prob = sorted_prob.clamp_min(0.0)
    cumulative = sorted_prob.cumsum(dim=-1)
    target = (float(top_p) - local_mass).clamp_min(0.0)
    available = remote.sum(dim=-1).view(1, query_blocks).expand(batch, -1)

    # Smallest n for which cumulative[n-1] >= target. A zero target permits
    # no fetch; otherwise add one to the count of entries still below target.
    needed = (cumulative < target[..., None]).sum(dim=-1) + (target > 0).long()
    needed = torch.where(available > 0, needed, torch.zeros_like(needed))
    if min_remote_pages:
        needed = torch.where(
            available > 0,
            torch.maximum(needed, needed.new_full((), min_remote_pages)),
            needed,
        )
    counts = torch.minimum(needed, available)
    counts = counts.clamp_max(max_remote_pages)

    indices = sorted_idx[..., :remote_capacity].to(torch.int32)
    if indices.shape[-1] < remote_capacity:
        indices = F.pad(indices, (0, remote_capacity - indices.shape[-1]))

    # Local blocks are partial blocks because exact causality/windowing is
    # applied inside them. There are local_page_span + 1 blocks at most: the
    # current page plus the possibly partial oldest local page.
    max_local = local_page_span + 1
    offsets = torch.arange(max_local, device=device)
    local_start = (qb - local_page_span).clamp_min(0)
    local_indices = local_start[:, None] + offsets[None]
    local_counts = qb - local_start + 1
    local_valid = offsets[None] < local_counts[:, None]
    local_indices = torch.where(local_valid, local_indices, torch.zeros_like(local_indices))
    local_indices = local_indices[None].expand(batch, -1, -1).to(torch.int32)
    local_counts = local_counts[None].expand(batch, -1).to(torch.int32)

    cap_rate = (counts >= max_remote_pages).float().mean() if max_remote_pages else counts.new_zeros((), dtype=torch.float32)
    return Route(
        page_indices=indices,
        page_counts=counts.to(torch.int32),
        local_indices=local_indices,
        local_counts=local_counts,
        page_scores=page_scores,
        cap_rate=cap_rate,
        mean_remote_pages=counts.float().mean(),
        local_only_rate=(counts == 0).float().mean(),
    )


def _local_mask(local_window: int) -> Callable:
    def mask_mod(batch, head, q_idx, kv_idx):
        del batch, head
        return (kv_idx <= q_idx) & (kv_idx > q_idx - local_window)

    return mask_mod


def build_block_mask(route: Route, *, heads: int, block_size: int, local_window: int):
    if not HAS_FLEX_ATTENTION:
        raise RuntimeError("PyTorch FlexAttention is unavailable")
    partial_counts = route.local_counts[:, None].expand(-1, heads, -1).contiguous()
    partial_indices = route.local_indices[:, None].expand(-1, heads, -1, -1).contiguous()
    full_counts = route.page_counts[:, None].expand(-1, heads, -1).contiguous()
    full_indices = route.page_indices[:, None].expand(-1, heads, -1, -1).contiguous()
    return BlockMask.from_kv_blocks(
        partial_counts,
        partial_indices,
        full_counts,
        full_indices,
        BLOCK_SIZE=(block_size, block_size),
        mask_mod=_local_mask(local_window),
    )


def _dense_allowed_mask(route: Route, *, sequence_length: int, block_size: int, local_window: int) -> Tensor:
    """Portable reference mask. It is deliberately forbidden for long sequences."""

    if sequence_length > 4096:
        raise RuntimeError(
            "the dense sparse-attention reference is limited to 4096 tokens; "
            "run 32K CPT on CUDA with the configured sparse kernel"
        )
    batch, query_blocks = route.page_counts.shape
    device = route.page_counts.device
    q = torch.arange(sequence_length, device=device)
    k = torch.arange(sequence_length, device=device)
    allowed = (k[None, :] <= q[:, None]) & (k[None, :] > q[:, None] - local_window)
    allowed = allowed[None].expand(batch, -1, -1).clone()
    for b in range(batch):
        for qb in range(query_blocks):
            q0, q1 = qb * block_size, min((qb + 1) * block_size, sequence_length)
            count = int(route.page_counts[b, qb].item())
            for slot in range(count):
                page = int(route.page_indices[b, qb, slot].item())
                k0, k1 = page * block_size, min((page + 1) * block_size, sequence_length)
                allowed[b, q0:q1, k0:k1] = True
    return allowed


class FovealAttention(nn.Module):
    """Wrap a trained ATMA attention layer with a learned MQA page index."""

    def __init__(
        self,
        base: CausalSelfAttention | PolarAttention,
        *,
        hidden_size: int,
        index_dim: int,
        page_size: int,
        local_window: int,
        remote_capacity: int,
        top_p: float,
        min_remote_pages: int,
        max_remote_pages: int,
        teacher_query_blocks: int,
        teacher_interval: int,
        teacher_mean_weight: float,
        adaptation_mode: str,
        compile_flex: bool,
        flex_kernel_options: dict | None = None,
    ):
        super().__init__()
        if not isinstance(base, (CausalSelfAttention, PolarAttention)):
            raise TypeError(f"unsupported ATMA attention module: {type(base).__name__}")
        self.base = base
        self.index_q = LinearNoBias(hidden_size, index_dim)
        self.index_k = LinearNoBias(hidden_size, index_dim)
        self.index_v = LinearNoBias(hidden_size, index_dim)
        self.index_out = LinearNoBias(index_dim, hidden_size)
        nn.init.normal_(self.index_q.weight, std=hidden_size ** -0.5)
        nn.init.normal_(self.index_k.weight, std=hidden_size ** -0.5)
        nn.init.normal_(self.index_v.weight, std=hidden_size ** -0.5)
        # Start as a small residual, while keeping a non-zero first-step gradient
        # for the 16D q/k/v stream in the LM-output ablations.
        nn.init.normal_(self.index_out.weight, std=1e-3)
        self.index_rotary = Rotary(index_dim) if getattr(base, "pos", None) == "rope" else None

        self.hidden_size = hidden_size
        self.index_dim = index_dim
        self.page_size = page_size
        self.local_window = local_window
        self.remote_capacity = remote_capacity
        self.top_p = top_p
        self.min_remote_pages = min_remote_pages
        self.max_remote_pages = max_remote_pages
        self.teacher_query_blocks = teacher_query_blocks
        self.teacher_interval = teacher_interval
        self.teacher_mean_weight = teacher_mean_weight
        self.adaptation_mode = adaptation_mode
        self.compile_flex = compile_flex
        self.flex_kernel_options = flex_kernel_options
        self.mode = "sparse"
        self.step = 0
        self.last_stats: dict[str, Tensor] = {}
        self._last_teacher_stats: dict[str, Tensor] = {}
        self._compiled_flex = None

    @property
    def num_heads(self) -> int:
        return self.base.num_heads

    @property
    def num_kv_heads(self) -> int:
        return self.base.num_kv_heads

    @property
    def head_dim(self) -> int:
        return self.base.head_dim

    @property
    def is_polar(self) -> bool:
        return isinstance(self.base, PolarAttention)

    def set_mode(self, mode: str) -> None:
        if mode not in {"sparse", "dense_teacher"}:
            raise ValueError(f"unknown Foveal attention mode: {mode}")
        self.mode = mode

    def set_step(self, step: int) -> None:
        self.step = int(step)

    def set_route(self, top_p: float, min_remote_pages: int, max_remote_pages: int) -> None:
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must lie in (0, 1]")
        if not 0 <= min_remote_pages <= max_remote_pages <= self.remote_capacity:
            raise ValueError("invalid remote page limits")
        self.top_p = float(top_p)
        self.min_remote_pages = int(min_remote_pages)
        self.max_remote_pages = int(max_remote_pages)

    @property
    def uses_lm_output(self) -> bool:
        return self.adaptation_mode in {"lm_output", "lm_output_kl"}

    @property
    def uses_kl(self) -> bool:
        return self.adaptation_mode in {"kl", "lm_output_kl"}

    def _index(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch, tokens, _ = x.shape
        if tokens % self.page_size:
            raise ValueError(f"token length {tokens} must be divisible by page_size={self.page_size}")
        # The auxiliary retrieval objective trains the index projections, not
        # the checkpoint's hidden-state geometry. The backbone still adapts via
        # the sparse LM objective during CPT.
        source = x.detach()
        qi = F.rms_norm(self.index_q(source), (self.index_dim,))
        ki = F.rms_norm(self.index_k(source), (self.index_dim,))
        vi = self.index_v(source)
        if self.index_rotary is not None:
            qi = self.index_rotary(qi[:, :, None, :]).squeeze(2)
            ki = self.index_rotary(ki[:, :, None, :]).squeeze(2)
        pages = tokens // self.page_size
        # First query in each block is the causal routing representative.
        q_block = qi[:, :: self.page_size]
        k_page = ki.view(batch, pages, self.page_size, self.index_dim).mean(dim=2)
        q_block = F.normalize(q_block.float(), dim=-1)
        k_page = F.normalize(k_page.float(), dim=-1)
        scores = torch.einsum("bqr,bpr->bqp", q_block, k_page) / math.sqrt(self.index_dim)
        v_page = vi.view(batch, pages, self.page_size, self.index_dim).mean(dim=2)
        return scores, v_page

    def _index_output(self, page_scores: Tensor, v_page: Tensor) -> Tensor:
        """Return the causal 16D MQA index stream in model width.

        Page selection remains discrete for the sparse attention kernel, while
        this soft page read makes q/k/v and the output projection trainable from
        the ordinary LM objective.
        """

        _, query_blocks, pages = page_scores.shape
        q_page = torch.arange(query_blocks, device=page_scores.device)
        k_page = torch.arange(pages, device=page_scores.device)
        complete = k_page[None, :] < q_page[:, None]
        probs = _masked_softmax(
            page_scores,
            complete[None].expand(page_scores.shape[0], -1, -1),
        ).to(v_page.dtype)
        context = torch.einsum("bqp,bpr->bqr", probs, v_page)
        context = context.repeat_interleave(self.page_size, dim=1)
        return self.index_out(context)

    def _route(self, page_scores: Tensor) -> Route:
        # Page selection itself is discrete; the differentiable score tensor is
        # retained separately for teacher distillation.
        detached = page_scores.detach()
        route = select_pages(
            detached,
            page_size=self.page_size,
            local_window=self.local_window,
            top_p=self.top_p,
            min_remote_pages=self.min_remote_pages,
            max_remote_pages=self.max_remote_pages,
            remote_capacity=self.remote_capacity,
        )
        route.page_scores = page_scores
        return route

    def _canon(self, value: Tensor, conv: nn.Conv1d, heads: int) -> Tensor:
        batch, tokens = value.shape[:2]
        flat = value.reshape(batch, tokens, -1).transpose(1, 2)
        weight = conv.weight.squeeze(1).to(dtype=flat.dtype)
        out = flat + causal_conv1d_fn(flat.contiguous(), weight)
        return out.transpose(1, 2).reshape(batch, tokens, heads, self.head_dim)

    def _project_qkv(self, x: Tensor):
        batch, tokens, _ = x.shape
        heads, kv_heads, dim = self.num_heads, self.num_kv_heads, self.head_dim
        q_gate = self.base.q(x).view(batch, tokens, heads, dim * 2)
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        k = self.base.k(x).view(batch, tokens, kv_heads, dim)
        v = self.base.v(x).view(batch, tokens, kv_heads, dim)
        q = F.rms_norm(q, (dim,))
        k = F.rms_norm(k, (dim,))

        if isinstance(self.base, CausalSelfAttention) and self.base.pos == "rope":
            q_attn, k_attn, v_attn = self.base.rotary(q), self.base.rotary(k), v
            q_mem, k_mem, v_mem = q, k, v
        else:
            q_attn = self._canon(q, self.base.canon_q, heads)
            k_attn = self._canon(k, self.base.canon_k, kv_heads)
            v_attn = self._canon(v, self.base.canon_v, kv_heads)
            q_mem, k_mem, v_mem = q_attn, k_attn, v_attn
        return gate, q_attn, k_attn, v_attn, q_mem, k_mem, v_mem

    def _flex(self, q: Tensor, k: Tensor, v: Tensor, **kwargs):
        if self._compiled_flex is None:
            self._compiled_flex = torch.compile(flex_attention, dynamic=False) if self.compile_flex else flex_attention
        if self.flex_kernel_options is not None:
            kwargs["kernel_options"] = self.flex_kernel_options
        return self._compiled_flex(q, k, v, **kwargs)

    def _flex_with_lse(self, q: Tensor, k: Tensor, v: Tensor, **kwargs) -> tuple[Tensor, Tensor]:
        if AuxRequest is not None:
            output, auxiliary = self._flex(q, k, v, return_aux=AuxRequest(lse=True), **kwargs)
            if auxiliary.lse is None:
                raise RuntimeError("FlexAttention did not return the requested log-sum-exp")
            return output, auxiliary.lse
        return self._flex(q, k, v, return_lse=True, **kwargs)

    def _softmax_sparse(self, q: Tensor, k: Tensor, v: Tensor, route: Route) -> Tensor:
        # q: (B,H,T,D); k/v: (B,KVH,T,D)
        if q.is_cuda and HAS_FLEX_ATTENTION:
            block_mask = build_block_mask(
                route,
                heads=self.num_heads,
                block_size=self.page_size,
                local_window=self.local_window,
            )
            scale = self.base.sdpa_scale if getattr(self.base, "sdpa_scale", None) is not None else None
            return self._flex(
                q,
                k,
                v,
                block_mask=block_mask,
                scale=scale,
                enable_gqa=True,
            )

        allowed = _dense_allowed_mask(
            route,
            sequence_length=q.shape[2],
            block_size=self.page_size,
            local_window=self.local_window,
        )
        groups = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        scale = self.base.sdpa_scale if getattr(self.base, "sdpa_scale", None) is not None else None
        return F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[:, None], scale=scale)

    def _polar_sparse(self, q: Tensor, k: Tensor, v: Tensor, route: Route) -> tuple[Tensor, Tensor]:
        tokens = q.shape[2]
        n_keys = torch.arange(1, tokens + 1, device=q.device, dtype=torch.float32)
        groups = self.num_heads // self.num_kv_heads
        if q.is_cuda and HAS_POLAR_TRITON and polar_attention_sparse is not None:
            # The Triton kernel consumes the selected route directly and fuses
            # Polar's (M,L,Q2,S) reductions into one sparse pass.  As in ATMA's
            # dense Polar path, expand GQA KV heads before entering the kernel.
            k_full = k.repeat_interleave(groups, dim=1)
            v_full = v.repeat_interleave(groups, dim=1)
            return polar_attention_sparse(
                q,
                k_full,
                v_full,
                route.page_indices,
                route.page_counts,
                page_size=self.page_size,
                local_window=self.local_window,
                v_null=self.base.v_null,
                null_base=self.base.null_base,
                null_slope_raw=self.base.null_slope_raw,
                len_gain_raw=self.base.len_gain_raw,
                mag_beta_raw=self.base.mag_beta_raw,
            )

        # Compatibility fallback for CUDA builds where FlexAttention is present
        # but the custom Triton extension cannot be imported.
        if q.is_cuda and HAS_FLEX_ATTENTION:
            block_mask = build_block_mask(
                route,
                heads=self.num_heads,
                block_size=self.page_size,
                local_window=self.local_window,
            )
            temp, null = polar_temp_null(
                n_keys,
                self.base.len_gain_raw,
                self.base.null_base,
                self.base.null_slope_raw,
            )

            def polar_score(score, batch, head, q_idx, kv_idx):
                del batch, kv_idx
                return score * temp[0, head, q_idx, 0]

            def polar_score_squared(score, batch, head, q_idx, kv_idx):
                del batch, kv_idx
                return 2.0 * score * temp[0, head, q_idx, 0]

            real_out, lse = self._flex_with_lse(
                q,
                k,
                v,
                score_mod=polar_score,
                block_mask=block_mask,
                enable_gqa=True,
            )
            # A small aligned value width keeps the second reduction cheap while
            # avoiding backend-specific edge cases around a one-channel value.
            dummy_v = v.new_zeros((*v.shape[:-1], min(16, self.head_dim)))
            _, lse2 = self._flex_with_lse(
                q,
                k,
                dummy_v,
                score_mod=polar_score_squared,
                block_mask=block_mask,
                enable_gqa=True,
            )
            null_logit = (null * temp).squeeze(0).squeeze(-1)
            real_conf = torch.sigmoid(lse.float() - null_logit.float())
            v_null = self.base.v_null.view(1, self.num_heads, 1, self.head_dim).to(real_out.dtype)
            mixed = real_conf[..., None].to(real_out.dtype) * real_out + (
                1.0 - real_conf[..., None].to(real_out.dtype)
            ) * v_null
            direction = F.normalize(mixed, dim=-1, eps=1e-6)
            log_neff = (2.0 * lse.float() - lse2.float()).clamp(min=0.0, max=math.log(max(tokens, 2)))
            n_eff = torch.exp(log_neff)
            m_eff = n_eff * real_conf
            beta = F.softplus(self.base.mag_beta_raw.float()).view(1, self.num_heads, 1)
            mag = torch.tanh(beta * torch.log1p(m_eff)).to(real_out.dtype)
            return direction, mag

        allowed = _dense_allowed_mask(
            route,
            sequence_length=tokens,
            block_size=self.page_size,
            local_window=self.local_window,
        )
        k_full = k.repeat_interleave(groups, dim=1)
        v_full = v.repeat_interleave(groups, dim=1)
        scores = torch.matmul(q, k_full.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~allowed[:, None], -torch.inf)
        return polar_reduce(
            scores,
            v_full,
            n_keys,
            v_null=self.base.v_null,
            null_base=self.base.null_base,
            null_slope_raw=self.base.null_slope_raw,
            len_gain_raw=self.base.len_gain_raw,
            mag_beta_raw=self.base.mag_beta_raw,
        )

    def _anchor_blocks(self, query_blocks: int, device: torch.device) -> Tensor:
        if query_blocks <= 1 or self.teacher_query_blocks <= 0:
            return torch.empty(0, device=device, dtype=torch.long)
        count = min(self.teacher_query_blocks, query_blocks - 1)
        # Avoid the trivial first block and rotate anchors through later context.
        stride = max(1, query_blocks // (count + 1))
        offset = self.step % stride
        blocks = stride + offset + torch.arange(count, device=device) * stride
        return blocks.clamp_max(query_blocks - 1).unique()

    def _teacher_loss(
        self,
        page_scores: Tensor,
        q: Tensor,
        k: Tensor,
        route: Route | None = None,
    ) -> Tensor:
        """Distill page mass at rotating query-block anchors.

        Calibration (``route is None``) uses all completed pages. Sparse CPT
        only gathers the selected local and remote pages, so it never executes
        a dense full-attention teacher in the training loop.
        """

        if self.teacher_query_blocks <= 0 or self.step % self.teacher_interval:
            self._last_teacher_stats = {}
            return page_scores.sum() * 0.0
        batch, query_blocks, pages = page_scores.shape
        anchors = self._anchor_blocks(query_blocks, page_scores.device)
        if anchors.numel() == 0:
            self._last_teacher_stats = {}
            return page_scores.sum() * 0.0
        groups = self.num_heads // self.num_kv_heads
        k_full = k.repeat_interleave(groups, dim=2).detach()
        q = q.detach()
        anchor_count = anchors.numel()
        if route is None:
            selected = torch.arange(pages, device=page_scores.device).view(1, 1, pages)
            selected = selected.expand(batch, anchor_count, -1)
            selected_valid = selected < anchors.view(1, anchor_count, 1)
        else:
            local = route.local_indices[:, anchors].long()
            remote = route.page_indices[:, anchors].long()
            local_slots = torch.arange(local.shape[-1], device=page_scores.device)
            remote_slots = torch.arange(remote.shape[-1], device=page_scores.device)
            local_valid = local_slots.view(1, 1, -1) < route.local_counts[:, anchors, None]
            remote_valid = remote_slots.view(1, 1, -1) < route.page_counts[:, anchors, None]
            selected = torch.cat((local, remote), dim=-1)
            selected_valid = torch.cat((local_valid, remote_valid), dim=-1)
            selected_valid &= selected < anchors.view(1, anchor_count, 1)

        safe_selected = selected.clamp(min=0, max=pages - 1)
        offsets = torch.arange(self.page_size, device=page_scores.device)
        token_ids = safe_selected[..., None] * self.page_size + offsets
        batch_ids = torch.arange(batch, device=page_scores.device).view(batch, 1, 1, 1)
        keys = k_full[batch_ids, token_ids]  # (B,A,S,page,H,D)
        query_pos = anchors * self.page_size
        q_anchor = q[:, :, query_pos].permute(0, 2, 1, 3)
        raw = torch.einsum("bahd,basthd->bahst", q_anchor, keys) / math.sqrt(self.head_dim)
        token_valid = selected_valid[:, :, None, :, None]
        raw = raw.masked_fill(~token_valid, -torch.inf)
        flat_raw = raw.flatten(start_dim=-2)

        if isinstance(self.base, CausalSelfAttention):
            if self.base.sdpa_scale is not None:
                flat_raw = flat_raw * (self.base.sdpa_scale * math.sqrt(self.head_dim))
            teacher = torch.softmax(flat_raw.float(), dim=-1)
        else:
            n = (query_pos + 1).to(dtype=torch.float32)
            temp, null = polar_temp_null(
                n,
                self.base.len_gain_raw.detach(),
                self.base.null_base.detach(),
                self.base.null_slope_raw.detach(),
            )
            null_logit = (null * temp).permute(0, 2, 1, 3)
            temp = temp.permute(0, 2, 1, 3)
            real = flat_raw.float() * temp
            weights = torch.softmax(
                torch.cat((real, null_logit.expand(batch, -1, -1, -1)), dim=-1),
                dim=-1,
            )
            teacher = weights[..., :-1]
            teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(1e-20)

        selected_count = selected.shape[-1]
        page_mass = teacher.view(
            batch, anchor_count, self.num_heads, selected_count, self.page_size
        ).sum(dim=-1)
        mean_mass = page_mass.mean(dim=2)
        max_mass = page_mass.amax(dim=2)
        target = self.teacher_mean_weight * mean_mass + (
            1.0 - self.teacher_mean_weight
        ) * max_mass
        target = target.masked_fill(~selected_valid, 0.0)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-20)

        student = page_scores[:, anchors].gather(2, safe_selected)
        student = student.masked_fill(~selected_valid, -torch.inf)
        log_student = torch.log_softmax(student.float(), dim=-1)
        log_student = torch.where(selected_valid, log_student, torch.zeros_like(log_student))
        loss = -(target.detach() * log_student).sum(dim=-1).mean()

        topk = min(self.max_remote_pages, selected_count)
        self._last_teacher_stats = {}
        if topk:
            student_top = student.topk(topk, dim=-1).indices
            oracle_top = target.topk(topk, dim=-1).indices
            chosen_valid = selected_valid.gather(2, student_top)
            matches = (
                student_top[..., :, None] == oracle_top[..., None, :]
            ).any(dim=-1)
            denominator = chosen_valid.sum().clamp_min(1)
            recall = (matches & chosen_valid).sum().float() / denominator
            captured = target.gather(2, student_top).sum(dim=-1).mean()
            head_oracle = page_mass.topk(topk, dim=-1).indices
            head_matches = (
                student_top[:, :, None, :, None] == head_oracle[:, :, :, None, :]
            ).any(dim=-1)
            head_denominator = chosen_valid.sum(dim=-1, keepdim=True).clamp_min(1)
            head_recall = (
                (head_matches & chosen_valid[:, :, None]).sum(dim=-1).float()
                / head_denominator
            )
            self._last_teacher_stats = {
                "teacher_topk_recall": recall.detach(),
                "teacher_mass_at_k": captured.detach(),
                "teacher_head_recall": head_recall.mean().detach(),
                "teacher_worst_head_recall": head_recall.amin(dim=2).mean().detach(),
            }
        return loss

    def forward(self, x: Tensor):
        if self.mode == "dense_teacher":
            page_scores, _ = self._index(x)
            out, _ = self.base(x)
            _, q_attn, k_attn, _, _, _, _ = self._project_qkv(x)
            index_loss = self._teacher_loss(page_scores, q_attn.transpose(1, 2), k_attn)
            self.last_stats = {"index_loss": index_loss.detach(), **self._last_teacher_stats}
            return out, index_loss

        if self.adaptation_mode == "local":
            pages = x.shape[1] // self.page_size
            page_scores = x.new_zeros((x.shape[0], pages, pages), dtype=torch.float32)
            route = select_pages(
                page_scores,
                page_size=self.page_size,
                local_window=self.local_window,
                top_p=1.0,
                min_remote_pages=0,
                max_remote_pages=0,
                remote_capacity=self.remote_capacity,
            )
            index_values = None
        else:
            page_scores, index_values = self._index(x)
            route = self._route(page_scores)
        gate, q_attn, k_attn, v_attn, q_mem, k_mem, v_mem = self._project_qkv(x)
        q_t = q_attn.transpose(1, 2).contiguous()
        k_t = k_attn.transpose(1, 2).contiguous()
        v_t = v_attn.transpose(1, 2).contiguous()
        index_loss = (
            self._teacher_loss(page_scores, q_t, k_attn, route)
            if self.uses_kl
            else page_scores.sum() * 0.0
        )

        if self.is_polar:
            direction, mag = self._polar_sparse(q_t, k_t, v_t, route)
            flat = direction.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
            content = self.base.proj(flat * torch.sigmoid(gate.reshape_as(flat)))
            out = content + self.base.mu_proj(mag.transpose(1, 2))
        else:
            attended = self._softmax_sparse(q_t, k_t, v_t, route)
            flat = attended.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
            out = self.base.proj(flat * torch.sigmoid(gate.reshape_as(flat)))

        if self.uses_lm_output:
            if index_values is None:
                raise RuntimeError("LM index output requested without an index value stream")
            out = out + self._index_output(page_scores, index_values)

        if self.base.mem is not None:
            groups = self.num_heads // self.num_kv_heads
            out = out + self.base.mem(
                x,
                q_mem.transpose(1, 2),
                k_mem.repeat_interleave(groups, dim=2).transpose(1, 2),
                v_mem.repeat_interleave(groups, dim=2).transpose(1, 2),
            )

        self.last_stats = {
            "index_loss": index_loss.detach(),
            "cap_rate": route.cap_rate.detach(),
            "mean_remote_pages": route.mean_remote_pages.detach(),
            "local_only_rate": route.local_only_rate.detach(),
            **self._last_teacher_stats,
        }
        return out, index_loss
