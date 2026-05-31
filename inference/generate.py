"""Self-contained Polar-Attention inference / text generation.

Uses the FlashAttention-style Triton polar kernel (kernel/polar_triton.py) for the
attention layers. Seeks a weights checkpoint by default and loads it; if none is
found, falls back to random initialization.

Architecture comes from the checkpoint's config.json when present, otherwise it is
inferred from the checkpoint tensor shapes — so it loads whatever Polar-Attention
model was trained without needing a matching config object.

Run:
    python inference/generate.py --prompt "Once upon a time" --max-new-tokens 50
    python inference/generate.py --ids 1 2 3 --max-new-tokens 20          # raw token ids
    python inference/generate.py --checkpoint /path/to/dir_or_weights.pt
"""
import os
import sys
import argparse

# Allow running as a standalone script (python inference/generate.py): put the repo
# root on sys.path so `model` / `kernel` resolve without importing the heavy inference
# package __init__ (which pulls in transformers).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from torch import nn
import torch.nn.functional as F

from model.blocks import AtmaAttnBase, AtmaConvBase, polar_reduce
from model.layers import RMSNorm, MLP
from kernel.polar_triton import polar_attention_fwd, HAS_TRITON

# Default checkpoint directories searched relative to the repo root: ../checkpoints
# (outside the atma dir, per the project layout) and ./checkpoints (inside). A
# checkpoint dir holds weights.pt (+ optional config.json / tokenizer.json).
_DEFAULT_CKPT_DIRS = ["../checkpoints", "checkpoints"]
_WEIGHT_NAMES = ["weights.pt", "model.pt", "pytorch_model.bin"]

# Polar structural-prior inits (match train/model.py & model/reference.py).
_LEN_GAIN_INIT, _NULL_SLOPE_INIT, _NULL_BASE_INIT, _MAG_BETA_INIT = -1.0, 0.5, 2.0, -1.5


class Linear(nn.Linear):
    """dtype-safe Linear (casts weight/bias to the activation dtype)."""
    def forward(self, x):
        b = self.bias.type_as(x) if self.bias is not None else None
        return F.linear(x, self.weight.type_as(x), b)


def _causal_conv1d(x_in, conv_mod):
    """Depthwise causal conv1d, pure PyTorch. x_in: (B, C, T)."""
    k = conv_mod.weight.size(2)
    return F.conv1d(F.pad(x_in, (k - 1, 0)), conv_mod.weight, stride=1, padding=0, groups=x_in.size(1))


class PolarAttention(AtmaAttnBase):
    """Polar attention for inference. Mirrors model/reference.py PolarAttention but
    runs the forward through the Triton kernel (falls back to polar_reduce on CPU)."""

    def __init__(self, dim, head_dim, num_heads, num_kv_heads, kernel_size=4, use_triton=True):
        super().__init__(dim, linear_cls=Linear, head_dim=head_dim, num_kv_heads=num_kv_heads,
                         kernel_size=kernel_size, num_heads=num_heads)
        H, dk = self.num_heads, self.head_dim
        self.mu_proj = Linear(H, dim)
        self.v_null = nn.Parameter(torch.zeros(H, dk))
        self.null_base = nn.Parameter(torch.full((H,), _NULL_BASE_INIT))
        self.null_slope_raw = nn.Parameter(torch.full((H,), _NULL_SLOPE_INIT))
        self.len_gain_raw = nn.Parameter(torch.full((H,), _LEN_GAIN_INIT))
        self.mag_beta_raw = nn.Parameter(torch.full((H,), _MAG_BETA_INIT))
        self.use_triton = use_triton

    def forward(self, x):
        B, T, _ = x.shape
        H, dk = self.num_heads, self.head_dim

        q_gate = self.q(x).view(B, T, H, dk * 2)
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        k = self.k(x).view(B, T, self.num_kv_heads, dk)
        v = self.v(x).view(B, T, self.num_kv_heads, dk)

        q = F.rms_norm(q, (dk,))
        k = F.rms_norm(k, (dk,))

        q_in = q.reshape(B, T, -1).transpose(1, 2)
        k_in = k.reshape(B, T, -1).transpose(1, 2)
        v_in = v.reshape(B, T, -1).transpose(1, 2)
        q_out = q_in + _causal_conv1d(q_in, self.canon_q)
        k_out = k_in + _causal_conv1d(k_in, self.canon_k)
        v_out = v_in + _causal_conv1d(v_in, self.canon_v)

        q_attn = q_out.transpose(1, 2).reshape(B, T, H, dk)
        k_attn = k_out.transpose(1, 2).reshape(B, T, self.num_kv_heads, dk)
        v_attn = v_out.transpose(1, 2).reshape(B, T, self.num_kv_heads, dk)

        groups = H // self.num_kv_heads
        k_attn = k_attn.repeat_interleave(groups, dim=2)
        v_attn = v_attn.repeat_interleave(groups, dim=2)

        q_t = q_attn.transpose(1, 2).contiguous()      # (B, H, T, dk)
        k_t = k_attn.transpose(1, 2).contiguous()
        v_t = v_attn.transpose(1, 2).contiguous()

        n_keys = torch.arange(1, T + 1, device=x.device, dtype=torch.float32)
        params = dict(v_null=self.v_null, null_base=self.null_base, null_slope_raw=self.null_slope_raw,
                      len_gain_raw=self.len_gain_raw, mag_beta_raw=self.mag_beta_raw)

        if self.use_triton and HAS_TRITON and q_t.is_cuda:
            c, mag = polar_attention_fwd(q_t, k_t, v_t, n_keys, is_causal=True, **params)
        else:
            sigma = torch.matmul(q_t.float(), k_t.float().transpose(-2, -1)) / (dk ** 0.5)
            sigma = sigma + torch.triu(torch.full((T, T), float("-inf"), device=x.device), 1)
            c, mag = polar_reduce(sigma, v_t, n_keys, **params)

        c_flat = c.transpose(1, 2).reshape(B, T, H * dk)
        content = self.proj(c_flat * torch.sigmoid(gate.reshape(B, T, -1)))
        count = self.mu_proj(mag.transpose(1, 2))
        return content + count


class LFM2Conv(AtmaConvBase):
    """LFM2 gated conv block (pure PyTorch), mirrors model/reference.py."""
    def __init__(self, dim, kernel_size=3):
        super().__init__(dim, linear_cls=Linear, kernel_size=kernel_size)

    def forward(self, x):
        B_gate, C, x_proj = self.in_proj(x).chunk(3, dim=-1)
        x_in = (B_gate * x_proj).transpose(1, 2)
        x_conv = _causal_conv1d(x_in, self.conv).transpose(1, 2)
        return self.out_proj(C * x_conv)


class Block(nn.Module):
    def __init__(self, cfg, attention):
        super().__init__()
        dim = cfg["hidden_size"]
        self.attn = (PolarAttention(dim, head_dim=cfg["head_dim"], num_heads=cfg["num_heads"],
                                    num_kv_heads=cfg["num_kv_heads"], kernel_size=cfg["attn_kernel_size"],
                                    use_triton=cfg["use_triton"])
                     if attention else LFM2Conv(dim, kernel_size=cfg["conv_kernel_size"]))
        self.mlp = MLP(dim, linear_cls=Linear)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PolarLM(nn.Module):
    """Polar-attention LM, structurally weight-compatible with the trained
    train.model.Model / model.reference.ReferenceModel checkpoints."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.blocks = nn.ModuleList([Block(cfg, attention=(i in cfg["attn_layers"]))
                                     for i in range(cfg["num_hidden_layers"])])
        self.proj = Linear(cfg["hidden_size"], cfg["vocab_size"])
        self.norm = RMSNorm(cfg["hidden_size"])

    @torch.no_grad()
    def forward(self, input_ids):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm(x)).float()
        return 15.0 * logits * (logits.square() + 225.0).rsqrt()


# --------------------------------------------------------------------------- #
# Architecture inference + checkpoint loading
# --------------------------------------------------------------------------- #

def _unwrap(sd):
    if isinstance(sd, dict) and not any(hasattr(v, "shape") for v in list(sd.values())[:5]):
        for w in ("model", "state_dict", "model_state_dict", "weights", "ema"):
            if w in sd and isinstance(sd[w], dict):
                return sd[w]
    return sd


def _layers_to_cfg(num_layers):
    return [i for i in range(num_layers) if i % 4 == 2]   # attention every 4th block (i%4==2)


def config_from_json(d, use_triton=True):
    """Build the build-config from a saved AtmaConfig-style config.json."""
    hidden, head_dim = d["hidden_size"], d["head_dim"]
    num_heads = hidden // head_dim
    n_layers = d["num_hidden_layers"]
    return dict(vocab_size=d["vocab_size"], hidden_size=hidden, num_hidden_layers=n_layers,
                head_dim=head_dim, num_heads=num_heads, num_kv_heads=max(1, num_heads // 4),
                attn_layers=_layers_to_cfg(n_layers),
                attn_kernel_size=d.get("attn_kernel_size", 4),
                conv_kernel_size=d.get("conv_kernel_size", 3), use_triton=use_triton)


def infer_config(sd, use_triton=True):
    """Infer the model architecture from a checkpoint's tensor shapes (fallback when
    no config.json is present)."""
    import re
    vocab, hidden = sd["embed.weight"].shape
    layer_ids = sorted({int(m.group(1)) for k in sd if (m := re.match(r"blocks\.(\d+)\.", k))})
    attn_layers = sorted({i for i in layer_ids if f"blocks.{i}.attn.q.weight" in sd})
    a = attn_layers[0]
    head_dim = sd[f"blocks.{a}.attn.v_null"].shape[1]
    num_heads = sd[f"blocks.{a}.attn.v_null"].shape[0]
    num_kv_heads = sd[f"blocks.{a}.attn.k.weight"].shape[0] // head_dim
    attn_k = sd[f"blocks.{a}.attn.canon_q.weight"].shape[2]
    conv_layer = next((i for i in layer_ids if f"blocks.{i}.attn.conv.weight" in sd), None)
    conv_k = sd[f"blocks.{conv_layer}.attn.conv.weight"].shape[2] if conv_layer is not None else 3
    return dict(vocab_size=vocab, hidden_size=hidden, num_hidden_layers=len(layer_ids),
                head_dim=head_dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
                attn_layers=attn_layers, attn_kernel_size=attn_k, conv_kernel_size=conv_k,
                use_triton=use_triton)


def default_config(use_triton=True):
    """Architecture used when no checkpoint is found (default AtmaConfig: hidden 1024,
    head_dim 128 -> 8 heads / 2 kv-heads, 16 layers, attn at i%4==2)."""
    return dict(vocab_size=50304, hidden_size=1024, num_hidden_layers=16,
                head_dim=128, num_heads=8, num_kv_heads=2,
                attn_layers=_layers_to_cfg(16),
                attn_kernel_size=4, conv_kernel_size=3, use_triton=use_triton)


def _tok_name(d):
    import json
    tp = os.path.join(d, "tokenizer.json")
    if os.path.isfile(tp):
        try:
            return json.load(open(tp)).get("tokenizer_name")
        except Exception:
            return None
    return None


def find_checkpoint(explicit=None):
    """Locate a checkpoint. Returns (weights_path|None, config_path|None,
    tokenizer_name|None, searched_list). `explicit` may be a weights file or a dir."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    searched, dirs = [], []
    if explicit:
        if os.path.isfile(explicit):
            d = os.path.dirname(explicit)
            cfg = os.path.join(d, "config.json")
            return explicit, (cfg if os.path.isfile(cfg) else None), _tok_name(d), [explicit]
        dirs.append(os.path.abspath(explicit))
    dirs += [os.path.normpath(os.path.join(repo_root, p)) for p in _DEFAULT_CKPT_DIRS]
    for d in dirs:
        for w in _WEIGHT_NAMES:
            wp = os.path.join(d, w)
            searched.append(wp)
            if os.path.isfile(wp):
                cfg = os.path.join(d, "config.json")
                return wp, (cfg if os.path.isfile(cfg) else None), _tok_name(d), searched
    return None, None, None, searched


def load_model(checkpoint=None, device="cuda", dtype=torch.bfloat16, use_triton=True):
    """Build the polar LM and load weights if a checkpoint is found, else random init.

    Returns (model, info). Architecture comes from config.json when present, otherwise
    is inferred from the checkpoint tensor shapes.
    """
    import json
    wpath, cfgpath, tok, searched = find_checkpoint(checkpoint)
    if wpath is None:
        cfg = default_config(use_triton)
        model = PolarLM(cfg).to(device=device, dtype=dtype).eval()
        return model, {"loaded": False, "path": None, "config": cfg, "tokenizer": "gpt2",
                       "searched": searched}

    sd = _unwrap(torch.load(wpath, map_location="cpu", weights_only=False))
    cfg = config_from_json(json.load(open(cfgpath)), use_triton) if cfgpath else infer_config(sd, use_triton)
    model = PolarLM(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model = model.to(device=device, dtype=dtype).eval()
    return model, {"loaded": True, "path": wpath, "config": cfg, "tokenizer": tok or "gpt2",
                   "missing": list(missing), "unexpected": list(unexpected), "searched": searched}


# --------------------------------------------------------------------------- #
# Generation (full-recompute; correct and simple, uses the fast polar kernel)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def generate(model, input_ids, max_new_tokens=50, temperature=1.0, top_k=None,
             eos_id=None, device="cuda"):
    ids = torch.as_tensor(input_ids, dtype=torch.long, device=device).view(1, -1)
    for _ in range(max_new_tokens):
        logits = model(ids)[:, -1, :]            # (1, vocab)
        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = torch.softmax(logits.float(), dim=-1)
            nxt = torch.multinomial(probs, 1)
        ids = torch.cat([ids, nxt], dim=1)
        if eos_id is not None and int(nxt) == eos_id:
            break
    return ids[0].tolist()


def _get_tokenizer(name="gpt2"):
    """Best-effort BPE tokenizer (the trained run used gpt2). Optional."""
    try:
        import tiktoken
        return tiktoken.get_encoding(name or "gpt2")
    except Exception:
        pass
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(name or "gpt2")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Polar-attention inference")
    ap.add_argument("--checkpoint", default=None, help="explicit checkpoint path or dir")
    ap.add_argument("--prompt", default=None, help="text prompt (needs tiktoken/transformers)")
    ap.add_argument("--ids", type=int, nargs="+", default=None, help="raw token ids prompt")
    ap.add_argument("--max-new-tokens", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-triton", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="run in fp32 (default: bf16, matching the checkpoint)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    dtype = torch.float32 if (args.fp32 or device == "cpu") else torch.bfloat16

    model, info = load_model(args.checkpoint, device=device, dtype=dtype, use_triton=not args.no_triton)
    cfg = info["config"]
    if info["loaded"]:
        print(f"[checkpoint] loaded {info['path']}")
    else:
        print("[checkpoint] NOT FOUND -> random init. Searched:")
        for p in info.get("searched", []):
            print(f"             {p}")
    print(f"[config] hidden={cfg['hidden_size']} layers={cfg['num_hidden_layers']} "
          f"heads={cfg['num_heads']} kv_heads={cfg['num_kv_heads']} head_dim={cfg['head_dim']} "
          f"vocab={cfg['vocab_size']} attn_layers={cfg['attn_layers']}")
    if info["loaded"] and (info.get("missing") or info.get("unexpected")):
        print(f"[load] missing={info['missing'][:4]} unexpected={info['unexpected'][:4]}")
    use_tri = cfg["use_triton"] and HAS_TRITON and device == "cuda"
    print(f"[attn] polar kernel: {'Triton' if use_tri else 'PyTorch polar_reduce'}")

    tok = _get_tokenizer(info.get("tokenizer"))
    if args.ids is not None:
        ids = args.ids
    elif args.prompt is not None and tok is not None:
        ids = tok.encode(args.prompt)
    elif args.prompt is not None:
        raise SystemExit("no tokenizer (pip install tiktoken); pass --ids instead of --prompt")
    else:
        ids = tok.encode("The") if tok else [0]

    out = generate(model, ids, max_new_tokens=args.max_new_tokens,
                   temperature=args.temperature, top_k=args.top_k, device=device)
    print("\n[token ids]", out)
    if tok is not None:
        print("\n[text]\n" + tok.decode(out))


if __name__ == "__main__":
    main()
