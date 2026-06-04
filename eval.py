"""
eval.py

Evaluate length extrapolation from a saved checkpoint.
Measures cross-entropy loss at multiples of the training sequence length
to quantify how well the model generalises beyond its training context window.

Usage:
    python eval.py
    python eval.py --checkpoint checkpoints --num_seqs 16
    python eval.py --multipliers 1 2 4 8 16 32 64
"""

import os
import json
import argparse

import torch
import torch.nn.functional as F

from model.config import AtmaConfig
from train.model import Model
from train.data import data_generator


def load_from_checkpoint(checkpoint_dir: str, device: torch.device,
                         compile_model: bool = True, force_probe_path: bool = False):
    config_path = os.path.join(checkpoint_dir, "config.json")
    weights_path = os.path.join(checkpoint_dir, "weights.pt")

    with open(config_path) as f:
        cfg = json.load(f)

    cfg["dtype"] = getattr(torch, cfg["dtype"])
    cfg["num_random_keys"] = 0  # distractor path is training-only; no weights attached

    if force_probe_path:
        # The materialized polar path is O(T^2) and OOMs past ~16x. Both the Triton
        # kernel and the torch online path are O(T*block) and now emit the probe sink;
        # prefer Triton (much faster at 64x), else fall back to torch online.
        from model.blocks import HAS_TRITON
        if HAS_TRITON:
            cfg["attn_kernel"], cfg["attn_online"] = "triton", False
        else:
            cfg["attn_kernel"], cfg["attn_online"] = "torch", True

    config = AtmaConfig(**cfg)
    model = Model(config).to(device)

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)["model"]
    model.load_state_dict(state_dict)
    if compile_model:
        model = torch.compile(model)
    model.eval()

    return model, config


def eval_length(model: torch.nn.Module, val_data: str, seq_len: int, num_seqs: int):
    gen = data_generator(val_data, seq_len, seq_len=seq_len)  # yields (1, seq_len) each call
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_seqs):
            inputs, targets = next(gen)  # one sequence at a time — no pre-loaded batch
            loss, _, _ = model(inputs, targets)
            total_loss += loss.item()
    return total_loss / (num_seqs * seq_len)


class _ChannelStats:
    """Streaming per-channel mean/std over (batch, position) — fp32, O(D) memory."""

    def __init__(self):
        self.count = 0
        self.sum = None      # (D,)
        self.sumsq = None    # (D,)

    def update(self, x):     # x: (B, T, D)
        xf = x.detach().float().reshape(-1, x.shape[-1])
        s, ss = xf.sum(0), xf.square().sum(0)
        if self.sum is None:
            self.sum, self.sumsq = s, ss
        else:
            self.sum += s
            self.sumsq += ss
        self.count += xf.shape[0]

    def mean(self):
        return self.sum / self.count

    def std(self):
        return (self.sumsq / self.count - self.mean().square()).clamp_min(0).sqrt()

    def rms(self):
        return (self.sumsq.sum() / (self.count * self.sum.numel())).sqrt().item()


class Probe:
    """Captures, per layer and per multiplier, the activation distribution the
    downstream (strictly local) stack consumes, plus the polar internals. The
    question it answers: as N grows, do these go out-of-distribution vs 1×?"""

    def __init__(self, model):
        from train.model import PolarAttention
        self.enabled = False
        self.handles = []
        self.attn_layers = []                       # block indices that are PolarAttention
        self.res = {}                               # idx -> _ChannelStats (residual stream)
        self.attn = {}                              # idx -> _ChannelStats (attn contribution)
        self.polar = {}                             # idx -> {n_eff_sum, mag_sum, mag_sat, w_null_sum, n}
        self.snapshots = {}                         # mult -> captured per-layer stats

        for i, block in enumerate(model.blocks):
            self.res[i] = _ChannelStats()
            self.handles.append(block.register_forward_hook(self._res_hook(i)))
            if isinstance(block.attn, PolarAttention):
                self.attn_layers.append(i)
                self.attn[i] = _ChannelStats()
                self.handles.append(block.attn.register_forward_hook(self._attn_hook(i)))

    def _res_hook(self, i):
        def hook(_m, _inp, out):
            if self.enabled:
                self.res[i].update(out[0])          # Block returns (x, reg_loss, align_loss)
        return hook

    def _attn_hook(self, i):
        def hook(_m, _inp, out):
            if self.enabled:
                self.attn[i].update(out[0])         # attn returns (content+count, align_loss)
        return hook

    def reset(self):
        for d in (self.res, self.attn):
            for k in d:
                d[k] = _ChannelStats()
        self.polar = {i: dict(n_eff=0.0, mag=0.0, mag_sat=0, w_null=0.0, n=0)
                      for i in self.attn_layers}

    def consume_sink(self, sink):
        # sink: one dict per attention layer per forward, in block order.
        for i, rec in zip(self.attn_layers, sink):
            acc = self.polar[i]
            acc["n_eff"] += rec["n_eff"].sum().item()
            acc["mag"] += rec["mag"].sum().item()
            acc["mag_sat"] += (rec["mag"] > 0.99).sum().item()
            acc["w_null"] += rec["w_null"].sum().item()
            acc["n"] += rec["mag"].numel()

    def finalize(self, mult):
        snap = {"res": {}, "attn": {}, "polar": {}}
        for i, st in self.res.items():
            snap["res"][i] = (st.mean(), st.std(), st.rms())
        for i, st in self.attn.items():
            snap["attn"][i] = (st.mean(), st.std(), st.rms())
        for i, acc in self.polar.items():
            n = max(acc["n"], 1)
            snap["polar"][i] = dict(n_eff=acc["n_eff"] / n, mag=acc["mag"] / n,
                                    mag_sat=100.0 * acc["mag_sat"] / n, w_null=acc["w_null"] / n)
        self.snapshots[mult] = snap


def _blocks_forward(model, inputs):
    """embed -> blocks only (NO LM head). The block/attn hooks capture activations and
    the polar sink fills here; skipping the (1, T, vocab) head avoids the 13-26 GB fp32
    logits spike that OOMs a 24 GB L4 at long T. Returns the pre-final-norm residual."""
    x = model.embed(inputs)
    for block in model.blocks:
        x, _, _ = block(x)
    return x


def _chunked_loss(model, x, targets, chunk=8192, pos_sum=None, pos_cnt=None):
    """Per-token CE via a time-chunked head: peak logits = chunk*vocab*4 (~1.6 GB at
    chunk=8192), not T*vocab*4. Mirrors Model.forward's logit squashing exactly.

    If pos_sum/pos_cnt (1D tensors) are given, also bins per-token loss by absolute target
    position into log2 buckets (bucket k = positions [2^k, 2^{k+1})) — the L(t) curve."""
    total, n = 0.0, 0
    T, B = x.shape[1], x.shape[0]
    for c0 in range(0, T, chunk):
        c1 = min(c0 + chunk, T)
        logits = model.proj(model.norm(x[:, c0:c1])).float()
        logits = 15 * logits * (logits.square() + 15 ** 2).rsqrt()
        tgt = targets[:, c0:c1].reshape(-1)
        loss = F.cross_entropy(logits.reshape(tgt.numel(), -1), tgt, reduction="none")
        total += loss.sum().item()
        n += tgt.numel()
        if pos_sum is not None:
            pos = torch.arange(c0, c1, device=loss.device) + 1     # 1-indexed predicted position
            if B > 1:
                pos = pos.repeat(B)                                # reshape order is (batch, position)
            bucket = torch.log2(pos.float()).floor().long().clamp_(max=pos_sum.numel() - 1)
            pos_sum.index_add_(0, bucket, loss.double())
            pos_cnt.index_add_(0, bucket, torch.ones_like(loss, dtype=torch.float64))
    return total, n


def _doc_source(args, mults, device):
    """Shared data source: single coherent long docs (--hf_dataset) or the concatenated
    .bin stream. Returns (docs, n_seqs, batches) where batches(seq_len) yields (inputs, targets).
    The .bin generator restarts from the same files each call, so every window/multiplier sees
    the SAME sequences — a fair cross-window comparison."""
    docs = None
    if getattr(args, "hf_dataset", None):
        min_tok = args.min_doc_tokens or args.base_len * max(mults)
        docs = select_long_docs(args.hf_dataset, args.hf_text_key, args.hf_split,
                                min_tok, args.num_seqs)
        if not docs:
            raise SystemExit(f"No docs with ≥ {min_tok + 1:,} tokens found — lower --multipliers "
                             f"or --min_doc_tokens, or pick a dataset with longer documents.")
    n_seqs = len(docs) if docs is not None else args.num_seqs

    def batches(seq_len):
        if docs is not None:                        # nested prefixes of the SAME coherent docs
            for d in docs:
                buf = d[:seq_len + 1]
                yield (buf[:-1].view(1, -1).to(device, torch.int32),
                       buf[1:].view(1, -1).to(device, torch.int64))
        else:
            gen = data_generator(args.val_data, seq_len, seq_len=seq_len)
            for _ in range(n_seqs):
                yield next(gen)

    return docs, n_seqs, batches


def select_long_docs(dataset_id, text_key, split, min_tokens, num_docs, tokenizer_name="gpt2"):
    """Select up to num_docs SINGLE documents with >= min_tokens+1 tokens, tokenized exactly
    as training (gpt2, EOT prepended as document start). Pre-filters by char length to avoid
    tokenizing short docs. Returns a list of 1D int64 token tensors truncated to min_tokens+1
    (we only ever test that prefix), so all multipliers see nested prefixes of the same doc."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    eot = tok.eos_token_id
    need = min_tokens + 1
    char_min = need               # ≥ need tokens ⇒ ≥ need chars (BPE token ≥ 1 char); safe lower bound

    print(f"Scanning '{dataset_id}' [{split}] for docs with ≥ {need:,} tokens "
          f"(pre-filter ≥ {char_min:,} chars)...")
    ds = load_dataset(dataset_id, split=split)
    docs, scanned = [], 0
    for row in ds:
        scanned += 1
        text = row.get(text_key)
        if not text or len(text) < char_min:
            continue
        ids = [eot] + tok.encode(text, add_special_tokens=False)
        if len(ids) >= need:
            docs.append(torch.tensor(ids[:need], dtype=torch.int64))
            if len(docs) >= num_docs:
                break
    print(f"  selected {len(docs)}/{num_docs} docs (scanned {scanned:,} rows).")
    return docs


def run_diagnose(model, args, device):
    from model import blocks
    from train.model import PolarAttention

    window = getattr(args, "window", None)
    if window is not None:
        for block in model.blocks:
            if isinstance(block.attn, PolarAttention):
                block.attn.window = window

    probe = Probe(model)
    eps = 1e-6
    mults = args.multipliers if 1 in args.multipliers else [1] + args.multipliers

    docs, n_seqs, batches = _doc_source(args, mults, device)
    src = f"{n_seqs} docs from {args.hf_dataset}" if docs is not None else f"{n_seqs} seqs/length"
    win_note = f", sliding window W={window}" if window is not None else ""
    print(f"\nActivation-distribution probe ({src}, "
          f"base_len={args.base_len}, reference = 1×{win_note}):\n")

    losses = {}
    for mult in mults:
        seq_len = args.base_len * mult
        torch.cuda.empty_cache()
        probe.reset()
        probe.enabled = True

        loss_sum, tok = 0.0, 0
        with torch.no_grad():
            for inputs, targets in batches(seq_len):
                blocks._PROBE = []                  # capture this forward's polar internals
                x = _blocks_forward(model, inputs)  # hooks + sink fill here; no LM head
                probe.consume_sink(blocks._PROBE)
                ls, n = _chunked_loss(model, x, targets, chunk=args.loss_chunk)
                loss_sum += ls
                tok += n
        blocks._PROBE = None
        probe.enabled = False
        losses[mult] = loss_sum / tok
        probe.finalize(mult)

    ref = probe.snapshots[1]

    def chan_drift(snap_layer, ref_layer):
        m, s, _ = snap_layer
        rm, rs, _ = ref_layer
        z = (m - rm).abs() / (rs + eps)             # per-channel shift in ref-σ units
        return z.mean().item(), (s / (rs + eps)).mean().item()

    # ---- residual stream: what every downstream local layer actually consumes ----
    print("Residual stream (mean over all 16 blocks, vs 1×):")
    hdr = f"{'mult':>6}  {'loss':>8}  {'Δloss':>8}  {'mean-shift(σ)':>14}  {'std-ratio':>10}  {'rms-ratio':>10}"
    print(hdr); print("-" * len(hdr))
    for mult in mults:
        snap = probe.snapshots[mult]
        shifts, stds, rmss = [], [], []
        for i in snap["res"]:
            d_mean, d_std = chan_drift(snap["res"][i], ref["res"][i])
            shifts.append(d_mean); stds.append(d_std)
            rmss.append(snap["res"][i][2] / (ref["res"][i][2] + eps))
        dl = losses[mult] - losses[1]
        print(f"{f'{mult}x':>6}  {losses[mult]:>8.4f}  {dl:>+8.4f}  "
              f"{sum(shifts)/len(shifts):>14.3f}  {sum(stds)/len(stds):>10.3f}  {sum(rmss)/len(rmss):>10.3f}")

    # ---- final block residual (closest to the logits — most predictive of loss) ----
    last = max(ref["res"])
    print(f"\nFinal block (#{last}) residual, vs 1×:")
    print(f"{'mult':>6}  {'mean-shift(σ)':>14}  {'std-ratio':>10}  {'rms-ratio':>10}")
    for mult in mults:
        snap = probe.snapshots[mult]
        d_mean, d_std = chan_drift(snap["res"][last], ref["res"][last])
        rr = snap["res"][last][2] / (ref["res"][last][2] + eps)
        print(f"{f'{mult}x':>6}  {d_mean:>14.3f}  {d_std:>10.3f}  {rr:>10.3f}")

    # ---- attention contribution (mean over the 4 polar layers) ----
    print("\nAttention contribution to residual (rms-ratio vs 1×, mean over polar layers):")
    print(f"{'mult':>6}  {'rms-ratio':>10}  {'mean-shift(σ)':>14}")
    for mult in mults:
        snap = probe.snapshots[mult]
        rmss, shifts = [], []
        for i in snap["attn"]:
            rmss.append(snap["attn"][i][2] / (ref["attn"][i][2] + eps))
            d_mean, _ = chan_drift(snap["attn"][i], ref["attn"][i])
            shifts.append(d_mean)
        print(f"{f'{mult}x':>6}  {sum(rmss)/len(rmss):>10.3f}  {sum(shifts)/len(shifts):>14.3f}")

    # ---- polar internals (mean over the 4 polar layers): is the score-level premise holding? ----
    print("\nPolar internals (mean over polar layers):")
    ph = f"{'mult':>6}  {'mag (mean)':>11}  {'mag>0.99 %':>11}  {'w_null':>9}  {'n_eff':>9}"
    print(ph); print("-" * len(ph))
    for mult in mults:
        pol = probe.snapshots[mult]["polar"]
        n = len(pol)
        mag = sum(p["mag"] for p in pol.values()) / n
        sat = sum(p["mag_sat"] for p in pol.values()) / n
        wn = sum(p["w_null"] for p in pol.values()) / n
        ne = sum(p["n_eff"] for p in pol.values()) / n
        print(f"{f'{mult}x':>6}  {mag:>11.4f}  {sat:>11.2f}  {wn:>9.4f}  {ne:>9.1f}")

    print("\nReading it:")
    print("  • residual mean-shift/std-ratio grow with N  → downstream local stack sees OOD")
    print("    activations → keeping N in-distribution (compression/memory) is the right attack.")
    print("  • mag saturates→1, n_eff blows up, or w_null→0 with N → the score-level premise")
    print("    (length-invariant count) is failing → Δ collapse; compression alone won't save it.")
    print("  • everything flat but loss still climbs → failure is positional, not aggregation.")

    for h in probe.handles:
        h.remove()


def _parse_window(token):
    """'full'/'none'/'0' -> (label, None); else -> (str(W), int(W))."""
    if token.lower() in ("full", "none", "0", "-1"):
        return ("full", None)
    return (token, int(token))


def run_window_sweep(model, args, device):
    """Compare multiple sliding windows + full attention in ONE pass on the same docs.
    Prints a loss matrix and an n_eff matrix (multiplier × window), and — with
    --per_position — the per-token loss L(t) by absolute position at the max multiplier
    (plateau = averaging artifact; keeps declining = genuine long-range gain)."""
    from model import blocks
    from train.model import PolarAttention

    win_cfgs = [_parse_window(w) for w in args.windows]
    labels = [lbl for lbl, _ in win_cfgs]
    mults = sorted(set(args.multipliers) | {1})
    max_mult = max(mults)
    polar_layers = [b.attn for b in model.blocks if isinstance(b.attn, PolarAttention)]

    docs, n_seqs, batches = _doc_source(args, mults, device)
    src = f"{n_seqs} docs from {args.hf_dataset}" if docs is not None else f"{n_seqs} seqs/length"
    print(f"\nWindow sweep ({src}, base_len={args.base_len}, "
          f"windows={labels}{', +L(t)' if args.per_position else ''}):\n")

    NBINS = 24
    loss_mat, neff_mat, posL = {}, {}, {}
    for label, W in win_cfgs:
        for attn in polar_layers:
            attn.window = W
        for mult in mults:
            seq_len = args.base_len * mult
            torch.cuda.empty_cache()
            do_pos = args.per_position and mult == max_mult
            ps = torch.zeros(NBINS, dtype=torch.float64, device=device) if do_pos else None
            pc = torch.zeros(NBINS, dtype=torch.float64, device=device) if do_pos else None
            loss_sum, tok, ne_sum, ne_cnt = 0.0, 0, 0.0, 0
            with torch.no_grad():
                for inputs, targets in batches(seq_len):
                    blocks._PROBE = []
                    x = _blocks_forward(model, inputs)
                    for rec in blocks._PROBE:               # mean n_eff over layers/positions/seqs
                        ne_sum += rec["n_eff"].sum().item()
                        ne_cnt += rec["n_eff"].numel()
                    ls, n = _chunked_loss(model, x, targets, args.loss_chunk, pos_sum=ps, pos_cnt=pc)
                    loss_sum += ls
                    tok += n
            blocks._PROBE = None
            loss_mat[(label, mult)] = loss_sum / tok
            neff_mat[(label, mult)] = ne_sum / max(ne_cnt, 1)
            if do_pos:
                posL[label] = (ps.cpu(), pc.cpu())
    for attn in polar_layers:
        attn.window = None

    def print_matrix(title, mat, fmt):
        print(title)
        print(f"{'mult':>6}  " + "  ".join(f"{l:>9}" for l in labels))
        print("-" * (8 + 11 * len(labels)))
        for mult in mults:
            print(f"{f'{mult}x':>6}  " + "  ".join(f"{mat[(l, mult)]:>9{fmt}}" for l in labels))
        print()

    print_matrix("Mean loss vs N  (cols = window; 'full' = no window):", loss_mat, ".4f")
    print_matrix("Mean n_eff vs N  (effective keys attended; cols = window):", neff_mat, ".1f")

    if args.per_position:
        print(f"Per-token loss L(t) by position bin at {max_mult}x  (cols = window):")
        print(f"{'pos':>14}  " + "  ".join(f"{l:>9}" for l in labels))
        print("-" * (16 + 11 * len(labels)))
        for k in range(NBINS):
            if all(posL[l][1][k].item() == 0 for l in labels):
                continue
            rng = f"{2**k:,}-{2**(k+1):,}"
            cells = []
            for l in labels:
                s, c = posL[l][0][k].item(), posL[l][1][k].item()
                cells.append(f"{s / c:>9.4f}" if c > 0 else f"{'—':>9}")
            print(f"{rng:>14}  " + "  ".join(cells))
        print("\nReading L(t): flat past the window width → the 'long < short' mean is the "
              "position-averaging artifact (later tokens easier), not long-range use.\n"
              "Keeps declining past the window under 'full' → genuine long-range gain.")


def _haystack_pool(args, min_len, n, device):
    """A pool of n real-text token tensors (CPU int64), each ≥ min_len tokens — the
    'haystack'. From --hf_dataset (single long docs) or the .bin stream. In-distribution
    text so the needle sits in realistic context."""
    if getattr(args, "hf_dataset", None):
        docs = select_long_docs(args.hf_dataset, args.hf_text_key, args.hf_split, min_len, n)
        if not docs:
            raise SystemExit(f"No haystack docs with ≥ {min_len + 1:,} tokens found.")
        return docs
    gen = data_generator(args.val_data, min_len, seq_len=min_len)
    return [next(gen)[0].view(-1).to("cpu", torch.int64) for _ in range(n)]


def run_needle(model, args, device):
    """Induction-style needle-in-haystack realistic for a small/undertrained LM: plant a
    natural sentence carrying a UNIQUE random key and a short spaced-digit VALUE early, then
    re-present the same sentence (minus the value) at the end and score next-token loss on the
    value. No instructions/QA — pure verbatim copy-at-distance (what induction heads do).

    Sweeps the needle→query distance and compares full attention vs sliding windows. A window
    W < distance physically can't see the needle (→ chance), so 'full' beating it at distance =
    genuine long-range retrieval; 'full' decaying to the needle-absent baseline = no long-range
    capability (and the OOD blowup is destroying it)."""
    import random
    from transformers import AutoTokenizer
    from model import blocks
    from train.model import PolarAttention

    random.seed(1234)
    tok = AutoTokenizer.from_pretrained("gpt2")
    eot = tok.eos_token_id

    distances = args.needle_distances or [256, 512, 1024, 2048, 4096, 8192]
    win_cfgs = [_parse_window(w) for w in (args.windows or ["full"])]
    labels = [lbl for lbl, _ in win_cfgs]
    vlen = args.needle_val_len
    n_trials = args.num_seqs
    polar_layers = [b.attn for b in model.blocks if isinstance(b.attn, PolarAttention)]

    # haystack must cover the largest gap + the needle/query scaffold
    overhead = 64
    pool = _haystack_pool(args, max(distances) + overhead, n_trials, device)

    def make_needle():
        key = random.randint(10 ** 6, 10 ** 7 - 1)          # unique key ⇒ unambiguous induction binding
        digits = [random.randint(0, 9) for _ in range(vlen)]
        cue = tok.encode(f" The access code for record {key} is")
        val = tok.encode("".join(f" {d}" for d in digits))   # one token per spaced digit
        return cue, val

    def value_logits(inputs):
        x = _blocks_forward(model, inputs)                   # (1, L-1, D)
        logits = model.proj(model.norm(x[:, -len(val):])).float()
        return (15 * logits * (logits.square() + 15 ** 2).rsqrt())[0]   # (vlen, vocab)

    # accumulators
    ce = {(l, d): 0.0 for l in labels for d in distances}
    acc = {(l, d): 0.0 for l in labels for d in distances}
    base_ce = 0.0
    print(f"\nNeedle retrieval ({n_trials} trials, value = {vlen} spaced random digits; "
          f"windows={labels}; haystack = "
          f"{args.hf_dataset or args.val_data}):\n")

    with torch.no_grad():
        for t in range(n_trials):
            hay = pool[t % len(pool)]
            cue, val = make_needle()
            needle = cue + val
            val_t = torch.tensor(val, device=device)
            # needle-absent baseline (cue appears only at the query → model's prior on the value)
            g = hay[:distances[0] + len(needle)].tolist()
            seq = [eot] + g + cue + val
            inp = torch.tensor(seq[:-1], dtype=torch.int32, device=device).view(1, -1)
            for attn in polar_layers:
                attn.window = None
            base_ce += F.cross_entropy(value_logits(inp), val_t, reduction="mean").item()

            for d in distances:
                gap = hay[:d].tolist()
                seq = [eot] + needle + gap + cue + val      # needle at start, query at end, gap=d
                inp = torch.tensor(seq[:-1], dtype=torch.int32, device=device).view(1, -1)
                for label, W in win_cfgs:
                    for attn in polar_layers:
                        attn.window = W
                    lg = value_logits(inp)
                    ce[(label, d)] += F.cross_entropy(lg, val_t, reduction="mean").item()
                    acc[(label, d)] += (lg.argmax(-1) == val_t).float().mean().item()
    for attn in polar_layers:
        attn.window = None

    base = base_ce / n_trials
    print(f"Needle-absent baseline (chance) CE/digit: {base:.3f}\n")

    def table(title, mat, fmt):
        print(title)
        print(f"{'distance':>9}  " + "  ".join(f"{l:>9}" for l in labels))
        print("-" * (11 + 11 * len(labels)))
        for d in distances:
            print(f"{d:>9,}  " + "  ".join(f"{mat[(l, d)] / n_trials:>9{fmt}}" for l in labels))
        print()

    table("value CE / digit  (lower = retrieved; ≈ baseline = not retrieved):", ce, ".3f")
    table("per-digit accuracy %  (greedy):", {k: v * 100 for k, v in acc.items()}, ".1f")
    print("Reading it:")
    print("  • full CE << baseline out to large distance → genuine long-range retrieval → "
          "compression could preserve it.")
    print("  • full CE → baseline past ~training length (and a window < distance also at "
          "baseline) → no long-range retrieval capability; compression can't add it.")
    print("  • a window W < distance should sit at baseline by construction (validates the metric).")


def main():
    parser = argparse.ArgumentParser(description="Length extrapolation evaluation")
    parser.add_argument("--checkpoint", default="checkpoints",
                        help="Checkpoint directory written by train.py (default: checkpoints)")
    parser.add_argument("--val_data", default="finewebedu10B/finewebedu_val_*.bin",
                        help="Glob pattern for validation shards")
    parser.add_argument("--base_len", type=int, default=1024,
                        help="Training sequence length (default: 1024)")
    parser.add_argument("--num_seqs", type=int, default=16,
                        help="Sequences evaluated per length — more gives a stabler estimate (default: 16)")
    parser.add_argument("--multipliers", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64],
                        help="Context length multipliers to evaluate (default: 1 2 4 8 16 32 64)")
    parser.add_argument("--diagnose", action="store_true",
                        help="Probe the per-layer activation distribution + polar internals vs N "
                             "(localizes the extrapolation failure) instead of the loss-only sweep. "
                             "Runs embed->blocks only (no LM head) with a time-chunked loss to fit a 24 GB L4.")
    parser.add_argument("--loss_chunk", type=int, default=8192,
                        help="Time-chunk for the diagnose loss head (bounds peak logits memory; default 8192).")
    parser.add_argument("--window", type=int, default=None,
                        help="Eval-only causal sliding window: cap each query to its last W keys. "
                             "Tests whether holding N in-distribution restores extrapolation "
                             "(set W≈training length). Implies the probe harness.")
    parser.add_argument("--hf_dataset", default=None,
                        help="HF dataset of SINGLE coherent long documents (e.g. codelion/finepdfs-100M). "
                             "Evaluates loss-vs-N on the same long docs (nested prefixes), not concatenated "
                             "unrelated docs — the honest long-range test. Implies the probe harness.")
    parser.add_argument("--hf_text_key", default="text", help="Text column in --hf_dataset (default: text).")
    parser.add_argument("--hf_split", default="train", help="Split of --hf_dataset (default: train).")
    parser.add_argument("--min_doc_tokens", type=int, default=None,
                        help="Min token length for selected docs (default: base_len * max(multipliers), so "
                             "every multiplier is a prefix of the same coherent doc).")
    parser.add_argument("--windows", nargs="+", default=None,
                        help="Compare multiple sliding windows + full attention in ONE run on the same "
                             "docs. Values are key widths; use 'full' (or 0) for no window. "
                             "E.g. --windows 128 512 2048 full. Prints loss + n_eff matrices.")
    parser.add_argument("--per_position", action="store_true",
                        help="With --windows: also dump per-token loss L(t) binned by absolute position "
                             "at the max multiplier — separates genuine long-range gain (curve keeps "
                             "declining under 'full') from the averaging artifact (curve plateaus past W).")
    parser.add_argument("--needle", action="store_true",
                        help="Induction needle-in-haystack: plant a natural sentence with a unique key + "
                             "spaced-digit value, re-present it at the end, score value next-token loss vs "
                             "needle→query distance. Tests long-range RETRIEVAL capability (vs perplexity). "
                             "Compares full attention vs --windows (default: full only).")
    parser.add_argument("--needle_distances", type=int, nargs="+", default=None,
                        help="Needle→query distances to sweep (default: 256 512 1024 2048 4096 8192).")
    parser.add_argument("--needle_val_len", type=int, default=5,
                        help="Number of spaced random digits in the needle value (default: 5).")
    parser.add_argument("--no_mem", action="store_true",
                        help="Ablation: disable the Titans memory branch (set attn.mem=None) on the "
                             "loaded checkpoint. Re-run any probe mode (--windows/--needle/--diagnose) "
                             "with and without this flag to isolate the memory's eval-time contribution.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required — data_generator writes directly to GPU memory.")

    device = torch.device("cuda")

    print(f"Loading checkpoint from '{args.checkpoint}'...")
    # Diagnose needs eager modules (hooks + the python-side probe sink survive only
    # uncompiled) and the streaming polar path (materialized O(T^2) OOMs past ~16x).
    probe_mode = (args.diagnose or args.window is not None or args.hf_dataset is not None
                  or args.windows is not None or args.needle)
    model, config = load_from_checkpoint(
        args.checkpoint, device,
        compile_model=not probe_mode, force_probe_path=probe_mode,
    )
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {num_params:.2f}M parameters  |  hidden={config.hidden_size}  "
          f"layers={config.num_hidden_layers}  vocab={config.vocab_size}")

    if args.no_mem:
        from train.model import PolarAttention
        m = getattr(model, "_orig_mod", model)        # unwrap torch.compile if present
        n = 0
        for block in m.blocks:
            if isinstance(block.attn, PolarAttention) and getattr(block.attn, "mem", None) is not None:
                block.attn.mem = None
                n += 1
        print(f"[--no_mem] Titans memory branch DISABLED in {n} attention layers (ablation).")

    if args.needle:
        run_needle(model, args, device)
        return
    if args.windows is not None:
        run_window_sweep(model, args, device)
        return
    if probe_mode:
        run_diagnose(model, args, device)
        return

    print(f"\nEvaluating {args.num_seqs} sequences per length "
          f"(base_len={args.base_len}, multipliers={args.multipliers}):\n")

    header = f"{'Multiplier':>12}  {'Seq length':>12}  {'Loss':>10}  {'Δ vs 1×':>10}"
    print(header)
    print("-" * len(header))

    results = {}
    for mult in args.multipliers:
        seq_len = args.base_len * mult

        torch.cuda.empty_cache()

        loss = eval_length(model, args.val_data, seq_len, args.num_seqs)
        results[mult] = loss

        baseline = results.get(1, loss)
        delta = loss - baseline
        delta_str = f"{delta:+.5f}" if mult != 1 else "—"
        print(f"{f'{mult}x':>12}  {seq_len:>12,}  {loss:>10.5f}  {delta_str:>10}")

    if len(results) > 1 and 1 in results:
        best_extrap = min((m for m in results if m > 1), key=lambda m: results[m])
        worst_extrap = max((m for m in results if m > 1), key=lambda m: results[m])
        print(f"\nBest extrapolation:  {best_extrap}×  loss={results[best_extrap]:.5f}")
        print(f"Worst extrapolation: {worst_extrap}×  loss={results[worst_extrap]:.5f}")


if __name__ == "__main__":
    main()
