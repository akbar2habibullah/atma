"""Config-driven training entrypoint for one ablation cell.

    python -m ablation.train --config ablation/configs/<run_id>.json --log ablation/logs/<run_id>.log

Reads ALL hyperparameters from the JSON RunConfig (no hardcode), mirrors the validated
train.py loop (AdamW+Muon split, stable/decay LR schedule, mbs grad-accum, clip, MFU,
proj zero-init, torch.compile), records a val-loss-vs-wall-clock curve, then runs the full
structured eval (ablation/evaluate.py) on the in-memory model. Emits ONE self-describing
.log holding three delimited JSON blocks (config / curve / eval) + human-readable lines.

Set FLA_CUSTOM_OP=1 for the compile-clean FLA path. ATMA_WALL_CUSTOM_OP=1 is only relevant when
reproducing diagnostic Wall runs.
"""
import argparse
import json
import os
import pickle
import socket
import time
import traceback
from pathlib import Path


def _log_open(path):
    # "w": one process per config -> a re-run (e.g. after --reset) overwrites the stale failed
    # log cleanly instead of appending two CONFIG/ERROR blocks the parser would trip on.
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return open(path, "w", buffering=1)


def _emit_block(fh, name, obj):
    fh.write(f"\n==={name}===\n{json.dumps(obj)}\n===END===\n")


def main():
    ap = argparse.ArgumentParser(description="Train one ablation cell from a JSON config.")
    ap.add_argument("--config", required=True, help="path to a RunConfig JSON")
    ap.add_argument("--log", required=True, help="path to the output .log")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--save_ckpt", action="store_true", help="persist a checkpoint (off by default; eval is in-process)")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    fh = _log_open(args.log)

    import torch
    from torch import nn
    from torch.optim import AdamW
    import torch.nn.functional as F

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def p0(s):
        print(s); fh.write(s + "\n")

    p0("=" * 100)
    p0(f"[ablation] run_id={cfg['run_id']} host={socket.gethostname()} device={device} "
       f"torch={torch.__version__} fla_custom_op={os.environ.get('FLA_CUSTOM_OP','0')} "
       f"wall_custom_op={os.environ.get('ATMA_WALL_CUSTOM_OP','0')}")
    p0("=" * 100)

    try:
        from train.data import get_data, data_generator
        from train.optimizer import Muon
        from train.model import Model
        from model.config import AtmaConfig
        from ablation.evaluate import run_eval

        # ---------------- data ----------------
        seq_len = cfg["seq_len"]
        batch_size = cfg["batch_size"]
        mbs = cfg["mbs"]
        num_chunks = cfg["num_chunks"]
        max_steps = cfg.get("max_steps")  # smoke cap (optional)

        get_data("finewebedu_val_%06d.bin" % 0)
        for i in range(1, num_chunks + 1):
            get_data("finewebedu_train_%06d.bin" % i)

        # bound the budget to exactly num_chunks train shards (sorted)
        train_files = sorted(Path.cwd().glob("finewebedu10B/finewebedu_train_*.bin"))[:num_chunks]
        train_steps = 0
        for f in train_files:
            num_tokens = int(torch.from_file(str(f), False, 256, dtype=torch.int32)[2])
            train_steps += (num_tokens - 2) // batch_size
        if max_steps is not None:
            train_steps = min(train_steps, max_steps)
        p0(f"[ablation] train_steps={train_steps} (num_chunks={num_chunks}, batch_size={batch_size}, seq_len={seq_len})")

        val_tokens = cfg["val_tokens"]
        val_inputs, val_targets = next(data_generator(cfg["val_data"], val_tokens, seq_len=seq_len))
        train_loader = data_generator("finewebedu10B/finewebedu_train_*.bin", batch_size, seq_len=seq_len)

        # ---------------- model ----------------
        ac = AtmaConfig(
            vocab_size=cfg["vocab_size"], num_hidden_layers=cfg["num_hidden_layers"],
            hidden_size=cfg["hidden_size"], head_dim=cfg["head_dim"],
            max_position_embeddings=seq_len, num_random_keys=cfg["num_random_keys"],
            attn_type=cfg["attn_type"], attn_window=cfg["attn_window"],
            mem_enabled=cfg["mem_enabled"], mem_chunk=cfg["mem_chunk"],
            mem_gamma_bias=cfg["mem_gamma_bias"], mem_beta_bias=cfg["mem_beta_bias"],
            mem_kernel=cfg["mem_kernel"], wall_gate_bias=cfg.get("wall_gate_bias"),
        )
        model = Model(ac, reg_mode=cfg["reg_mode"], sketch_dim=cfg["sketch_dim"]).to(device)
        num_params = sum(p.numel() for p in model.parameters())
        p0(f"[ablation] params={num_params/1e6:.2f}M")
        _emit_block(fh, "ABLATION_CONFIG_JSON", {**cfg, "num_params": num_params,
                                                 "host": socket.gethostname(), "device": str(device)})

        if device.type == "cuda":
            model = torch.compile(model)

        # proj zero-init (matches train.py)
        for name, p in model.named_parameters():
            if "proj" in name:
                p.data.zero_()

        # ---------------- optimizers (mirror train.py) ----------------
        opt1 = AdamW([dict(params=[model.embed.weight], lr=0.3),
                      dict(params=[model.proj.weight], lr=1 / 320),
                      dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01)],
                     betas=(0.8, 0.95), eps=1e-10, weight_decay=0, fused=(device.type == "cuda"))
        opt2 = Muon([p for p in model.blocks.parameters() if p.ndim >= 2], lr=0.02, weight_decay=0.01)
        optimizers = [opt1, opt2]
        assert set(p for opt in optimizers for g in opt.param_groups for p in g["params"]) == set(model.parameters())
        for opt in optimizers:
            for g in opt.param_groups:
                g["initial_lr"] = g["lr"]

        cooldown = cfg["cooldown_frac"]

        def set_hparams(step):
            progress = step / train_steps
            eta = 1.0 if progress < 1 - cooldown else (1 - progress) / cooldown
            for opt in optimizers:
                for g in opt.param_groups:
                    g["lr"] = g["initial_lr"] * eta

        # ---------------- MFU ----------------
        flops_per_token = 6 * num_params + 12 * ac.num_hidden_layers * ac.hidden_size * seq_len
        flops_per_step = flops_per_token * batch_size
        peak_flops = 65e12
        if device.type == "cuda":
            name = torch.cuda.get_device_name()
            peak_flops = {"A100": 312e12, "V100": 125e12, "T4": 65e12, "L4": 121e12,
                          "H100": 989e12, "H200": 989e12}.get(next((k for k in
                          ["A100", "V100", "T4", "L4", "H100", "H200"] if k in name), "T4"), 65e12)

        sigr_alpha = cfg["sigr_alpha"]
        dist_w = cfg["dist_align_loss_weight"]

        mem_profile = device.type == "cuda" and os.environ.get("ATMA_MEM_PROFILE", "0") == "1"
        mem_trace = os.environ.get("ATMA_MEM_TRACE") if device.type == "cuda" else None

        def mem_mark(label):
            if not mem_profile:
                return
            torch.cuda.synchronize()
            p0(
                f"[mem] {label} "
                f"alloc={torch.cuda.memory_allocated() / 1024**2:.1f}MiB "
                f"reserved={torch.cuda.memory_reserved() / 1024**2:.1f}MiB "
                f"max_alloc={torch.cuda.max_memory_allocated() / 1024**2:.1f}MiB "
                f"max_reserved={torch.cuda.max_memory_reserved() / 1024**2:.1f}MiB"
            )

        if mem_trace:
            torch.cuda.memory._record_memory_history(
                enabled="all",
                context="all",
                stacks="all",
                max_entries=int(os.environ.get("ATMA_MEM_TRACE_MAX_ENTRIES", "200000")),
                clear_history=True,
                compile_context=True,
            )
            p0(f"[mem] trace recording enabled -> {mem_trace}")

        # ---------------- train loop ----------------
        curve = []
        val_freq = max(1, min(125, train_steps // 4))
        training_time, last_val_step, t0 = 0.0, 0, time.perf_counter()

        def validate(step):
            model.eval()
            vl = 0.0
            with torch.no_grad():
                for i in range(len(val_inputs) // mbs):
                    ls, _, _ = model(val_inputs[i*mbs:(i+1)*mbs], val_targets[i*mbs:(i+1)*mbs])
                    vl += ls.item()
            model.train()
            return vl / val_tokens

        for step in range(train_steps + 1):
            if step == train_steps or step % val_freq == 0:
                dt = time.perf_counter() - t0
                step_avg = dt / max(step - last_val_step, 1)
                last_val_step = step
                training_time += dt
                val_loss = validate(step)
                mfu = (flops_per_step / step_avg / peak_flops * 100) if (step > 0 and device.type == "cuda") else 0.0
                tok_s = (batch_size / step_avg) if step > 0 else 0.0
                curve.append(dict(step=step, wall_s=round(training_time, 3), val_loss=val_loss,
                                  mfu=round(mfu, 2), step_ms=round(1000 * step_avg, 2), tok_s=round(tok_s, 1)))
                p0(f"step:{step}/{train_steps} val_loss:{val_loss:.5f} wall:{training_time:.1f}s "
                   f"step_avg:{1000*step_avg:.1f}ms MFU:{mfu:.1f}%")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                t0 = time.perf_counter()

            if step == train_steps:
                break

            inputs, targets = next(train_loader)
            assert len(inputs) % mbs == 0
            if mem_profile and step == 0:
                torch.cuda.reset_peak_memory_stats()
                mem_mark("train_step0_start")
            train_loss = 0.0
            for i in range(len(inputs) // mbs):
                if mem_profile and step == 0:
                    mem_mark(f"micro{i}_before_forward")
                ls, reg_loss, align_loss = model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs])
                if mem_profile and step == 0:
                    mem_mark(f"micro{i}_after_forward")
                train_loss += ls.item()
                loss = (1 - sigr_alpha) * ls + sigr_alpha * reg_loss + dist_w * align_loss
                loss.backward()
                if mem_profile and step == 0:
                    mem_mark(f"micro{i}_after_backward")
            if mem_profile and step == 0:
                mem_mark("after_all_microbatches")
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if mem_profile and step == 0:
                mem_mark("after_clip_grad_norm")
            set_hparams(step)
            for opt in optimizers:
                opt.step()
            if mem_profile and step == 0:
                mem_mark("after_optimizer_step")
            model.zero_grad(set_to_none=True)
            if mem_profile and step == 0:
                mem_mark("after_zero_grad")
                if mem_trace:
                    trace_path = Path(mem_trace)
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    with trace_path.open("wb") as trace_f:
                        pickle.dump(torch.cuda.memory._snapshot(), trace_f)
                    p0(f"[mem] trace snapshot saved -> {trace_path}")

        mfu_final = curve[-1]["mfu"] if curve else 0.0
        _emit_block(fh, "ABLATION_CURVE_JSON", curve)
        p0(f"[ablation] training done: {training_time:.1f}s, final MFU {mfu_final:.1f}%")

        if args.save_ckpt:
            ckpt_dir = os.path.join("checkpoints", cfg["run_id"])
            os.makedirs(ckpt_dir, exist_ok=True)
            clean = {k.removeprefix("_orig_mod."): v.cpu() for k, v in model.state_dict().items()}
            torch.save({"model": clean}, os.path.join(ckpt_dir, "weights.pt"))
            p0(f"[ablation] checkpoint -> {ckpt_dir}")

        # ---------------- structured eval ----------------
        p0("[ablation] evaluating (clean_ppl / junk_ppl / needle at full context)...")
        eval_res = run_eval(model, cfg, device)
        eval_res.update(dict(mfu_final=mfu_final, train_elapsed_s=round(training_time, 3),
                             num_params=num_params, train_steps=train_steps))
        _emit_block(fh, "ABLATION_EVAL_JSON", eval_res)
        p0("[ablation] eval:")
        p0(f"  clean_ppl(nats): {eval_res.get('clean_ppl')}")
        p0(f"  junk_ppl(nats):  {eval_res.get('junk_ppl')}")
        p0(f"  needle:          { {d: round(v['acc'],1) for d,v in (eval_res.get('needle') or {}).items()} }")
        p0(f"  needle_baseline CE: {eval_res.get('needle_baseline')}")
        p0("[ablation] DONE")

    except Exception:
        tb = traceback.format_exc()
        p0("[ablation] FAILED:\n" + tb)
        _emit_block(fh, "ABLATION_ERROR_JSON", {"error": tb})
        fh.close()
        raise
    fh.close()


if __name__ == "__main__":
    main()
