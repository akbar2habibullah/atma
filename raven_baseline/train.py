"""Train one Raven baseline config and emit Atma-compatible ablation logs."""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
import traceback
from pathlib import Path


def _log_open(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return open(path, "w", buffering=1, encoding="utf-8")


def _emit_block(fh, name, obj):
    fh.write(f"\n==={name}===\n{json.dumps(obj)}\n===END===\n")


def _fill_runtime_defaults(cfg):
    cfg.setdefault("optimizer", "adamw_raven")
    cfg.setdefault("adamw_lr", 3e-4)
    cfg.setdefault("adamw_lr_min_frac", 0.1)
    cfg.setdefault("adamw_warmup_frac", 0.05)
    cfg.setdefault("adamw_beta1", 0.9)
    cfg.setdefault("adamw_beta2", 0.95)
    cfg.setdefault("adamw_eps", 1e-15)
    cfg.setdefault("adamw_weight_decay", 0.1)
    cfg.setdefault("skip_nan_inf", True)
    cfg.setdefault("compile_model", True)
    cfg.setdefault("atma_head_match", True)
    if cfg.get("arch_type") != "raven_native" and cfg.get("atma_head_match", True) and cfg.get("num_heads") == 4:
        cfg["num_heads"] = 8
    cfg.setdefault(
        "num_kv_heads",
        cfg["num_heads"] if cfg.get("arch_type") == "raven_native" else max(1, cfg["num_heads"] // 4),
    )
    return cfg


def _save_checkpoint(model, cfg, ckpt_root, log_fn):
    import torch

    ckpt_dir = Path(ckpt_root) / cfg["run_id"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    clean = {k.removeprefix("_orig_mod."): v.cpu() for k, v in model.state_dict().items()}
    torch.save({"model": clean}, ckpt_dir / "weights.pt")
    (ckpt_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (ckpt_dir / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (ckpt_dir / "tokenizer.json").write_text(json.dumps({"tokenizer_name": "gpt2"}, indent=2), encoding="utf-8")
    log_fn(f"[raven_baseline] checkpoint -> {ckpt_dir}")
    return ckpt_dir


def main():
    ap = argparse.ArgumentParser(description="Train one Raven baseline from a JSON config.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--no_save_ckpt", action="store_true")
    args = ap.parse_args()

    cfg = _fill_runtime_defaults(json.load(open(args.config, encoding="utf-8")))
    fh = _log_open(args.log)

    import torch
    from torch import nn
    from torch.optim import AdamW

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def p0(s):
        print(s)
        fh.write(s + "\n")

    p0("=" * 100)
    p0(
        f"[raven_baseline] run_id={cfg['run_id']} arch_type={cfg['arch_type']} "
        f"host={socket.gethostname()} device={device} torch={torch.__version__} "
        f"fla_custom_op={os.environ.get('FLA_CUSTOM_OP', '0')}"
    )
    p0("=" * 100)

    try:
        from train.data import data_generator, get_data
        from raven_baseline.evaluate import run_eval
        from raven_baseline.model import create_model

        seq_len = cfg["seq_len"]
        batch_size = cfg["batch_size"]
        mbs = cfg["mbs"]
        num_chunks = cfg["num_chunks"]
        max_steps = cfg.get("max_steps")

        get_data("finewebedu_val_%06d.bin" % 0)
        for i in range(1, num_chunks + 1):
            get_data("finewebedu_train_%06d.bin" % i)

        train_files = sorted(Path.cwd().glob("finewebedu10B/finewebedu_train_*.bin"))[:num_chunks]
        train_steps = 0
        for f in train_files:
            num_tokens = int(torch.from_file(str(f), False, 256, dtype=torch.int32)[2])
            train_steps += (num_tokens - 2) // batch_size
        if max_steps is not None:
            train_steps = min(train_steps, max_steps)
        p0(
            f"[raven_baseline] train_steps={train_steps} "
            f"(num_chunks={num_chunks}, batch_size={batch_size}, seq_len={seq_len})"
        )

        val_tokens = cfg["val_tokens"]
        val_inputs, val_targets = next(data_generator(cfg["val_data"], val_tokens, seq_len=seq_len))
        train_loader = data_generator("finewebedu10B/finewebedu_train_*.bin", batch_size, seq_len=seq_len)

        model = create_model(cfg).to(device)
        num_params = sum(p.numel() for p in model.parameters())
        p0(f"[raven_baseline] params={num_params/1e6:.2f}M")
        _emit_block(
            fh,
            "ABLATION_CONFIG_JSON",
            {**cfg, "num_params": num_params, "host": socket.gethostname(), "device": str(device)},
        )

        if device.type == "cuda" and cfg.get("compile_model", True):
            model = torch.compile(model)

        for name, p in model.named_parameters():
            if (
                name in {"proj.weight", "_orig_mod.proj.weight"}
                or name.endswith(".o_proj.weight")
                or name.endswith(".mlp.proj.weight")
            ):
                p.data.zero_()

        if cfg.get("optimizer", "adamw_raven") == "atma_muon":
            from train.optimizer import Muon
            opt1 = AdamW(
                [
                    dict(params=[model.embed.weight], lr=0.3),
                    dict(params=[model.proj.weight], lr=1 / 320),
                    dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01),
                ],
                betas=(0.8, 0.95),
                eps=1e-10,
                weight_decay=0,
                fused=(device.type == "cuda"),
            )
            opt2 = Muon([p for p in model.blocks.parameters() if p.ndim >= 2], lr=0.02, weight_decay=0.01)
            optimizers = [opt1, opt2]
        else:
            opt = AdamW(
                model.parameters(),
                lr=cfg.get("adamw_lr", 3e-4),
                betas=(cfg.get("adamw_beta1", 0.9), cfg.get("adamw_beta2", 0.95)),
                eps=cfg.get("adamw_eps", 1e-15),
                weight_decay=cfg.get("adamw_weight_decay", 0.1),
                fused=(device.type == "cuda"),
            )
            optimizers = [opt]
        assert set(p for opt in optimizers for g in opt.param_groups for p in g["params"]) == set(model.parameters())
        for opt in optimizers:
            for g in opt.param_groups:
                g["initial_lr"] = g["lr"]

        cooldown = cfg["cooldown_frac"]

        def set_hparams(step):
            if cfg.get("optimizer", "adamw_raven") == "atma_muon":
                progress = step / train_steps
                eta = 1.0 if progress < 1 - cooldown else (1 - progress) / cooldown
            else:
                warmup_steps = max(1, int(cfg.get("adamw_warmup_frac", 0.05) * train_steps))
                min_frac = cfg.get("adamw_lr_min_frac", 0.1)
                if step < warmup_steps:
                    eta = (step + 1) / warmup_steps
                else:
                    progress = (step - warmup_steps) / max(train_steps - warmup_steps, 1)
                    eta = min_frac + 0.5 * (1.0 - min_frac) * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)).item())
            for opt in optimizers:
                for g in opt.param_groups:
                    g["lr"] = g["initial_lr"] * eta

        flops_per_token = 6 * num_params + 12 * cfg["num_hidden_layers"] * cfg["hidden_size"] * seq_len
        flops_per_step = flops_per_token * batch_size
        peak_flops = 65e12
        if device.type == "cuda":
            name = torch.cuda.get_device_name()
            peak_flops = {
                "A100": 312e12,
                "V100": 125e12,
                "T4": 65e12,
                "L4": 121e12,
                "L40S": 362e12,
                "H100": 989e12,
                "H200": 989e12,
            }.get(next((k for k in ["A100", "V100", "T4", "L4", "L40S", "H100", "H200"] if k in name), "T4"), 65e12)

        sigr_alpha = cfg["sigr_alpha"]
        dist_w = cfg["dist_align_loss_weight"]
        curve = []
        val_freq = max(1, min(125, train_steps // 4))
        training_time, last_val_step, t0 = 0.0, 0, time.perf_counter()

        def validate():
            model.eval()
            vl = 0.0
            with torch.no_grad():
                for i in range(len(val_inputs) // mbs):
                    ls, _, _ = model(val_inputs[i * mbs:(i + 1) * mbs], val_targets[i * mbs:(i + 1) * mbs])
                    vl += ls.item()
            model.train()
            return vl / val_tokens

        for step in range(train_steps + 1):
            if step == train_steps or step % val_freq == 0:
                dt = time.perf_counter() - t0
                step_avg = dt / max(step - last_val_step, 1)
                last_val_step = step
                training_time += dt
                val_loss = validate()
                mfu = (flops_per_step / step_avg / peak_flops * 100) if (step > 0 and device.type == "cuda") else 0.0
                tok_s = (batch_size / step_avg) if step > 0 else 0.0
                curve.append(
                    dict(
                        step=step,
                        wall_s=round(training_time, 3),
                        val_loss=val_loss,
                        mfu=round(mfu, 2),
                        step_ms=round(1000 * step_avg, 2),
                        tok_s=round(tok_s, 1),
                    )
                )
                p0(
                    f"step:{step}/{train_steps} val_loss:{val_loss:.5f} wall:{training_time:.1f}s "
                    f"step_avg:{1000 * step_avg:.1f}ms MFU:{mfu:.1f}%"
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                t0 = time.perf_counter()

            if step == train_steps:
                break

            inputs, targets = next(train_loader)
            assert len(inputs) % mbs == 0
            for i in range(len(inputs) // mbs):
                ls, reg_loss, align_loss = model(inputs[i * mbs:(i + 1) * mbs], targets[i * mbs:(i + 1) * mbs])
                loss = (1 - sigr_alpha) * ls + sigr_alpha * reg_loss + dist_w * align_loss
                loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if cfg.get("skip_nan_inf", True) and not torch.isfinite(grad_norm):
                p0(f"[raven_baseline] non-finite grad_norm at step {step}; skipping optimizer step")
                model.zero_grad(set_to_none=True)
                continue
            set_hparams(step)
            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)

        mfu_final = curve[-1]["mfu"] if curve else 0.0
        _emit_block(fh, "ABLATION_CURVE_JSON", curve)
        p0(f"[raven_baseline] training done: {training_time:.1f}s, final MFU {mfu_final:.1f}%")

        if not args.no_save_ckpt:
            _save_checkpoint(model, cfg, args.ckpt_dir, p0)

        p0("[raven_baseline] evaluating (clean_ppl / junk_ppl / needle at full context)...")
        eval_res = run_eval(model, cfg, device)
        eval_res.update(
            dict(
                mfu_final=mfu_final,
                train_elapsed_s=round(training_time, 3),
                num_params=num_params,
                train_steps=train_steps,
            )
        )
        _emit_block(fh, "ABLATION_EVAL_JSON", eval_res)
        p0("[raven_baseline] eval:")
        p0(f"  clean_ppl(nats): {eval_res.get('clean_ppl')}")
        p0(f"  junk_ppl(nats):  {eval_res.get('junk_ppl')}")
        p0(f"  needle:          { {d: round(v['acc'], 1) for d, v in (eval_res.get('needle') or {}).items()} }")
        p0(f"  needle_baseline CE: {eval_res.get('needle_baseline')}")
        p0("[raven_baseline] DONE")

    except Exception:
        tb = traceback.format_exc()
        p0("[raven_baseline] FAILED:\n" + tb)
        _emit_block(fh, "ABLATION_ERROR_JSON", {"error": tb})
        fh.close()
        raise
    fh.close()


if __name__ == "__main__":
    main()
