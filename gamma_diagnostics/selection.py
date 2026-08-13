"""Pure result-selection logic shared by the GPU sweep and CPU tests."""

from __future__ import annotations


def recommend(
    conditions: dict,
    lengths: list[int],
    tolerance: float,
    min_needle_improvement: float = 0.05,
) -> dict | None:
    baseline = conditions.get("baseline", {}).get("metrics", {})
    shortest, longest = str(min(lengths)), str(max(lengths))
    base_short = baseline.get("clean", {}).get(shortest, {}).get("loss_nats")
    base_needle = baseline.get("needle", {}).get("by_distance", {}).get(longest, {}).get("ce_nats")
    # Promotion is intentionally unavailable when either guardrail metric was
    # omitted. Exploratory clean-only/needle-only runs must not select a clamp.
    if base_short is None or base_needle is None:
        return None
    candidates = []
    for label, condition in conditions.items():
        if label == "baseline":
            continue
        metrics = condition["metrics"]
        short = metrics.get("clean", {}).get(shortest, {}).get("loss_nats")
        needle = metrics.get("needle", {}).get("by_distance", {}).get(longest, {}).get("ce_nats")
        short_delta = None if short is None else short - base_short
        needle_delta = None if needle is None else needle - base_needle
        eligible = short_delta is not None and short_delta <= tolerance
        if eligible and needle_delta is not None and needle_delta <= -min_needle_improvement:
            candidates.append((needle_delta, short_delta, label))
    if not candidates:
        return None
    needle_delta, short_delta, label = min(candidates)
    return {
        "condition": label,
        "reason": (
            "lowest longest-distance needle CE meeting the minimum improvement "
            "and short-loss guardrail"
        ),
        "longest_needle_ce_delta_nats": needle_delta,
        "shortest_clean_loss_delta_nats": short_delta,
        "spec": conditions[label]["spec"],
    }
