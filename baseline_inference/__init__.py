"""Isolated inference forks for ablation baselines."""

__all__ = ["BaselineLLM"]


def __getattr__(name):
    if name == "BaselineLLM":
        from .engine import BaselineLLM

        return BaselineLLM
    raise AttributeError(name)
