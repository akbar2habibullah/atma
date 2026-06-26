"""Minimal backend dispatch shim.

The upstream FLA decorator routes operations to optional backends. The local Wall
kernel only needs the decorator shape, so this returns the function unchanged.
"""


def dispatch(_operation):
    def decorator(fn):
        return fn
    return decorator
