"""Opt-in host-backed Engram for the pinned vLLM-Ascend DSV4.1 model."""

def install(module):
    from .integration import install as apply
    apply(module)
