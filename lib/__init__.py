"""Lightweight package init for inference/runtime imports.

Avoid eager imports of training-only modules so utility imports like
``lib.utils.*`` work in minimal Colab environments.
"""

__all__ = []

try:
    from .humanlrm_wrapper_sa_v1 import SapiensGS_SA_v1

    __all__.append("SapiensGS_SA_v1")
except Exception:
    # Optional dependency path (training/runtime wrapper).
    pass
