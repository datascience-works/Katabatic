try:
    from .original.original_api import PG_ORIGINAL
except ImportError:
    from original.original_api import PG_ORIGINAL

# For now, just export the original
__all__ = ['PG_ORIGINAL']