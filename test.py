# Try to import optional dependencies
try:
    import mlx.core as mx
    import mlx.nn as nn
    HAS_MLX = True
    print(HAS_MLX)
except ImportError:
    HAS_MLX = False
    print("Warning: MLX not available. Will use PyTorch for embeddings if available.")

