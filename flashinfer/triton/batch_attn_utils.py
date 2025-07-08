import torch
import triton
from .kernels.batch_attn_utils import _augment_head_major_kernel, _augment_head_major_3d_kernel

def augment_head_major_triton_3d(
    per_head_kv_indices: torch.Tensor,  # (num_layers, num_kv_heads, total_indices)
    device: str = "cuda"
) -> torch.Tensor:
    """
    Triton-accelerated head-major augmentation for 3D tensors.
    
    This function augments indices for head-major virtual batch organization
    across multiple layers in parallel.
    
    Args:
        per_head_kv_indices: Input indices for each layer and head 
                           (num_layers, num_kv_heads, total_indices)
        device: Device to run on
        
    Returns:
        Augmented indices (num_layers, num_kv_heads * total_indices)
    """
    num_layers, num_kv_heads, total_indices = per_head_kv_indices.shape
    
    # Allocate output tensor
    output_indices = torch.empty(
        (num_layers, num_kv_heads * total_indices),
        dtype=per_head_kv_indices.dtype,
        device=device
    )
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 1024
    total_size_per_layer = num_kv_heads * total_indices
    num_blocks_per_layer = triton.cdiv(total_size_per_layer, BLOCK_SIZE)
    
    # Launch kernel with 2D grid: (num_layers, num_blocks_per_layer)
    _augment_head_major_3d_kernel[(num_layers, num_blocks_per_layer)](
        per_head_kv_indices,
        output_indices,
        num_layers,
        num_kv_heads,
        total_indices,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output_indices


def augment_head_major_triton(
    per_head_kv_indices: torch.Tensor,  # (num_kv_heads, total_indices) or (num_layers, num_kv_heads, total_indices)
    device: str = "cuda"
) -> torch.Tensor:
    """
    Triton-accelerated head-major augmentation.
    
    This function augments indices for head-major virtual batch organization.
    The output is directly in head-major order, no interleaving needed.
    
    Automatically handles both 2D and 3D input tensors.
    
    Args:
        per_head_kv_indices: Input indices for each head 
                           - 2D: (num_kv_heads, total_indices)
                           - 3D: (num_layers, num_kv_heads, total_indices)
        device: Device to run on
        
    Returns:
        Augmented indices in head-major order
        - 2D input: (num_kv_heads * total_indices,) flattened
        - 3D input: (num_layers, num_kv_heads * total_indices)
    """
    # Check if input is 3D
    if per_head_kv_indices.dim() == 3:
        return augment_head_major_triton_3d(per_head_kv_indices, device)
    
    # Original 2D implementation
    num_kv_heads, total_indices = per_head_kv_indices.shape
    
    # Allocate output tensor
    output_indices = torch.empty(
        num_kv_heads * total_indices,
        dtype=per_head_kv_indices.dtype,
        device=device
    )
    
    # Configure kernel launch parameters
    BLOCK_SIZE = 1024  # Larger block size for better performance
    total_size = num_kv_heads * total_indices
    num_blocks = triton.cdiv(total_size, BLOCK_SIZE)
    
    # Launch kernel
    _augment_head_major_kernel[(num_blocks,)](
        per_head_kv_indices,
        output_indices,
        num_kv_heads,
        total_indices,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output_indices
