"""
Triton kernel for head-major augmentation (without interleaving).

This kernel performs the augmentation step for head-major virtual batch organization:
- Augments indices: block_idx * num_kv_heads + head_idx
- Directly outputs in head-major order (no interleaving needed)
"""

import triton
import triton.language as tl


@triton.jit
def _augment_head_major_kernel(
    per_head_kv_indices_ptr,  # (num_kv_heads, total_indices)
    output_indices_ptr,       # Output augmented indices
    num_kv_heads,
    total_indices,
    BLOCK_SIZE: tl.constexpr
):
    """
    Optimized Triton kernel for head-major augmentation.
    
    Each thread block processes a contiguous chunk of the flattened output.
    The augmentation formula is: augmented_idx = original_idx * num_kv_heads + head_idx
    """
    # Get the program ID
    pid = tl.program_id(0)
    
    # Calculate the starting position in the flattened output
    start_pos = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = start_pos + tl.arange(0, BLOCK_SIZE)
    
    # Calculate total size
    total_size = num_kv_heads * total_indices
    
    # Mask for valid positions
    mask = offsets < total_size
    
    # Calculate head index and position within head for each element
    # Since output is head-major: position = head_idx * total_indices + idx_within_head
    head_idx = offsets // total_indices
    idx_within_head = offsets % total_indices
    
    # Calculate input positions
    input_pos = head_idx * total_indices + idx_within_head
    
    # Load original indices
    original_indices = tl.load(per_head_kv_indices_ptr + input_pos, mask=mask, other=0)
    
    # Apply augmentation: block_idx * num_kv_heads + head_idx
    augmented_indices = original_indices * num_kv_heads + head_idx
    
    # Store augmented indices
    tl.store(output_indices_ptr + offsets, augmented_indices, mask=mask)


@triton.jit
def _augment_head_major_3d_kernel(
    per_head_kv_indices_ptr,  # (num_layers, num_kv_heads, total_indices)
    output_indices_ptr,       # Output augmented indices (num_layers, num_kv_heads * total_indices)
    num_layers,
    num_kv_heads,
    total_indices,
    BLOCK_SIZE: tl.constexpr
):
    """
    Optimized Triton kernel for head-major augmentation with 3D input.
    
    Processes all layers in parallel.
    The augmentation formula is: augmented_idx = original_idx * num_kv_heads + head_idx
    """
    # Get the program ID (2D grid)
    pid_layer = tl.program_id(0)
    pid_block = tl.program_id(1)
    
    # Skip if layer index is out of bounds
    if pid_layer >= num_layers:
        return
    
    # Calculate the starting position within the layer
    start_pos = pid_block * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = start_pos + tl.arange(0, BLOCK_SIZE)
    
    # Calculate total size per layer
    total_size_per_layer = num_kv_heads * total_indices
    
    # Mask for valid positions within the layer
    mask = offsets < total_size_per_layer
    
    # Calculate head index and position within head for each element
    head_idx = offsets // total_indices
    idx_within_head = offsets % total_indices
    
    # Calculate input positions (3D tensor layout)
    input_pos = pid_layer * total_size_per_layer + head_idx * total_indices + idx_within_head
    
    # Calculate output positions (2D tensor layout)
    output_pos = pid_layer * total_size_per_layer + offsets
    
    # Load original indices
    original_indices = tl.load(per_head_kv_indices_ptr + input_pos, mask=mask, other=0)
    
    # Apply augmentation: block_idx * num_kv_heads + head_idx
    augmented_indices = original_indices * num_kv_heads + head_idx
    
    # Store augmented indices
    tl.store(output_indices_ptr + output_pos, augmented_indices, mask=mask)
