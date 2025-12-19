"""
Utilities for calculating Bits Per Pixel (BPP) for MS-ILLM compression.
"""
import torch
from typing import Dict, Optional, Tuple, Any, List, Union

try:
    from neuralcompression.metrics import pickle_size_of
except ImportError:
    # Fallback if neuralcompression is not available
    import pickle
    def pickle_size_of(obj):
        return len(pickle.dumps(obj))


def calculate_bpp_from_latent(
    latent: torch.Tensor,
    original_shape: Tuple[int, ...],
    bits_per_element: int = 8,
) -> float:
    """
    Calculate Bits Per Pixel (BPP) from a compressed latent representation.
    
    Args:
        latent: Compressed latent tensor from encoder (any shape)
        original_shape: Original image shape (C, H, W) or (B, C, H, W)
        bits_per_element: Number of bits per element in the latent (default: 8 for quantized)
    
    Returns:
        BPP value (bits per pixel)
    """
    # Get number of pixels in original image
    if len(original_shape) == 3:
        # (C, H, W)
        num_pixels = original_shape[1] * original_shape[2]
    elif len(original_shape) == 4:
        # (B, C, H, W)
        num_pixels = original_shape[2] * original_shape[3]
    else:
        raise ValueError(f"Unexpected original_shape: {original_shape}")
    
    # Calculate total bits in latent
    num_elements = latent.numel()
    total_bits = num_elements * bits_per_element
    
    # BPP = total_bits / num_pixels
    bpp = total_bits / num_pixels
    return bpp


def calculate_bpp_from_encoder_output(
    encoder_output: torch.Tensor,
    original_image: torch.Tensor,
    bits_per_element: int = 8,
) -> float:
    """
    Calculate BPP from encoder output and original image.
    
    Args:
        encoder_output: Output from MS-ILLM encoder (latent z)
        original_image: Original image tensor (B, C, H, W) or (C, H, W)
        bits_per_element: Number of bits per element in latent
    
    Returns:
        BPP value
    """
    return calculate_bpp_from_latent(
        encoder_output,
        original_image.shape,
        bits_per_element=bits_per_element,
    )


def calculate_bpp_from_hyperprior_output(
    compressed_output: Any,
    original_shape: Tuple[int, ...],
) -> float:
    """
    Calculate BPP from HyperpriorCompressedOutput (actual bitstream).
    Uses pickle_size_of (same as official MS-ILLM evaluation code).
    
    Args:
        compressed_output: HyperpriorCompressedOutput object containing latent_strings and hyper_latent_strings
        original_shape: Original image shape (C, H, W) or (B, C, H, W)
        
    Returns:
        BPP value
    """
    # Get number of pixels in original image
    if len(original_shape) == 3:
        # (C, H, W)
        num_pixels = original_shape[1] * original_shape[2]
        batch_size = 1
    elif len(original_shape) == 4:
        # (B, C, H, W)
        num_pixels = original_shape[2] * original_shape[3]
        batch_size = original_shape[0]
    else:
        raise ValueError(f"Unexpected original_shape: {original_shape}")
    
    # Use pickle_size_of (same as official MS-ILLM evaluation code)
    num_bytes = pickle_size_of(compressed_output)
    total_bits = num_bytes * 8
    
    # BPP = total_bits / (num_pixels * batch_size)
    bpp = total_bits / (num_pixels * batch_size)
    return bpp


def accumulate_bpp_stats(
    bpp_dict: Dict[str, float],
    stats_dict: Optional[Dict[str, list]] = None,
) -> Dict[str, list]:
    """
    Accumulate BPP statistics across multiple measurements.
    
    Args:
        bpp_dict: Dictionary with keys like 'rgb_static', 'rgb_gripper' and BPP values
        stats_dict: Optional existing stats dictionary to update
    
    Returns:
        Updated stats dictionary with lists of BPP values
    """
    if stats_dict is None:
        stats_dict = {}
    
    for key, bpp_value in bpp_dict.items():
        if key not in stats_dict:
            stats_dict[key] = []
        stats_dict[key].append(bpp_value)
    
    return stats_dict


def compute_average_bpp(stats_dict: Dict[str, list]) -> Dict[str, float]:
    """
    Compute average BPP from accumulated statistics.
    
    Args:
        stats_dict: Dictionary with lists of BPP values
    
    Returns:
        Dictionary with average BPP values
    """
    avg_bpp = {}
    for key, bpp_list in stats_dict.items():
        if len(bpp_list) > 0:
            avg_bpp[key] = sum(bpp_list) / len(bpp_list)
        else:
            avg_bpp[key] = 0.0
    
    return avg_bpp
