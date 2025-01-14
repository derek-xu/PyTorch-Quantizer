import torch

def quantize_tensor(tensor: torch.Tensor, num_bits: int = 8):
    """
    Naive uniform quantization of a float tensor to a given bit-width.
    
    Args:
        tensor (torch.Tensor): Input float tensor.
        num_bits (int): Number of bits for quantization (e.g., 8 for uint8).
    
    Returns:
        tuple:
            - quantized (torch.Tensor): Quantized tensor (integers).
            - scale (float): Scale used for quantization.
            - zero_point (float): Zero point used for quantization.
    """
    # Determine the range of the input data
    min_val, max_val = tensor.min(), tensor.max()
    
    # Handle edge case where all values might be equal
    # to avoid division-by-zero in scale calculation
    if min_val == max_val:
        # If all elements are the same, just return a zero tensor
        # (or all the same value) for quantized data
        return (
            torch.zeros_like(tensor, dtype=torch.int), 
            1.0,  # scale
            0.0   # zero_point
        )
    
    # Number of quantization levels: for 8 bits => 256
    q_levels = 2 ** num_bits
    
    # Calculate scale and zero_point
    scale = (max_val - min_val) / (q_levels - 1)
    zero_point = -min_val / scale
    
    # Quantize
    #   1) shift by zero_point
    #   2) scale
    #   3) round to nearest integer
    #   4) clamp to [0, q_levels-1]
    quantized = torch.round(tensor / scale + zero_point)
    quantized = torch.clamp(quantized, 0, q_levels - 1)
    
    # Convert to integer type
    quantized = quantized.to(torch.int)
    
    return quantized, scale.item(), zero_point.item()


def dequantize_tensor(q_tensor: torch.Tensor, scale: float, zero_point: float):
    """
    Dequantize a tensor back to float given the scale and zero point.
    
    Args:
        q_tensor (torch.Tensor): Quantized tensor (integers).
        scale (float): Scale used during quantization.
        zero_point (float): Zero point used during quantization.
    
    Returns:
        torch.Tensor: Dequantized float tensor.
    """
    return (q_tensor - zero_point) * scale


if __name__ == "__main__":
    # Example usage
    # Create a random float tensor
    float_data = torch.randn(5) * 10  # random data in some range
    print("Original float tensor:\n", float_data, "\n")

    # Quantize
    q_data, scale, z_point = quantize_tensor(float_data, num_bits=8)
    print("Quantized tensor (int):\n", q_data)
    print(f"Scale: {scale}, Zero Point: {z_point}\n")

    # Dequantize
    dq_data = dequantize_tensor(q_data, scale, z_point)
    print("Dequantized tensor (float):\n", dq_data)
