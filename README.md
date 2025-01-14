PyTorch Quantizer

Naive uniform quantization of a float tensor to a given bit-width.
    
    Args:
        tensor (torch.Tensor): Input float tensor.
        num_bits (int): Number of bits for quantization (e.g., 8 for uint8).
    
    Returns:
        tuple:
            - quantized (torch.Tensor): Quantized tensor (integers).
            - scale (float): Scale used for quantization.
            - zero_point (float): Zero point used for quantization.

  
