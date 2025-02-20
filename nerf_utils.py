import torch

def get_minibatches(inputs, chunksize=1024*8):
    return [inputs[i : i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

def cumprod_exclusive(tensor: torch.Tensor):
    cumprod = torch.cumprod(tensor, -1)
    cumprod = torch.roll(cumprod, 1, -1)
    cumprod[..., 0] = 1.0
    return cumprod

def positional_encoding(tensor, num_encoding_functions=6, include_input=True, log_sampling=True):
    encoding = [tensor.to(torch.float32)] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=torch.float32,
            device=tensor.device)
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=torch.float32,
            device=tensor.device)

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # For no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)