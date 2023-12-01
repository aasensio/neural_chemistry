import torch
import numpy as np
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """Positional encoding for the input vector

    gamma(v) = [..., cos(2 * pi * sigma ** (j / m) * v), sin(2 * pi * sigma ** (j / m) * v), ...]

    Parameters
    ----------
    nn : _type_
        _description_
    """
    def __init__(self, sigma, n_freqs, input_size):
        super().__init__()
        self.sigma = sigma
        self.n_freqs = n_freqs
        self.encoding_size = (2 * n_freqs + 1) * input_size

    def forward(self, v, alpha=None):

        n_batch, n_input = v.shape

        if (alpha is None):
            alpha = 1.0
        
        k = torch.arange(self.n_freqs, device=v.device)

        weight = 0.5 * (1.0 - torch.cos((alpha * self.n_freqs - k) * np.pi))
        weight[alpha * self.n_freqs < k] = 0.0
        weight[alpha * self.n_freqs - k >= 1] = 1.0        
        weight = weight[None, None, :]

        coeffs = 2 * np.pi * self.sigma ** (1.0 * k / self.n_freqs)
        vp = coeffs * torch.unsqueeze(v, -1)
        vp_cat = torch.cat((weight * torch.cos(vp), weight * torch.sin(vp)), dim=-1)

        out = vp_cat.flatten(-2, -1)

        out = torch.cat((v, out), dim=-1)

        return out
        
    
class GaussianEncoding(nn.Module):
    def __init__(self, input_size, encoding_size, sigma=None):
        super().__init__()
        self.sigma = sigma
        self.input_size = input_size
        self.encoding_size_half = encoding_size
        self.encoding_size = 2 * encoding_size + input_size

        # Fourier matrix        
        B = self.sigma * torch.randn((encoding_size, self.input_size))
        self.B_max = torch.max(torch.abs(B))

        # Compute the frequency to reorder the Fourier matrix
        freq = torch.sqrt(torch.sum(B**2, dim=1))
        _, idx = torch.sort(freq, descending=False)
        B = B[idx, :]
        
        self.register_buffer("B", B)

    def forward(self, v, alpha=None):                
        if (alpha is None):
            alpha = 1.0

        k = torch.arange(self.encoding_size_half, device=v.device)
            
        weight = 0.5 * (1.0 - torch.cos((alpha * self.encoding_size_half - k) * np.pi))
        weight[alpha * self.encoding_size_half < k] = 0.0
        weight[alpha * self.encoding_size_half - k >= 1] = 1.0
        
        vp = 2.0 * np.pi * v @ self.B.T

        out = torch.cat([weight * torch.cos(vp), weight * torch.sin(vp)], dim=-1)
    
        out = torch.cat((v, out), dim=-1)

        return out

class IdentityEncoding(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.encoding_size = input_size
        
    def forward(self, v, alpha=None):

        return v