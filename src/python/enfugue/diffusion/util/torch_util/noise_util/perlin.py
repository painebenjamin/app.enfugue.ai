# type: ignore
# Adapted from https://raw.githubusercontent.com/WASasquatch/PowerNoiseSuite/main/modules/latent_noise.py
import torch
import torch.nn as nn

def normalize(latent, target_min=None, target_max=None):
    """
    Normalize a tensor `latent` between `target_min` and `target_max`.
    """
    min_val = latent.min()
    max_val = latent.max()
    
    if target_min is None:
        target_min = min_val
    if target_max is None:
        target_max = max_val
        
    normalized = (latent - min_val) / (max_val - min_val)
    scaled = normalized * (target_max - target_min) + target_min
    return scaled

# PERLIN POWER FRACTAL
class PerlinPowerFractal(nn.Module):
    """
    Generate a batch of images with a Perlin power fractal effect.
    """
    def __init__(self, width, height):
        """
        Initialize the PerlinPowerFractal.
        """
        super(PerlinPowerFractal, self).__init__()
        self.width = width
        self.height = height

    def forward(self, batch_size, X, Y, Z, frame, device='cpu', generator=None, evolution_factor=0.1,
                octaves=4, persistence=0.5, lacunarity=2.0, exponent=4.0, scale=100,
                brightness=0.0, contrast=0.0, min_clamp=0.0, max_clamp=1.0):
        """
        Generate a batch of images with Perlin power fractal effect.
        """

        def fade(t):
            return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

        def lerp(t, a, b):
            return a + t * (b - a)

        def grad(hash, x, y, z):
            h = hash & 15
            u = torch.where(h < 8, x, y)
            v = torch.where(h < 4, y, torch.where((h == 12) | (h == 14), x, z))
            return torch.where(h & 1 == 0, u, -u) + torch.where(h & 2 == 0, v, -v)

        def noise(x, y, z, p):
            X = (x.floor() % 255).to(torch.int32)
            Y = (y.floor() % 255).to(torch.int32)
            Z = (z.floor() % 255).to(torch.int32)

            x -= x.floor()
            y -= y.floor()
            z -= z.floor()

            u = fade(x)
            v = fade(y)
            w = fade(z)

            A = p[X] + Y
            AA = p[A] + Z
            AB = p[A + 1] + Z
            B = p[X + 1] + Y
            BA = p[B] + Z
            BB = p[B + 1] + Z

            r = lerp(w, lerp(v, lerp(u, grad(p[AA], x, y, z), grad(p[BA], x - 1, y, z)),
                              lerp(u, grad(p[AB], x, y - 1, z), grad(p[BB], x - 1, y - 1, z))),
                     lerp(v, lerp(u, grad(p[AA + 1], x, y, z - 1), grad(p[BA + 1], x - 1, y, z - 1)),
                              lerp(u, grad(p[AB + 1], x, y - 1, z - 1), grad(p[BB + 1], x - 1, y - 1, z - 1))))

            return r

        device = 'cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu'

        p = torch.randperm(max(self.width, self.height) ** 2, dtype=torch.int32, device=device, generator=generator)
        p = torch.cat((p, p))

        noise_map = torch.zeros(batch_size, self.height, self.width, dtype=torch.float32, device=device)

        X = torch.arange(self.width, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) + X
        Y = torch.arange(self.height, dtype=torch.float32, device=device).unsqueeze(1).unsqueeze(0) + Y
        Z = evolution_factor * torch.arange(batch_size, dtype=torch.float32, device=device).unsqueeze(1).unsqueeze(1) + Z + frame

        for octave in range(octaves):
            frequency = lacunarity ** octave
            amplitude = persistence ** octave

            nx = X / scale * frequency
            ny = Y / scale * frequency
            nz = (Z + frame * evolution_factor) / scale * frequency

            noise_values = noise(nx, ny, nz, p) * (amplitude ** exponent)

            noise_map += noise_values.squeeze(-1) * amplitude

        noise_map = normalize(noise_map, min_clamp, max_clamp)

        latent = (noise_map + brightness) * (1.0 + contrast)
        latent = normalize(latent)
        latent = latent.unsqueeze(-1)

        return latent
