# type: ignore
# Adapted from https://raw.githubusercontent.com/WASasquatch/PowerNoiseSuite/main/modules/latent_noise.py
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.fft as fft

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

# POWER-LAW NOISE
class PowerLawNoise(nn.Module):
    """
    Generate various types of power-law noise.
    """
    def __init__(self, device='cpu'):
        """
        Initialize the PowerLawNoise.

        Args:
            device (str, optional): The device to use for computation ('cpu' or 'cuda'). Default is 'cpu'.
            alpha (float, optional): The exponent of the power-law distribution. Default is 2.0.
        """
        super(PowerLawNoise, self).__init__()
        self.device = device
        
    @staticmethod
    def get_noise_types():
        """
        Return the valid noise types

        Returns:
            (list): a list of noise types to use for noise_type parameter
        """
        return ["white", "grey", "pink", "green", "blue", "random_mix", "brownian_fractal", "velvet", "violet"]

    @classmethod
    def get_random_noise_type(cls, generator=None):
        """
        Returns a random noise type
        """
        noise_types = cls.get_noise_types()
        return noise_types[int(torch.randint(0, len(noise_types), (1,), generator=generator)[0])]

    def get_generator(self, noise_type):
        if noise_type in self.get_noise_types():
            if noise_type == "white":
                return self.white_noise
            elif noise_type == "grey":
                return self.grey_noise
            elif noise_type == "pink":
                return self.pink_noise
            elif noise_type == "green":
                return self.green_noise
            elif noise_type == "blue":
                return self.blue_noise
            elif noise_type == "velvet":
                return self.velvet_noise
            elif noise_type == "violet":
                return self.violet_noise
            elif noise_type == "random_mix":
                return self.mix_noise
            elif noise_type == "brownian_fractal":
                return self.brownian_fractal_noise
        else:
            raise ValueError(f"`noise_type` is invalid. Valid types are {', '.join(self.get_noise_types())}")

    def white_noise(self, batch_size, width, height, scale, alpha=0.0, generator=None, **kwargs):
        """
        Generate white noise with a power-law distribution.
        """
        scale = scale
        noise_real = torch.randn((batch_size, 1, height, width), device=self.device, generator=generator)
        noise_power_law = torch.sign(noise_real) * torch.abs(noise_real) ** alpha
        noise_power_law *= scale
        return noise_power_law.to(self.device)

    def grey_noise(self, batch_size, width, height, scale, alpha=1.0, generator=None, **kwargs):
        """
        Generate grey noise with a flat power spectrum and modulation.
        """
        scale = scale
        noise_real = torch.randn((batch_size, 1, height, width), device=self.device, generator=generator)
        modulation = torch.abs(noise_real) ** (alpha - 1)
        noise_modulated = noise_real * modulation
        noise_modulated *= scale
        return noise_modulated.to(self.device)

    def blue_noise(self, batch_size, width, height, scale, alpha=2.0, generator=None, **kwargs):
        """
        Generate blue noise using the power spectrum method.
        """
        noise = torch.randn(batch_size, 1, height, width, device=self.device, generator=generator)
        
        freq_x = fft.fftfreq(width, 1.0)
        freq_y = fft.fftfreq(height, 1.0)
        Fx, Fy = torch.meshgrid(freq_x, freq_y, indexing="ij")
        
        power = (Fx**2 + Fy**2)**(alpha / 2.0)
        power[0, 0] = 1.0
        power = power.unsqueeze(0).expand(batch_size, 1, width, height).permute(0, 1, 3, 2).to(device=self.device)
        
        noise_fft = fft.fftn(noise)
        power = power.to(noise_fft)
        noise_fft = noise_fft / torch.sqrt(power)
        
        noise_real = fft.ifftn(noise_fft).real
        noise_real = noise_real - noise_real.min()
        noise_real = noise_real / noise_real.max()
        noise_real = noise_real * scale
        
        return noise_real.to(self.device)

    def green_noise(self, batch_size, width, height, scale, alpha=1.5, generator=None, **kwargs):
        """
        Generate green noise using the power spectrum method.
        """
        noise = torch.randn(batch_size, 1, height, width, device=self.device, generator=generator)
        
        freq_x = fft.fftfreq(width, 1.0)
        freq_y = fft.fftfreq(height, 1.0)
        Fx, Fy = torch.meshgrid(freq_x, freq_y, indexing="ij")
        
        power = (Fx**2 + Fy**2)**(alpha / 2.0)
        power[0, 0] = 1.0
        power = power.unsqueeze(0).expand(batch_size, 1, width, height).permute(0, 1, 3, 2).to(device=self.device)
        
        noise_fft = fft.fftn(noise)
        power = power.to(noise_fft)
        noise_fft = noise_fft / torch.sqrt(power)
        
        noise_real = fft.ifftn(noise_fft).real
        noise_real = noise_real - noise_real.min()
        noise_real = noise_real / noise_real.max()
        noise_real = noise_real * scale
        
        return noise_real.to(self.device)
        
    def pink_noise(self, batch_size, width, height, scale, alpha=1.0, generator=None, **kwargs):
        """
        Generate pink noise using the power spectrum method.
        """
        noise = torch.randn(batch_size, 1, height, width, device=self.device, generator=generator)
        
        freq_x = fft.fftfreq(width, 1.0)
        freq_y = fft.fftfreq(height, 1.0)
        Fx, Fy = torch.meshgrid(freq_x, freq_y, indexing="ij")
        
        power = (Fx**2 + Fy**2)**(alpha / 2.0)
        power[0, 0] = 1.0
        power = power.unsqueeze(0).expand(batch_size, 1, width, height).permute(0, 1, 3, 2).to(device=self.device)

        noise_fft = fft.fftn(noise)
        noise_fft = noise_fft / torch.sqrt(power.to(noise_fft.dtype))

        noise_real = fft.ifftn(noise_fft).real
        noise_real = noise_real - noise_real.min()
        noise_real = noise_real / noise_real.max()
        noise_real = noise_real * scale
        
        return noise_real.to(self.device)
    
    def velvet_noise(self, batch_size, width, height, alpha=1.0, device='cpu', generator=None, **kwargs):
        """
        Generate true Velvet noise with specified width and height using PyTorch.
        """
        white_noise = torch.randn((batch_size, 1, height, width), device=device, generator=generator)
        velvet_noise = torch.sign(white_noise) * torch.abs(white_noise) ** (1 / alpha)
        velvet_noise /= torch.max(torch.abs(velvet_noise))
        
        return velvet_noise

    def violet_noise(self, batch_size, width, height, alpha=1.0, device='cpu', generator=None, **kwargs):
        """
        Generate true Violet noise with specified width and height using PyTorch.
        """
        white_noise = torch.randn((batch_size, 1, height, width), device=device, generator=generator)
        violet_noise = torch.sign(white_noise) * torch.abs(white_noise) ** (alpha / 2.0)
        violet_noise /= torch.max(torch.abs(violet_noise))
        
        return violet_noise

    def brownian_fractal_noise(self, batch_size, width, height, scale, alpha=1.0, modulator=1.0, generator=None, **kwargs):
        """
        Generate Brownian fractal noise using the power spectrum method.
        """
        def add_particles_to_grid(grid, particle_x, particle_y):
            for x, y in zip(particle_x, particle_y):
                grid[y, x] = 1

        def move_particles(particle_x, particle_y):
            dx = torch.randint(-1, 2, (batch_size, n_particles), device=self.device, generator=generator)
            dy = torch.randint(-1, 2, (batch_size, n_particles), device=self.device, generator=generator)
            particle_x = torch.clamp(particle_x + dx, 0, width - 1)
            particle_y = torch.clamp(particle_y + dy, 0, height - 1)
            return particle_x, particle_y

        n_iterations = int(5000 * modulator)
        fy = fft.fftfreq(height).unsqueeze(1) ** 2
        fx = fft.fftfreq(width) ** 2
        f = fy + fx
        power = torch.sqrt(f) ** alpha
        power[0, 0] = 1.0

        grid = torch.zeros(height, width, dtype=torch.uint8, device=self.device)

        n_particles = n_iterations // 10 
        particle_x = torch.randint(0, int(width), (batch_size, n_particles), device=self.device, generator=generator)
        particle_y = torch.randint(0, int(height), (batch_size, n_particles), device=self.device, generator=generator)

        neighborhood = torch.tensor([[1, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1]], dtype=torch.uint8, device=self.device)

        for _ in range(n_iterations):
            add_particles_to_grid(grid, particle_x, particle_y)
            particle_x, particle_y = move_particles(particle_x, particle_y)

        brownian_tree = grid.clone().detach().float().to(self.device)
        brownian_tree = brownian_tree / brownian_tree.max()
        brownian_tree = F.interpolate(brownian_tree.unsqueeze(0).unsqueeze(0), size=(height, width), mode='bilinear', align_corners=False)
        brownian_tree = brownian_tree.squeeze(0).squeeze(0)

        fy = fft.fftfreq(height).unsqueeze(1) ** 2
        fx = fft.fftfreq(width) ** 2
        f = fy + fx
        power = torch.sqrt(f) ** alpha
        power[0, 0] = 1.0
        noise_real = brownian_tree * scale

        amplitude = 1.0 / (scale ** (alpha / 2.0))
        noise_real *= amplitude

        noise_fft = fft.fftn(noise_real.to(self.device))
        noise_fft = noise_fft / power.to(self.device)
        noise_real = fft.ifftn(noise_fft).real
        noise_real *= scale

        return noise_real.unsqueeze(0).unsqueeze(0)

    def noise_masks(self, batch_size, width, height, scale, num_masks=3, alpha=2.0, generator=None):
        """
        Generate a fixed number of random masks.
        """
        masks = []
        for i in range(num_masks):
            noise_type = self.get_random_noise_type(generator=generator)
            mask = self.get_generator(noise_type)(batch_size, width, height, scale=scale, alpha=alpha, generator=generator)
            masks.append(mask)
        return masks

    def mix_noise(self, batch_size, width, height, scale, alpha=2.0, generator=None, **kwargs):
        """
        Mix white, grey, and pink noise with blue noise masks.
        """
        noise_types = [self.get_random_noise_type(generator=generator) for _ in range(3)]
        scales = [scale] * 3
        noise_alpha = 0.5 + (float(torch.rand((1,), generator=generator)[0]) * 1.5)

        mixed_noise = torch.zeros(batch_size, 1, height, width, device=self.device)
        
        for noise_type in noise_types:
            noise = self.get_generator(noise_type)(batch_size, width, height, scale=scale, alpha=noise_alpha, generator=generator).to(self.device)
            mixed_noise += noise

        return mixed_noise

    def forward(self, batch_size, width, height, alpha=2.0, scale=1.0, modulator=1.0, noise_type="white", generator=None):
        """
        Generate a noise image with options for type, frequency, and generator
        """

        if noise_type not in self.get_noise_types():
            raise ValueError(f"`noise_type` is invalid. Valid types are {', '.join(self.get_noise_types())}")
        
        channels = []
        for i in range(3):
            noise = normalize(self.get_generator(noise_type)(batch_size, width, height, scale=scale, generator=generator, alpha=alpha, modulator=modulator))
            channels.append(noise)

        noise_image = torch.cat((channels[0], channels[1], channels[2]), dim=1)
        noise_image = (noise_image - noise_image.min()) / (noise_image.max() - noise_image.min())
        noise_image = noise_image.permute(0, 2, 3, 1).float()

        return noise_image.to(device="cpu")
