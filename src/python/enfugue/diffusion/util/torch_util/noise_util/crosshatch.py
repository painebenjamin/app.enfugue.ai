# type: ignore
# Adapted from https://raw.githubusercontent.com/WASasquatch/PowerNoiseSuite/main/modules/latent_noise.py
import torch
import torch.nn.functional as F
import torch.nn as nn
import math

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
    
# CROSS-HATCH POWER FRACTAL
class CrossHatchPowerFractal(nn.Module):
    """
    Generate a batch of crosshatch-patterned images with a power fractal effect.
    """
    def __init__(self, width, height, frequency=320, octaves=12, persistence=1.5, num_colors=16, color_tolerance=0.05, angle_degrees=45, blur=2, brightness=0.0, contrast=0.0, clamp_min=0.0, clamp_max=1.0):
        """
        Initialize the CrossHatchPowerFractal.

        Args:
            width (int): Width of each image in pixels.
            height (int): Height of each image in pixels.
            frequency (int, optional): Frequency of the crosshatch pattern. Default is 320.
            octaves (int, optional): Number of octaves for fractal generation. Default is 12.
            persistence (float, optional): Persistence parameter for fractal generation. Default is 1.5.
            num_colors (int, optional): Number of colors to map the generated noise to. Default is 16.
            color_tolerance (float, optional): Color tolerance for mapping noise values to colors. Default is 0.05.
            angle_degrees (float, optional): Angle in degrees for the crosshatch pattern orientation. Default is 45.
            blur (int, optional): Amount of blur to apply to the generated image. Default is 2.
            brightness (float, optional): Adjusts the overall brightness of the generated images. Default is 0.0.
            contrast (float, optional): Adjusts the contrast of the generated images. Default is 0.0.
            clamp_min (float, optional): Minimum value to clamp the pixel values to. Default is 0.0.
            clamp_max (float, optional): Maximum value to clamp the pixel values to. Default is 1.0.
        """
        super(CrossHatchPowerFractal, self).__init__()
        self.width = width
        self.height = height
        self.frequency = frequency
        self.num_octaves = octaves
        self.persistence = persistence
        self.angle_radians = math.radians(angle_degrees)
        self.num_colors = num_colors
        self.color_tolerance = color_tolerance
        self.blur = blur
        self.brightness = brightness
        self.contrast = contrast
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, batch_size=1, device='cpu', generator=None):
        """
        Generate a batch of crosshatch-patterned images.
        """
        x = torch.linspace(0, 1, self.width, dtype=torch.float32, device=device)
        y = torch.linspace(0, 1, self.height, dtype=torch.float32, device=device)
        x, y = torch.meshgrid(x, y, indexing="ij")

        batched_noises = []

        for i in range(batch_size):
            noise = torch.zeros(self.width, self.height, device=device)

            for octave in range(self.num_octaves):
                frequency = self.frequency * 2 ** octave
                octave_noise = self.generate_octave(x, y, frequency)
                noise += octave_noise * self.persistence ** octave

            noise = (noise - noise.min()) / (noise.max() - noise.min())
            colored_noise = self.apply_color_mapping(noise, device, generator)
            colored_noise = colored_noise.cpu()

            r_channel = colored_noise[:, :, 0]
            g_channel = colored_noise[:, :, 1]
            b_channel = colored_noise[:, :, 2]

            kernel_size = int(self.blur * 2 + 1)
            uniform_kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)

            blurred_r = F.conv2d(r_channel.unsqueeze(0).unsqueeze(0), uniform_kernel, padding=int(self.blur))
            blurred_g = F.conv2d(g_channel.unsqueeze(0).unsqueeze(0), uniform_kernel, padding=int(self.blur))
            blurred_b = F.conv2d(b_channel.unsqueeze(0).unsqueeze(0), uniform_kernel, padding=int(self.blur))
            blurred_noise = torch.cat((blurred_r, blurred_g, blurred_b), dim=1)
            blurred_noise = F.interpolate(blurred_noise, size=(self.height, self.width), mode='bilinear')

            batched_noises.append(blurred_noise.permute(0, 2, 3, 1))

        batched_noises = torch.cat(batched_noises, dim=0).to(device='cpu')
        batched_noises = (batched_noises + self.brightness) * (1.0 + self.contrast)
        
        return normalize(batched_noises, self.clamp_min, self.clamp_max)

    def generate_octave(self, x, y, frequency):
        """
        Generate an octave of the crosshatch pattern.

        Args:
            x (torch.Tensor): X-coordinate grid.
            y (torch.Tensor): Y-coordinate grid.
            frequency (int): Frequency of the crosshatch pattern.

        Returns:
            torch.Tensor: Octave of the crosshatch pattern.
        """
        grid_hatch_x = torch.sin(x * frequency * math.pi)
        grid_hatch_y = torch.sin(y * frequency * math.pi)

        grid_hatch_x = (grid_hatch_x - grid_hatch_x.min()) / (grid_hatch_x.max() - grid_hatch_x.min())
        grid_hatch_y = (grid_hatch_y - grid_hatch_y.min()) / (grid_hatch_y.max() - grid_hatch_y.min())
        grid_hatch = grid_hatch_x + grid_hatch_y

        return grid_hatch

    def apply_color_mapping(self, noise, device, generator):
        """
        Apply color mapping to noise values fir a consisten look
        """
        random_colors = torch.rand(self.num_colors, 3, generator=generator, dtype=torch.float32, device=device)

        noise_scaled = noise * (self.num_colors - 1)
        tolerance = self.color_tolerance * (self.num_colors - 1)
        noise_scaled_rounded = torch.round(noise_scaled)
        colored_noise = random_colors[noise_scaled_rounded.long()]

        return colored_noise

# LINEAR CROSS-HATCH POWER FRACTAL
class CrossHatchLinearPowerFractal(nn.Module):
    """
    Generate a batch of linear crosshatch-patterned images with a power fractal effect.
    """
    def __init__(self, width, height, frequency=320, octaves=12, persistence=1.5, angle_degrees=45, gain=0.1, add_noise_tolerance=0.25, mapping_range=24, brightness=0.0, contrast=0.0):
        """
        Initialize the CrossHatchLinearPowerFractal.
        """
        super(CrossHatchLinearPowerFractal, self).__init__()
        self.width = width
        self.height = height
        self.frequency = frequency
        self.num_octaves = octaves
        self.persistence = persistence
        self.angle_radians = math.radians(angle_degrees)
        self.gain = gain
        self.noise_tolerance = add_noise_tolerance
        self.mapping_range = mapping_range
        self.brightness = brightness
        self.contrast = contrast

    def forward(self, batch_size=1, device='cpu', generator=None):
        """
        Generate a batch of crosshatch-patterned images.
        """
        x = torch.linspace(0, 1, self.width, dtype=torch.float32, device=device)
        y = torch.linspace(0, 1, self.height, dtype=torch.float32, device=device)
        x, y = torch.meshgrid(x, y, indexing="ij")

        noises = []
        for batch_idx in range(batch_size):
            noise = torch.zeros(self.width, self.height, dtype=torch.float32, device=device)

            for octave in range(self.num_octaves):
                frequency = self.frequency * 2 ** octave
                octave_noise = self.generate_octave(x, y, frequency, device, generator=generator)
                noise += octave_noise * self.persistence ** octave

            noise = normalize(noise, 0, 1)
            mapped_noise = self.apply_mapping(noise.permute(1, 0), device)

            # Expand the tensor to have 3 channels
            mapped_noise = mapped_noise.unsqueeze(-1).expand(-1, -1, 3)

            noises.append(mapped_noise)

        batched_noises = torch.stack(noises, dim=0)

        return batched_noises.to(device='cpu')

    def generate_octave(self, x, y, frequency, device, generator):
        """
        Generate an octave of the crosshatch pattern.
        """
        grid_hatch_x = torch.sin(x * (frequency * self.gain) * math.pi)
        grid_hatch_y = torch.sin(y * (frequency * self.gain) * math.pi)

        noise = torch.randint(int(self.frequency + 1), (grid_hatch_x.shape[0], grid_hatch_x.shape[1]), device=grid_hatch_x.device, dtype=grid_hatch_x.dtype, generator=generator)
        
        grid_hatch = (normalize(grid_hatch_x) + normalize(grid_hatch_y)) + (noise * self.noise_tolerance)

        return grid_hatch

    def apply_mapping(self, noise, device):
        """
        Apply mapping to noise values for consistent look.

        Args:
            noise (torch.Tensor): Noise values.
            device (str): The device to use for computation ('cpu' or 'cuda').

        Returns:
            torch.Tensor: Noise values after mapping.
        """
        steps = min(max(self.mapping_range, 4), 256)

        step_mapping = torch.linspace(0, 1, steps, dtype=torch.float32, device=device)
        noise_scaled = noise * (steps - 1)
        noise_scaled_rounded = torch.round(noise_scaled)
        noise_scaled_rounded = torch.clamp(noise_scaled_rounded, 0, steps - 1)

        noise = step_mapping[noise_scaled_rounded.long()]

        return noise
