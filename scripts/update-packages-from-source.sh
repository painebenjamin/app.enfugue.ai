#!/usr/bin/env sh

# Installs packages that are always installed from source
pip install git+https://github.com/CompVis/stable-diffusion.git@main#egg=latent-diffusion git+https://github.com/openai/CLIP.git@main#egg=clip git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers -I
