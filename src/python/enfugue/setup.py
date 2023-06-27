import os
from setuptools import setup, find_packages

package_name = "enfugue"
version_major = "0"
version_minor = "1"
version_patch = "0"

install_requires = [
    "cheroot>=9.0.0",
    "nvidia-pyindex>=1.0.9",
    "pibble[cherrypy]>=0.5",
    "torch>=1.13.1,<2.0",
    "numpy>=1.24.0",
    "colored>=1.4,<1.5",
    "diffusers>=0.16,<0.17",
    "albumentations>=0.4.3,<0.5",
    "opencv-python>=4.7.0.72,<4.8",
    "pudb==2019.2",
    "invisible-watermark>=0.1,<0.2",
    "imageio>=2.9,<3.0",
    "imageio-ffmpeg>=0.4.2,<0.5",
    "pytorch-lightning>=2.0.2,<2.1",
    "omegaconf>=2.1.1,<2.2",
    "test-tube>=0.7.5,<0.8",
    "streamlit>=0.73,<0.74",
    "einops>=0.3,<0.4",
    "torch-fidelity>=0.3,<0.4",
    "transformers>=4.28,<5.0",
    "torchmetrics==0.11.4",
    "kornia>=0.6,<0.7",
    "polygraphy>=0.47,<0.48",
    "accelerate>=0.18,<0.19",
    "tqdm>=4.27",
    "safetensors>=0.3,<0.4",
    "realesrgan>=0.3,<0.4",
    "gfpgan>=1.3.8,<1.4",
    "backgroundremover>=0.2.3,<0.3",
    "beautifulsoup4>=4.12,<5",
    "pystray>=0.19,<0.24",
    "html2text",
    "taming-transformers",
    "clip",
    "latent-diffusion",
]

extras_require = {
    "tensorrt": [
        "onnx==1.12",
        "onnxruntime-gpu==1.12.1",
        "onnx-graphsurgeon==0.3.26",
        "tensorrt>=8.6,<8.7",
    ],
    "build": [
        "types-protobuf>=4.23.0.1,<5.0",
        "types-requests>=2.30,<3.0",
        "types-setuptools>=67.7,<68.0",
        "types-urllib3>=1.26.25,<2.0",
        "types-tabulate>=0.9,<0.10",
        "importchecker>=2.0,<3.0",
    ],
}

packaged_files = []

here = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.join(here, package_name)
data_dirs = ["static", "config"]
for data_dir in data_dirs:
    data_dir_path = os.path.join(package_dir, data_dir)
    for dir_path, sub_dirs, file_names in os.walk(data_dir_path):
        for file_name in file_names:
            packaged_files.append(os.path.join(dir_path, file_name))

package_data = [os.path.relpath(packaged_file, package_dir) for packaged_file in packaged_files]

setup(
    name=package_name,
    author="Benjamin Paine",
    author_email="painebenjamin@gmail.com",
    version=f"{version_major}.{version_minor}.{version_patch}",
    packages=find_packages("."),
    package_data={package_name: ["py.typed", *package_data]},
    license="agpl-3.0",
    entry_points={"console_scripts": ["enfugue = enfugue.__main__:main"]},
    install_requires=install_requires,
    extras_require=extras_require,
)
