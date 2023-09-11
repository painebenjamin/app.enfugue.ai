<p align="center">
    <img src="https://github.com/painebenjamin/app.enfugue.ai/blob/main/docs/github-header.png?raw=true" alt="ENFUGUE">
</p>

<h2 align="center">
Enfugue is a feature-rich Stable Diffusion web app for desktop or server.
</h2>
<p align="center">
<em>Forever open source and totally free.</em>
</p>

<p align="center">
    <img src="https://github.com/painebenjamin/app.enfugue.ai/assets/57536852/6f6c6df9-dbb2-40d7-bd8c-834de9c22b20" alt="The ENFUGUE interface" />
</p>

# Feature Summary

- 🚀 **One Click Install:** Enfugue is available as a simple `.exe` to make it as easy as possible to get busy making images, not configuring environments.
- 🤝 **Plays Nice with Others:** If you want more control in managing your environment, simply install `enfugue` via `pip` into your existing stable diffusion workflow. Worried about a whole webapp clogging up your installation? Don't be, Enfugue is only `1 MB` in size on it's own.
- 👥 **Owners and Users:** Optional authentication and authorization keeps your installation settings locked down.
- 🗃 **Easy Model Management:** In-app CivitAI browser makes community checkpoints, LoRA, and Textual Inversions one click away. Drop any other models directly into the app to keep files organized.
- 🧷 **Safety's On:** Safety checker is on by default, and can be disabled by owners right in the UI.
- ⚡ **Turbocharged:** Have a powerful Nvidia GPU? TensorRT support is built-in; speed up inference by up to 100% using state-of-the-art AI technology from Nvidia.
- ♻️ **Waste Not, Want Not:** AI can take a lot of resources, and Enfugue takes care to only use what it needs. It will free your GPU for desktop applications or gaming as soon as it's no longer needed, and clean up unneeded files as it goes.
- 🧈 **Unified Pipeline:** Never choose between `txt2img`, `img2img`, `inpainting`, or any upscaling pipeline again, with or without multi-diffusion. Just ask what you want, and Enfugue will take care of the rest.
- 🕹️ **Take Control:** Region prompting and Controlnet are standard.
- 🔌 **Plug Away:** All features are available via JSON API, or can be added to your Python scripts using our `diffusers` extensions.
- 👁️ **Eye Queue:** Have things to do? Send an unlimited\* number of invocations at once, let Enfugue take care of making sure they all get done.
- ☁️ **Your Own Cloud:** All of the best features you would expect from a SaaS application, with the security of knowing nothing ever leaves your computer. Results are kept by the app until you no longer need them, and your browser keeps a lengthy history of workspaces so you can always revisit where you left off.
- ⚙️ **Configurable:** Numerous configuration options are available, both in the GUI and via configuration file. Change IP addresses, ports, SSL configuration, directories, and much more.

*\* configurable in-app, defaults to five queued invocations*

# Installation and Running

## As Easy as Possible: Self-Contained Executable

1. Navigate to [the Releases page](https://github.com/painebenjamin/app.enfugue.ai/releases) and download the latest release as `.zip` (Windows) or `.tar.gz` (MacOS & Linux).
2. Extract the archive anywhere. See the releases page for details on extraction.
3. Navigate to the archive folder and run the executable file - `enfugue-server.exe` for Windows, or `enfugue.sh` for Linux and MacOS. Some situations may require additional commands, see the releases page for more details.

On windows, you will now see the Enfugue icon in the bottom-right-hand corner of your screen. Click on this to exit the server when you wish. To enable TensorRT for Windows follow the steps under **Windows TensorRT Support** below.

## Advanced: Creating your Own Environment and Running from Command Line

This instruction assumes you are using a variant of [Conda](https://docs.conda.io/projects/conda/en/stable/).
1. Choose an environment in in the `environments/` directory that corresponds to your platform and hardware.
   1. If you have a powerful next-generation Nvidia GPU (3000 series and better with at least 12 GB of VRAM), use `tensorrt` for all of the capabilities of `cuda` with the added ability to compile TensorRT engines.
   2. If you have any other Nvidia GPU or CUDA-capable device, or do not plan to use `tensorrt`, use `cuda`.
   3. If you are on a MacOS M1 or M2 device, use `macos-mps`. Other MacOS devices are not supported.
   4. Additional graphics APIs for AMD devices coming soon.
2. Run the command `conda env create -f <file_downloaded_above>`
3. Run the command `conda activate enfugue`
4. Run the command `enfugue run` to run the server. Issue a keyboard interrupt (Ctrl+C) to stop it.

## À la Carte

You can install `enfugue` into any other latent diffusion Python environment using `pip install enfugue`. If you are on Linux and want to install TensorRT support as well, use `pip install enfugue[tensorrt]`. If you are on Windows, this will not work, you will need to install the python packages from source as detailed below.

## Windows Nvidia TensorRT Support

In order to use Nvidia TensorRT on Windows, some additional steps must be taken. This is temporary (hopefully) as TensorRT support for Windows is very new.

You will be asked to add a number of directories to your PATH. On windows, the easiest way to reach it is:
1. Open the start menu and begin typing "Environment". You will see an option that says "Edit the system environment variables," click this.
2. In the bottom-right-hand corner of the System Properties window, click "Environment Variables."
3. Under your user, click the "Path" variable and then click "Edit".
4. Add a new entry pointing to the requested path.

<p align="center">
    <img src="https://github.com/painebenjamin/app.enfugue.ai/blob/main/docs/github-windows-env.png?raw=true" alt="Windows Configuration for Enabling TensorRT" />
</p>

Before downloading anything, you will need to make an account with Nvidia and [Join the Nvidia Developer Program](https://developer.nvidia.com/developer-program).

Once that is complete, download the following packages and install them anywhere to your system. 
1. [Install CUDA](https://developer.nvidia.com/cuda-11-7-1-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local), add `/bin` to PATH
2. [Install CUDNN](https://developer.nvidia.com/rdp/cudnn-download), add `/lib` to PATH
3. [Install TensorRT](https://developer.nvidia.com/nvidia-tensorrt-8x-download), add `/lib` to PATH. If you are creating your own environment, you should also use `pip` to install `python/tensorrt-8.*-cp310-none-win_amd64.whl` from this directory.
4. If you are creating your own environment, now run `pip install enfugue[tensorrt]`

## Configuration

Many configuration options are available, but none are required. If you want to specify things such as the host and port that the server listens on, please review the documentation for this on [the wiki](https://github.com/painebenjamin/app.enfugue.ai/wiki/Configuration-for-Advanced-Users).

# App Quickstart Guide

Once the Enfugue server is running on your computer, you can start using the app. Simply open any browser (Chromium-based browsers are recommended for all features, but Firefox is also supported) and navigate to `https://app.enfugue.ai:45554`, or for short, `my.enfugue.ai` (this redirects you to the first address.) If you specified your own configuration, then the server will instead be listening on your configured address. You'll be greeted by your application home screen and some initial documentation - please read it in it's entirety.

## Windows

The Enfugue interface uses a custom frontend framework which features a windows-like interface. Many interface elements will spawn windows; these can be moved around, minimized, maximized, resized and closed as you would expect if you've ever worked in a window-focused interface.

<p align="center">
    <img src="https://github.com/painebenjamin/app.enfugue.ai/blob/main/docs/windows.png?raw=true" alt="Windows in ENFUGUE" />
</p>

## Components

The User Interface is broken up into a small handful of components:

### The Sidebar

<p align="center">
    <img src="https://github.com/painebenjamin/app.enfugue.ai/blob/main/docs/sidebar.png?raw=true" alt="The ENFUGUE interface sidebar" />
</p>

The sidebar is where all of your settings are for the current image you are creating. Most import is the `prompt` field, which must be filled with what you want the image to contain before invoking the engine. All other settings are optional; click on any of the headers to expand them and view the settings beneath. Hold your mouse over the relevant input sections to see details about what it does, or visit [the Wiki](https://github.com/painebenjamin/app.enfugue.ai/edit/main/README.md).

The button at the button, labeled `ENFUGUE`, to send your current invocation settings to the engine and start the image creation process.

### The Canvas

The main feature that sets Enfugue apart from other Stable Diffusion interfaces is the Canvas. 

With nothing on it, the canvas shows you a preview of the shape of your inference, with convenient 8-pixel and 64-pixel demarcations provided.

<p align="center">
    <img src="https://github.com/painebenjamin/app.enfugue.ai/blob/main/docs/canvas.png?raw=true" alt="The ENFUGUE interface canvas with nothing in it" />
</p>

While making images, the canvas will be replaced with in-progress samples, and then by the final images when complete. When making multiple samples at once, you can choose between the samples here.

<p align="center">
    <img src="https://github.com/painebenjamin/app.enfugue.ai/blob/main/docs/samples.png?raw=true" alt="The ENFUGUE interface canvas showing in-progress samples" />
</p>

You can move the entire canvas (pan) by placing your cursor over it then holding down the **middle-mouse** button, or alternatively **Ctrl+Left-Mouse-Button** or **Alt+Left-Mouse-Button** (**Option⌥+Left-Mouse-Button on MacOS**) , and move the canvas around.

Zoom in and out using the scroll wheel or scroll gestures. You can also click the `+` and `-` icons in the bottom-right-hand corner. Click 'RESET' at any time to bring the canvas back to the initial position.

<p align="center">
    <img src="https://github.com/painebenjamin/app.enfugue.ai/blob/main/docs/zoom-and-pan.png?raw=true" alt="The ENFUGUE interface zoomed in and panned to the side." />
</p>

To take better control of the image, there are different **nodes** available that can be placed on the canvas. See the **toolbar** section below for descriptions of the nodes that can be added. Nodes can be moved, removed, and resized just like windows, within the confines of the canvas. 

<p align="center">
    <img src="https://github.com/painebenjamin/app.enfugue.ai/blob/main/docs/canvas-nodes.png?raw=true" alt="The ENFUGUE interface canvas with multiple nodes" />
</p>

Nodes on the canvas often feature additional buttons on their headers. Place your cursor over each to see what they do. Some nodes hide their headers when your cursor is not in them so you can better see the contents underneath.

<p align="center">
    <img src="https://github.com/painebenjamin/app.enfugue.ai/blob/main/docs/scribble.png?raw=true" width="300" alt="A scribble node on the ENFUGUE interface." />
    <img src="https://github.com/painebenjamin/app.enfugue.ai/blob/main/docs/scribble-result.png?raw=true" width="300" alt="The result of a scribble on the ENFUGUE interface." />
</p>

Drag an image toward the top area of another image to merge them together. This allows you to use multiple different images for different methods of control.

Additionally, some nodes feature the ability to draw black and white images using simple tools. These nodes all feature an array of buttons at the top to control various things about the brush you are drawing with. There are some additional controls available when drawing:
1. Use the Scroll Wheel or Scroll Gestures to increase/decrease the size of the brush.
2. Hold `Control` when scrolling up/down to stop this behavior and instead perform the previous behavior of zooming in and out.
3. Left-click to draw, or hold `Alt` and left-click to erase.
4. After releasing left-click, if you then draw somewhere else while holding `shift`, a line will be drawn between the last point and the new point using the current brush.

### The Toolbar

<p align="center">
    <img src="https://github.com/painebenjamin/app.enfugue.ai/blob/main/docs/toolbar.png?raw=true" alt="The ENFUGUE interface toolbar" />
</p>

1. **Image**: Upload an image from your computer and place it on the canvas. Paste an image in the window to quickly make an image node without having to save it.
2. **Scribble**: Draw the shape you're looking for.
3. **Prompt**: Denote a section of the image as a different prompt with it's own settings.

### The Menu Bar

<p align="center">
    <img src="https://github.com/painebenjamin/app.enfugue.ai/blob/main/docs/menu.png?raw=true" alt="The ENFUGUE menu bar" />
</p>

1. **File**: Save, load, and reset settings, including the entire content of the canvas, uploaded images and all. Also review your history here.
2. **Models**: Manage models - download fine-tuned models from CivitAI and create configurations of models and prompts.
3. **System**: Manage your installation. Find here the settings window to enable or disable authentication, enable or disable the safety checker, and manage queue sizes. If authentication is enabled, you will also manager users and passwords here.
4. **Help**: View information about Enfugue, and find links to resources such as this page.

### The Header

<p align="center">
    <img src="https://github.com/painebenjamin/app.enfugue.ai/blob/main/docs/header.png?raw=true" alt="The ENFUGUE header" />
</p>

The header contains useful information about your GPU, as well as two important icons. The **Download** icon shows you the progress of any active downloads, and the **Status** icon shows you the current engine status - Green (ready), Yellow (busy) or Gray (Idle). Whenever the status indicator shows any state other than **Idle**, you can click on it to terminate the engine, stopping any active diffusion process.

<p align="center">
    <img src="https://github.com/painebenjamin/app.enfugue.ai/blob/main/docs/terminate.png?raw=true" alt="A window in the ENFUGUE interface offering to terminate any active invocations." />
</p>

### The Model Picker

A special callout should be made to the Model Picker, the input in the top-left-hand corner of the Canvas. This allows you to pick between installed checkpoints and pre-configured models:

<p align="center">
	<img src="https://github.com/painebenjamin/app.enfugue.ai/assets/57536852/82f32e7e-1775-40ab-96ca-771352dd22bc" /><br />
 	<em>Selecting between installed checkpoints and preconfigured models.</em>
</p>

After downloading a model from Civit AI, or uploading one through the menu at `System` -> `Installation`, or manually playing one into the correct directory (`~/.cache/enfugue/checkpoint`, `~/.cache/enfugue/lora`, etc, by default, or as configured by the user during initialization or using the `System > Installation Manager` menu item,) use the **Model Manager** from the `Models` menu to create a pre-configured set of model, LoRA, LyCORIS, Textual Inversions, default/trigger prompts, and other default values.

![image](https://github.com/painebenjamin/app.enfugue.ai/assets/57536852/78c7c05f-4af5-47a0-ab2b-da80ae38e035)

You can also create configurations on-the-fly when selecting a checkpoint from the model picker.

<p align="center">
	<img src="https://github.com/painebenjamin/app.enfugue.ai/assets/57536852/c4cb4497-dd60-4e47-90a4-64810d04c447" /><br />
 	<em>Using advanced configuration after selecting a checkpoint.</em>
</p>

## Tensor RT

TensorRT is a technology created by Nvidia that transforms an AI model into one that takes advantage of hardware acceleration available on Nvidia GPUs.

As there are numerous varying architectures used by Nvidia that support this technology, these engines must be compiled by an architecture compatible with your actual hardware, rather than distributed by AI model providers. The compilation time for each model varies, but generally takes between 15 and 30 minutes each. You can expect between 50% and 100% faster inference speeds during the engine's respective step(s).

This is **only** available for modern 30xx and 40xx Nvidia GPU's.

After selecting a model, you will see a small icon next to the model name with a number. This is the number of TensorRT engines that are prepared. Each engine is used in a different portion of the inference process. Click the icon to see a small window that allows you to begin the engine build process. You will receive a notification in this window (and any others) when the build is complete, and the engine will automatically be used when it is able to be used. Build all engines to ensure you are using the fastest possible inference method for all image generation techniques.

<p align="center">
    <img src="https://github.com/painebenjamin/app.enfugue.ai/blob/main/docs/tensorrt-build.png?raw=true" alt="A window in the ENFUGUE interface offering to build TensorRT engines." />
</p>

**Note:** *This is experimental.* Engine builds can fail for a variety of reasons, some of which are not immediately apparent. There is a hard cap of one hour of unresponsiveness from the build process before it will be canceled.

## Authentication

When enabled, authentication will be required when using Enfugue. This enables system administrators to create a two-tiered hierarchy of users and administrators, where users are not permitted to modify models or the installation itself; they are only allowed to use the pre-configured setup. The primary impetus behind this was to create the ability for parents to curate an environment for children to safely experiment with generative AI.

<p align="center">
    <img src="https://github.com/painebenjamin/app.enfugue.ai/assets/57536852/c1603304-ad73-48a3-91c9-aecbffbe4841" alt="A window in the ENFUGUE interface offering multiple settings options" />
</p>

Once enabled in the settings menu, you will be taken to a login screen. The default user and password are both `enfugue`, all lowercase. You can change any other user's password as an administrator. 

If you ever forget your password, you can reset the root password by creating a file named `password_reset.txt` in the enfugue cache directory with the desired new password, then restart the server. The cache directory is located at `~/.cache/enfugue/`, where `~` is your home directory or user directory, depending on platform.

## Tips

Here are a few quick tips to getting great results:
1. The base Stable Diffusion model can create some pretty good images, but it's with the fine-tuned models from the larger AI community that you will see the best images. Open up the CivitAI browser and download the latest and greatest models to give them a try.
2. The very first invocation with a new model will always take quite a bit longer than subsequent ones, as Enfugue moves files and downloads any other necessary files.
3. Upscaling can take an image from mediocre to great with surprising frequency. Try out the AI upscale methods and upscale diffusion to take your image generation to the next level.

# Troubleshooting

Here are a few steps to take if you're having trouble getting Enfugue to work for you:
1. Ensure your Firewall is not blocking port inbound port `45554`. This would be uncommon, but possible in strict environments. See [here](https://learn.microsoft.com/en-us/windows/security/operating-system-security/network-security/windows-firewall/create-an-inbound-port-rule) for details on how to create an Inbound Port rule.
2. Instead of typing in `my.enfugue.ai`, go directly to `https://app.enfugue.ai:45554/`
3. Clear you cache and cookies, then try reload the app.
4. Report your issue [in the issues tab](https://github.com/painebenjamin/app.enfugue.ai/issues)

If all else fails, you can always try deleting the `enfugue` folder and `enfugue.db` file in your `~/.cache` directory to re-initialize the application.

# FAQ

## What is "Full FP32", "Pruned PickleTensor", etc.?

These flags are the precision level and format of the model. In order to use all of Enfugue's features, Enfugue will change these formats and precisions as necessary, so in theory any of them will work the same as the other. In practice, if one is available, pick `Pruned` and `FP16/Half` for the fastest download and processes.

## What are the best settings for the best images?

There is still much to be learned about generative AI in general, and Stable Diffusion in specific. A great starting point has been preconfigured in Enfugue, but there will be no one-size-fits-all set of parameters that works for every kind of image you want to generate. The best way to learn is simply to play with the values and see what the effect is on the final image.

## Where can I learn more?

1. [The Wiki](https://github.com/painebenjamin/app.enfugue.ai/wiki) (in progress)
2. [The Discussion Boards](https://github.com/painebenjamin/app.enfugue.ai/discussions)

Additional resources will be made available as they are needed, so don't hesitate to ask for what you think will work best for you.

# For Developers

## The Enfugue Diffusion Pipeline

Enfugue uses an extension of `diffusers.StableDiffusionPipeline` that provides a number of additional arguments over the typical signature, weaving between `txt2img`, `img2img`, `inpaint` and `controlnet` as necessary. It also has TensorRT support for all models in the pipeline. Start [here](https://github.com/painebenjamin/app.enfugue.ai/tree/main/src/python/enfugue) for documentation on how it is used.

## Enfugue JSON API

The entirety of Enfugue's capabilities are available via JSON API. Find the documentation in [the wiki.](https://github.com/painebenjamin/app.enfugue.ai/wiki/JSON-API)

## Building

For anyone interested in building from source themselves, simply check out this repository and issue a `make` command to build the associated binary release. See below for all make targets.

| Build Step | Description | Depends On |
| ---------- | ----------- | ---------- |
| **clean** | This target removes build artifacts. | None |
| **typecheck** | This step runs `mypy` against each source file. See [mypy-lang.org](https://mypy-lang.org/) for details on python static typing. Mypy is ran with the `--strict` flag, meaning all constraints are opted in. |  Python source files |
| **importcheck** | This step runs `importcheck` against each source file. See [github](https://github.com/python-coincidence/importcheck) for details on importcheck; simply put, it will produce an error if an imported module is not used. |  Python source files | 
| **unittest** | This step runs `doctest` against each source file. See [the Python Documentation](https://docs.python.org/3/library/doctest.html) for details on doctest. This will run all tests placed in docstrings, you will see these as python commands in the documentation, prepended by `>>>` |  Python source files |
| **test** | This step runs `enfugue.test.run`. This will run the `main` method in `<n>*.py` files places in the `test` directory. |  Python source files, python test files |
| **vendor** | This step fetches vendor resources by running all scripts under the `vendor/` directory. | Script files |
| **js** | This step compresses and mangles the `.mjs` files using `terser`. | `src/js/*.mjs` files |
| **css** | This step minifies all `css` files using `cssminify`. | `src/css/*.css` files |
| **html**, **img** | These step simply copy the relevant static directories (`/html`, `/img`) to the build directory. | `src/html/*.html` files, `src/img/*.*` files |
| **sdist** | This step compiles the source distribution into an installable `.tar.gz` file, suitable for passing to `pip install`. Contains all the results of the previous steps | Python source files, passing `typecheck`, `importcheck`, `unittest`, and `test`, running `vendor`, compiling `js`, `css`, `html`, and `img` |
| **dist** | This step compiles the relevant executable artifact (and zips/tars it). | sdist |
| **dockerfile** | This step compiles the dockerfile and prepares it for building. | sdist |
| **docker** | This step builds the docker image. | dockerfile |

## Running directly from Source

To run directly from source (in development mode,) use the `scripts/run-dev.sh` script. This works on Windows (in Cygwin) and on Linux, it has not been tested on MacOS.
