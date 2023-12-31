<p align="center">
    <img src="https://github.com/painebenjamin/app.enfugue.ai/assets/57536852/2af58adc-6921-4ee9-b668-e741eba80e7f" alt="ENFUGUE Web UI v0.3.2">
</p>
<h3 align="center">ENFUGUE is an open-source app for making studio-grade AI-generated images and video.</h3>
<p align="center"><i>For server or desktop, beginners or pros.</i></p><hr />
<p align="center">
  <img src="https://img.shields.io/static/v1?label=painebenjamin&message=app.enfugue.ai&color=ff3366&logo=github" alt="painebenjamin - app.enfugue.ai">
    <img src="https://img.shields.io/github/stars/painebenjamin/app.enfugue.ai?style=social" alt="stars - app.enfugue.ai">
    <img src="https://img.shields.io/github/forks/painebenjamin/app.enfugue.ai?style=social" alt="forks - app.enfugue.ai"><br />
    <a href="https://github.com/painebenjamin/app.enfugue.ai/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-AGPL3-ff3366" alt="License"></a>
    <a href="https://github.com/painebenjamin/app.enfugue.ai/releases/"><img src="https://img.shields.io/github/tag/painebenjamin/app.enfugue.ai?include_prereleases=&sort=semver&color=ff3366" alt="GitHub tag"></a>
    <a href="https://github.com/painebenjamin/app.enfugue.ai/releases/"><img alt="GitHub release (with filter)" src="https://img.shields.io/github/v/release/painebenjamin/app.enfugue.ai?color=ff3366"></a>
    <a href="https://pypi.org/project/enfugue"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/enfugue?color=ff3366"></a>
    <a href="https://github.com/painebenjamin/app.enfugue.ai/releases/"><img alt="GitHub all releases" src="https://img.shields.io/github/downloads/painebenjamin/app.enfugue.ai/total?logo=github&color=ff3366"></a>
    <a href="https://pypi.org/project/enfugue"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/enfugue?logo=python&logoColor=white&color=ff3366"></a>
</p>
<div align="center">
<img src="https://github.com/painebenjamin/app.enfugue.ai/assets/57536852/28b4bc57-ab6c-4c83-b1cb-b8c16332779d" width="260" />
<img src="https://github.com/painebenjamin/app.enfugue.ai/assets/57536852/7ec7528d-de24-4c08-b5d0-061dcc675e21" width="260" />
<img src="https://github.com/painebenjamin/app.enfugue.ai/assets/57536852/2215ef54-d820-4e78-869d-330eee5f297c" width="260" />
<br />
<img src="https://github.com/painebenjamin/app.enfugue.ai/assets/57536852/f0bca178-9a50-45f8-94a3-fa54cabf23bd" width="260" />
<img src="https://github.com/painebenjamin/app.enfugue.ai/assets/57536852/7e1adeb3-642a-43b0-9a2b-b244d50fa44b" width="260" />
<img src="https://github.com/painebenjamin/app.enfugue.ai/assets/57536852/1596f720-e609-4e24-ab1d-c90a34821687" width="260" />
<br />
<img src="https://github.com/painebenjamin/app.enfugue.ai/assets/57536852/85749fca-dd0b-47e3-a491-6d6219168bb0" width="260" />
<img src="https://github.com/painebenjamin/app.enfugue.ai/assets/57536852/98a6ac73-9b03-446c-91b2-9d09d6ce428a" width="260" />
<img src="https://github.com/painebenjamin/app.enfugue.ai/assets/57536852/44a7c5fb-87f0-4594-a8ee-f54de7e8f46d" width="260" />
<br />
<em>Views of the ENFUGUE user interface in various configurable themes, click to view full-size.</em>
</div><br />

# Feature Summary

- 🚀 **One Click Install:** Use our convenient installation script to install, update and launch ENFUGUE without any configuration needed.
-  **Plays Nice with Others:** Share AI models or entire server environments between ENFUGUE and all the most popular open-source AI applications.
- 🔪 **Cutting Edge:** All the best open-source image/video generartion models are implemented and available as soon as they're released to the public.
- 👥 **Owners and Users:** Optional authentication and authorization keeps your installation settings locked down for shared environments.
- 🗃 **Easy Model Management:** In-app CivitAI browser brings all the best community models to 
- 🧷 **Safety's On:** Safety checker is on by default, and can be disabled by owners right in the UI. You can feel safe 
- ♻️ **Waste Not, Want Not:** AI can take a lot of resources, and ENFUGUE takes care to only use what it needs. It will free your GPU as soon as it's no longer needed and clean up unneeded files as it goes.
- 🧈 **Unified Pipeline:** You never need to switch tabs to change input modes. Text-to-image, image-to-video, and all kinds of advanced operations are all immediately available through a combination of the layered canvas and input roles.
- 🛈 **Tooltipped:** Wondering what an input does? Hover your mouse over it and find out; documentation is available right in-app to help ease you into learning features as you need them.
- 🔌 **Plug Away:** All features are available via JSON API, or can be added to your Python scripts using our `diffusers` extensions.
- ☁️ **Your Own Cloud:** All of the best features you would expect from a SaaS application, with the security of knowing nothing ever leaves your computer. Results are kept by the app until you no longer need them, and your browser keeps a lengthy history of workspaces so you can always revisit where you left off.
- ⚙️ **Configurable:** Numerous configuration options are available, both in the GUI and via configuration file. Change IP addresses, ports, SSL configuration, directories, and much more.

# Installation and Running

A script is provided for Windows and Linux machines to install, update, and run ENFUGUE. Copy the relevant command below and answer the on-screen prompts to choose your installation type and install optional dependencies.

## Windows
Access the command prompt from the start menu by searching for "command." Alternatively, hold the windows key on your keyboard and click `x`, then press `r` or click `run`, then type `cmd` and press enter or click `ok`.
```cmd
curl https://raw.githubusercontent.com/painebenjamin/app.enfugue.ai/main/enfugue.bat -o enfugue.bat
.\enfugue.bat
```

## Linux
```sh
curl https://raw.githubusercontent.com/painebenjamin/app.enfugue.ai/main/enfugue.sh -o enfugue.sh
chmod u+x enfugue.sh
./enfugue.sh
```

Both of these commands accept the same flags.

```sh
USAGE: enfugue.(bat|sh) [OPTIONS]
Options:
 --help                   Display this help message.
 --conda / --portable     Automatically set installation type (do not prompt.)
 --update / --no-update   Automatically apply or skip updates (do not prompt.)
 --mmpose / --no-mmpose   Automatically install or skip installing MMPose (do not prompt.)
```

## Manual Installation

If you want to install without using the installation scripts, see this [Wiki page](https://github.com/painebenjamin/app.enfugue.ai/wiki/Installation-and-Running#manual-installation).

## Configuration

Many configuration options are available, but none are required. If you want to specify things such as the host and port that the server listens on, please review the documentation for this on [this Wiki page](https://github.com/painebenjamin/app.enfugue.ai/wiki/Configuration-for-Advanced-Users).

# App Quickstart Guide

Once the Enfugue server is running on your computer, you can start using the app. Simply open any browser (Chromium-based browsers are recommended for all features, but Firefox is also supported) and navigate to `https://app.enfugue.ai:45554`, or for short, `my.enfugue.ai` (this redirects you to the first address.) If you specified your own configuration, then the server will instead be listening on your configured address. You'll be greeted by your application home screen and some initial documentation - please read it in it's entirety.

## Interface Windows

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

The button at the bottom, labeled `ENFUGUE`, to send your current invocation settings to the engine and start the image creation process.

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

To use TensorRT, first **create a pre-configured model as detailed above.** TensorRT requires all weights be "frozen" prior to engine compilation - which means that certain parameters must be made static and unchanging (or else you must re-compile the engine.) These are:

1. The checkpoint (base model)
2. The size of the TensorRT engine (default 512px)
3. LoRA and their scales
4. LyCORIS and their scales
5. Textual Inversion

Since this is the case, you **must** use model configuration sets to compile TensorRT engines. After creating a configuration set and selecting a model, you will see a small icon next to the model name with a number. This is the number of TensorRT engines that are prepared. Each engine is used in a different portion of the inference process. Click the icon to see a small window that allows you to begin the engine build process. You will receive a notification in this window (and any others) when the build is complete, and the engine will automatically be used when it is able to be used. Build all engines to ensure you are using the fastest possible inference method for all image generation techniques.

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
