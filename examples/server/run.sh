#!/usr/bin/env bash
#
# This script serves as an example configuration file as well as run command.
# See the full documentation for more details on configurability.
#
# We write the following configuration to a `.yml` file in the current working directory.
# Most of these are self-explanatory (or able to be explained quickly,) but there are some
# important tuning values that can determine how your server behaves that are explained here.
#
# -----------------------
# enfugue.pipeline.switch
# -----------------------
# 'switch' determines how to swap between pipelines when required, like going from inpainting to non-inpainting or loading a refiner.
# The default behavior, 'offload,' sends unneeded pipelines to the CPU and promotes active pipelines to the GPU when requested.
# This usually provides the best balance of speed and memory usage, but can result in heavy overhead on some systems.
#
# If this proves too much, or you wish to minimize memory usage, set this to 'unload,'which will always completely unload a pipeline 
# and free memory before a different pipeline is used.
#
# If you set this to 'null,' _all models will remain in memory_. This is by far thefastest but consumes the most memory, this is only
# suitable for enterprise GPUs.
#
# --------------------------
# enfugue.pipeline.inpainter
# --------------------------
# 'inpainter' determines how to inpaint when no inpainter is specified.
#
# When the value is 'true', and the user is using a stable diffusion 1.5 model for their base model, enfugue will look for another
# checkpoint with the same name but the suffix `-inpainting`, and when one is not found, it will create one using the model merger.
# Fine-tuned inpainting checkpoints perform significantly better at the task, however they are roughly equivalent in size to the
# main model, effectively doubling storage required.
#
# When the value is 'null' or 'false', Enfugue will still search for a fine-tuned inpainting model, but will not create one if it does not exist.
# Instead, enfugue will use 4-dim inpainting, which in 1.5 is less effective.
# SDXL does not have a fine-tuned inpainting model (yet,) so this procedure does not apply, and 4-dim inpainting is always used.
#
# ----------------------
# enfugue.pipeline.cache
# ----------------------
# 'cache' determines when to create diffusers caches. A diffusers cache will always load faster than a checkpoint, but is once again
# approximately the same size as the checkpoint, so this will also effectively double storage size.
#
# When the value is 'null' or 'false', diffusers caches will _only_ be made for TensorRT pipelines, as it is required. This is the default value.
#
# When the value is 'xl', enfugue will cache XL checkpoints. These load _significantly_ faster than when loading from file, between
# 2 and 3 times as quickly. You may wish to consider using this setting to speed up changing between XL checkpoints.
#
# When the value is 'true', diffusers caches will be created for all pipelines. This is not recommended as it only provides marginal
# speed advantages for 1.5 models.
#
# ---------------------------
# enfugue.pipeline.sequential
# ---------------------------
# 'sequential' enables sequential onloading and offloading of AI models.
#
# When the value is 'true', AI models will only ever be loaded to the GPU when they are needed.
# At all other times, they will be in normal memory, waiting for the next time they are requested, at which time they will be loaded
# to the GPU, and afterward unloaded.
#
# These operations take time, so this is only recommended to enable if you are experiencing issues with out-of-memory errors.

cat << EOF > $PWD/config.yml
---
sandboxed: false                                # true = disable system management in UI
server:
    host: 0.0.0.0                               # listens on any connection
    port: 45554                                 # ports < 1024 require sudo
    domain: app.enfugue.ai                      # this is a loopback domain
    secure: true                                # enables SSL
    # If you change the domain, you must provide your own certificates or disable SSL
    # key: /path/to/key.pem
    # cert: /path/to/cert.pem
    # chain: /path/to/chain.pem
    logging:
        file: ~/.cache/enfugue.log              # server logs (NOT diffusion logs)
        level: error                            # server only logs errors
enfugue:
    noauth: true                                # authentication default
    queue: 4                                    # queue size default
    safe: true                                  # safety checker default
    model: v1-5-pruned.ckpt                     # default model (repo or file)
    inpainter: null                             # default inpainter (repo or file), null = sd15 base inpainter
    refiner: null                               # default refiner (repo or fill), null = none
    refiner_start: 0.85                         # default start for refining when refiner set and no args (0.0-1.0)
    dtype: null                                 # set to float32 to disable half-precision, you probably dont want to
    engine:
        logging:
            file: ~/.cache/enfugue-engine.log   # diffusion logs (shown in UI)
            level: debug                        # logs everything, helpful for debugging
        root: ~/.cache/enfugue                  # root engine directory, images save in /images
        cache: ~/.cache/enfugue/cache           # diffusers cache, controlnets, VAE
        checkpoint: ~/.cache/enfugue/checkpoint # checkpoints only
        lora: ~/.cache/enfugue/lora             # lora only
        lycoris: ~/.cache/enfugue/lycoris       # lycoris only
        inversion: ~/.cache/enfugue/inversion   # textual inversion only
        other: ~/.cache/enfugue/other           # other AI models (upscalers, preprocessors, etc.)
    pipeline:
        switch: "offload"                   # See comment above
        inpainter: true                     # See comment above
        cache: null                         # See comment above
        sequential: false                   # See comment above
EOF

# Now we run enfugue with the current configuration file.
# Usually there is an 'enfugue' binary available on the system path, so you can
# get by with just running 'enfugue run', though if that doesn't work you can
# execute it using `python -m enfugue run`.
#
# We pass our configuration with -c <file>.
# We also pass `-m` to merge our configuration with the rest of the default configuration values.
enfugue run -c $PWD/config.yml -m
