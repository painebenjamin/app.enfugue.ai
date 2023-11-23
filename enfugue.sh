#!/usr/bin/env bash
#
# This shell script serves as a one-and-done download, update, configure and run command for enfugue.
# It's able to download enfuge via conda or in portable form.
#
# Below is the configuration for enfugue. Most values are self-explanatory, but some others need some explanation.
# See below the configuration for explanation on these values.
#
# AN IMPORTANT NOTE ABOUT NETWORKING
# 
# By default enfugue uses a domain-based networking scheme with the domain `app.enfugue.ai`.
# This is a registered domain name that resolves to `127.0.0.1`, the loopback domain.
# If you're running enfugue on a local machine with internet access, this should be fine, and you shouldn't need to change any configuration.
# If you're running enfugue on a remote machine, you will want to change the following:
# - `server.domain` should be changed to the domain you want to access the interface through _unless_ you're using a proxy. This can be an IP address.
# - `server.secure` should be set to false _unless_ you also configure an SSL certificate. See the commented-out (#) lines in the configuration below.
# If you're additionally using a proxy like ngrok, you should configure `server.cms.path.root` to be the proxied URL. See the commented-out (#) lines below.

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
    cms:
        path:                                   # only configure when different from top-level domain
            # root: http(s)://my-proxy-host.ngrok-or-other/
enfugue:
    noauth: true                                # authentication default
    queue: 4                                    # queue size default
    safe: true                                  # safety checker default
    model: v1-5-pruned.ckpt                     # default model (repo or file)
    inpainter: null                             # default inpainter (repo or file), null = sd15 base inpainter
    refiner: null                               # default refiner (repo or file), null = none
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
        motion: ~/.cache/enfugue/motion         # motion modules only
        other: ~/.cache/enfugue/other           # other AI models (upscalers, preprocessors, etc.)
    pipeline:
        switch: "offload"                   # See comment above
        inpainter: true                     # See comment above
        cache: null                         # See comment above
        sequential: false                   # See comment above
EOF
trap "rm $PWD/config.yml" EXIT
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
#
# -- end of configuration --

# This function prints out the help message.
usage() {
    echo "USAGE: $0 [OPTIONS]"
    echo "Options:"
    echo " -h                   Display this help message."
    echo " -t <conda|portable>  Automatically set installation type (do not prompt.)"
    echo " -u <yes|no>          Automatically apply or skip updates (do not prompt.)"
    echo " -d <directory>       Automatically extract portable installation to this directory (do not prompt.)"
    echo " -s <yes|no>          Automatically apply or skip symlinking portable binary (do not prompt.)"
    echo " -c <yes|no>          Automatically install miniconda if needed. Exit if not installed and [no] passed (do not prompt.)"
    echo " -m <yes|no>          Automatically install or skip mmpose if needed (do not prompt.)"
}

# Declare default options, then iterate through command line arguments and set variables.
INSTALL_TYPE=""
INSTALL_MINICONDA=""
INSTALL_MMPOSE=""
INSTALL_UPDATE=""
INSTALL_DIRECTORY=""
INSTALL_SYMLINK=""

while getopts ":ht:u:i:c:m:d:s:" ARG; do
    case $ARG in
        t)
            INSTALL_TYPE=$OPTARG
            ;;
        d)
            INSTALL_DIRECTORY=$OPTARG
            ;;
        u)
            INSTALL_UPDATE=${OPTARG:0:1}
            INSTALL_UPDATE=${INSTALL_UPDATE,,}
            ;;
        m)
            INSTALL_MMPOSE=${OPTARG:0:1}
            INSTALL_MMPOSE=${INSTALL_MMPOSE,,}
            ;;
        c)
            INSTALL_MINICONDA=${OPTARG:0:1}
            INSTALL_MINICONDA=${INSTALL_MINICONDA,,}
            ;;
        s)
            INSTALL_SYMLINK=${OPTARG:0:1}
            INSTALL_SYMLINK=${INSTALL_SYMLINK,,}
            ;;
        h)
            usage
            exit 0
            ;;
        *)
            echo "Invalid option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

# This function compares versions and prompts you to download when relevant.
# Pass --no-update when executing this script to never check versions.
# Pass --update when executing this script to automatically update when available.
compare_prompt_update() {
    COMPARE=$($PYTHON -c "from semantic_version import compare; print(compare('$1', '$2'))")
    if [ "$COMPARE" == "-1" ]; then
        if [ "$INSTALL_UPDATE" == "y" ]; then
            echo "y"
        else
            # A new version of enfugue is available
            read -p "Version $2 of enfugue is available, you have version $1 installed. Download update? [Yes]: " DOWNLOAD_LATEST
            DOWNLOAD_LATEST=${DOWNLOAD_LATEST:-Yes}
            DOWNLOAD_LATEST=${DOWNLOAD_LATEST:0:1}
            echo "${DOWNLOAD_LATEST,,}"
        fi
    fi
}

# This function downloads and extracts the latest portable version when relevant.
download_portable() {
    PORTABLE_DIR="$1"
    if [ "$PORTABLE_DIR" == "" ]; then
        DEFAULT_DIR="$HOME/enfugue"
        read -p "Where would you like to extract enfugue? [$DEFAULT_DIR]: " PORTABLE_DIR
        PORTABLE_DIR=${PORTABLE_DIR:-$DEFAULT_DIR}
    fi
    RELEASE_PACKAGES=$(curl -s https://api.github.com/repos/painebenjamin/app.enfugue.ai/releases/latest | grep browser_download_url | grep manylinux | cut -d : -f 2,3 | tr -d ' ,\"')
    mkdir -p $PORTABLE_DIR
    while IFS= read -r TARBALL; do
        FILENAME=$(basename $TARBALL)
        curl -L $TARBALL -o $FILENAME
    done <<< "$RELEASE_PACKAGES"
    DOWNLOADED=$(echo "$RELEASE_PACKAGES" | sed 's|.*/||')
    cat $DOWNLOADED | tar -xvz --directory $PORTABLE_DIR --strip-components=2 1>&2
    rm $DOWNLOADED
    echo $PORTABLE_DIR
}

# Second gather some variables from the current environment.
ENFUGUE=$(which enfugue)
ENFUGUE_SERVER=$(which enfugue-server)
CONDA=$CONDA_EXE
# Make sure conda can be executed.
if [ ! -x $CONDA ]; then
    CONDA=""
fi

# These will be populated later if relevant.
ENFUGUE_INSTALLED_PIP_VERSION=""
ENFUGUE_AVAILABLE_PIP_VERSION=""
ENFUGUE_INSTALLED_PORTABLE_VERSION=""
ENFUGUE_AVAILABLE_PORTABLE_VERSION=""

# Check if we can simply activate an existing conda environment.
if [[ "$ENFUGUE" == "" && "$CONDA" != "" ]]; then
    if conda env list | grep -q enfugue; then
        echo "Found enfugue environment, activating."
        source $(dirname $CONDA)/activate enfugue
        ENFUGUE=$(which enfugue)
    fi
fi

# Get the current python executable
PYTHON=$(which python3)

# Check if either of the above tactics found enfugue. If so, and it's not disabled, check for updates.
if [ "$ENFUGUE" != "" ]; then
    if [[ "$PYTHON" != "" && "$INSTALL_UPDATE" != "n" && "$INSTALL_UPDATE" != "f" ]]; then
        # Get installed version from pip
        ENFUGUE_INSTALLED_PIP_VERSION=$($PYTHON -m pip freeze | grep enfugue | awk -F= '{print $3}' | sed -e 's/\.post/\-/g')
    fi
    if [ "$ENFUGUE_INSTALLED_PIP_VERSION" != "" ]; then
        # Enfugue was installed in this current environment and update allowed, get available versions from pip
        ENFUGUE_AVAILABLE_PIP_VERSION=$($PYTHON -m pip install enfugue== 2>&1 | grep 'from versions' | awk '{n=split($0,v,/, /); print v[n]}')
        ENFUGUE_AVAILABLE_PIP_VERSION=$(echo ${ENFUGUE_AVAILABLE_PIP_VERSION::-1} | sed -e 's/\.post/\-/g')
        # Compare versions and prompt if necessary
        DOWNLOAD_UPDATE="$(compare_prompt_update $ENFUGUE_INSTALLED_PIP_VERSION $ENFUGUE_AVAILABLE_PIP_VERSION)"
        if [[ "$DOWNLOAD_UPDATE" == "y" || "$DOWNLOAD_UPDATE" == "t" ]]; then
            echo "Downloading update."
            pip install --upgrade enfugue
        fi
    fi
elif [ "$ENFUGUE_SERVER" != "" ]; then
    # Portable found
    ENFUGUE_PORTABLE_DIR=$(dirname $(realpath $ENFUGUE_SERVER))
    if [[ "$INSTALL_UPDATE" != "n" && "$INSTALL_UPDATE" != "f" ]]; then
        # Get versions
        ENFUGUE_AVAILABLE_PORTABLE_VERSION=$(curl -s https://api.github.com/repos/painebenjamin/app.enfugue.ai/releases/latest | grep "tag_name" | cut -d : -f 2,3 | tr -d ' ,\"')
        ENFUGUE_INSTALLED_PORTABLE_VERSION=$(cat $ENFUGUE_PORTABLE_DIR/enfugue/version.txt)
        # Compare versions and prompt if necessary
        DOWNLOAD_UPDATE="$(compare_prompt_update $ENFUGUE_INSTALLED_PORTABLE_VERSION $ENFUGUE_AVAILABLE_PORTABLE_VERSION)"
        if [[ "$DOWNLOAD_UPDATE" == "y" || "$DOWNLOAD_UPDATE" == "t" ]]; then
            echo "Downloading update."
            download_portable $(dirname $ENFUGUE_PORTABLE_DIR)
        fi
    fi
fi

# Check if we've found enfugue by now. If not, prompt what to do.
if [[ "$ENFUGUE" == "" && "$ENFUGUE_SERVER" == "" ]]; then
    echo "Enfugue is not currently installed."
    # Prompt how to install enfugue.
    DOWNLOAD_TYPE=""
    if [ "$INSTALL_TYPE" == "conda" ]; then
        DOWNLOAD_TYPE="1"
    elif [ "$INSTALL_TYPE" == "portable" ]; then
        DOWNLOAD_TYPE="2"
    fi
    while [ "$DOWNLOAD_TYPE" == "" ]; do
        echo "How would you like to install it?"
        echo "1) Anaconda/Miniconda (Recommended)"
        echo "2) Portable Installation"
        read -p "Please make a selection: [1] " DOWNLOAD_TYPE
        if [ "$DOWNLOAD_TYPE" == "" ]; then
            DOWNLOAD_TYPE="1"
        else
            DOWNLOAD_TYPE=${DOWNLOAD_TYPE:0:1}
        fi
        if [[ "$DOWNLOAD_TYPE" != "1" && "$DOWNLOAD_TYPE" != "2" ]]; then
            echo "'$DOWNLOAD_TYPE' is not a valid response."
            DOWNLOAD_TYPE=""
        fi
    done
    if [ "$DOWNLOAD_TYPE" == "1" ]; then
        # Check if conda is already installed.
        if [ "$CONDA" == "" ]; then
            # Prompt if we should install miniconda
            if [[ "${INSTALL_MINICONDA,,}" != "y" && "${INSTALL_MINICONDA,,}" != "t" ]]; then
                read -p "Conda not found. Install miniconda? [Yes]: " INSTALL_MINICONDA
                INSTALL_MINICONDA=${INSTALL_MINICONDA:-Yes}
                INSTALL_MINICONDA=${INSTALL_MINICONDA:0:1}
            fi
            if [[ "${INSTALL_MINICONDA,,}" != "y" && "${INSTALL_MINICONDA,,}" != "t" ]]; then
                echo "Exiting installer. Install anaconda or miniconda and ensure it is available on your PATH, then try again."
                exit 1
            fi
            # Run the miniconda installer
            MINICONDA_DEFAULT_DIR="${HOME}/miniconda3"
            read -p "Enter miniconda installation directory: [$MINICONDA_DEFAULT_DIR]" MINICONDA_INSTALL_DIR
            MINICONDA_INSTALL_DIR=${MINICONDA_INSTALL_DIR:-$MINICONDA_DEFAULT_DIR}
            mkdir -p $MINICONDA_INSTALL_DIR
            curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o ${MINICONDA_INSTALL_DIR}/miniconda.sh --silent
            bash $MINICONDA_INSTALL_DIR/miniconda.sh -b -u -p $MINICONDA_INSTALL_DIR
            # Remove the miniconda installer
            rm -rf $MINICONDA_INSTALL_DIR/miniconda.sh
            CONDA=$MINICONDA_INSTALL_DIR/condabin/conda
        fi
        echo "Creating enfugue environment. This can take up to 15 minutes depending on download speeds, please be patient."
        # Download the latest environment file from the github
        curl https://raw.githubusercontent.com/painebenjamin/app.enfugue.ai/main/environments/linux-cuda.yml -o $PWD/enfugue-environment.yml --silent
        # Use conda to create the environment
        $CONDA env create -f $PWD/enfugue-environment.yml
        # Remove the environment file
        rm $PWD/enfugue-environment.yml
        # Activate the environment
        source $(dirname $CONDA)/activate enfugue
        ENFUGUE=$(which enfugue)
        PYTHON=$(which python)
    elif [ "$DOWNLOAD_TYPE" == "2" ]; then
        # Download and extract the latest portable
        PORTABLE_DIR=$(download_portable $INSTALL_DIRECTORY)
        ENFUGUE_SERVER="$PORTABLE_DIR/enfugue-server"
        # Prompt if we should add a symlink so this script can find it in the future
        if [[ "$INSTALL_SYMLINK" != "n" && "$INSTALL_SYMLINK" != "f" ]]; then
            if [[ "$INSTALL_SYMLINK" != "y" && "$INSTALL_SYMLINK" != "t" ]]; then
                read -p "Successfully extracted enfugue. Add symlink to /usr/local/bin? [Yes]: " INSTALL_SYMLINK
                INSTALL_SYMLINK=${INSTALL_SYMLINK:-Yes}
                INSTALL_SYMLINK=${INSTALL_SYMLINK:0:1}
            fi
        fi
        if [ "${INSTALL_SYMLINK,,}" == "y" ]; then
            sudo ln -s $ENFUGUE_SERVER /usr/local/bin/enfugue-server
        fi
    fi
fi

# Now enfugue is installed, check if we can install MMPose
if [[ "$ENFUGUE" != "" && "$PYTHON" != "" ]]; then
    MMPOSE_INSTALLED=$($PYTHON -m pip freeze | grep mmpose)
    if [[ "$MMPOSE_INSTALLED" == "" && "${INSTALL_MMPOSE,,}" != 'f' && "${INSTALL_MMPOSE,,}" != 'n' ]]; then
        if [[ "${INSTALL_MMPOSE,,}" != "y" && "${INSTALL_MMPOSE,,}" != "t" ]]; then
            read -p "MMPose not installed. Install it? [Yes]: " INSTALL_MMPOSE
            INSTALL_MMPOSE=${INSTALL_MMPOSE:-Yes}
            INSTALL_MMPOSE=${INSTALL_MMPOSE:0:1}
        fi
        if [[ "${INSTALL_MMPOSE,,}" == "y" || "${INSTALL_MMPOSE,,}" == "t" ]]; then
            $PYTHON -m mim install mmengine
            $PYTHON -m mim install "mmcv>=2.0.1"
            $PYTHON -m mim install "mmdet>=3.1.0"
            $PYTHON -m mim install "mmpose>=1.1.0"
        fi
    fi
fi

# Now we should have enfugue, run it.
if [ "$ENFUGUE" != "" ]; then
    # Run enfugue via python module script
    $PYTHON -m enfugue run
elif [ "$ENFUGUE_SERVER" != "" ]; then
    # Run enfugue via portable executable
    PORTABLE_DIR=$(dirname $(realpath $ENFUGUE_SERVER))
    # Check if this is the first launch (it will take longer)
    LAUNCH_FILE=${PORTABLE_DIR}/.launched
    if [ ! -f $LAUNCH_FILE ]; then
        echo "First launch detected, it may take a minute or so for the server to start.";
        if [ "${NO_BROWSER}" == false ]; then
            echo "A window will be opened when the server is ready to respond to requests.";
        fi
        touch $LAUNCH_FILE
    fi
    # Set path variables so binaries can be found
    export PATH=$PORTABLE_DIR/torch/lib:$PORTABLE_DIR/tensorrt:$PATH
    export LD_LIBRARY_PATH=$PORTABLE_DIR/torch/lib:$PORTABLE_DIR/tensorrt:$LD_LIBRARY_PATH
    # Allow lazy CUDA loading
    export CUDA_MODULA_LOADING=LAZY
    # Disable unsafe legacy crytography suites (MacOS requires this)
    export CRYTOGRAPHY_OPENSSL_NO_LEGACY=1
    # Enable MPS fallback (MacOS)
    export PYTORCH_ENABLE_MPS_FALLBACK=1
    # Allow enfugue to spawn unlimited processes
    # This is essential for enfugue to operate - enfugue uses process isolation,
    # so it will very frequently create and destroy processes. Without this, enfugue
    # will enfugue not be able to create any more processes and it will start failing.
    ulimit -n unlimited 2>/dev/null > /dev/null || true
    # Echo and go
    echo "Starting Enfugue server. Press Ctrl+C to exit."
    ENFUGUE_CONFIG=$PWD/config.yml $PORTABLE_DIR/enfugue-server
    echo "Goodbye!"
else
    echo "Could not find or download enfugue. Exiting."
    exit 1
fi

exit 0
