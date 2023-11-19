#!/usr/bin/env bash
#
# This shell script serves as a one-and-done download, update and run
# command for enfugue. It's able to download enfuge via conda or in
# portable form.

# We first declare default options, these will be populated later.
NO_UPDATE=false
NO_BROWSER=false
UPDATE=false
CONFIG=""

# This function prints out the help message.
usage() {
    echo "USAGE: $0 [OPTIONS]"
    echo "Options:"
    echo " --help           Display this help message."
    echo " --update         Automatically download any update when it is available."
    echo " --config         An optional configuration file to use instead of the default."
    echo " --no-update      Do not fetch versions and prompt to update prior to launching. Takes precedence over --update."
}

# This function compares versions and prompts you to download when relevant.
# Pass --no-update when executing this script to never check versions.
# Pass --update when executing this script to automatically update when available.
compare_prompt_update() {
    COMPARE=$($PYTHON -c "from semantic_version import compare; print(compare('$1', '$2'))")
    if [ "$COMPARE" == "-1" ]; then
        # A new version of enfugue is available
        read -p "Version $2 of enfugue is available, you have version $1 installed. Download update? [Yes]: " DOWNLOAD_LATEST
        DOWNLOAD_LATEST=${DOWNLOAD_LATEST:-Yes}
        DOWNLOAD_LATEST=${DOWNLOAD_LATEST:0:1}
        echo "${DOWNLOAD_LATEST,,}"
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
        echo "Downloading $FILENAME"
        curl -L $TARBALL -o $FILENAME
    done <<< "$RELEASE_PACKAGES"
    DOWNLOADED=$(echo "$RELEASE_PACKAGES" | sed 's|.*/||')
    cat $DOWNLOADED | tar -xvz --directory $PORTABLE_DIR
    rm $DOWNLOADED
    echo $PORTABLE_DIR
}

# Iterate through command flags and set variables
while [ $# -gt 0 ]; do
    case $1 in
        --help)
            usage
            exit 0
            ;;
        --no-update)
            NO_UPDATE=true
            ;;
        --no-browser)
            NO_BROWSER=true
            export ENFUGUE_OPEN="false"
            ;;
        --update)
            UPDATE=true
            ;;
        *)
            echo "Invalid option: $1" >&2
            usage
            exit 1
            ;;
    esac
    shift
done

# Second gather some variables from the current environment.
ENFUGUE=$(which enfugue)
ENFUGUE_SERVER=$(which enfugue-server)
CONDA=$(which conda)

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
    if [[ "$PYTHON" != "" && "$NO_UPDATE" == false ]]; then
        # Get installed version from pip
        ENFUGUE_INSTALLED_PIP_VERSION=$($PYTHON -m pip freeze | grep enfugue | awk -F= '{print $3}' | sed -e 's/\.post/\-/g')
    fi
    if [ "$ENFUGUE_INSTALLED_PIP_VERSION" != "" ]; then
        # Enfugue was installed in this current environment and update allowed, get available versions from pip
        ENFUGUE_AVAILABLE_PIP_VERSION=$($PYTHON -m pip install enfugue== 2>&1 | grep 'from versions' | awk '{n=split($0,v,/, /); print v[n]}')
        ENFUGUE_AVAILABLE_PIP_VERSION=$(echo ${ENFUGUE_AVAILABLE_PIP_VERSION::-1} | sed -e 's/\.post/\-/g')
        # Compare versions and prompt if necessary
        if [ "$(compare_prompt_update $ENFUGUE_INSTALLED_PIP_VERSION $ENFUGUE_AVAILABLE_PIP_VERSION)" == "y" ]; then
            echo "Downloading update."
            pip install --upgrade enfugue
        fi
    fi
elif [ "$ENFUGUE_SERVER" != "" ]; then
    # Portable found
    ENFUGUE_PORTABLE_DIR=$(dirname $(realpath $ENFUGUE_SERVER))
    if [ "$NO_UPDATE" == false ]; then
        # Get versions
        ENFUGUE_AVAILABLE_PORTABLE_VERSION=$(curl -s https://api.github.com/repos/painebenjamin/app.enfugue.ai/releases/latest | grep "tag_name" | cut -d : -f 2,3 | tr -d ' ,\"')
        ENFUGUE_INSTALLED_PORTABLE_VERSION=$(cat $ENFUGUE_PORTABLE_DIR/enfugue/version.txt)
        # Compare versions and prompt if necessary
        if [ "$(compare_prompt_update $ENFUGUE_INSTALLED_PORTABLE_VERSION $ENFUGUE_AVAILABLE_PORTABLE_VERSION)" == "y" ]; then
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
            read -p "Conda not found. Install miniconda? [Yes]: " INSTALL_MINICONDA
            INSTALL_MINICONDA=${INSTALL_MINICONDA:-Yes}
            INSTALL_MINICONDA=${INSTALL_MINICONDA:0:1}
            if [ "${INSTALL_MINICONDA,,}" != "y" ]; then
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
        curl https://raw.githubusercontent.com/painebenjamin/app.enfugue.ai/main/environments/linux-cuda.yml -o ./enfugue-environment.yml --silent
        # Use conda to create the environment
        $CONDA env create -f ./enfugue-environment.yml
        # Remove the environment file
        rm ./enfugue-environment.yml
        # Activate the environment
        source $(dirname $CONDA)/activate enfugue
        ENFUGUE=$(which enfugue)
        PYTHON=$(which python)
    elif [ "$DOWNLOAD_TYPE" == "2" ]; then
        # Download and extract the latest portable
        PORTABLE_DIR=$(download_portable)
        ENFUGUE_SERVER="$PORTABLE_DIR/enfugue-server"
        # Prompt if we should add a symlink so this script can find it in the future
        read -p "Successfully extracted enfugue. Add symlink to /usr/local/bin? [Yes]: " ADD_SYMLINK
        ADD_SYMLINK=${ADD_SYMLINK:-Yes}
        ADD_SYMLINK=${ADD_SYMLINK:0:1}
        if [ "${ADD_SYMLINK,,}" == "y" ]; then
            sudo ln -s /usr/local/bin/enfugue-server $ENFUGUE_SERVER
        fi
    fi
fi

if [ "$ENFUGUE" != "" ]; then
    # Run enfugue via python module script
    $PYTHON -m enfugue run
elif [ "$ENFUGUE_SERVER" != "" ]; then
    # Run enfugue via portable executable
    PORTABLE_DIR=$(dirname $(realpath ${ENFUGUE_SERVER}))
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
    $PORTABLE_DIR/enfugue-server
    echo "Goodbye!"
else
    echo "Could not find or download enfugue. Exiting."
    exit 1
fi

exit 0
