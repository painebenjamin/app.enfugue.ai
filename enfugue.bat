@SETLOCAL EnableExtensions EnableDelayedExpansion
@ECHO OFF
SET ENFUGUE_CONFIG=%CD%\.enfugue.config.yml
:: This batch script serves as a one-and-done download, update, configure and run command for enfugue.
:: It's able to download enfuge via conda or in portable form.
::
:: Below is the configuration for enfugue. Most values are self-explanatory, but some others need some explanation.
:: See below the configuration for explanation on these values.
::
:: AN IMPORTANT NOTE ABOUT NETWORKING
:: 
:: By default enfugue uses a domain-based networking scheme with the domain `app.enfugue.ai`.
:: This is a registered domain name that resolves to `127.0.0.1`, the loopback domain.
:: If you're running enfugue on a local machine with internet access, this should be fine, and you shouldn't need to change any configuration.
:: If you're running enfugue on a remote machine, you will want to change the following:
:: - `server.domain` should be changed to the domain you want to access the interface through _unless_ you're using a proxy. This can be an IP address.
:: - `server.secure` should be set to false _unless_ you also configure an SSL certificate. See the commented-out (#) lines in the configuration below.
:: If you're additionally using a proxy like ngrok, you should configure `server.cms.path.root` to be the proxied URL. See the commented-out (#) lines in the configuration below.

@ECHO --- ^

sandboxed: false                                # true = disable system management in UI ^

server: ^

    host: 0.0.0.0                               # listens on any connection ^

    port: 45554                                 # ports less than 1024 require sudo ^

    domain: app.enfugue.ai                      # this is a loopback domain ^

    secure: true                                # enables SSL ^

    # If you change the domain, you must provide your own certificates or disable SSL ^

    # key: /path/to/key.pem ^

    # cert: /path/to/cert.pem ^

    # chain: /path/to/chain.pem ^

    logging: ^

        file: ~/.cache/enfugue.log              # server logs (NOT diffusion logs) ^

        level: error                            # server only logs errors ^

    # cms:                                      # only configure when using a proxy ^

    #     path: ^

    #         root: http(s)://my-proxy-host.ngrok-or-other/ ^

enfugue: ^

    noauth: true                                # authentication default ^

    queue: 4                                    # queue size default ^

    safe: true                                  # safety checker default ^

    model: v1-5-pruned.ckpt                     # default model (repo or file) ^

    inpainter: null                             # default inpainter (repo or file), null = sd15 base inpainter ^

    refiner: null                               # default refiner (repo or file), null = none ^

    refiner_start: 0.85                         # default start for refining when refiner set and no args (0.0-1.0) ^

    dtype: null                                 # set to float32 to disable half-precision, you probably dont want to ^

    engine: ^

        logging: ^

            file: ~/.cache/enfugue-engine.log   # diffusion logs (shown in UI) ^

            level: debug                        # logs everything, helpful for debugging ^

        root: ~/.cache/enfugue                  # root engine directory, images save in /images ^

        cache: ~/.cache/enfugue/cache           # diffusers cache, controlnets, VAE ^

        checkpoint: ~/.cache/enfugue/checkpoint # checkpoints only ^

        lora: ~/.cache/enfugue/lora             # lora only ^

        lycoris: ~/.cache/enfugue/lycoris       # lycoris only ^

        inversion: ~/.cache/enfugue/inversion   # textual inversion only ^

        motion: ~/.cache/enfugue/motion         # motion modules only ^

        other: ~/.cache/enfugue/other           # other AI models (upscalers, preprocessors, etc.) ^

    pipeline: ^

        switch: "offload"                   # See comment below ^

        inpainter: true                     # See comment below ^

        cache: null                         # See comment below ^

        sequential: false                   # See comment below ^

> %ENFUGUE_CONFIG%

:: -----------------------
:: enfugue.pipeline.switch
:: -----------------------
:: 'switch' determines how to swap between pipelines when required, like going from inpainting to non-inpainting or loading a refiner.
:: The default behavior, 'offload,' sends unneeded pipelines to the CPU and promotes active pipelines to the GPU when requested.
:: This usually provides the best balance of speed and memory usage, but can result in heavy overhead on some systems.
::
:: If this proves too much, or you wish to minimize memory usage, set this to 'unload,'which will always completely unload a pipeline 
:: and free memory before a different pipeline is used.
::
:: If you set this to 'null,' _all models will remain in memory_. This is by far thefastest but consumes the most memory, this is only
:: suitable for enterprise GPUs.
::
:: --------------------------
:: enfugue.pipeline.inpainter
:: --------------------------
:: 'inpainter' determines how to inpaint when no inpainter is specified.
::
:: When the value is 'true', and the user is using a stable diffusion 1.5 model for their base model, enfugue will look for another
:: checkpoint with the same name but the suffix `-inpainting`, and when one is not found, it will create one using the model merger.
:: Fine-tuned inpainting checkpoints perform significantly better at the task, however they are roughly equivalent in size to the
:: main model, effectively doubling storage required.
::
:: When the value is 'null' or 'false', Enfugue will still search for a fine-tuned inpainting model, but will not create one if it does not exist.
:: Instead, enfugue will use 4-dim inpainting, which in 1.5 is less effective.
:: SDXL does not have a fine-tuned inpainting model (yet,) so this procedure does not apply, and 4-dim inpainting is always used.
::
:: ----------------------
:: enfugue.pipeline.cache
:: ----------------------
:: 'cache' determines when to create diffusers caches. A diffusers cache will always load faster than a checkpoint, but is once again
:: approximately the same size as the checkpoint, so this will also effectively double storage size.
::
:: When the value is 'null' or 'false', diffusers caches will _only_ be made for TensorRT pipelines, as it is required. This is the default value.
::
:: When the value is 'xl', enfugue will cache XL checkpoints. These load _significantly_ faster than when loading from file, between
:: 2 and 3 times as quickly. You may wish to consider using this setting to speed up changing between XL checkpoints.
::
:: When the value is 'true', diffusers caches will be created for all pipelines. This is not recommended as it only provides marginal
:: speed advantages for 1.5 models.
::
:: ---------------------------
:: enfugue.pipeline.sequential
:: ---------------------------
:: 'sequential' enables sequential onloading and offloading of AI models.
::
:: When the value is 'true', AI models will only ever be loaded to the GPU when they are needed.
:: At all other times, they will be in normal memory, waiting for the next time they are requested, at which time they will be loaded
:: to the GPU, and afterward unloaded.
::
:: These operations take time, so this is only recommended to enable if you are experiencing issues with out-of-memory errors.
::
:: -- end of configuration --

:: Static Variables
SET LF=-
SET VERSION_FILE_PATH=enfugue\version.txt
SET CONDA_INSTALL_PATH=%UserProfile%\miniconda3
SET GET_PIP_VERSIONS="python -m pip install enfugue== 2>&1"

:: Capability Flags
SET ZIP_AVAILABLE=0
SET ZIP_R_AVAILABLE=0
SET CONDA_AVAILABLE=0

:: Available/Installed Versions
SET ENFUGUE_INSTALLED_PIP_VERSION=
SET ENFUGUE_AVAILABLE_PIP_VERSION=
SET ENFUGUE_INSTALLED_PORTABLE_VERSION=
SET ENFUGUE_AVAILABLE_PORTABLE_VERSION=

:: Options
SET INSTALL_TYPE=
SET INSTALL_UPDATE=
SET INSTALL_MMPOSE=

:: Iterate over command-line arguments
FOR %%I IN (%*) DO (
    IF "%%I"=="--help" GOTO :Usage
    IF "%%I"=="--conda" SET INSTALL_TYPE=conda
    IF "%%I"=="--portable" SET INSTALL_TYPE=portable
    IF "%%I"=="--update" SET INSTALL_UPDATE=1
    IF "%%I"=="--no-update" SET INSTALL_UPDATE=0
    IF "%%I"=="--mmpose" SET INSTALL_MMPOSE=1
    IF "%%I"=="--no-mmpose" SET INSTALL_MMPOSE=0
)

:: Add PATHS
SET PATH=%PATH%;%CD%;%CD%\enfugue-server;%CONDA_INSTALL_PATH%\condabin

:: Gather capabilities
where 7z >NUL 2>NUL && (
    SET ZIP_AVAILABLE=1
) || (
    where 7zr >NUL 2>NUL && (
        SET ZIP_R_AVAILABLE=1
    )
)
where conda.bat >NUL 2>NUL && (
    SET CONDA_AVAILABLE=1
)

:: Look for enfugue environment if conda exists and activate it
IF "!INSTALL_TYPE!" NEQ "portable" IF "!CONDA_AVAILABLE!" == "1" (
    FOR /f "delims=" %%I IN ('conda.bat env list') DO (
        ECHO "%%I" | findstr /C:"enfugue">NUL && (
            ECHO Found ENFUGUE conda environment, activating it.
            CALL conda.bat activate enfugue
        )
    )
)

:: Check for an installed version of enfugue installed via conda/python
IF "!INSTALL_TYPE!" NEQ "portable" where enfugue.exe >NUL 2>NUL && (
    REM Get installed version
    FOR /f "delims=" %%I IN ('python -m pip freeze') DO (
        ECHO "%%I" | findstr /C:"enfugue">NUL && (
            SET ENFUGUE_INSTALLED_PIP_VERSION=%%I
        )
    )
    IF "!INSTALL_UPDATE!" NEQ "0" IF "!INSTALL_TYPE!" NEQ "portable" (
        REM Get available version
        FOR /f "delims=" %%I IN ('%GET_PIP_VERSIONS%') DO (
            ECHO "%%I" | findstr /C:"versions">NUL && (
                FOR %%J IN ('ECHO %%I') DO (
                    FOR /f "delims=) tokens=1" %%K IN ('ECHO %%J') DO (
                        SET ENFUGUE_AVAILABLE_PIP_VERSION=%%K
                    )
                )
            )
        )
    )
)
IF NOT "!ENFUGUE_INSTALLED_PIP_VERSION!"=="" (
    REM Trim ==
    SET ENFUGUE_INSTALLED_PIP_VERSION=!ENFUGUE_INSTALLED_PIP_VERSION:~9,30!
    REM Replace .post with .
    CALL SET ENFUGUE_INSTALLED_PIP_VERSION=!!ENFUGUE_INSTALLED_PIP_VERSION:post=!!
    ECHO Found conda ENFUGUE v.!ENFUGUE_INSTALLED_PIP_VERSION! installation.
)
IF NOT "!ENFUGUE_AVAILABLE_PIP_VERSION!"=="" (
    REM Replace .post with .
    CALL SET ENFUGUE_AVAILABLE_PIP_VERSION=!!ENFUGUE_AVAILABLE_PIP_VERSION:post=!!
)

:: Check for an installed version of enfugue via portable
if "!INSTALL_TYPE!" NEQ "conda" where enfugue-server.exe >NUL 2>NUL && (
    REM Get installed version
    FOR /f "tokens=1" %%I IN ('where enfugue-server.exe') DO (
        FOR /f "tokens=1" %%J IN (%%~dpI%VERSION_FILE_PATH%) DO (
            ECHO Found portable ENFUGUE v.%%J installation.
            SET ENFUGUE_INSTALLED_PORTABLE_VERSION=%%J
        )
    )
    IF "!INSTALL_UPDATE!" NEQ "0" IF "!INSTALL_TYPE!" NEQ "conda" (
        REM Get available version
        FOR /f "delims=" %%I IN ('curl -s https://api.github.com/repos/painebenjamin/app.enfugue.ai/releases/latest') DO (
            ECHO "%%I" | findstr /C:"tag_name">NUL && (
                SET ENFUGUE_AVAILABLE_PORTABLE_VERSION=%%I
            )
        )
    )
)
IF "!ENFUGUE_AVAILABLE_PORTABLE_VERSION!" NEQ "" (
    REM Trim by quotes
    FOR /f delims^=^"^ ^tokens^=4 %%I IN ('ECHO !ENFUGUE_AVAILABLE_PORTABLE_VERSION!') DO (
        SET ENFUGUE_AVAILABLE_PORTABLE_VERSION=%%I
    )
)

:: If an installation was found and updates aren't disabled, compare versions
IF "!ENFUGUE_AVAILABLE_PIP_VERSION!" NEQ "" IF "!ENFUGUE_INSTALLED_PIP_VERSION!" NEQ "" (
    :: Compare installed pip version
    CALL :CompareVersion ENFUGUE_INSTALLED_PIP_VERSION ENFUGUE_AVAILABLE_PIP_VERSION
    IF ERRORLEVEL 1 (
        IF "!INSTALL_UPDATE!" == "1" (
            python -m pip install enfugue -U
        ) ELSE (
            CALL :PromptYesNo "There is a new version of ENFUGUE available (v.!ENFUGUE_AVAILABLE_PIP_VERSION!). Download and install update? (Yes): "
            IF ERRORLEVEL 1 (
                python -m pip install enfugue -U
            )
        )
    )
)
IF "!ENFUGUE_AVAILABLE_PORTABLE_VERSION!" NEQ "" IF "!ENFUGUE_INSTALLED_PORTABLE_VERSION!" NEQ "" (
    :: Compare installed portable versions
    CALL :CompareVersion ENFUGUE_INSTALLED_PORTABLE_VERSION ENFUGUE_AVAILABLE_PORTABLE_VERSION
    IF ERRORLEVEL 1 (
        IF "!INSTALL_UPDATE!" == "1" (
            ECHO There is a new portable version of ENFUGUE available ^(v.!ENFUGUE_AVAILABLE_PORTABLE_VERSION!^). Downloading now.
            CALL :DownloadPortable
        ) ELSE (
            CALL :PromptYesNo "There is a new portable version of ENFUGUE available (v.!ENFUGUE_AVAILABLE_PORTABLE_VERSION!). Download and install update? (Yes): "
            IF ERRORLEVEL 1 (
                CALL :DownloadPortable
            )
        )
    )
)

:: If no installation was found, install it
IF "!ENFUGUE_INSTALLED_PIP_VERSION!"=="" IF "!ENFUGUE_INSTALLED_PORTABLE_VERSION!"=="" (
    ECHO ENFUGUE is not currently installed.
    :GetInstallType
    IF "!INSTALL_TYPE!" == "" (
        SET /P "INSTALL_TYPE=How would you like to install? Type either 'conda' or 'portable' and press enter (or simply press enter for conda): "
        IF "!INSTALL_TYPE!" == "" SET INSTALL_TYPE=conda
    )
    IF "!INSTALL_TYPE!" NEQ "conda" IF "!INSTALL_TYPE!" NEQ "portable" (
        ECHO Invalid response '!INSTALL_TYPE!'
        SET INSTALL_TYPE=
        GOTO :GetInstallType
    )
    IF "!INSTALL_TYPE!" == "conda" (
        CALL :DownloadConda
    ) ELSE (
        CALL :DownloadPortable
    )
)

:: The requested version should be available
IF "!INSTALL_TYPE!" NEQ "portable" where enfugue.exe 2>NUL >NUL && (
    :: Perform MMPose checks if requested
    SET MMPOSE_INSTALLED=0
    IF "!INSTALL_MMPOSE!" NEQ "0" (
        FOR /f "delims=" %%I IN ('python -m pip freeze') DO (
            ECHO "%%I" | findstr /C:"mmpose">NUL && (
                SET MMPOSE_INSTALLED=1
            )
        )
        IF "!MMPOSE_INSTALLED!" == "0" (
            IF "!INSTALL_MMPOSE!" == "1" (
                CALL :InstallMMPose
            ) ELSE (
                CALL :PromptYesNo "MMPose is not installed. Install it? (Yes): "
                IF ERRORLEVEL 1 (
                    CALL :InstallMMPose
                )
            )
        )
    )
    :: Execute (synchronous)
    python -m enfugue run -c %ENFUGUE_CONFIG% -m
)
IF "!INSTALL_TYPE!" NEQ "conda" where enfugue-server.exe 2>NUL >NUL && (
    :: Execute (asynchronous)
    START enfugue-server
    ECHO ENFUGUE launched. A browser window will open shortly - this window can now be closed. To terminate the ENFUGUE server, right-click the icon in the bottom-right hand corner.
    ECHO This window will automatically close in 15 seconds.
    %WINDIR%\System32\timeout.exe /T 15
)

EXIT /b 0

:: -- execution end, functions start --

:LCase
:UCase
REM Source: https://www.robvanderwoude.com/battech_convertcase.php
REM Converts to upper/lower case variable contents
REM Syntax: CALL :UCase _VAR1 _VAR2
REM Syntax: CALL :LCase _VAR1 _VAR2
REM _VAR1 = Variable NAME whose VALUE is to be converted to upper/lower case
REM _VAR2 = NAME of variable to hold the converted value
REM Note: Use variable NAMES in the CALL, not values (pass "by reference")

SET _UCase=A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
SET _LCase=a b c d e f g h i j k l m n o p q r s t u v w x y z
SET _Lib_UCase_Tmp=!%1!
IF /I "%0"==":UCase" SET _Abet=%_UCase%
IF /I "%0"==":LCase" SET _Abet=%_LCase%
FOR %%Z IN (%_Abet%) DO SET _Lib_UCase_Tmp=!_Lib_UCase_Tmp:%%Z=%%Z!
SET %2=%_Lib_UCase_Tmp%
EXIT /b

:: -- version compare function --

:CompareVersion
REM Adapted from https://stackoverflow.com/questions/15807762/compare-version-numbers-in-batch-file
REM Compares two version numbers and returns the result in the ERRORLEVEL

SET "v1=!%~1!"
SET "v2=!%~2!"

CALL :DivideLetters v1
CALL :DivideLetters v2

:CompareVersionLoop
CALL :ParseVersionNode "%v1%" n1 v1
CALL :ParseVersionNode "%v2%" n2 v2
IF %n1% gtr %n2% EXIT /b -1
IF %n1% lss %n2% EXIT /b 1
IF NOT DEFINED v1 IF NOT DEFINED v2 EXIT /b 0
IF NOT DEFINED v1 EXIT /b 1
IF NOT DEFINED v2 EXIT /b -1
GOTO :CompareVersionLoop

:ParseVersionNode  version  nodeVar  remainderVar
FOR /f "tokens=1* delims=.,-" %%A IN ("%~1") DO (
  SET "%~2=%%A"
  SET "%~3=%%B"
)
EXIT /b

:DivideLetters  versionVar
FOR %%C IN (a b c d e f g h i j k l m n o p q r s t u v w x y z) DO SET "%~1=!%~1:%%C=.%%C!"
EXIT /b

:: -- prompt user function --
:PromptYesNo
REM Echoes a passed used prompt and returns 0 or 1 in the error level.
SET /P USER_INPUT=%1
SET USER_INPUT=!USER_INPUT:~0,1!
CALL :UCase USER_INPUT USER_INPUT
IF "!USER_INPUT!"=="~0,1" EXIT /b 1
IF "!USER_INPUT!"=="~" EXIT /b 1
IF "!USER_INPUT!"=="1" EXIT /b 1
IF "!USER_INPUT!"=="Y" EXIT /b 1
IF "!USER_INPUT!"=="T" EXIT /b 1
EXIT /b 0

:: -- download portable installation function --
:DownloadPortable
REM This function downloads and extracts the latest portable release.
SET ARCHIVE_NAME=
SET ARCHIVES=
FOR /f "delims=" %%I IN ('curl -s https://api.github.com/repos/painebenjamin/app.enfugue.ai/releases/latest') DO (
    ECHO "%%I" | findstr /C:"browser_download_url">NUL && (
        ECHO "%%I" | findstr /C:"win">NUL && (
            FOR /f delims^=^"^ ^tokens^=4 %%J IN ('ECHO %%I') DO (
                IF "!ARCHIVE_NAME!"=="" (
                    SET ARCHIVE_NAME=%%~nJ
                    SET ARCHIVES=%%~nJ%%~xJ
                ) ELSE (
                    SET ARCHIVES=!ARCHIVES! %%~nJ%%~xJ
                )
                curl -L %%J -o %%~nJ%%~xJ
            )
        )
    )
)
IF NOT "!ARCHIVE_NAME!"=="" (
    IF "!ZIP_AVAILABLE!"=="1" (
        7z x !ARCHIVE_NAME!.001 -y
    ) ELSE (
        IF "!ZIP_R_AVAILABLE!"=="0" (
            ECHO Downloading 7zr.exe [7-Zip Standalone Executable]
            curl -L https://www.7-zip.org/a/7zr.exe -o 7zr.exe
        )
        7zr x !ARCHIVE_NAME!.001 -y
        REM Specifically call built-in tar just in case we're in cygwin
        %WINDIR%\System32\tar.exe -xvf !ARCHIVE_NAME!
        DEL !ARCHIVE_NAME!
        FOR %%I IN (!ARCHIVES!) DO (
            DEL %%I
        )
    )
)
EXIT /b

:: -- download conda installation function --
:DownloadConda
REM Download conda if it's not available
IF "%CONDA_AVAILABLE%"=="0" (
    ECHO Downloading miniconda [Package Manager]
    curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe
    START /wait "" miniconda.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%CONDA_INSTALL_PATH%
    DEL miniconda.exe
)
REM Download environment file
curl https://raw.githubusercontent.com/painebenjamin/app.enfugue.ai/main/environments/windows-cuda.yml -o environment.yml 2>NUL
ECHO Creating Environment
FOR /f %%I IN ('where conda.bat') DO (
    %%I env create -f environment.yml
    %%I activate enfugue
)
DEL environment.yml
EXIT /b

:: -- install mmpose function --
:InstallMMPose
REM This function installs MMPose and dependencies
python -m mim install mmengine
python -m mim install "mmcv>=2.0.1"
python -m mim install "mmdet>=3.1.0"
python -m mim install "mmpose>=1.1.0"
EXIT /b 0

:: -- usage help function --
:Usage
REM This function prints out the help message.
@ECHO USAGE: %~n0%~x0 (options) ^

Options: ^

  --help                    Display this help message. ^

  --conda  / --portable     Automatically set installation type (do not prompt.) ^

  --update / --no-update    Automatically apply or skip updates (do not prompt.) ^

  --mmpose / --no-mmpose    Automatically install mmpose if needed (do not prompt.)

EXIT /b 0
