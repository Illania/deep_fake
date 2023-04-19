
# CPU installation manual 



## Mac Intel x86-64

    1. Install homebrew if it is not installed on your Mac yet. Open terminal and execute the following command:

```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
    2. Install pyenv (a plugin for creating virtual environments) and install python 3.7:
```bash
  brew update
  brew install pyenv
  pyenv install 3.7
``` 

    3. Go to your repo directory and execute the following command, which tells which python version to use:
```bash
  cd MY_REPO_PATH
  pyenv local 3.7
```  
    
    4. Install wget:
```bash
  brew install wget
```  

    5. In your repo folder make init.sh script executable and run it. The shell script will clone SimSwap repository 
    into your repository folder and will install the required models. It will require some time, please wait until 
    script execution will be completed.
```bash
  chmod +x init.sh
  ./init.sh
``` 

    6. Install Powershell and apply a CPU patch on SimSwap project. Wait till execution finishes.
```bash
  brew install --cask powershell
  cd SimSwap
  pwsh ../patch_cpu.ps1
```  
    7. Verify that the patch has been applied correctly. Go to SimSwap/models/fs_model.py, find line 51 and 
    check that it equals the following "device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')".

    8. Install onnxruntime:
```bash
  brew install onnxruntime
```  
    9. Install requirements:
```bash
  cd MY_REPO_PATH
  pipenv install -r requirements/mac_x86_64.txt
```  
    10. Add SimSwap directory as a source root directory (change PATH_TO_SIMSWAP_FOLDER with your actual path 
    to SimSwap folder):
```bash
  export PYTHONPATH="${PYTHONPATH}:/PATH_TO_SIMSWAP_FOLDER
  pipenv install -r requirements.txt
```  
    11. Activate your virtual environment:
```bash
  pipenv shell
```
    12. Run main.py:
```bash
  python3 main.py
```
   Follow steps described in [API usage](USAGE.md).

## Mac ARM64

    1. Execute steps 1, 4-8 from Mac Intel x86-64 manual.

    2. Install rosetta:
```bash
  sudo softwareupdate --install-rosetta
```

    3. Install ffmpeg:
```bash
  brew install ffmpeg
```
    4. Download Anaconda from https://repo.anaconda.com/archive/Anaconda3-2023.03-MacOSX-x86_64.pkg and install 
    it by double clicking on the pkg file. Follow the installation steps.

    5. Restart the terminal window.

    6. Go to the repository folder, create and activate Conda environment by executing the following commands:
```bash
  conda create -n deepfake python=3.9
  conda activate deepfake
```
    7. Install pytorch, protobuf and numpy:
```bash
    conda install pytorch torchvision -c pytorch-nightly
    conda install -c conda-forge protobuf numpy
```
    8. Add SimSwap directory as a source root directory to your conda path (change PATH_TO_SIMSWAP_FOLDER with 
    your actual path to SimSwap folder, for example "/Users/illania/deep_fake/SimSwap"):
```bash
    conda develop PATH_TO_SIMSWAP_FOLDER
```  

    9. Install requirements:
```bash
  pip install -r requirements/mac_arm64.txt
```

    10. Run main.py:
```bash
  python main.py
```
    11. Configure Anaconda in PyCharm. 
        11.1. Open your repo folder in PyCharm.
        11.2. Close the window, that asks you about which interpreter you want to use.
        11.3. Click on "No interpreter" in the bottom right corner of PyCharm window.
        11.4. Select "Add new interpreter" -> "Add local interpreter".
        11.5. In the "Add Python interpreter window" select "Conda environment", then "Use existing environment", 
        then select "deepfake" from the dropdown list. It is the name of Conda environment, that you have 
        created on step 6.
        11.6. Now right-click on main.py and select "Run 'main'".
        
   Follow steps described in [API usage](USAGE.md).

