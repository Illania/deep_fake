# <a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&pause=500&color=1D46F7&width=435&lines=DeepFakeEvolution" alt="Typing SVG" /></a><br>  

Deep Fake Evolution is an API that can perform arbitrary face swapping on videos using one single trained model.

That allows you to perform face-swapping on videos using a single and efficient trained model. 
### Examples of usage:

![_](./guides/images/sample2.png)
![_](./guides/images/sample1.png)

### Team:

- Anna Gulich 
- Sergey Shvets 
- Alexey Naymushin 
- Liudmila Mihaylichenko 

### Key features:

1. Arbitrary single-face and multi-face swapping using modern, fast and efficient <a href="https://github.com/neuralchen/SimSwap" target="_blank">SimSwap Framework</a>
2. High-quality output video for professional use
3. Easy-to-use interface for users

### [Installation on CPU](/guides/INSTALLATION.md)

### [API usage](/guides/USAGE.md)

### Project structure description:

<pre>
├── FaceDetectAntelopeModel.py              - Face detection model class
├── LICENSE                                 - License file
├── README.MD                               - Readme
├── SwapManager.py                          - Class that manages face swapping using SimSwap framework
├── init.py                                 - Module initialization class
├── demo                                    - Demo files folder
│ ├── multi                                 - Demo folder for multi swap testing
│ └── single                                - Demo folder for single swap testing
├── downloads                               - Folder used for keeping result videos files available for download
├── guides                                  - Guides and instructions for users
│ ├── INSTALLATION.md                       - Installation guide
│ └── USAGE.md                              - Usage guide
├── init.sh                                 - SimSwap installation shell script
├── main.py                                 - API file and program entry point
├── main_test.py                            - API tests
├── patch_cpu.ps1                           - SimSwap patch for CPU usage
├── requirements                            - Project requirements
│ ├── common_mac.txt                        - MacOS common requirements
│ ├── git_actions_ubuntu.txt                - Linux Ubuntu requirements
│ ├── mac_arm64.txt                         - MacOS ARM64 specific requirements
│ └── mac_x86_64.txt                        - MacOS x86-64 specific requirements
├── uploads                                 - API users' uploads folder
│ ├── images                                - Image files uploads
│ ├── multispecific.                        - Multispecific archive files uploads
│ └── videos                                - Video files uploads
└── utilities                               - Utilities module
├── init.py                                 - Module initialization files
├── api_constants.py                        - Constants used in API
├── api_utils.py                            - Helper functions used by API
├── enums.py                                - Enumeration classes
└── hasher.py                               - Class used for calculating video file hashes
</pre>

### Dependencies:
  * Python 3.7+
  * opencv-python~=4.7.0.72
  * torch~=1.12.0
  * torchvision~=0.13.0
  * torchaudio
  * onnxruntime 
  * moviepy
  * insightface==0.2.1 
  * imageio==2.5.0
  * numpy~=1.21.6
  * uvicorn==0.21.1
  * starlette==0.26.1
  * Werkzeug==2.2.3
  * fastapi==0.95.1
  * python-multipart
  * pytest
  * httpx
  

