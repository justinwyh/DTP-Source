# DTP-Source
Final Year Project: Drone Tracking Program (DTP)
# Requirements:
1. Windows 10
2. Nvidia GPU that supports CUDA 10.1
# Installation
1. Download DTP_Data folder and the DTP program on the Github Release page.
2. Run requirements.bat under DTP_Data, please visit [link](https://github.com/facebookresearch/pytorch3d/issues/10) if you encounter any installation issues.
3. Install the DTP program by right-clicking the DTP_Package_1.0.0.0_Test\Add-AppDevPackage.ps1 in the release app package.
4. Replace "from pysot." to "from PySOT.pysot" and replace "from toolkit." to "from PySOT.toolkit." in all Python files under DTP_Data\Source\PySOT
5. Start the program and update the DTP data path in the program.
# Build the DTP from scratch
Please reference [here](https://developer.dji.com/windows-sdk/documentation/application-development-workflow/workflow-integrate.html) and the download the SDK [here](https://developer.dji.com/windows-sdk/downloads).
# References
https://github.com/STVIR/pysot/

https://github.com/jorge-pessoa/pytorch-msssim
