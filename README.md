# Monocular Depth Estimation in Adverse Conditions


This project aims to improve image enhancement under adverse conditions through monocular depth estimation using advanced image processing algorithms and AI techniques. The system is designed to generate accurate depth maps in challenging environments, such as low light, fog, and rain.
## Features
- Modular Design: Independent components specialized for different adverse conditions.
- Depth Map Generation: Accurate depth maps for improved image enhancement.
- Standardized Quantification: Objective evaluation of system performance.

To get more information about the implemantion, refer to the [Full Report](Docs/Reporte-Escrito.pdf).

The main idea behind this project follows the following diagram:
![Flow Diagram](https://github.com/FabianRamirez03/TFG_Monocular_Depth_Stimation/assets/38987714/62fcb5a0-5e45-4db2-9c29-b37f2d50de6a)



## Installation

1. Clone the repository:
```sh
git clone https://github.com/FabianRamirez03/TFG_Monocular_Depth_Stimation.git
cd TFG_Monocular_Depth_Stimation
```

2. Install the required dependencies:
```sh
pip install -r requirements.txt
```

3. Download the pretrained models:

Create a directory "models", download the pretrained models from [here](https://drive.google.com/drive/folders/1vkp2olfbOAa49DCGpQFov3BT3dVOVK2K?usp=drive_link) and save them in this directory.

4. Modify the imports in AdaBins submodule:

This project is using the [AdaBins](https://github.com/shariqfarooq123/AdaBins) project as Depth Map Generator. AdaBins is included as a Submodule. 
To be able to run the code correctly, is necesary to a change in the file [Infer.py](https://github.com/shariqfarooq123/AdaBins/blob/0952d91e9e762be310bb4cd055cbfe2448c0ce20/infer.py) downloaded locally. 

In the imports change:

```python
import model_io
import utils
from models import UnetAdaptiveBin
```
To:

```python
import AdaBins.model_io as model_io
import AdaBins.utils as utils
from AdaBins.models import UnetAdaptiveBins
```

## Usage

1. Run the all in one GUI.

```sh
py .\main.py
```

2. GUI usage:

The following GUI will open.

![image](https://github.com/FabianRamirez03/TFG_Monocular_Depth_Stimation/assets/38987714/1d0d1036-5779-439b-b3ef-90c77d901f0e)

  1. Upload an image using the button.
  2. Verify if the tagger did the job. If there is an error, clicking in the led of each condition will change the state.
  3. Finally, process the image. The result should look like this:

![image](https://github.com/FabianRamirez03/TFG_Monocular_Depth_Stimation/assets/38987714/ea10cbf7-bd39-4707-9411-c373e4020764)

