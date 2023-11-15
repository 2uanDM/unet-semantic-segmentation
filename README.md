# BKAI-IGH NeoPolyp

Student Name: Đường Minh Quân
Student ID: 20210710

# Inferencing guideline

Step 1:
Add data "bkai-igh-neopolyp" to /kaggle/input/

Step 2:
First, we need to download the "model.pth" from Google Drive and put it in "/kaggle/working/"

```python
import requests
import os

drive_url = f'https://drive.google.com/uc?id=11X5lrZV2QAklZ6n9ReQUmb2yyE_eQCuT&export=download&confirm=t&uuid=501d3c0c-6f65-438c-9857-3a70f62ef5b4'

save_dir = '/kaggle/working/'

response = requests.get(drive_url)

with open(os.path.join(save_dir, 'model.pth'), 'wb') as f:
    f.write(response.content)

print('Save "model.pth" successfully!')
```

Inferring

```python
!git clone https://github.com/2uanDM/unet-semantic-segmentation.git
!cp /kaggle/working/model.pth /kaggle/working/unet-semantic-segmentation/
!pip install -r /kaggle/working/unet-semantic-segmentation/requirements.txt
!python /kaggle/working/unet-semantic-segmentation/infer.py
```
