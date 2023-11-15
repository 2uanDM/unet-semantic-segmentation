import os 
import sys
sys.path.append(os.getcwd())

import numpy as np 
import pandas as pd 
import torch 
import cv2
import segmentation_models_pytorch as smp

from albumentations import Compose, Normalize, Resize
from albumentations.pytorch.transforms import ToTensorV2
from mask_io import mask2string
import albumentations as A 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = smp.UnetPlusPlus(
    encoder_name="resnet101",
    encoder_weights="imagenet",
    in_channels=3,
    classes=3
)
model.to(device)

# Load the pretrained model 
check_point = torch.load('model.pth', map_location=device)
model.load_state_dict(check_point['model'])

color_mapping = {
    0: (0, 0, 0), # Background
    1: (255, 0, 0), # Neoplastic polyp
    2: (0, 255, 0) # Non-neoplastic polyp
}

def mask_to_rgb(mask, color_mapping):
    output = np.zeros((mask.shape[0], mask.shape[1], 3))
    for key in color_mapping.keys():
        output[mask == key] = color_mapping[key]
    
    return np.uint8(output)

# Transform for the test set
transformer = A.Compose([
    A.Normalize(),
    ToTensorV2(),
])


# Evaluation the test set
model.eval()

test_dir = '/kaggle/input/bkai-igh-neopolyp/test/test'

for idx, img_name in enumerate(os.listdir(test_dir)):
    print(f'Predicted {idx+1}/400 ...')
    test_img_path = os.path.join(test_dir, img_name)
    
    img = cv2.imread(test_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    width, height = img.shape[1], img.shape[0]
    
    # Resize the image to 256x256
    img = cv2.resize(img, (256, 256))
    
    # Transform the image
    transformed_img = transformer(image=img)['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        out_mask = model.forward(transformed_img).squeeze(0).cpu().numpy().transpose(1, 2, 0) # (256, 256, 3)
        
    # Resize the mask to the original size
    out_mask = cv2.resize(out_mask, (width, height))
    out_mask = np.argmax(out_mask, axis=2)
    
    # Convert the mask to RGB
    rgb_mask = mask_to_rgb(out_mask, color_mapping)
    rgb_mask = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR)
    
    # Save the mask
    save_dir = os.makedirs('/kaggle/working/predict_mask', exist_ok=True)
    save_path = os.path.join(save_dir, img_name)
    cv2.imwrite(save_path, rgb_mask)

# Convert the mask to string for submission
result = mask2string('/kaggle/working/predict_mask')

df = pd.DataFrame(result, colum = ['Id', 'Expected'])
df['Id'] = result['idx']
df['Expected'] = result['result_str']
df.to_csv('/kaggle/working/submission.csv', index=False)

print('--- Finish ---')

    

    