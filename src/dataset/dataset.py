import numpy as np
from torch.utils.data import Dataset
from dataset.transform import train_transform, val_transform, test_transform
import os 
import cv2

class NeoPolypDataset(Dataset):
    def __init__(self, img_dir, gt_img_dir, status: str = 'train') -> None:
        super().__init__()
        
        self.status = status
        self.img_dir = img_dir
        self.gt_img_dir = gt_img_dir
    
    def __len__(self) -> int:
        return len(self.img_dir)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.img_dir, os.listdir(self.img_dir)[idx])
        gt_img_path = os.path.join(self.gt_img_dir, os.listdir(self.gt_img_dir)[idx])
        
        img = cv2.imread(img_path)
        gt = self._mask_to_class(gt_img_path)
        
        if self.status == 'train':
            img, gt = train_transform(img, gt)
        elif self.status == 'val':
            img, gt = val_transform(img, gt)
        elif self.status == 'test':
            img = test_transform(img)
        else:
            raise ValueError('Invalid status')
        
        return img, gt
        
    
    def _mask_to_class(self, mask_path: str):
        """
            This method used to convert mask to class pixels:
            1 (neoplastic) and 2 (non-neoplastic) and 0 (background)
        Args:
            mask_path (str): The path of the mask image
        """
        
        # Read the mask image using cv2 
        mask = cv2.imread(mask_path)
        
        # Convert the mask to HSV color space
        mask_hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
        
        # Red pixels are neoplastic
        lower_red_1 = np.array([0, 50, 50])
        upper_red_1 = np.array([10, 255, 255])
        
        lower_red_2 = np.array([170, 50, 50])
        upper_red_2 = np.array([179, 255, 255])
        
        lower_mask_red = cv2.inRange(mask_hsv, lower_red_1, upper_red_1)
        upper_mask_red = cv2.inRange(mask_hsv, lower_red_2, upper_red_2)
        
        red_mask = lower_mask_red + upper_mask_red
        red_mask[red_mask > 0] = 1 # Set the neoplastic pixels to 1
        
        # Green pixels are non-neoplastic
        lower_green = np.array([36, 25, 25])
        upper_green = np.array([86, 255, 255])
        
        green_mask = cv2.inRange(mask_hsv, lower_green, upper_green)
        green_mask[green_mask > 0] = 2 # Set the non-neoplastic pixels to 2
        
        
        # Combine the red and green masks
        full_mask = cv2.bitwise_or(red_mask, green_mask) # If the pixel is red or green, it will be 1 or 2
        full_mask = full_mask.astype(np.uint8) # Convert the mask to uint8 
        
        return full_mask
        
if __name__ == '__main__':
    img_dir = r'C:\Users\hokag\Documents\GitHub\unet-semantic-segmentation\train\train'
    gt_img_dir = r'C:\Users\hokag\Documents\GitHub\unet-semantic-segmentation\train_gt\train_gt'
    
    dataset = NeoPolypDataset(img_dir=img_dir, gt_img_dir=gt_img_dir, status='train')
    print(dataset[2])
        
        
