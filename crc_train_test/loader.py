import os
from skimage import io
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensor

def Normalization():
   return A.Compose(
       [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ])
class binary_class(Dataset):
        def __init__(self,path,data, transform=None):
            self.path = path
            self.folders = data
            self.transforms = transform
        
        def __len__(self):
            return len(self.folders)
              
        
        def __getitem__(self,idx):
            np.set_printoptions(threshold=np.inf)
            image_path = os.path.join(self.path,'images/',self.folders[idx])
            mask_path = os.path.join(self.path,'masks/',self.folders[idx])
            image_id = self.folders[idx]
            img = io.imread(image_path)
            mask = io.imread(mask_path, as_gray=True)
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=-1) 
            elif len(img.shape) == 3 and img.shape[2] > 3: 
                img = img[:, :, :3] 
            img = img.astype('float32')
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            
            return (img,mask,image_id)
        
class binary_class2(Dataset):
        def __init__(self,path,data, transform=None):
            self.path = path
            self.folders = data
            self.transforms = transform
        
        def __len__(self):
            return len(self.folders)
              
        
        def __getitem__(self,idx):
            image_path = os.path.join(self.path,self.folders[idx],'images/',self.folders[idx])
            mask_path = os.path.join(self.path,self.folders[idx],'masks/',self.folders[idx])
            image_id = self.folders[idx]
            img = io.imread(f'{image_path}.png')[:,:,:3].astype('float32')
            mask = io.imread(f'{mask_path}.png', as_gray=True)

            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
   
            return (img,mask,image_id)
        
