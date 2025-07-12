import numpy as np
import pandas as pd
import SimpleITK as sitk
import os

root = rf'D:\DLFS\Unet\LiTS17_LiS\train'

hu_values = []
image_list = os.listdir(os.path.join(root, 'image'))
target_list = os.listdir(os.path.join(root, 'target'))
for image_name, target_name in zip(image_list, target_list):
    
    image_path = os.path.join(root, 'image', image_name)
    target_path = os.path.join(root, 'target', target_name)

    image = sitk.ReadImage(image_path)
    target = sitk.ReadImage(target_path)
    
    image_array = sitk.GetArrayFromImage(image)
    target_array = sitk.GetArrayFromImage(target)
    target_array[target_array==2]=1
       
    tumor_mask = (target_array == 1)    
    tumor_hu = image_array[tumor_mask]
    hu_values.extend(tumor_hu.flatten())
    
if len(hu_values) == 0:
    raise ValueError("No tumor HU values found in the training set.")

hu_values = np.array(hu_values)

lower_bound = np.percentile(hu_values, 0.5)
upper_bound = np.percentile(hu_values, 99.5)

# In kết quả
print(f"Determined HU range: [{lower_bound}, {upper_bound}]")