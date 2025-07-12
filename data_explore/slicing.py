import numpy as np
import pandas as pd
import SimpleITK as sitk
import os
slice = 0
root = r'D:\DLFS\Unet\LiTS17'
for ct in range(131):
    image_path = os.path.join(root, f'volume-{ct}.nii')
    mask_path = os.path.join(root, f'segmentation-{ct}.nii')

    image = sitk.ReadImage(image_path)
    image_array  = sitk.GetArrayFromImage(image)
    
    mask = sitk.ReadImage(mask_path)
    mask_array  = sitk.GetArrayFromImage(mask)
    
    C = mask_array.shape[0]
    tumor_slice = []
    for c in range(C):
        if np.any(mask_array[c] == 1) or np.any(mask_array[c] == 2):
            tumor_slice.append(c)
    try: 
        image_array = image_array[tumor_slice, :, :]
        mask_array = mask_array[tumor_slice, :, :]

        image_array = image_array.reshape(-1, 1, 512, 512)
        mask_array = mask_array.reshape(-1, 1, 512, 512)
        B = image_array.shape[0]
        for b in range(B):
            image_slice = image_array[b]
            mask_slice = mask_array[b]

            image_slice = sitk.GetImageFromArray(image_slice)
            mask_slice = sitk.GetImageFromArray(mask_slice)

            sitk.WriteImage(image_slice, rf'D:\DLFS\Unet\LiTS17_LiS\volume-{slice}.nii')
            sitk.WriteImage(mask_slice, rf'D:\DLFS\Unet\LiTS17_LiS\segmentation-{slice}.nii')
            slice += 1
    except: 
        print(f'no liver in {ct}')
        continue
    print(f'done ct{ct}')