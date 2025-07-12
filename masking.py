import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import numpy as np
image_path = rf'D:\DLFS\Unet\sample\train\image\volume-103.nii'
mask_path = rf'D:\DLFS\Unet\sample\train\target\segmentation-103.nii'

# Đọc ảnh gốc và mask
image = sitk.ReadImage(image_path)
mask = sitk.ReadImage(mask_path)

binary_mask = sitk.Cast(sitk.Or(mask == 1, mask == 2), sitk.sitkUInt8)

# Áp dụng masking



clamp_filter = sitk.ClampImageFilter()
clamp_filter.SetLowerBound(-45)  
clamp_filter.SetUpperBound(167)

clamped_image = clamp_filter.Execute(image)
masked_image = sitk.Mask(clamped_image, binary_mask, outsideValue=0)
origin_array = sitk.GetArrayFromImage(image)
image_array = sitk.GetArrayFromImage(clamped_image)
mask_array = sitk.GetArrayFromImage(mask)
masked_image_array = sitk.GetArrayFromImage(masked_image)
masked_image_array[masked_image_array==0] = 0.0
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(image_array[0], cmap='gray')
plt.title("Ảnh gốc")
plt.axis("on")

plt.subplot(1, 3, 2)
plt.imshow(mask_array[0], cmap='gray')
plt.title("mask gốc")
plt.axis("on")

plt.subplot(1, 3, 3)
plt.imshow(masked_image_array[0], cmap='gray')
plt.title("mask pred")

plt.tight_layout()
plt.show()
print("Unique values ngoài mask:", np.unique(masked_image_array[masked_image_array == 0]))