import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
# Đọc file .nii
image = sitk.ReadImage(r"D:\DLFS\Unet\preprocess_data\ct0\volume5.nii")
seg = sitk.ReadImage(r"D:\DLFS\Unet\preprocess_data\ct0\segmentation5.nii")
# Chuyển thành NumPy array
image_array = sitk.GetArrayFromImage(image)  # Shape: (depth, height, width)
seg_array = sitk.GetArrayFromImage(seg)  # Shape: (depth, height, width)
print(image_array.shape)
print(seg_array.shape)


# Lấy lát cắt ở giữa (theo chiều depth)
slice_idx = 2
print(np.unique(seg_array))

# Vẽ ảnh gốc và mask trên cùng 1 hình
plt.figure(figsize=(10, 5))

# Ảnh gốc
plt.subplot(1, 2, 1)
plt.imshow(image_array[slice_idx], cmap='gray')
plt.title("Ảnh gốc")
plt.axis("off")

# Ảnh mask (overlay lên ảnh gốc)
plt.subplot(1, 2, 2)
plt.imshow(image_array[slice_idx], cmap='gray')
plt.imshow(seg_array[slice_idx], cmap='Reds', alpha=0.4)  # alpha điều chỉnh độ trong suốt
plt.title("Overlay mask")
plt.axis("off")

plt.tight_layout()
plt.show()