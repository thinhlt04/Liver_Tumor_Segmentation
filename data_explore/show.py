import SimpleITK as sitk
import numpy as np
import cv2
import os

# Load ảnh gốc và mask
root = r".\LiTS17v2\ct0"
# for i in range(131):
image_path = os.path.join(root, rf'volume-0.nii')
mask_path = os.path.join(root, rf'segmentation-0.nii')
image = sitk.ReadImage(image_path)
mask = sitk.ReadImage(mask_path)

image_np = sitk.GetArrayFromImage(image)  
mask_np = sitk.GetArrayFromImage(mask)  

depth, h, w = image_np.shape

# Chuẩn hóa ảnh về 0–255
image_norm = ((image_np - image_np.min()) / (image_np.max() - image_np.min()) * 255).astype(np.uint8)

# Đảm bảo mask là nhị phân 0–255
liver_mask = (mask_np == 1).astype(np.uint8) * 255
tumor_mask = (mask_np == 2).astype(np.uint8) * 255

# Tạo video writer (ghi màu, do có overlay)
out = cv2.VideoWriter(rf'.\overlay_mask0.avi', cv2.VideoWriter_fourcc(*'XVID'), 10, (w, h), isColor=True)

for i in range(depth):
    base = cv2.cvtColor(image_norm[i], cv2.COLOR_GRAY2BGR)

    # Tạo overlay: vùng mask màu đỏ
    red = np.zeros_like(base)
    red[:, :, 2] = 255  # kênh Red
    green = np.zeros_like(base)
    green[:, :, 1] = 255
    # Áp overlay bằng mask
    mask_3c = np.stack([liver_mask[i]]*3, axis=-1)
    overlay = np.where(mask_3c == 255, red, base)

    mask_tumor = np.stack([tumor_mask[i]]*3, axis=-1)
    tumor_overlay = np.where(mask_tumor == 255, green, overlay)

    # Trộn nhẹ overlay và base (alpha blending)
    blended = cv2.addWeighted(base, 0.5, tumor_overlay, 0.5, 0)

    out.write(blended)

out.release()
print("Video đã lưu")

