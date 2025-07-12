import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh 2D
image = sitk.ReadImage(r"D:\DLFS\Unet\sample\train\image\volume-103.nii")
liver_mask = sitk.ReadImage(r"D:\DLFS\Unet\sample\train\target\segmentation-103.nii")

# Chuyển 2 → 1 nếu có tumor
liver_mask_np = sitk.GetArrayFromImage(liver_mask)[0]  # (1, 512, 512) → (512, 512)
liver_mask_np[liver_mask_np == 2] = 1


def get_liver_bbox(mask_np, padding=5):
    pos = np.argwhere(mask_np > 0)
    miny, minx = pos.min(axis=0)
    maxy, maxx = pos.max(axis=0)

    miny = max(miny - padding, 0)
    minx = max(minx - padding, 0)
    maxy = min(maxy + padding, mask_np.shape[0] - 1)
    maxx = min(maxx + padding, mask_np.shape[1] - 1)

    return (miny, maxy, minx, maxx)
def pad_to_shape(image_crop, target_shape=(256, 256)):
    padded = np.zeros(target_shape, dtype=image_crop.dtype)
    h, w = image_crop.shape
    start_y = (target_shape[0] - h) // 2
    start_x = (target_shape[1] - w) // 2
    padded[start_y:start_y+h, start_x:start_x+w] = image_crop
    return padded

bbox = get_liver_bbox  (liver_mask_np)

image_np = sitk.GetArrayFromImage(image)[0]  

image_np = np.clip(image_np, -45, 167)


miny, maxy, minx, maxx = bbox
cropped_image = image_np[miny:maxy+1, minx:maxx+1]
cropped_mask = liver_mask_np[miny:maxy+1, minx:maxx+1]

cropped_image_padded = pad_to_shape(cropped_image, (256, 256))
cropped_mask_padded = pad_to_shape(cropped_mask, (256, 256))

# Hiển thị
plt.figure(figsize=(8, 8))  # Tăng chiều cao để phù hợp 2 hàng

plt.subplot(2, 2, 1)
plt.imshow(image_np, cmap='gray')
plt.title("Ảnh gốc")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(liver_mask_np, cmap='gray')
plt.title("Mask gan")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(cropped_image_padded, cmap='gray')
plt.title("Ảnh crop gan")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(cropped_mask_padded, cmap='gray')
plt.title("Mask gan crop")
plt.axis("off")

plt.tight_layout()
plt.show()
