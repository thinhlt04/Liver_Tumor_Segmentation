import numpy as np
import os
from sklearn.model_selection import train_test_split

root = r'D:\DLFS\Unet\LiTS17_LiS'
X = []
y = []
i = 0
while os.path.exists(os.path.join(root, f'volume-{i}.nii')):
    image_path = os.path.join(root, f'volume-{i}.nii')
    mask_path = os.path.join(root, f'segmentation-{i}.nii')
    X.append(image_path)
    y.append(mask_path)
    i+=1

X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=0.25, random_state=42)

image_des_train = os.path.join(root, 'train', 'image')
target_des_train = os.path.join(root, 'train', 'target')
image_des_dev = os.path.join(root, 'dev', 'image')
target_des_dev = os.path.join(root, 'dev', 'target')
image_des_test = os.path.join(root, 'test', 'image')
target_des_test = os.path.join(root, 'test', 'target')


os.makedirs(image_des_train, exist_ok=True)
os.makedirs(target_des_train, exist_ok=True)
os.makedirs(image_des_dev, exist_ok=True)
os.makedirs(target_des_dev, exist_ok=True)
os.makedirs(image_des_test, exist_ok=True)
os.makedirs(target_des_test, exist_ok=True)

for x, y in zip(X_test, y_test):

    image_name = os.path.basename(x)
    mask_name = os.path.basename(y)

    image_des = os.path.join(image_des_test, image_name)
    target_des = os.path.join(target_des_test, mask_name)

    os.rename(x, image_des)
    os.rename(y, target_des)

for x, y in zip(X_train, y_train):

    image_name = os.path.basename(x)
    mask_name = os.path.basename(y)

    image_des = os.path.join(image_des_train, image_name)
    target_des = os.path.join(target_des_train, mask_name)

    os.rename(x, image_des)
    os.rename(y, target_des)

for x, y in zip(X_dev, y_dev):

    image_name = os.path.basename(x)
    mask_name = os.path.basename(y)

    image_des = os.path.join(image_des_dev, image_name)
    target_des = os.path.join(target_des_dev, mask_name)

    os.rename(x, image_des)
    os.rename(y, target_des)
