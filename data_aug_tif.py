import cv2
import os
import rasterio
from tqdm import tqdm
import numpy as np
from PIL import Image
from albumentations import RandomRotate90, GridDistortion, HorizontalFlip, VerticalFlip, RandomCrop, RandomBrightness, \
    RandomShadow, Transpose, ShiftScaleRotate, RandomSizedCrop, SafeRotate

def augment(image, mask, aug_img, aug_mask):
    c = 0
    for i, j in tqdm(zip(os.listdir(image), os.listdir(mask))):

        with rasterio.open(image+i, 'r') as ds:
            z = ds.read()
            x = z.reshape(2000, 2000, 1)
        y = cv2.imread(mask + i[:-3] + "png")

        if y is None:
            print(i[:-3])
            continue

        aug = RandomRotate90(p=1.0)
        augmented = aug(image=x, mask=y)
        x2 = augmented['image']
        y2 = augmented['mask']

        aug = GridDistortion(p=0.2)
        augmented = aug(image=x, mask=y)
        x3 = augmented['image']
        y3 = augmented['mask']

        aug = HorizontalFlip(p=1.0)
        augmented = aug(image=x, mask=y)
        x4 = augmented['image']
        y4 = augmented['mask']

        aug = VerticalFlip(p=1.0)
        augmented = aug(image=x, mask=y)
        x5 = augmented['image']
        y5 = augmented['mask']

        aug = RandomBrightness(p=1.0)
        augmented = aug(image=x, mask=y)
        x6 = augmented['image']
        y6 = augmented['mask']

        # aug = RandomShadow(p=0.2)
        # augmented = aug(image=x, mask=y)
        # x7 = augmented['image']
        # y7 = augmented['mask']

        aug = Transpose(p=1)
        augmented = aug(image=x, mask=y)
        x8 = augmented['image']
        y8 = augmented['mask']

        aug = ShiftScaleRotate(p=1)
        augmented = aug(image=x, mask=y)
        x9 = augmented['image']
        y9 = augmented['mask']

        aug = SafeRotate(p=1)
        augmented = aug(image=x, mask=y)
        x10 = augmented['image']
        y10 = augmented['mask']

        #Find loss of each operation

        save_images = [z, x2, x3, x4, x5, x6, x8, x9, x10]
        save_masks = [y, y2, y3, y4, y5, y6, y8, y9, y10]
        for p, q in zip(save_images, save_masks):
            p = p.reshape(1, 2000, 2000)
            with rasterio.open(aug_img + str(c) + ".tif", 'w', driver='GTiff', height=2000, width=2000, count=1, dtype=p.dtype) as s:
                s.write(p)
                cv2.imwrite(aug_mask + str(c) + ".png", cv2.resize(q, (2000, 2000)))
                c = c + 1

def augment_aug(image, mask, aug_img, aug_mask):
    c = 0
    for i, j in tqdm(zip(os.listdir(image), os.listdir(mask))):

        x = cv2.imread(image + i[:-3] + "png")
        y = cv2.imread(mask + i[:-3] + "png")

        if y is None:
            print(i[:-3])
            continue

        aug = RandomRotate90(p=1.0)
        augmented = aug(image=x, mask=y)
        x2 = augmented['image']
        y2 = augmented['mask']

        aug = GridDistortion(p=0.2)
        augmented = aug(image=x, mask=y)
        x3 = augmented['image']
        y3 = augmented['mask']

        aug = HorizontalFlip(p=1.0)
        augmented = aug(image=x, mask=y)
        x4 = augmented['image']
        y4 = augmented['mask']

        aug = VerticalFlip(p=1.0)
        augmented = aug(image=x, mask=y)
        x5 = augmented['image']
        y5 = augmented['mask']

        aug = RandomBrightness(p=1.0)
        augmented = aug(image=x, mask=y)
        x6 = augmented['image']
        y6 = augmented['mask']

        # aug = RandomShadow(p=0.2)
        # augmented = aug(image=x, mask=y)
        # x7 = augmented['image']
        # y7 = augmented['mask']

        aug = Transpose(p=1)
        augmented = aug(image=x, mask=y)
        x8 = augmented['image']
        y8 = augmented['mask']

        aug = ShiftScaleRotate(p=1)
        augmented = aug(image=x, mask=y)
        x9 = augmented['image']
        y9 = augmented['mask']

        aug = SafeRotate(p=1)
        augmented = aug(image=x, mask=y)
        x10 = augmented['image']
        y10 = augmented['mask']

        #Find loss of each operation

        save_images = [x, x2, x3, x4, x5, x6, x8, x9, x10]
        save_masks = [y, y2, y3, y4, y5, y6, y8, y9, y10]
        for p, q in zip(save_images, save_masks):
            cv2.imwrite(aug_img + str(c) + ".png", cv2.resize(p, (2000, 2000)))
            cv2.imwrite(aug_mask + str(c) + ".png", cv2.resize(q, (2000, 2000)))
            c = c + 1

aug_image = "./data_final/augmented/png_image/"
aug_mask = "./data_final/augmented/png_mask/"
data_image = "./data_final/original/png_image/"
data_mask = "./data_final/original/mask/"
x = [i[:-3] for i in list(os.listdir(data_image))]
y = [i[:-3] for i in list(os.listdir(data_mask))]

for i in x:
    if i not in y:
        print(i)

augment_aug(data_image, data_mask, aug_image, aug_mask)