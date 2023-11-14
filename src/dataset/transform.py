import albumentations as tf
from albumentations.pytorch import ToTensorV2
import cv2


def train_transform(img, mask):
    transform = tf.Compose([
        tf.HorizontalFlip(p=0.3),
        tf.VerticalFlip(p=0.3),
        tf.RandomRotate90(p=0.3),
        tf.Resize(256, 256, interpolation=cv2.INTER_LINEAR),
        tf.Normalize(),
        ToTensorV2(),
    ])

    return transform(image=img, mask=mask)


def val_transform(img, mask):
    transform = tf.Compose([
        tf.Resize(256, 256, interpolation=cv2.INTER_LINEAR),
        ToTensorV2(),
    ])

    return transform(image=img, mask=mask)


def test_transform(img):
    transform = tf.Compose([
        tf.Resize(256, 256, interpolation=cv2.INTER_LINEAR),
        tf.Normalize(), # This is important since we will use pretrained model
        ToTensorV2(),
    ])

    return transform(image=img)['image']