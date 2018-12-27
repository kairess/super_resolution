import cv2, os, glob
import matplotlib.pyplot as plt
from DataGenerator import DataGenerator

base_path = '/mnt/sda1/celeba-dataset/processed'

x_train_list = glob.glob(os.path.join(base_path, 'x_train', '*.npy'))
y_train_list = glob.glob(os.path.join(base_path, 'y_train', '*.npy'))
x_val_list = glob.glob(os.path.join(base_path, 'x_val', '*.npy'))
y_val_list = glob.glob(os.path.join(base_path, 'y_val', '*.npy'))

print(len(x_train_list), len(y_train_list), len(x_val_list), len(y_val_list))

print(x_val_list[0])

train_gen = DataGenerator(list_IDs=x_train_list, labels=None, batch_size=16, dim=(44,44), n_channels=3, n_classes=None, shuffle=True)

val_gen = DataGenerator(list_IDs=x_val_list, labels=None, batch_size=16, dim=(44,44), n_channels=3, n_classes=None, shuffle=False)

a = train_gen.__getitem__(0)

print(a[2].shape)