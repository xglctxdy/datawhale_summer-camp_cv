import numpy as np
import os
import nibabel as nib
import shutil
from PIL import Image  # 保存图像
import random
import cv2

# folder nii
classes = ['MCI', 'NC']
current_dir = 'D:\\人工智能\\暑期夏令营\\进阶'
data_dir = '脑PET图像分析和疾病预测挑战赛数据集'
os.chdir(f'{current_dir}')


def mkdir(path):
    floder = os.path.exists(path)
    if not floder:
        os.mkdir(path)
    else:
        pass


for i in range(2):
    file = f'{data_dir}\\Val\\{classes[i]}'
    mkdir(file)

# count nii samples
sample_number = []
for i in range(2):
    train_dir = f'{data_dir}\\Train\\{classes[i]}'
    file = os.listdir(train_dir)
    sample_number.append(len(file))
print('nii:', sample_number)


# generate val set
def moveFile(train_Dir, val_Dir, rate=0.15):
    pathDir = os.listdir(train_Dir)
    filenumber = len(pathDir)
    sample = random.sample(pathDir, int(filenumber * rate))
    for name in sample:
        shutil.move(train_Dir + '\\' + name, val_Dir + '\\' + name)
    return


for i in range(2):
    train_dir = f'{data_dir}\\Train\\{classes[i]}'
    val_dir = f'{data_dir}\\Val\\{classes[i]}'
    moveFile(train_dir, val_dir)

sample_number = []
for i in range(2):
    train_dir = f'{data_dir}\\Train\\{classes[i]}'
    file = os.listdir(train_dir)
    sample_number.append(len(file))
print('training nii', sample_number)  # 单词拼写错误

sample_number = []
for i in range(2):
    val_dir = f'{data_dir}\\Val\\{classes[i]}'
    file = os.listdir(val_dir)  # 这里本来写的是train_dir
    sample_number.append(len(file))
print('val nii:', sample_number)


# recover training set
def Recover(val_Dir, train_Dir):
    pathDir = os.listdir(val_Dir)
    for name in pathDir:
        shutil.move(val_Dir + '\\' + name, train_Dir + '\\' + name)
    return


for i in range(2):
    train_dir = f'{data_dir}\\Train\\{classes[i]}'
    val_dir = f'{data_dir}\\Val\\{classes[i]}'
    Recover(val_dir, train_dir)

sample_number = []
for i in range(2):
    train_dir = f'{data_dir}\\Train\\{classes[i]}'
    file = os.listdir(train_dir)
    sample_number.append(len(file))
print('training nii:', sample_number)

sample_number = []
for i in range(2):
    val_dir = f'{data_dir}\\Val\\{classes[i]}'
    file = os.listdir(val_dir)
    sample_number.append(len(file))
print('val nii:', sample_number)

# folder jpg
folder = [
    'PETjpg\\Test',
    'PETjpg\\pred',
    'PETjpg\\Train\\MCI',
    'PETjpg\\Train\\NC',
    'PETjpg\\Val\\MCI',
    'PETjpg\\Val\\NC'
]

for i in range(len(folder)):
    mkdir(folder[i])


# function
def nii_to_image(filepath, imgfile):
    filenames = os.listdir(filepath)

    for f in filenames:
        img_path = os.path.join(filepath, f)
        img = nib.load(img_path)
        img_fdata = img.get_fdata()
        img_fdata = img_fdata[:, :, :, 0]
        fname = f.replace('.nii', '')
        img_f_path = os.path.join(imgfile, fname)
        mkdir(img_f_path)

        (x, y, z) = img_fdata.shape
        for j in range(int(z / 8), int(7 * z / 8)):  # 舍弃边上黑色的部分
            slice = img_fdata[:, :, :, i]
            slice_min = np.min(slice)
            slice_max = np.max(slice)
            slice_normalized = (slice - slice_min) / (slice_max - slice_min)
            slice_scaled = (slice_normalized * 255).astype(np.uint8)
            slice_image = Image.fromarray(slice_scaled)
            slice_image.save(os.path.join(img_f_path, f'{j}.jpg'))


# nii_to_jpg
base = [f'{data_dir}\\Train\\MCI',
        f'{data_dir}\\Train\\NC',
        f'{data_dir}\\Val\\MCI',
        f'{data_dir}\\Val\\NC',
        f'{data_dir}\\Test']

output = ['PETjpg\\Train\\MCI',
          'PETjpg\\Train\\NC',
          'PETjpg\\Val\\MCI',
          'PETjpg\\Val\\NC',
          'PETjpg\\Test']

for i in range(5):
    nii_to_image(base[i], output[i])

# unpack
all_dir = output

for i in range(5):
    work_dir = all_dir[i]
    work_dir = os.listdir(work_dir)
    folder_number = len(work_dir)
    for j in range(folder_number):
        folder_dir = os.listdir(f'{current_dir}' + all_dir[i] + work_dir[j])
        file_number = len(folder_dir)
        os.chdir(f'{current_dir}' + all_dir[i] + work_dir[j])
        for k in range(file_number):
            os.rename(folder_dir[k], work_dir[j] + '_' + folder_dir[k])
            shutil.move(f'{current_dir}' + all_dir[i] + work_dir[j] +
                        '\\' + work_dir[j] + '_' + folder_dir[k],
                        f'{current_dir}' + all_dir[i] + '\\')

    os.chdir(f'{current_dir}')
    for j in range(folder_number):
        shutil.rmtree(f'{current_dir}' + all_dir[i] + work_dir[j])

# count jpg samples
training_jpg = []
for i in range(2):
    train_dir = f'PETjpg\\Train\\{classes[i]}'
    file = os.listdir(train_dir)
    training_jpg.append(len(file))
print('training jpg', training_jpg)

val_jpg = []
for i in range(2):
    val_dir = f'PETjpg\\Val\\{classes[i]}'
    file = os.listdir(val_dir)
    val_jpg.append(len(file))
print('val ipg:', val_jpg)

# random_generate_balanced_training_sample
for i in range(2):
    train_dir = f'PETjpg\\Train\\{classes[i]}'
    pathDir = os.listdir(train_dir)
    filenumber = len(pathDir)
    os.chdir(train_dir)
    if filenumber != 1800:
        for j in range(1800 - filenumber):
            index = np.random.randint(filenumber)
            image = cv2.imread(pathDir[index])
            cv2.imwrite(f'synthetic{j}.jpg', image)
    os.chdir(f'{current_dir}')

sample_number = []
for i in range(2):
    train_dir = f'PETjpg\\Train\\{classes[i]}'
    file = os.listdir(train_dir)
    sample_number.append(len(file))
print('training jpg:', sample_number)

for i in range(2):
    train_dir = f'PETjpg\\Val\\{classes[i]}'
    pathDir = os.listdir(train_dir)
    filenumber = len(pathDir)
    os.chdir(train_dir)
    if filenumber != 200:
        for j in range(200 - filenumber):
            index = np.random.randint(filenumber)
            image = cv2.imread(pathDir[index])
            cv2.imwrite(f'synthetic{j}.jpg', image)
    os.chdir(f'{current_dir}')

sample_number = []
for i in range(2):
    val_dir = f'PETjpg\\Val\\{classes[i]}'
    file = os.listdir(val_dir)
    sample_number.append(len(file))
print('val jpg:', sample_number)

# recover_generated_training_sample
training_jpg = [948, 1748]
for i in range(2):
    train_dir = f'PETjpg\\Train\\{classes[i]}'
    pathDir = os.listdir(train_dir)
    filenumber = len(pathDir)
    os.chdir(train_dir)
    for j in range(0, 1800 - training_jpg[i]):
        os.remove(f'synthetic{j}.jpg')
    os.chdir(f'{current_dir}')

sample_number = []
for i in range(2):
    train_dir = f'PETjpg\\Train\\{classes[i]}'
    file = os.listdir(train_dir)
    sample_number.append(len(file))
print('training jpg:', sample_number)

# recover_generated_val_sample
val_jpg = [132, 132]
for i in range(2):
    val_dir = f'dataset\\PETjpg\\Val\\{classes[i]}'
    pathDir = os.listdir(val_dir)
    filenumber = len(pathDir)
    os.chdir(val_dir)
    for j in range(0, 200 - val_jpg[i]):
        os.remove(f'synthetic{j}.jpg')
    os.chdir(f'{current_dir}')

sample_number = []
for i in range(2):
    val_dir = f'PETjpg\\Val\\{classes[i]}'
    file = os.listdir(val_dir)
    sample_number.append(len(file))
print('val jpg:', sample_number)


def count_nii_z(filepath):
    filenames = os.listdir(filepath)
    z_number = []

    for f in filenames:
        img_path = os.path.join(filepath, f)
        img = nib.load(img_path)
        img_fdata = img.get_fdata()
        img_fdata = img_fdata[:, :, :, 0]

        (x, y, z) = img_fdata.shape
        z_number.append(int(7 * z / 8) - int(z / 8))
    return z_number


z_num1 = count_nii_z('{data_dir}\\Test')
print(z_num1)
