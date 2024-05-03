# %%
from tqdm import tqdm
import shutil
import os
import random
brats_train_path = "brats_data/train/"
brats_val_path = "brats_data/val/"
brats_test_path = "brats_data/test/"
os.makedirs(brats_val_path, exist_ok=True)
os.makedirs(brats_test_path, exist_ok=True)
file_names = os.listdir(brats_train_path)
sample_size = int(len(file_names) * 0.2)
len(file_names)

# %%
random_indices = random.sample(range(len(file_names)), sample_size)
_half = len(random_indices) // 2

# %%
for i in tqdm(random_indices[:_half], desc="Splitting val"):
    file_name = file_names[i]
    shutil.move(f'{brats_train_path}/{file_name}',
                f'{brats_val_path}/{file_name}')

for i in tqdm(random_indices[_half:], desc="Splitting test"):
    file_name = file_names[i]
    shutil.move(f'{brats_train_path}/{file_name}',
                f'{brats_test_path}/{file_name}')

# %%
print('train:', len(os.listdir(brats_train_path)))
print('val:', len(os.listdir(brats_val_path)))
print('test:', len(os.listdir(brats_test_path)))
