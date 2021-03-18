import os
import shutil

# this script organizes data into proper dir structure for training
# should add a shuffle, but it's pretty random as is

train_set_ratio = 0.8
val_set_ratio = 0.1
test_set_ratio = 0.1

os.makedirs('./data/train/1', exist_ok=True)
os.makedirs('./data/train/0', exist_ok=True)
os.makedirs('./data/valid/1', exist_ok=True)
os.makedirs('./data/valid/0', exist_ok=True)
os.makedirs('./data/test/predict', exist_ok=True)

train_set_size = round(len(os.listdir('./train_dir/1')) * train_set_ratio)
val_set_size = round(len(os.listdir('./train_dir/1')) * val_set_ratio)
test_set_size = len(os.listdir('./train_dir/1')) - train_set_size - val_set_size

ones = os.listdir('./train_dir/1')
zeroes = os.listdir('./train_dir/0')

train_1_fn = ones[:train_set_size]
test_1_fn = ones[train_set_size:train_set_size+test_set_size]
val_1_fn = ones[train_set_size+test_set_size:]

train_0_fn = zeroes[:train_set_size]
test_0_fn = zeroes[train_set_size:train_set_size+test_set_size]
val_0_fn = zeroes[train_set_size+test_set_size:]

fns = [train_1_fn, test_1_fn, val_1_fn, train_0_fn, test_0_fn, val_0_fn]
bases = ['./train_dir/1/', './train_dir/1/', './train_dir/1/', './train_dir/0/', './train_dir/0/', './train_dir/0/']
new_dirs = ['./data/train/1', './data/test/predict', './data/valid/1', './data/train/0', './data/test/predict', './data/valid/0']

for i in range(6):
    for fn in fns[i]:
        shutil.copyfile(bases[i] + fn, new_dirs[i] + os.sep + fn)

for b in new_dirs:
    print(len(os.listdir(b)))