import os
import shutil

# this script organizes data into proper dir structure for training

train_set_ratio = 0.8
val_set_ratio = 0.1
test_set_ratio = 0.1

os.makedirs('./data/tr/1', exist_ok=True)
os.makedirs('./data/tr/0', exist_ok=True)
os.makedirs('./data/v/1', exist_ok=True)
os.makedirs('./data/v/0', exist_ok=True)
os.makedirs('./data/te/1', exist_ok=True)
os.makedirs('./data/te/0', exist_ok=True)

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
new_dirs = ['./data/tr/1', './data/te/1', './data/v/1', './data/tr/0', './data/te/0', './data/v/0']
for i in range(6):
    for fn in fns[i]:
        shutil.copyfile(bases[i] + fn, new_dirs[i] + os.sep + fn)

for b in new_dirs:
    print(len(os.listdir(b)))