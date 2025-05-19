import os
import shutil
import random

# Path to the original dataset
original_dataset_dir = r"C:\Users\BISWADAS\PycharmProjects\DOG_CLASSIFIER\Images" # your current folder
base_dir = r"C:\Users\BISWADAS\PycharmProjects\DOG_CLASSIFIER\Dataset"            # new organized folder
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Make target directories
for split in ['train', 'val', 'test']:
    split_path = os.path.join(base_dir, split)
    os.makedirs(split_path, exist_ok=True)

# Loop over each class (breed)
for breed in os.listdir(original_dataset_dir):
    breed_path = os.path.join(original_dataset_dir, breed)
    if not os.path.isdir(breed_path):
        continue

    images = os.listdir(breed_path)
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_imgs = images[:n_train]
    val_imgs = images[n_train:n_train + n_val]
    test_imgs = images[n_train + n_val:]

    for split, split_imgs in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
        split_class_dir = os.path.join(base_dir, split, breed)
        os.makedirs(split_class_dir, exist_ok=True)

        for img in split_imgs:
            src_path = os.path.join(breed_path, img)
            dst_path = os.path.join(split_class_dir, img)
            shutil.copy2(src_path, dst_path)

print("✅ Dataset split into train, val, and test folders in 'dataset/'")