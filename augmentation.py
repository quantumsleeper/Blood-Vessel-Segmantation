import os 
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio    # to read masks which are in gif format 
import gif2numpy as gf
from albumentations import HorizontalFlip, VerticalFlip, Rotate


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def augment_data(images, masks, save_path, augment=True):
    '''
        images: list of image paths
        masks: list of corresponding masks
        augment: if augment is false then you'll just resize    
    '''
    size = (512, 512)
    for idx, (img, mask) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        # Extracting the name e.g. 21_training 
        name = img.split('/')[-1].split('.')[0]
        
        # Reading the img and mask
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        mask, _, _ = gf.convert(mask)   # mask is a grayscale image
        
        if augment:
            pass
        else:
            X = [img]
            Y = [mask[0]]


        idx = 0
        for i, m in zip(X, Y):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            # Using a temp_name because after augmentation, same image will have different variants
            tmp_image_name = f"{name}_{idx}.png"
            tmp_mask_name = f"{name}_{idx}.png"

            img_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(img_path, i)
            cv2.imwrite(mask_path, m)

            idx += 1
        break


def load_data(path):

    np.random.seed(42)

    train_X = sorted(glob(os.path.join(path, "training", "images", "*.tif")))
    train_y = sorted(glob(os.path.join(path, "training", "mask", "*.gif")))

    test_X = sorted(glob(os.path.join(path, "test", "images", "*.tif")))
    test_y = sorted(glob(os.path.join(path, "test", "mask", "*.gif")))

    return (train_X, train_y), (test_X, test_y)


if __name__ == '__main__':
    data_path = "./DRIVE/"
    (train_X, train_y), (test_X, test_y) = load_data(data_path)

    print(f"Train: {len(train_X)} - {len(train_y)}")
    print(f"Test: {len(test_X)} - {len(test_y)}")

    create_dir("new_data/train/image")
    create_dir("new_data/test/image")
    create_dir("new_data/train/mask")
    create_dir("new_data/test/mask")    

    augment_data(train_X, train_y, "new_data/train/", augment=False)    
