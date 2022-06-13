import sysconfig

sys.path.append('/usr/local/google/home/asawant/.local/lib/python3.9/site-packages/')

import cv2
import random
from glob import glob
import numpy as np
import os
from skimage import measure
import tensorflow as tf



ROOT = '/usr/local/google/home/asawant/Void-Segmentation'
## TODO(anshul): All of this file uses numpy not, tf. This will use CPU. Don't know
## if that's good or bad but can use tf functions instead. That will probably be more
## efficient.
def load_image_paths(dir = ROOT):
  dirs = (os.path.join(dir, 'images/*.png'),
          os.path.join(dir, 'masks/*.png'),
          os.path.join(dir, 'bboxes/*.tf'),
          os.path.join(dir, 'bboxes/*.txt'))
  return sorted(glob(dirs[0])), sorted(glob(dirs[1])), sorted(glob(dirs[2])), sorted(glob(dirs[3]))

def load_image(path, eq=True, scale=True):
  x = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
  if eq:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    x = clahe.apply(x)
  if scale:
    x = x/255.0
  x = np.expand_dims(x.astype(np.float32), axis=2)
  return tf.convert_to_tensor(x)

def load_mask(path, scale = True):
  _, mask = cv2.threshold(cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE), 127, 255, cv2.THRESH_BINARY)
  if scale:
    mask = mask/255.0
  return tf.convert_to_tensor(np.expand_dims(mask.astype(np.float32), axis=2))

def load_bb(path):
  return tf.cast(tf.io.parse_tensor(tf.io.read_file(path), out_type=tf.int64), tf.float32)

def load_bb_np(path):
  return np.loadtxt(path)


def split(image, size=1024):
  half = size//2
  tl = image[:half, :half]
  tr = image[:half, half:]
  bl = image[half:, :half]
  br = image[half:, half:]
  return tf.stack([tl, tr, bl, br])

## image is (None, None, 1) array
def generate_noisy_images(image, mask = False, n = 2, min = 0.9, max = 1.1):
  if mask:
    return tf.stack([image for i in range(n+1)])
  return tf.stack([image] + [image * tf.random.uniform(image.shape, min, max) for i in range(n)])

def rotated_and_flipped_images(image):
  x = [None]*8
  x[0] = image
  x[1] = tf.image.rot90(x[0])
  x[2] = tf.image.rot90(x[1])
  x[3] = tf.image.rot90(x[2])
  x[4] = tf.image.flip_left_right(image)
  x[5] = tf.image.rot90(x[4])
  x[6] = tf.image.rot90(x[5])
  x[7] = tf.image.rot90(x[6])
  return tf.stack(x)

def rotate_and_flip_all_images(images):
  ## x = np.empty((s[0]*8, s[1], s[2], s[3]))
  return tf.concat([rotated_and_flipped_images(image) for image in images], axis=0)

def add_noise_to_all_images(images, n = 2, mask=False):
  return tf.concat([generate_noisy_images(image) for image in images], axis=0)

def augment_image(image):
  return add_noise_to_all_images(rotate_and_flip_all_images(split(image)), mask = False)

def augment_mask(mask):
  return add_noise_to_all_images(rotate_and_flip_all_images(split(mask)), mask=True)

## bboxes in format (y_min, x_min, y_max, x_max).
## This is the same format as used by Mask R-CNN code.
## This operation cannot easily be done inside tf, so should be done as a preprocessing step
def get_bboxes(mask):
  x = mask.numpy()
  x = x.reshape((x.shape[0], x.shape[1]))
  return np.concatenate([np.reshape(prop["bbox"], (1, 4)) for prop in measure.regionprops(measure.label(x))])


def split_and_write_images(
    in_dir=os.path.join(ROOT, 'Images', 'raw'),
    out_dir='/content/drive/MyDrive/Segmentation/dataset'):
  dirs = (os.path.join(in_dir, 'images/*.tif'), os.path.join(in_dir, 'masks/*.tif'))
  image_paths, mask_paths = sorted(glob(dirs[0])), sorted(glob(dirs[1]))
  assert(len(image_paths) > 0)
  assert([os.path.basename(path) for path in image_paths]
        == [os.path.basename(path) for path in mask_paths])
  def name(path):
    return os.path.splitext(os.path.basename(path))[0]
  split_images = [[(f'{name(path)}_{i}', sp) for i, sp in enumerate(split(load_image(path, scale=False)))] for path in image_paths]
  split_images = [x for xs in split_images for x in xs]
  split_masks = [[(f'{name(path)}_{i}', sp) for i, sp in enumerate(split(load_mask(path, scale=False)))] for path in mask_paths]
  split_masks = [x for xs in split_masks for x in xs]
  assert([n for n, _ in split_masks] == [n for n, _ in split_images])
  seed = 42
  random.Random(seed).shuffle(split_images)
  random.Random(seed).shuffle(split_masks)
  assert([n for n, _ in split_masks] == [n for n, _ in split_images])
  sz = len(split_images)
  train_set = range(0, (2*sz)//3)
  test_set = range((2*sz)//3, (5*sz)//6)
  holdout_set = range((5*sz)//6, sz)
  def full_path(set_name, image_type, name):
    x = os.path.join(out_dir, set_name, image_type, f'{name}.png')
    print(x)
    return x
  for i in train_set:
    cv2.imwrite(full_path('train', 'images', split_images[i][0]), split_images[i][1].numpy())
    cv2.imwrite(full_path('train', 'masks', split_masks[i][0]), split_masks[i][1].numpy())
  for i in test_set:
    cv2.imwrite(full_path('test', 'images', split_images[i][0]), split_images[i][1].numpy())
    cv2.imwrite(full_path('test', 'masks', split_masks[i][0]), split_masks[i][1].numpy())
  for i in holdout_set:
    cv2.imwrite(full_path('holdout', 'images', split_images[i][0]), split_images[i][1].numpy())
    cv2.imwrite(full_path('holdout', 'masks', split_masks[i][0]), split_masks[i][1].numpy())

def rotate_images_in_dir(
    in_path = '/content/drive/MyDrive/Segmentation/dataset/train/images'):
  paths = glob(in_path + '/*.png')
  def name(path):
    return os.path.splitext(os.path.basename(path))[0]
  def is_transformed(name):
    return (name.endswith('rot90') or
            name.endswith('rot180') or
            name.endswith('rot270') or
            name.endswith('flip'))

  images_in_dir = [(name(path), load_image(path, scale=False)) for path in paths if not is_transformed(name(path))]
  def f(name, image):
    x = [None]*7
    x[0] = (name + '_rot90', tf.image.rot90(image))
    x[1] = (name + '_rot180', tf.image.rot90(x[0][1]))
    x[2] = (name + '_rot270', tf.image.rot90(x[1][1]))
    x[3] = (name + '_flip', tf.image.flip_left_right(image))
    x[4] = (name + '_flip_rot90', tf.image.rot90(x[3][1]))
    x[5] = (name + '_flip_rot180', tf.image.rot90(x[4][1]))
    x[6] = (name + '_flip_rot270', tf.image.rot90(x[5][1]))
    return x
  rotated_images = [f(x[0], x[1]) for x in images_in_dir]
  rotated_images = [x for xs in rotated_images for x in xs]
  for i in rotated_images:
    n = os.path.join(in_path, i[0]+'.png')
    print(n)
    cv2.imwrite(n, i[1].numpy())

def compute_and_write_bboxes(
    in_path = os.path.join(ROOT, 'dataset', 'train'),
    masks_dir = 'masks', bboxes_dir = 'bboxes'):
  in_paths = glob(os.path.join(in_path, masks_dir) + '/*.png')
  names = [os.path.splitext(os.path.basename(p))[0] for p in in_paths]
  out_tf_paths = [os.path.join(in_path, bboxes_dir, n + '.tf') for n in names]
  out_np_paths = [os.path.join(in_path, bboxes_dir, n + '.txt') for n in names]
  for i, p in enumerate(in_paths):
    print(p)
    print(out_tf_paths[i])
    print(out_np_paths[i])
    bboxes = get_bboxes(load_mask(p, scale=False))
    np.savetxt(out_np_paths[i], bboxes)
    contents = tf.io.serialize_tensor(tf.convert_to_tensor(bboxes))
    tf.io.write_file(out_tf_paths[i], contents)

def recreate_dataset():
  split_and_write_images()
  rotate_images_in_dir('/content/drive/MyDrive/Segmentation/dataset/train/images')
  rotate_images_in_dir('/content/drive/MyDrive/Segmentation/dataset/train/masks')
  compute_and_write_bboxes(in_path = '/content/drive/MyDrive/Segmentation/dataset/train')
  compute_and_write_bboxes(in_path = '/content/drive/MyDrive/Segmentation/dataset/test')
  compute_and_write_bboxes(in_path = '/content/drive/MyDrive/Segmentation/dataset/holdout')

def verify_dataset():
  dataset_path = os.path.join(ROOT,'dataset')
  print('Verifying: ' + dataset_path)
  def name(path):
    return os.path.splitext(os.path.basename(path))[0]

  train_images = sorted(glob(os.path.join(dataset_path, 'train', 'images/*.png')))
  train_masks = sorted(glob(os.path.join(dataset_path, 'train', 'masks/*.png')))
  train_bboxes = sorted(glob(os.path.join(dataset_path, 'train', 'bboxes/*.txt')))
  test_images = sorted(glob(os.path.join(dataset_path, 'test', 'images/*.png')))
  test_masks = sorted(glob(os.path.join(dataset_path, 'test', 'masks/*.png')))
  test_bboxes = sorted(glob(os.path.join(dataset_path, 'test', 'bboxes/*.txt')))
  holdout_images = sorted(glob(os.path.join(dataset_path, 'holdout', 'images/*.png')))
  holdout_masks = sorted(glob(os.path.join(dataset_path, 'holdout', 'masks/*.png')))
  holdout_bboxes = sorted(glob(os.path.join(dataset_path, 'holdout', 'bboxes/*.txt')))

  print(f'Number of training images: {len(train_images)}.')
  print(f'Number of test images: {len(test_images)}.')
  print(f'Number of holdout images: {len(holdout_images)}.')
  assert(len(train_images) == 1024)
  assert(len(test_images) == 32)
  assert(len(holdout_images) == 32)

  def names(paths):
    return [name(p) for p in paths]

  assert(names(train_images) == names(train_masks))
  assert(names(train_masks) == names(train_bboxes))
  assert(names(test_images) == names(test_masks))
  assert(names(test_masks) == names(test_bboxes))
  assert(names(holdout_images) == names(holdout_masks))
  assert(names(holdout_masks) == names(holdout_bboxes))
  print('Verified: ' + dataset_path)
