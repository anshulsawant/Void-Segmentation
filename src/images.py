### Methods for preprocessing and prepping data for a tf pipeline.

from glob import glob
from scipy import spatial
from skimage import measure
import cv2
import json
import numpy as np
import os
import platform
import random
import tensorflow as tf

ROOT = ('/usr/local/google/home/asawant/Void-Segmentation'
        if platform.node().endswith(
            'corp.google.com') or platform.node().endswith('googlers.com')
        else '/content/Void-Segmentation')

def load_image_paths(base=os.path.join(ROOT, 'dataset'), segment='train'):
  print(f'Loading images from {os.path.join(base, segment)}.')
  dirs = (os.path.join(base, segment, 'images/*.png'),
          os.path.join(base, segment, 'masks/*.png'),
          os.path.join(base, segment, 'bboxes/*.tf'),
          os.path.join(base, segment, 'bboxes/*.txt'))
  return sorted(glob(dirs[0])), sorted(glob(dirs[1])), sorted(glob(dirs[2])), sorted(glob(dirs[3]))

## Load an image from the raw data
def load_equalized_image(path):
  x = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
  x = clahe.apply(x)
  x = np.expand_dims(x.astype(np.float32), axis=2)
  return tf.convert_to_tensor(x)

## Load a mask from the raw data
def load_binary_mask(path):
  _, mask = cv2.threshold(cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE), 127, 255, cv2.THRESH_BINARY)
  return tf.convert_to_tensor(np.expand_dims(mask.astype(np.float32), axis=2))

## Load an image from the dataset
def load_image(path):
  x = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
  x = np.expand_dims(x.astype(np.float32), axis=2)
  return tf.convert_to_tensor(x)

## Load a mask from the dataset
def load_mask(path):
  ## These masks should already be binary, but lets just threshold them anyway.
  _, mask = cv2.threshold(cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE), 127, 255, cv2.THRESH_BINARY)
  return tf.convert_to_tensor(np.expand_dims(mask.astype(np.float32), axis=2))

def load_bb(path):
  return tf.cast(tf.io.parse_tensor(tf.io.read_file(path), out_type=tf.int64), tf.float32)

def load_bb_np(path):
  return np.loadtxt(path)

## For generating dataset from the raw images.
def split(image, size=1024):
  half = size//2
  tl = image[:half, :half]
  tr = image[:half, half:]
  bl = image[half:, :half]
  br = image[half:, half:]
  return tf.stack([tl, tr, bl, br])

## bboxes in format (y_min, x_min, y_max, x_max).
## This is the same format as used by Mask R-CNN code.
## This operation cannot easily be done inside tf, so should be done as a preprocessing step
def get_bboxes(mask):
  x = mask.numpy()
  x = x.reshape((x.shape[0], x.shape[1]))
  return np.concatenate([np.reshape(prop["bbox"], (1, 4)) for prop in measure.regionprops(measure.label(x))])


## For generating dataset from the raw images
def split_and_write_images(in_dir=os.path.join(ROOT, 'raw_data'),
                           out_dir = os.path.join(ROOT, 'dataset')):
  dirs = (os.path.join(in_dir, 'images/*.tif'), os.path.join(in_dir, 'masks/*.tif'))
  image_paths, mask_paths = sorted(glob(dirs[0])), sorted(glob(dirs[1]))
  print(f'Reading {len(image_paths)} images and masks.')
  assert(len(image_paths) > 0)
  print(f'Name of an image file: {image_paths[0]}.')
  assert([os.path.basename(path) for path in image_paths]
        == [os.path.basename(path) for path in mask_paths])
  def name(path):
    return os.path.splitext(os.path.basename(path))[0]
  split_images = [[(f'{name(path)}_{i}', sp) for i, sp in enumerate(split(load_equalized_image(path)))] for path in image_paths]
  split_images = [x for xs in split_images for x in xs]
  split_masks = [[(f'{name(path)}_{i}', sp) for i, sp in enumerate(split(load_binary_mask(path)))] for path in mask_paths]
  split_masks = [x for xs in split_masks for x in xs]
  ## Images 13_2 and 13_3 have a big overalay specifying scale of the images. A small part is also in 14_2, but we
  ## can probably live with that.
  split_images = [(name, image) for (name, image) in split_images if ('13_2' not in name) and ('13_3' not in name)]
  split_masks = [(name, image) for (name, image) in split_masks if ('13_2' not in name) and ('13_3' not in name)]
  print(f'Split into {len(split_images)} images and masks.')
  assert([n for n, _ in split_masks] == [n for n, _ in split_images])
  seed = 42
  random.Random(seed).shuffle(split_images)
  random.Random(seed).shuffle(split_masks)
  assert([n for n, _ in split_masks] == [n for n, _ in split_images])
  sz = len(split_images)
  train_segment = range(0, (2*sz)//3)
  print(f'Selected {len(train_segment)} image for training.')
  test_segment = range((2*sz)//3, (5*sz)//6)
  print(f'Selected {len(test_segment)} image for testing.')
  holdout_segment = range((5*sz)//6, sz)
  print(f'Selected {len(holdout_segment)} image for holdout.')
  def full_path(segment, image_type, name):
    x = os.path.join(out_dir, segment, image_type, f'{name}.png')
    return x
  for i in train_segment:
    cv2.imwrite(full_path('train', 'images', split_images[i][0]), split_images[i][1].numpy())
    cv2.imwrite(full_path('train', 'masks', split_masks[i][0]), split_masks[i][1].numpy())
  for i in test_segment:
    cv2.imwrite(full_path('test', 'images', split_images[i][0]), split_images[i][1].numpy())
    cv2.imwrite(full_path('test', 'masks', split_masks[i][0]), split_masks[i][1].numpy())
  for i in holdout_segment:
    cv2.imwrite(full_path('holdout', 'images', split_images[i][0]), split_images[i][1].numpy())
    cv2.imwrite(full_path('holdout', 'masks', split_masks[i][0]), split_masks[i][1].numpy())

def is_transformed(path):
  name = os.path.splitext(os.path.basename(path))[0]
  return (name.endswith('rot90') or
          name.endswith('rot180') or
          name.endswith('rot270') or
          name.endswith('flip'))

## For generating dataset from the raw images
def rotate_images_in_dir(
    in_path = os.path.join(ROOT, 'dataset', 'train', 'images'),
    mask = False):
  paths = glob(in_path + '/*.png')
  def name(path):
    return os.path.splitext(os.path.basename(path))[0]
  def is_transformed(name):
    return (name.endswith('rot90') or
            name.endswith('rot180') or
            name.endswith('rot270') or
            name.endswith('flip'))

  load = load_mask if mask else load_image
  ## Do not rotate an already rotated or flipped image
  images_in_dir = [(name(path), load(path)) for path in paths if not is_transformed(name(path))]
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
  print(f'Writing {len(rotated_images)} rotated and flipped images.')
  for i in rotated_images:
    n = os.path.join(in_path, i[0]+'.png')
    cv2.imwrite(n, i[1].numpy())

## For generating dataset from the raw images
def compute_and_write_bboxes(
    in_path = os.path.join(ROOT, 'dataset', 'train'),
    masks_dir = 'masks', bboxes_dir = 'bboxes'):
  in_paths = glob(os.path.join(in_path, masks_dir) + '/*.png')
  names = [os.path.splitext(os.path.basename(p))[0] for p in in_paths]
  out_tf_paths = [os.path.join(in_path, bboxes_dir, n + '.tf') for n in names]
  out_np_paths = [os.path.join(in_path, bboxes_dir, n + '.txt') for n in names]
  print(f'Writing bounding boxes for {len(out_tf_paths)} images.')
  count = 0
  for i, p in enumerate(in_paths):
    bboxes = get_bboxes(load_mask(p))
    count += bboxes.shape[0]
    ## Human readable text
    np.savetxt(out_np_paths[i], bboxes)
    ## Binary format
    contents = tf.io.serialize_tensor(tf.convert_to_tensor(bboxes))
    tf.io.write_file(out_tf_paths[i], contents)
  print(f'Wrote {count} bounding boxes for {len(out_tf_paths)} images.')

## For generating dataset from the raw images
def recreate_dataset():
  split_and_write_images()
  rotate_images_in_dir(os.path.join(ROOT, 'dataset', 'train', 'images'))
  rotate_images_in_dir(os.path.join(ROOT, 'dataset', 'train', 'masks'), mask = True)
  compute_and_write_bboxes(in_path = os.path.join(ROOT, 'dataset', 'train'))
  compute_and_write_bboxes(in_path = os.path.join(ROOT, 'dataset', 'test'))
  compute_and_write_bboxes(in_path = os.path.join(ROOT, 'dataset', 'holdout'))

def summarise_dataset():
  dataset_path = os.path.join(ROOT,'dataset')
  print('Summarizing: ' + dataset_path)
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
  print(f'Number of training masks: {len(train_masks)}.')
  print(f'Number of training boxes: {len(train_bboxes)}.')
  print(f'Number of test images: {len(test_images)}.')
  print(f'Number of test masks: {len(test_masks)}.')
  print(f'Number of test boxes: {len(test_bboxes)}.')
  print(f'Number of holdout images: {len(holdout_images)}.')
  print(f'Number of holdout masks: {len(holdout_masks)}.')
  print(f'Number of holdout boxes: {len(holdout_bboxes)}.')

## Some sanity check for the generated dataset
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

## Returns list of unroated/unflipped files for images in a dir
def vanilla_files(dir = os.path.join(ROOT, 'dataset', 'train', 'masks'), ext='png'):
  all_files = glob(os.path.join(dir, f'*.{ext}'))
  return [file_name for file_name in all_files if not is_transformed(file_name)]


def clear_dataset(dir = os.path.join(ROOT, 'dataset')):
  train_images = os.path.join(dir, 'train', 'images')
  train_masks = os.path.join(dir, 'train', 'masks')
  train_bboxes = os.path.join(dir, 'train', 'bboxes')
  test_images = os.path.join(dir, 'test', 'images')
  test_masks = os.path.join(dir, 'test', 'masks')
  test_bboxes = os.path.join(dir, 'test', 'bboxes')
  holdout_images = os.path.join(dir, 'holdout', 'images')
  holdout_masks = os.path.join(dir, 'holdout', 'masks')
  holdout_bboxes = os.path.join(dir, 'holdout', 'bboxes')

  def delete(folder):
    files = glob(os.path.join(folder, '*'))
    print(f'Deleting {len(files)} from directory {folder}')
    for f in files:
      os.remove(f)

  delete(train_images)
  delete(train_masks)
  delete(train_bboxes)
  delete(test_images)
  delete(test_masks)
  delete(test_bboxes)
  delete(holdout_images)
  delete(holdout_masks)
  delete(holdout_bboxes)

def polyline_mask(xs, ys):
  ps = np.dstack((xs, ys))[0]
  image = np.zeros((512, 512))
  x = cv2.fillPoly(image, pts=[ps], color=1)
  return x > 0

def ellipse_mask(cx, cy, rx, ry, theta):
  xs, ys = np.meshgrid(range(512), range(512))
  xs = xs - cx
  ys = ys - cy

  d1 = np.square((xs/rx)*np.cos(theta) + (ys/rx)*np.sin(theta))
  d2 = np.square((xs/ry)*np.sin(theta) - (ys/ry)*np.cos(theta))

  return d1 + d2 - 1 <= 0

def circle_mask(cx, cy, r):
  xs, ys = np.meshgrid(range(512), range(512))
  xs = xs - cx
  ys = ys - cy

  d1 = np.square(ys/r)
  d2 = np.square(xs/r)

  return d1 + d2 - 1 <= 0

def trivial_mask():
  return np.zeros((512, 512))!=0

def region_to_mask(region_json):
  attributes = region_json['shape_attributes']
  if attributes['name'] == 'ellipse':
    cx = attributes['cx']
    cy = attributes['cy']
    rx = attributes['rx']
    ry = attributes['ry']
    theta = attributes['theta']
    ## Masks take up lot of memory. Do the actual computation right before saving them
    return lambda: ellipse_mask(cx, cy, rx, ry, theta)
  elif attributes['name'] == 'polygon':
    xs = np.array(attributes['all_points_x'])
    ys = np.array(attributes['all_points_y'])
    ## Masks take up lot of memory. Do the actual computation right before saving them
    return lambda: polyline_mask(xs, ys)
  elif attributes['name'] == 'circle':
    xs = np.array(attributes['cx'])
    ys = np.array(attributes['cy'])
    r = np.array(attributes['r'])
    ## Masks take up lot of memory. Do the actual computation right before saving them
    return lambda: circle_mask(xs, ys, r)
  else:
    raise RuntimeError(f'Unknown shape: {attributes}.')

def regions_to_masks(image_metadata):
  regions = image_metadata['regions']
  filename = image_metadata['filename']
  if len(regions) == 0:
    print(f'Warning: no masks provided for {filename}.')
  return filename, [region_to_mask(r) for r in regions]

def flatten_masks(mask_computations):
  x = trivial_mask()
  for c in mask_computations:
    x = np.logical_or(x, c())
  return x*255

def load_masks_json(base_dir = ROOT, fn = '0636_masks.json'):
  image_data = list(json.load(open(os.path.join(base_dir, fn)))[
      '_via_img_metadata'].values())
  return [regions_to_masks(i) for i in image_data]

def compute_flattened_masks(image_mask_computations):
  return [(i, flatten_masks(computations))
          for i, computations in image_mask_computations
          if len(computations) > 0]

def save_mask(image_mask, outdir = os.path.join(ROOT, 'raw_data', 'masks')):
  image, mask = image_mask
  p = os.path.join(outdir, image)
  print(f'Writing {p}.')
  cv2.imwrite(os.path.join(outdir, image), mask)

def load_and_write_masks(
    base_dir = ROOT,
    fn='0636_masks.json',
    outdir = os.path.join(ROOT, 'raw_data', 'masks')):
  for image_mask in compute_flattened_masks(load_masks_json(base_dir = base_dir, fn = fn)):
    save_mask(image_mask, outdir = outdir)

def load_and_write_masks_in_dir(
    base_dir = os.path.join(ROOT, 'raw_data', 'json_masks')):
  files = glob(os.path.join(base_dir, '*.json'))
  for f in files:
    print(f'Loading file: {f}.')
    load_and_write_masks(base_dir=base_dir, fn=f, outdir=base_dir)
