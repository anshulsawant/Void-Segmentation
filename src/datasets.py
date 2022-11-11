## TF datasets to be fed into ML (Keras) models.

import images
import utils
import tensorflow as tf
import numpy as np
import os

def _load_data(image, mask):
  def f(x, y):
    return (images.load_image(x.decode())/255.,
            tf.reshape(images.load_mask(y.decode()), (-1,1))/255.)
  img, msk = tf.numpy_function(f, [image, mask], [tf.float32, tf.float32])
  img.set_shape([512, 512, 1])
  msk.set_shape([512*512, 1])
  return (img, msk)


def _load_distance_data(image, mask, distance):
  def f(x, y, z):
    return (images.load_image(x.decode())/255.,
            tf.reshape(images.load_mask(y.decode()), (-1))/255.,
            tf.reshape(images.load_distances(z.decode()), (-1)))
  img, msk, dist = tf.numpy_function(
      f, [image, mask, distance], [tf.float32, tf.float32, tf.float32])
  msk_dist = tf.squeeze(tf.reshape(tf.concat([msk, dist], axis = 0), (-1, 1)))
  img.set_shape([512, 512, 1])
  msk_dist.set_shape([2*512*512])
  return (img, msk_dist)

anchors = utils.anchor_pyramid()

def _load_mask_rcnn_data(image, bbs, anchors):
  def f(i, b, a):
    img = images.load_image(i.decode())/255.
    boxes = images.load_bb(b.decode())
    rpn_labels = utils.anchor_gt_assignment(tf.squeeze(a), boxes)
    return (img, rpn_labels)
  (img, rpn_labels) = tf.numpy_function(f, [image, bbs, anchors], [tf.float32, tf.float32])
  img.set_shape([512, 512, 1])
  return (img, rpn_labels)

def create_dataset(dir=os.path.join(images.ROOT, 'dataset'), batch=8):
  image_paths, masks, _, _ = images.load_image_paths(base=dir, segment = 'train')
  train_size = len(image_paths)*4//5
  print(f'Creating dataset with {len(image_paths)} images.')
  print(f'Using {train_size} images for training.')
  ds = tf.data.Dataset.from_tensor_slices(
      (image_paths, masks)).shuffle(buffer_size=100000).map(_load_data)
  train_ds = ds.take(train_size).batch(batch).prefetch(2)
  val_ds = ds.skip(train_size).batch(batch)
  return (train_ds, val_ds)

def create_test_dataset(dir = os.path.join(images.ROOT, 'dataset'), batch=8):
  image_paths, masks, _, _ = images.load_image_paths(base=dir, segment = 'test')
  print(f'Loading {len(image_paths)} images for testing.')
  return tf.data.Dataset.from_tensor_slices((image_paths, masks)).map(_load_data).batch(batch)

def create_holdout_dataset(dir = os.path.join(images.ROOT, 'dataset'), batch=8):
  image_paths, masks, _, _ = images.load_image_paths(base=dir, segment = 'holdout')
  print(f'Loading {len(image_paths)} images for testing.')
  return tf.data.Dataset.from_tensor_slices((image_paths, masks)).map(_load_data).batch(batch)

def create_distance_dataset(dir=os.path.join(images.ROOT, 'dataset'), batch=8):
  image_paths, masks, _, distance = images.load_image_paths(base=dir, segment = 'train')
  train_size = len(image_paths)*4//5
  print(f'Creating dataset with {len(image_paths)} images.')
  print(f'Using {train_size} images for training.')
  ds = tf.data.Dataset.from_tensor_slices(
      (image_paths, masks, distance)).shuffle(buffer_size=100000).map(_load_distance_data)
  train_ds = ds.take(train_size).batch(batch).prefetch(2)
  val_ds = ds.skip(train_size).batch(batch)
  return (train_ds, val_ds)

def create_distance_test_dataset(dir = os.path.join(images.ROOT, 'dataset'), batch=8):
  image_paths, masks, _, distance = images.load_image_paths(base=dir, segment = 'test')
  print(f'Loading {len(image_paths)} images for testing.')
  return tf.data.Dataset.from_tensor_slices((image_paths, masks, distance)).map(_load_distance_data).batch(batch)

def create_distance_holdout_dataset(dir = os.path.join(images.ROOT, 'dataset'), batch=8):
  image_paths, masks, _, distance = images.load_image_paths(base=dir, segment = 'holdout')
  print(f'Loading {len(image_paths)} images for testing.')
  return tf.data.Dataset.from_tensor_slices((image_paths, masks, distance)).map(_load_distance_data).batch(batch)

def create_mask_rcnn_dataset(dir=os.path.join(images.ROOT, 'dataset'), batch=1):
  image_paths, _, bboxes, _ = images.load_image_paths(base=dir, segment = 'train')
  train_size = (len(image_paths)*4)//5
  print(f'Creating dataset with {len(image_paths)} images.')
  print(f'Using {train_size} images for training.')
  imgs = tf.data.Dataset.from_tensor_slices(image_paths)
  bbs = tf.data.Dataset.from_tensor_slices(bboxes)
  anchors = tf.data.Dataset.from_tensors([utils.anchor_pyramid()]).repeat()
  ds = (tf.data.Dataset.zip((imgs, bbs, anchors))
      .shuffle(buffer_size=100000)
      .map(_load_mask_rcnn_data))
  train_ds = ds.take(train_size).batch(batch).prefetch(2)
  val_ds = ds.skip(train_size).batch(batch).prefetch(2)
  return (train_ds, val_ds)
