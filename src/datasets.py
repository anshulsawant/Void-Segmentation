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

def _load_mask_rcnn_data(image, mask, bboxes, anchors):
  def f(x, y, z):
    positive_anchors, deltas, negative_anchors = utils.anchor_gt_assignment(anchors, bboxes)
    return (
        images.load_image(x.decode())/255.,
        tf.reshape(images.load_mask(y.decode()), (-1,1))/255.,
        positive_anchors,
        deltas,
        negative_anchors)
  img, msk, positive_anchors, deltas, negative_anchors = \
  tf.numpy_function(f, [image, mask, bboxes], [tf.float32, tf.float32, tf.float32])
  img.set_shape([512, 512, 1])
  msk.set_shape([512*512, 1])
  return (img, msk, positive_anchors, deltas, negative_anchors)

def create_dataset(dir=os.path.join(images.ROOT, 'dataset'), batch=8):
  image_paths, masks, _, _ = images.load_image_paths(dir=dir, segment = 'train')
  train_size = len(image_paths)*4//5
  print(f'Creating dataset with {len(image_paths)} images.')
  print(f'Using {train_size} images for training.')
  ds = tf.data.Dataset.from_tensor_slices(
      (image_paths, masks)).shuffle(buffer_size=100000).map(_load_data)
  train_ds = ds.take(train_size).batch(batch).prefetch(2)
  val_ds = ds.skip(train_size).batch(batch)
  return (train_ds, val_ds)

def create_test_dataset(dir = os.path.join(images.ROOT, 'dataset'), batch=8):
  image_paths, masks, _, _ = images.load_image_paths(dir=dir, segment = 'test')
  print(f'Loading {len(image_paths)} images for testing.')
  return tf.data.Dataset.from_tensor_slices((image_paths, masks)).map(_load_data).batch(batch)

def create_mask_rcnn_dataset(dir=os.path.join(images.ROOT, 'dataset'), batch=1):
  image_paths, masks, bboxes, _ = images.load_image_paths(dir=dir, segment = 'train')
  train_size = len(image_paths)*4//5
  anchors = utils.anchor_pyramid()
  print(f'Creating dataset with {len(image_paths)} images.')
  print(f'Using {train_size} images for training.')
  ds = (tf.data.Dataset
        .from_tensor_slices((image_paths, masks, bboxes))
        .shuffle(buffer_size=100000)
        .map(_load_mask_rcnn_data))
  train_ds = ds.take(train_size).batch(batch).prefetch(2)
  val_ds = ds.skip(train_size).batch(batch).prefetch(2)
  return (train_ds, val_ds)
