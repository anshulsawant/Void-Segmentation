import images
import tensorflow as tf
import numpy as np

def create_dataset(dir='/content/drive/MyDrive/Segmentation/Images/split16', batch=16):
  images, masks, _, _ = images.load_image_paths(dir=dir)
  seed = random.randint(0, 1000000)
  random.seed(seed)
  random.shuffle(images)
  random.seed(seed)
  random.shuffle(masks)
  def load_data(image, mask):
    def f(x,y):
      return (images.augment_image(images.load_image(x.decode())).reshape((64, 512, 512)), images.augment_mask(load_mask(y.decode())).reshape((64,512*512)))
    img, msk = tf.py_function(f, [image, mask], [tf.float32, tf.float32])
    img.set_shape([64, 512, 512])
    msk.set_shape([64, 512*512])
    return tf.data.Dataset.from_tensor_slices((img, msk))
  return (tf.data.Dataset.from_tensor_slices((images, masks)).take(28).flat_map(load_data).shuffle(buffer_size=10000).batch(batch)
      ,tf.data.Dataset.from_tensor_slices((images, masks)).skip(28).flat_map(load_data).shuffle(buffer_size=10000).batch(batch))

def create_test_dataset(dir='/content/drive/MyDrive/Segmentation/Images/split16', batch=16):
  _, _, images, masks = load_image_paths(dir=dir)
  def load_data(image, mask):
    def f(x,y):
      return (split(load_image(x.decode())).reshape(4, 512, 512), split(load_mask(y.decode())).reshape((4, 512*512)))
    img, msk = tf.numpy_function(f, [image, mask], [tf.float32, tf.float32])
    img.set_shape([4, 512, 512])
    msk.set_shape([4, 512*512])
    return tf.data.Dataset.from_tensor_slices((img, msk))
  return tf.data.Dataset.from_tensor_slices((images, masks)).flat_map(load_data).batch(batch)


def create_rpn_dataset(dir = '/content/drive/MyDrive/Segmentation/Images/split16'):
  dataset = create_dataset(dir = dir, batch=1)

