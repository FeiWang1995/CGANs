# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 19:58:48 2018

@author: home
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

import tensorflow as tf

def _crop(image, offset_height, offset_width, crop_height, crop_width):
  """Crops the given image using the provided offsets and sizes.
  Note that the method doesn't assume we know the input image size but it does
  assume we know the input image rank.
  Args:
    image: an image of shape [height, width, channels].
    offset_height: a scalar tensor indicating the height offset.
    offset_width: a scalar tensor indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.
  Returns:
    the cropped (and resized) image.
  Raises:
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
      less than the crop size.
  """
  
  original_shape = tf.shape(image)
  #(tf.rank(image), 3)
  rank_assertion = tf.Assert(
      tf.equal(tf.rank(image), 3),
      ['Rank of image must be equal to 3.'])
  #first execute the control dependencies
  with tf.control_dependencies([rank_assertion]):
    cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

  size_assertion = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
      ['Crop size greater than the image size.'])

  offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

  # Use tf.strided_slice instead of crop_to_bounding box as it accepts tensors
  # to define the crop size.
  with tf.control_dependencies([size_assertion]):
    #begin = [offset_height, offset_width, 0]
    #end = [offset_height+crop_height, offset_width+crop_width, 3]
    #strides = [1,1,1]
    image = tf.strided_slice(image, offsets, offsets + cropped_shape,
                             strides=tf.ones_like(offsets))
  return tf.reshape(image, cropped_shape)


def _central_crop(image_list, crop_height, crop_width):
  """Performs central crops of the given image list.
  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.
  Returns:
    the list of cropped images.
  """
  outputs = []
  for image in image_list:
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    offset_height = (image_height - crop_height) / 2
    offset_width = (image_width - crop_width) / 2

    outputs.append(_crop(image, offset_height, offset_width,
                         crop_height, crop_width))
  return outputs

def _random_crop(image_list, crop_height, crop_width):
  """Performs random crops of the given image list.
  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.
  Returns:
    the list of cropped images.
  """
  outputs = []


  for image in image_list:
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    h = image_height - crop_height + 1
    w = image_width - crop_width + 1
    offset_height = tf.random_uniform([], minval = 0, maxval = h, dtype=tf.int32)
    offset_width = tf.random_uniform([], minval = 0, maxval = w, dtype=tf.int32)

    outputs.append(_crop(image, offset_height, offset_width,
                         crop_height, crop_width))
  return outputs  
  

def _smallest_size_at_least(height, width, smallest_side):
  """Computes new shape with the smallest side equal to `smallest_side`.
  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.
  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.
  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  height = tf.to_float(height)
  width = tf.to_float(width)
  smallest_side = tf.to_float(smallest_side)

  scale = tf.cond(tf.greater(height, width),
                  lambda: smallest_side / width,
                  lambda: smallest_side / height)
  new_height = tf.to_int32(height * scale)
  new_width = tf.to_int32(width * scale)
  return new_height, new_width

def _aspect_preserving_resize(image, smallest_side):
  """Resize images preserving the original aspect ratio.
  Args:
    image: A 3-D image or a 4-D batch of images `Tensor`.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.
  Returns:
    resized_image: A 3-D or 4-D tensor containing the resized image(s).
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  input_rank = len(image.get_shape())
  if input_rank == 3:
    image = tf.expand_dims(image, 0)

  shape = tf.shape(image)
  height = shape[1]
  width = shape[2]
  new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
  resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                           align_corners=False)
  if input_rank == 3:
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
  else:
    resized_image.set_shape([None, None, None, 3])
  return resized_image
  
  

def parse_single_example_for_image(x, image_size = 286, square_crop = True, crop_size = 256):
    assert(image_size >= crop_size)
    features = tf.parse_single_example(x, features={'image_raw': tf.FixedLenFeature([], tf.string)})
    image = tf.image.decode_jpeg(features['image_raw'])
    image.set_shape([None, None, 3])
    if image_size:
        if square_crop:
            image = _aspect_preserving_resize(image, image_size)
            image = _random_crop([image], crop_size, crop_size)[0]
            image.set_shape([crop_size, crop_size, 3])
    else:
        image = _aspect_preserving_resize(image, image_size)
    
    image = tf.to_float(image)
    return image / 127.5 - 1.0