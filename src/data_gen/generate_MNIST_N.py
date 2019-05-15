from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2 as cv
import os
import random
import pickle
import csv
import time
from PIL import Image
from tqdm import tqdm
from os.path import join, isdir


def img_shear(img):
  """cut the image to fit the number."""
  height = img.shape[0]
  width = img.shape[1]
  top, bottom, left, right = height, 0, width, 0
  # find ROI
  for row in range(height):
    for col in range(width):
      if img[row, col] > 100:
        if row <= top:
          top = row
        if row >= bottom:
          bottom = row
        if col >= right:
          right = col
        if col <= left:
          left = col
  cropped = img[top - 1:bottom + 1, left - 1:right + 1]
  return cropped


def img_rotate(image, rotate_range=(-10, 10)):
  (h, w) = image.shape[:2]
  (cX, cY) = (w // 2, h // 2)
  angle = random.uniform(*rotate_range)
  # grab the rotation matrix (applying the negative of the
  # angle to rotate clockwise), then grab the sine and cosine
  # (i.e., the rotation components of the matrix)
  m = cv.getRotationMatrix2D((cX, cY), angle, 1.0)
  cos = np.abs(m[0, 0])
  sin = np.abs(m[0, 1])

  # compute the new bounding dimensions of the image
  nw = int((h * sin) + (w * cos))
  nh = int((h * cos) + (w * sin))

  # adjust the rotation matrix to take into account translation
  m[0, 2] += (nw / 2) - cX
  m[1, 2] += (nh / 2) - cY

  # perform the actual rotation and return the image
  return cv.warpAffine(image, m, (nw, nh), borderValue=(0, 0, 0))


def cal_margin(image_, images, height, width):
  """cal the suitable pos of images."""
  height_available = height - image_.shape[0]
  total_width = 0
  for image_i in images:
    total_width += image_i.shape[1]
  width_available = (width - total_width) // 2
  return height_available // 2, width_available


def img_resize(img, images):
  height_, width_ = cal_margin(img, images, 56, 56)
  s = random.randint(0, min(height_, width_))
  dst = cv.resize(
      img, (img.shape[1] + s, img.shape[0] + s), interpolation=cv.INTER_CUBIC)
  return dst


def merge(images, img_size=(56, 56)):
  """merge images."""
  target = Image.new('L', img_size)
  left = 0
  for image in images:
    h, w = cal_margin(image, images, *img_size)
    x = random.randint(0, w)
    y = random.randint(0, h)
    target.paste(
        Image.fromarray(image.astype('uint8')).convert('L'),
        box=((x + left), y, (x + left + image.shape[1]), (y + image.shape[0])))
    left += x + image.shape[1]
  return target


def csv_gen():
  title = ['num', 'loop', 'turn', 'cross']
  for i in range(10):
    title.append(i)

  list_ = []
  csv_file = open('./feature.csv', 'a', newline='')
  writer = csv.writer(csv_file, dialect='excel')
  writer.writerow(title)
  for i in range(100):
    list_.append(i)
    # feature 1
    if (str(0) in str(i)) or (str(6) in str(i)) or \
            (str(8) in str(i)) or (str(9) in str(i)):
      list_.append(1)
    else:
      list_.append(0)
    # feature 2
    if (str(2) in str(i)) or (str(4) in str(i)) or \
            (str(5) in str(i)) or (str(7) in str(i)):
      list_.append(1)
    else:
      list_.append(0)
    # feature 3
    if str(4) in str(i):
      list_.append(1)
    else:
      list_.append(0)
    # nums
    for j in range(10):
      if str(j) in str(i):
        list_.append(1)
      else:
        list_.append(0)
    writer.writerow(list_)
    list_ = []
  csv_file.close()


def load_data_from_pkl(data_path,  verbose=True):
  """Load data from pickle file."""

  with open(data_path, 'rb') as f:
    if verbose:
      print('Loading {}...'.format(f.name))
    return pickle.load(f)


def save_data_to_pkl(data, data_path, verbose=True):
  """data to pickle file."""
  file_size = data.nbytes
  with open(data_path, 'wb') as f:
    if verbose:
      print('Saving {}...'.format(f.name))
      print('Shape: {}'.format(np.array(data).shape))
      print('Size: {:.4}Mb'.format(file_size / (10**6)))
    pickle.dump(data, f)


def load_img_array(file_path):
  train_images = load_data_from_pkl(join(file_path, 'train_images.p'))
  train_labels = load_data_from_pkl(join(file_path, 'train_labels.p'))
  test_images = load_data_from_pkl(join(file_path, 'test_images.p'))
  test_labels = load_data_from_pkl(join(file_path, 'test_labels.p'))
  
  return train_images, train_labels, test_images, test_labels


def class_img(train_images, train_labels, num):
  # num [0-9]
  index_array = np.argwhere(train_labels == num)
  r = np.random.choice(index_array.reshape((index_array.shape[0])), 1)
  return train_images[r]


def data_gen(train_images,
             train_labels,
             num_in_class,
             num_range=(10, 99),
             img_size=(56, 56),
             rotate_range=(-10, 10)):
  # choose img
  #   |
  # random rotate
  #   |
  # shear
  #   |
  #
  # num : numbers of every class
  images = []
  labels = []
  count = 0
  # choose image and process
  for i in range(10):
    for j in range(10):
      if (i * 10 + j) in range(num_range[0], num_range[1]+1):
        print('Generating: ', i * 10 + j)
        for _ in tqdm(range(num_in_class), ncols=100):
          img1 = class_img(train_images, train_labels, i).reshape((28, 28))
          img2 = class_img(train_images, train_labels, j).reshape((28, 28))
          img1_rotated = img_rotate(img1, rotate_range=rotate_range)
          img2_rotated = img_rotate(img2, rotate_range=rotate_range)
          img1_sheared = img_shear(img1_rotated)
          img2_sheared = img_shear(img2_rotated)
          images_list = [img1_sheared, img2_sheared]
          merged_img = np.array(merge(images_list, img_size=img_size))
          images.append(merged_img)
          labels.append(i * 10 + j)
          count += 1
  return np.array(images, dtype=int), np.array(labels, dtype=int)


if __name__ == '__main__':

  start_time = time.time()

  source_data_path = '../data/source_data/mnist'
  save_path = '../data/source_data/mnist_100'

  if not isdir(save_path):
    os.makedirs(save_path)

  train_images_, train_labels_, \
      test_images_, test_labels_ = load_img_array(source_data_path)

  train_images_new, train_labels_new = data_gen(
      train_images_,
      train_labels_,
      num_in_class=5000,
      num_range=(10, 99),
      img_size=(56, 56),
      rotate_range=(-10, 10)
  )
  test_images_new, test_labels_new = data_gen(
      test_images_,
      test_labels_,
      num_in_class=1000,
      num_range=(10, 99),
      img_size=(56, 56),
      rotate_range=(-10, 10)
  )

  save_data_to_pkl(train_images_new, join(save_path, 'train_images.p'))
  save_data_to_pkl(train_labels_new, join(save_path, 'train_labels.p'))
  save_data_to_pkl(test_images_new, join(save_path, 'test_images.p'))
  save_data_to_pkl(test_labels_new, join(save_path, 'test_labels.p'))

  # import matplotlib.pyplot as plt
  #   # with open(join(save_path, 'train_images.p'), 'rb') as f_imgs:
  #   #     train_imgs_ = pickle.load(f_imgs)
  #   # with open(join(save_path, 'train_labels.p'), 'rb') as f_labels:
  #   #     train_labels_ = pickle.load(f_labels)
  #   # for i_ in range(20):
  #   #     plt.subplot(4, 5, i_ + 1)
  #   #     plt.imshow(train_imgs_[i_])
  #   # print(train_labels_[0:20])

  print('Done! Using {:.4}s'.format(time.time() - start_time))
