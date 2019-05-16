from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import gc
import re
import math
import gzip
import shutil
import pickle
import tarfile
from os import listdir

import numpy as np
from os.path import isdir
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from urllib.request import urlretrieve


def save_data_to_pkl(data, data_path, verbose=True):
  """data to pickle file."""
  file_size = data.nbytes
  if file_size / (2**30) > 4:
    if verbose:
      print('Saving {}...'.format(data_path))
      print('File is too large (>4Gb) for pickle to save: {:.4}Gb'.format(
          file_size / (10**9)))
    save_large_data_to_pkl(data, data_path[:-2], verbose=verbose)
  else:
    with open(data_path, 'wb') as f:
      if verbose:
        print('Saving {}...'.format(f.name))
        print('Shape: {}'.format(np.array(data).shape))
        print('Size: {:.4}Mb'.format(file_size / (10**6)))
      pickle.dump(data, f)


def save_large_data_to_pkl(data,
                           data_path,
                           max_part_size=2**31,
                           verbose=True,
                           return_n_parts=False):
  """Save large data to pickle file."""
  file_size = data.nbytes
  n_parts = file_size // max_part_size + 1
  len_part = int(((len(data) // n_parts) // 2048) * 2048)
  n_parts = int(len(data) // len_part + 1)

  if verbose:
    print('Saving {} into {} parts...'.format(data_path, n_parts))
    print('Total Size: {:.4}Gb'.format(file_size / (10**9)))
    print('Data Shape: ', data.shape)

  for i in range(n_parts):
    if i == n_parts - 1:
      data_part = data[i * len_part:]
    else:
      data_part = data[i * len_part:(i + 1) * len_part]
    data_path_i = data_path + '_{}.p'.format(i)
    with open(data_path_i, 'wb') as f:
      if verbose:
        file_size_part = data_part.nbytes
        print('Saving {}... Part Size: {:.4}Gb'.format(
            f.name, file_size_part / (10**9)))
        print('Part Shape: ', data_part.shape)
      pickle.dump(data_part, f)
    del data_part
    gc.collect()

  if return_n_parts:
    return n_parts


def load_pkls(dir_path, file_name, verbose=True):
  """Load data from pickle file or files."""
  indices = []
  for f_name in listdir(dir_path):
    m = re.match(file_name + '_(\d*).p', f_name)
    if m:
      indices.append(int(m.group(1)))
  if indices:
    return load_large_data_from_pkl(
        '{}/{}'.format(dir_path, file_name),
        n_parts=len(indices),
        verbose=verbose)
  else:
    return load_data_from_pkl(
        '{}/{}.p'.format(dir_path, file_name), verbose=verbose)


def load_data_from_pkl(data_path, verbose=True):
  """Load data from pickle file."""
  with open(data_path, 'rb') as f:
    if verbose:
      print('Loading {}...'.format(f.name))
    return pickle.load(f)


def load_large_data_from_pkl(data_path, n_parts=2, verbose=True):
  """Save large data to pickle file."""
  if verbose:
    print('Loading {}.p from {} parts...'.format(data_path, n_parts))
  data = None
  for i in range(n_parts):
    data_path_i = data_path + '_{}.p'.format(i)
    with open(data_path_i, 'rb') as f:
      if verbose:
        print('Loading {}...'.format(f.name))
      if i == 0:
        data = pickle.load(f)
      else:
        data = np.concatenate([data, pickle.load(f)], axis=0)

  if verbose:
    print('Total Size: {:.4}Gb'.format(data.nbytes / (10**9)))
    print('Data Shape: ', data.shape)

  return data


def check_dir(path_list):
  """Check if directories exit or not."""
  for dir_path in path_list:
    if not isdir(dir_path):
      os.makedirs(dir_path)


def thin_line():
  print('-' * 55)


def thick_line():
  print('=' * 55)


def _read32(bytestream):
  """Read 32-bit integer from bytesteam.

  Args:
    bytestream: A bytestream

  Returns:
    32-bit integer
  """
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_image(save_path, extract_path):
  """Extract the images into a 4D unit8 numpy array [index, y, x, depth]."""
  # Get data from save_path
  with open(save_path, 'rb') as f:

    print('Extracting {}...'.format(f.name))

    with gzip.GzipFile(fileobj=f) as bytestream:

      magic = _read32(bytestream)
      if magic != 2051:
        raise ValueError(
            'Invalid magic number {} in file: {}'.format(magic, f.name))
      num_images = _read32(bytestream)
      rows = _read32(bytestream)
      cols = _read32(bytestream)
      buf = bytestream.read(rows * cols * num_images)
      data = np.frombuffer(buf, dtype=np.uint8)
      data = data.reshape(num_images, rows, cols, 1)
      save_data_to_pkl(data, extract_path + '.p')


def extract_labels_mnist(save_path, extract_path):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  # Get data from save_path
  with open(save_path, 'rb') as f:

    print('Extracting {}...'.format(f.name))

    with gzip.GzipFile(fileobj=f) as bytestream:

      magic = _read32(bytestream)
      if magic != 2049:
        raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                         (magic, f.name))
      num_items = _read32(bytestream)
      buf = bytestream.read(num_items)
      labels = np.frombuffer(buf, dtype=np.uint8)
      save_data_to_pkl(labels, extract_path + '.p')


def download_and_extract_mnist(url, save_path, extract_path, data_type):

  if not os.path.exists(save_path):
    with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
      urlretrieve(url, save_path, pbar.hook)

  try:
    if data_type == 'images':
      extract_image(save_path, extract_path)
    elif data_type == 'labels':
      extract_labels_mnist(save_path, extract_path)
    else:
      raise ValueError('Wrong data_type!')
  except Exception as err:
    # Remove extraction folder if there is an error
    shutil.rmtree(extract_path)
    raise err

  # Remove compressed data
  os.remove(save_path)


def load_cifar10_batch(dataset_path, mode, batch_id=None):
  """Load a batch of the dataset."""
  if mode == 'train':
    with open(dataset_path + '/data_batch_' + str(batch_id),
              mode='rb') as file:
      batch = pickle.load(file, encoding='latin1')
  elif mode == 'test':
    with open(dataset_path + '/test_batch',
              mode='rb') as file:
      batch = pickle.load(file, encoding='latin1')
  else:
    raise ValueError('Wrong mode!')

  features = batch['data'].reshape(
      (len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
  labels = batch['labels']

  return features, np.array(labels)


def download_and_extract_cifar10(url, save_path, file_name, extract_path):

  archive_save_path = os.path.join(save_path, file_name)
  extracted_dir_path = os.path.join(save_path, 'cifar-10-batches-py')

  if not os.path.exists(os.path.join(save_path, 'cifar10')):
      with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
        urlretrieve(url, archive_save_path, pbar.hook)
  else:
    raise ValueError('Files already exist!')

  try:
    if not os.path.exists(extracted_dir_path):
      with tarfile.open(archive_save_path) as tar:
        tar.extractall(extract_path)
        tar.close()
  except Exception as err:
    # Remove extraction folder if there is an error
    shutil.rmtree(extract_path)
    raise err

  # Extract images and labels from batches
  features = []
  labels = []
  for batch_i in range(1, 6):
    features_i, labels_i = load_cifar10_batch(
        extracted_dir_path, 'train', batch_i)
    features.append(features_i)
    labels.append(labels_i)
  train_images = np.concatenate(features, axis=0)
  train_labels = np.concatenate(labels, axis=0)
  test_images, test_labels = load_cifar10_batch(
      extracted_dir_path, 'test')

  # Save concatenated images and labels to pickles
  pickle_path = os.path.join(save_path, 'cifar10')
  check_dir([pickle_path])
  save_data_to_pkl(train_images, pickle_path + '/train_images.p')
  save_data_to_pkl(train_labels, pickle_path + '/train_labels.p')
  save_data_to_pkl(test_images, pickle_path + '/test_images.p')
  save_data_to_pkl(test_labels, pickle_path + '/test_labels.p')

  # Remove compressed data
  os.remove(archive_save_path)
  shutil.rmtree(extracted_dir_path)


def square_grid_show_imgs(images, mode=None):
  """Save images as a square grid."""
  # Get maximum size for square grid of images
  img_size = math.floor(np.sqrt(images.shape[0]))

  # Scale to 0-255
  images = (
        ((images - images.min()) * 255) / (images.max() - images.min())).astype(
    np.uint8)

  # Put images in a square arrangement
  images_in_square = np.reshape(
      images[:img_size * img_size],
      (img_size, img_size, images.shape[1], images.shape[2], images.shape[3]))

  # images_in_square.shape = (5, 5, 28, 28, 1)

  if mode == 'L':
    cmap = 'gray'
    images_in_square = np.squeeze(images_in_square, 4)
  else:
    cmap = None

  # Combine images to grid image
  new_im = Image.new(mode,
                     (images.shape[1] * img_size, images.shape[2] * img_size))
  for row_i, row_images in enumerate(images_in_square):
    for col_i, image in enumerate(row_images):
      im = Image.fromarray(image, mode)
      new_im.paste(im, (col_i * images.shape[1], row_i * images.shape[2]))

  plt.imshow(np.array(new_im), cmap=cmap)
  plt.show()


class DLProgress(tqdm):
  """Handle Progress Bar while Downloading."""
  last_block = 0

  def hook(self, block_num=1, block_size=1, total_size=None):
    """
    A hook function that will be called once on establishment of the network
    connection and once after each block read thereafter.

    Args:
      block_num: A count of blocks transferred so far
      block_size: Block size in bytes
      total_size: The total size of the file. This may be -1 on older FTP
                  servers which do not return a file size in response to a
                  retrieval request.
    """
    self.total = total_size
    self.update((block_num - self.last_block) * block_size)
    self.last_block = block_num
