from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import gc
import re
import math
import pickle
import argparse
from PIL import Image
import numpy as np
import sklearn.utils
from copy import copy
from tqdm import tqdm
from os import listdir
from os.path import join
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

from models import utils
from config import config as cfg
from experiments.baseline_config import config as basel_cfg
from models.get_transfer_learning_codes import GetBottleneckFeatures

from keras.preprocessing.image import ImageDataGenerator
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))


class DataPreProcess(object):

  def __init__(self,
               config,
               seed=None,
               data_base_name=None,
               tl_encode=False,
               data_type=np.float16,
               show_img=False):
    """
    Preprocess data and save as pickle files.

    Args:
      config: configuration
    """
    self.cfg = config
    self.seed = seed
    self.data_base_name = data_base_name
    self.preprocessed_path = None
    self.source_data_path = None
    self.show_img = show_img
    self.data_type = data_type

    # Use encode transfer learning
    if tl_encode:
      self.tl_encode = True
    else:
      self.tl_encode = False

    if 'mnist' in self.data_base_name:
      self.input_size = (28, 28)
      self.img_mode = 'L'
    elif self.data_base_name == 'cifar10':
      self.input_size = (32, 32)
      self.img_mode = 'RGB'
    else:
      raise ValueError('Wrong database name!')

    if self.cfg.RESIZE_IMAGES:
      self.image_size = self.cfg.IMAGE_SIZE
    else:
      self.image_size = self.input_size

    if self.cfg.RESIZE_INPUTS:
      self.input_size = self.cfg.INPUT_SIZE

  def _load_data(self):
    """Load data set from files."""
    utils.thin_line()
    print('Loading {} data set...'.format(self.data_base_name))

    self.x = utils.load_pkls(
        self.source_data_path, 'train_images').astype(self.data_type)
    self.y = utils.load_pkls(self.source_data_path, 'train_labels')
    self.x_test = utils.load_pkls(
        self.source_data_path, 'test_images').astype(self.data_type)
    self.y_test = utils.load_pkls(self.source_data_path, 'test_labels')

    # Data augment
    if self.cfg.USE_DATA_AUG:
      utils.thin_line()
      print('Augmenting data...'.format(self.data_base_name))

      x_y_dict = self._get_x_y_dict(self.x, self.y)
      x_new = []
      y_new = []
      for y_ in tqdm(x_y_dict.keys(),
                     ncols=100,
                     unit=' class'):
        x_ = x_y_dict[y_]
        x_ = self._augment_data(
            x_,
            self.cfg.DATA_AUG_PARAM,
            img_num=self.cfg.MAX_IMAGE_NUM,
            add_self=self.cfg.DATA_AUG_KEEP_SOURCE)
        x_new.append(x_)
        y_new.extend([int(y_) for _ in range(len(x_))])

      if self.tl_encode:
        self.imgs = self.x
        self.imgs_test = self.x_test

      self.x = np.array(
          x_new, dtype=self.data_type).reshape((-1, *self.x[0].shape))
      self.y = np.array(y_new, dtype=np.int)

    if self.show_img:
      self._grid_show_imgs(self.x, self.y, 25, mode='L')

  @staticmethod
  def _get_x_y_dict(x, y, y_encoded=False):
    """Get y:x dictionary."""
    if y_encoded:
      # [[1, 0, ..., 0], ..., [0, 1, ..., 0]] -> [1, ..., 2]
      y = [np.argmax(y_) for y_ in y]
    classes = set(y)
    x_y_dict = {c: [] for c in classes}
    for idx, y_ in enumerate(y):
      x_y_dict[y_].append(x[idx])
    return x_y_dict

  def _augment_data(self, tensor, data_aug_param, img_num, add_self=True):
    """Augment data set and add noises."""
    data_generator = ImageDataGenerator(**data_aug_param)
    if add_self:
      new_x_tensors = copy(tensor)
    else:
      new_x_tensors = []
    while True:
      for i in range(len(tensor)):
        if len(new_x_tensors) >= img_num:
          return np.array(new_x_tensors, dtype=self.data_type)
        augmented = data_generator.random_transform(
            tensor[i].astype(np.float32))
        new_x_tensors.append(augmented)

  def _train_test_split(self):
    """Split data set for training and testing."""
    utils.thin_line()
    print('Splitting train/test set...')
    self.x, self.x_test, self.y, self.y_test = train_test_split(
        self.x,
        self.y,
        test_size=self.cfg.TEST_SIZE,
        shuffle=True,
        random_state=self.seed
    )

  def _scaling(self):
    """Scaling input images to (0, 1)."""
    utils.thin_line()
    print('Scaling features...')

    self.x = np.divide(self.x, 255.).astype(self.data_type)
    self.x_test = np.divide(self.x_test, 255.).astype(self.data_type)

  def _one_hot_encoding(self):
    """One-hot-encoding labels."""
    utils.thin_line()
    print('One-hot-encoding labels...')

    encoder = LabelBinarizer()
    encoder.fit(self.y)
    self.y = encoder.transform(self.y)
    self.y_test = encoder.transform(self.y_test)

  def _shuffle(self):
    """Shuffle data sets."""
    utils.thin_line()
    print('Shuffling images and labels...')
    self.x, self.y = sklearn.utils.shuffle(
        self.x, self.y, random_state=self.seed)
    self.x_test, self.y_test = sklearn.utils.shuffle(
        self.x_test, self.y_test, random_state=self.seed)

  def _generate_multi_obj_img(self,
                              x_y_dict=None,
                              data_aug=False):
    """Generate images of superpositions of multi-objects"""
    utils.thin_line()
    print('Generating images of superpositions of multi-objects...')
    self.x_test_mul = []
    self.y_test_mul = []
    y_list = list(x_y_dict.keys())

    for _ in tqdm(range(self.cfg.NUM_MULTI_IMG), ncols=100, unit=' images'):
      # Get images for merging
      if self.cfg.REPEAT:
        # Repetitive labels
        mul_img_idx_ = np.random.choice(
            len(self.x_test), self.cfg.NUM_MULTI_OBJECT, replace=False)
        mul_imgs = list(self.x_test[mul_img_idx_])
        mul_y = [0 if y_ == 0 else 1 for y_ in np.sum(
            self.y_test[mul_img_idx_], axis=0)]
      else:
        # No repetitive labels
        y_list_ = np.random.choice(
            y_list, self.cfg.NUM_MULTI_OBJECT, replace=False)
        mul_imgs = []
        mul_y = []
        for y_ in y_list_:
          x_ = x_y_dict[y_]
          x_ = x_[np.random.choice(len(x_))]
          mul_imgs.append(x_)
          mul_y.append(y_)
        mul_y = [1 if i in mul_y else 0 for i in range(len(x_y_dict.keys()))]

      # Data augment
      if data_aug:
        mul_imgs = np.array(self._augment_data(
            mul_imgs,
            self.cfg.DATA_AUG_PARAM,
            img_num=len(mul_imgs),
            add_self=False))

      # Merge images
      if self.cfg.OVERLAP:
        mul_imgs = utils.img_add_overlap(mul_imgs, merge=False, gamma=0)
      else:
        mul_imgs = utils.img_add_no_overlap(
            mul_imgs, self.cfg.NUM_MULTI_OBJECT,
            img_mode=self.img_mode, resize_filter=Image.ANTIALIAS)

      self.x_test_mul.append(mul_imgs)
      self.y_test_mul.append(mul_y)

    self.x_test_mul = np.array(self.x_test_mul).astype(self.data_type)
    self.y_test_mul = np.array(self.y_test_mul).astype(self.data_type)

    if self.show_img:
      y_show = np.argsort(
          self.y_test_mul, axis=1)[:, -self.cfg.NUM_MULTI_OBJECT:]
      self._grid_show_imgs(self.x_test_mul, y_show, 25, mode='L')

  def _train_valid_split(self):
    """Split data set for training and validation"""
    utils.thin_line()
    print('Splitting train/valid set...')

    if self.cfg.VALID_SIZE > 1:
      train_stop = len(self.x) - self.cfg.VALID_SIZE
    else:
      train_stop = len(self.x) - int(len(self.x) * self.cfg.VALID_SIZE)
    if self.cfg.DPP_TEST_AS_VALID:
      self.x_train = self.x
      self.y_train = self.y
      self.x_valid = self.x_test
      self.y_valid = self.y_test
    else:
      self.x_train = self.x[:train_stop]
      self.y_train = self.y[:train_stop]
      self.x_valid = self.x[train_stop:]
      self.y_valid = self.y[train_stop:]

  def _resize_imgs(self, imgs, img_size, mode):
    imgs = utils.img_resize(utils.imgs_scale_to_255(imgs),
                            img_size,
                            img_mode=mode,
                            resize_filter=Image.ANTIALIAS
                            ).astype(self.data_type)
    for i in range(len(imgs)):
      imgs[i] = imgs[i] / 255.
    return imgs.astype(self.data_type)

  @staticmethod
  def _grid_show_imgs(x, y, n_img_show, mode='L'):
    sample_idx_ = np.random.choice(
        len(y), n_img_show, replace=False)
    utils.square_grid_show_imgs(x[sample_idx_], mode=mode)
    y_show = y[sample_idx_]
    size = math.floor(np.sqrt(n_img_show))
    print(y_show.reshape(size, size, -1))

  def _save_images(self):
    """Get and save images"""

    img_shape = self.x_train.shape[1:3]

    if tuple(img_shape) != tuple(self.image_size):
      utils.thin_line()
      print('Resizing images...')
      print('Before: {}'.format(tuple(img_shape)))
      print('After: {}'.format(tuple(self.image_size)))

      self.imgs_train = self._resize_imgs(
          self.x_train, self.image_size, self.img_mode)
      self.imgs_valid = self._resize_imgs(
          self.x_valid, self.image_size, self.img_mode)
      self.imgs_test = self._resize_imgs(
          self.x_test, self.image_size, self.img_mode)
      if self.cfg.NUM_MULTI_OBJECT:
        self.imgs_test_mul = self._resize_imgs(
            self.x_test_mul, self.image_size, self.img_mode)
    else:
      self.imgs_train = self.x_train.astype(self.data_type)
      self.imgs_valid = self.x_valid.astype(self.data_type)
      self.imgs_test = self.x_test.astype(self.data_type)
      if self.cfg.NUM_MULTI_OBJECT:
        self.imgs_test_mul = self.x_test_mul.astype(self.data_type)

    if self.show_img:
      utils.square_grid_show_imgs(np.array(
          self.imgs_train[:25], dtype=self.data_type), mode=self.img_mode)
      if self.cfg.NUM_MULTI_OBJECT:
        utils.square_grid_show_imgs(np.array(
            self.imgs_test_mul[:25], dtype=self.data_type), mode=self.img_mode)

    # Save data to pickle files
    utils.thin_line()
    print('Saving images files...')
    utils.check_dir([self.preprocessed_path])
    utils.save_data_to_pkl(
        self.imgs_train, join(self.preprocessed_path, 'imgs_train.p'))
    utils.save_data_to_pkl(
        self.imgs_valid, join(self.preprocessed_path, 'imgs_valid.p'))
    utils.save_data_to_pkl(
        self.imgs_test, join(self.preprocessed_path, 'imgs_test.p'))

    if self.cfg.NUM_MULTI_OBJECT:
      utils.save_data_to_pkl(
          self.imgs_test_mul,
          join(self.preprocessed_path, 'imgs_test_multi_obj.p'))
      del self.imgs_test_mul

    del self.imgs_train
    del self.imgs_valid
    del self.imgs_test
    gc.collect()

  def _resize_inputs(self):
    """Resize input data"""
    img_shape = self.x_train.shape[1:3]

    if tuple(img_shape) != tuple(self.input_size):
      utils.thin_line()
      print('Resizing inputs...')
      print('Before: {}'.format(tuple(img_shape)))
      print('After: {}'.format(tuple(self.input_size)))

      self.x_train = self._resize_imgs(
          self.x_train, self.input_size, self.img_mode)
      self.x_valid = self._resize_imgs(
          self.x_valid, self.input_size, self.img_mode)
      self.x_test = self._resize_imgs(
          self.x_test, self.input_size, self.img_mode)
      if self.cfg.NUM_MULTI_OBJECT:
        self.x_test_mul = self._resize_imgs(
            self.x_test_mul, self.input_size, self.img_mode)

    if self.show_img:
      utils.square_grid_show_imgs(np.array(
          self.x_train[:25], dtype=self.data_type), mode=self.img_mode)
      if self.cfg.NUM_MULTI_OBJECT:
        utils.square_grid_show_imgs(np.array(
            self.x_test_mul[:25], dtype=self.data_type), mode=self.img_mode)

  def _check_data(self):
    """Check data format."""
    utils.thin_line()
    print('Checking data shapes...')
    assert self.x_train.max() <= 1, self.x_train.max()
    assert self.y_train.max() <= 1, self.y_train.max()
    assert self.x_valid.max() <= 1, self.x_valid.max()
    assert self.y_valid.max() <= 1, self.y_valid.max()
    assert self.x_test.max() <= 1, self.x_test.max()
    assert self.y_test.max() <= 1, self.y_test.max()

    assert self.x_train.min() >= 0, self.x_train.min()
    assert self.y_train.min() >= 0, self.y_train.min()
    assert self.x_valid.min() >= 0, self.x_valid.min()
    assert self.y_valid.min() >= 0, self.y_valid.min()
    assert self.x_test.min() >= 0, self.x_test.min()
    assert self.y_test.min() >= 0, self.y_test.min()

    if self.data_base_name == 'mnist':
      n_classes = 10
      input_size = (*self.input_size, 1)
    elif 'mnist' in self.data_base_name:
      n_classes = len(range(*self.cfg.MN_NUM_RANGE)) + 1
      input_size = (*self.input_size, 1)
    elif self.data_base_name == 'cifar10':
      n_classes = 10
      input_size = (*self.input_size, 3)
    else:
      raise ValueError('Wrong database name!')

    train_num = len(self.x_train)
    test_num = len(self.x_test)
    valid_num = len(self.x_valid)

    assert self.x_train.shape == (train_num, *input_size), self.x_train.shape
    assert self.y_train.shape == (train_num, n_classes), self.y_train.shape
    assert self.x_valid.shape == (valid_num, *input_size), self.x_valid.shape
    assert self.y_valid.shape == (valid_num, n_classes), self.y_valid.shape
    assert self.x_test.shape == (test_num, *input_size), self.x_test.shape
    assert self.y_test.shape == (test_num, n_classes), self.y_test.shape

    if self.cfg.NUM_MULTI_OBJECT:
      assert self.x_test_mul.max() <= 1, self.x_test_mul.max()
      assert self.y_test_mul.max() <= 1, self.y_test_mul
      assert self.x_test_mul.min() >= 0, self.x_test_mul.min()
      assert self.y_test_mul.min() >= 0, self.y_test_mul
      assert self.x_test_mul.shape == (self.cfg.NUM_MULTI_IMG, *input_size), \
          self.x_test_mul.shape
      assert self.y_test_mul.shape == (self.cfg.NUM_MULTI_IMG, n_classes), \
          self.y_test_mul.shape

  def _save_cache_data(self):
    """Save cache data for transfer learning."""
    max_part_size = 2**30

    def _save_data(data, data_dir, data_name):
      if data.nbytes > max_part_size:
        print('{} is too large!'.format(data_name))
        utils.save_large_data_to_pkl(
            data, join(data_dir, data_name), max_part_size=max_part_size)
      else:
        utils.save_data_to_pkl(data, join(data_dir, data_name) + '.p')

    # Get bottleneck features
    utils.thin_line()
    print('Saving cache data for transfer learning...')

    # Save x_train
    _save_data(self.x_train, self.preprocessed_path, 'x_train_cache')
    del self.x_train
    gc.collect()

    _save_data(self.x_valid, self.preprocessed_path, 'x_valid_cache')
    _save_data(self.x_test, self.preprocessed_path, 'x_test_cache')

    if self.cfg.NUM_MULTI_OBJECT:
      _save_data(self.x_test_mul, self.preprocessed_path,
                 'x_test_multi_obj_cache')

  def _save_data(self):
    """Save data set to pickle files."""
    utils.thin_line()
    print('Saving inputs files...')
    utils.check_dir([self.preprocessed_path])

    if self.tl_encode:
      self._save_cache_data()
    else:
      utils.save_data_to_pkl(
          self.x_train, join(self.preprocessed_path, 'x_train.p'))
      utils.save_data_to_pkl(
          self.x_valid, join(self.preprocessed_path, 'x_valid.p'))
      utils.save_data_to_pkl(
          self.x_test, join(self.preprocessed_path, 'x_test.p'))
      if self.cfg.NUM_MULTI_OBJECT:
        utils.save_data_to_pkl(
            self.x_test_mul, join(self.preprocessed_path, 'x_test_multi_obj.p'))

    utils.save_data_to_pkl(
        self.y_train, join(self.preprocessed_path, 'y_train.p'))
    utils.save_data_to_pkl(
        self.y_valid, join(self.preprocessed_path, 'y_valid.p'))
    utils.save_data_to_pkl(
        self.y_test, join(self.preprocessed_path, 'y_test.p'))
    if self.cfg.NUM_MULTI_OBJECT:
      utils.save_data_to_pkl(
          self.y_test_mul, join(self.preprocessed_path, 'y_test_multi_obj.p'))

  def pipeline(self):
    """Pipeline of preprocessing data."""
    utils.thick_line()
    print('Start preprocessing...')

    start_time = time.time()

    self.preprocessed_path = join(self.cfg.DPP_DATA_PATH, self.data_base_name)
    self.source_data_path = join(self.cfg.SOURCE_DATA_PATH, self.data_base_name)

    # Load data
    self._load_data()

    # Scaling images to (0, 1)
    self._scaling()

    # One-hot-encoding labels
    self._one_hot_encoding()

    # Shuffle data set
    # self._shuffle()

    # Generate multi-objects test images
    if self.cfg.NUM_MULTI_OBJECT:
      x_y_dict = self._get_x_y_dict(self.x_test, self.y_test, y_encoded=True)
      self._generate_multi_obj_img(
          x_y_dict=x_y_dict, data_aug=False)

    # Split data set into train/valid
    self._train_valid_split()

    # Save images
    self._save_images()

    # Resize images and inputs
    self._resize_inputs()

    # Check data format
    self._check_data()

    # Save data to pickles
    self._save_data()

    utils.thin_line()
    print('Done! Using {:.4}s'.format(time.time() - start_time))
    utils.thick_line()


def get_and_save_bf(config,
                    dir_path,
                    cache_file_name,
                    file_name,
                    bf_batch_size=128,
                    pooling='avg',
                    data_type=np.float32):
  utils.thin_line()
  print('Calculating bottleneck features of {}...'.format(file_name))

  indices = []
  for f_name in listdir(dir_path):
    m = re.match(cache_file_name + '_(\d*).p', f_name)
    if m:
      indices.append(int(m.group(1)))
  if indices:
    for i in np.sort(indices).tolist():
      part_path = join(dir_path, cache_file_name + '_{}.p'.format(i))
      print('Get bottleneck features of {}_{}.p'.format(cache_file_name, i))
      with open(part_path, 'rb') as f:
        data_part = pickle.load(f)
        print('Data cache shape: ', data_part.shape)
        GetBottleneckFeatures(
            config.TL_MODEL).save_bottleneck_features(
            data_part,
            file_path=join(dir_path, '{}_{}.p'.format(file_name, i)),
            batch_size=bf_batch_size,
            pooling=pooling,
            data_type=data_type)
        del data_part
        gc.collect()
      os.remove(part_path)
  else:
    part_path = join(dir_path, cache_file_name + '.p')
    print('Get bottleneck features of {}.p'.format(cache_file_name))
    with open(part_path, 'rb') as f:
      data_part = pickle.load(f)
      GetBottleneckFeatures(
          config.TL_MODEL).save_bottleneck_features(
          data_part,
          file_path=join(dir_path, '{}.p'.format(file_name)),
          batch_size=bf_batch_size,
          pooling=pooling,
          data_type=data_type)
    os.remove(part_path)


def save_bottleneck_features(config,
                             data_base_name,
                             mul_imgs=False):
  utils.thick_line()
  print('Start calculating bottleneck features...')
  start_time = time.time()

  pooling = config.BF_POOLING
  dir_path = join(config.DPP_DATA_PATH, data_base_name)
  get_and_save_bf(config, dir_path, 'x_train_cache',
                  'x_train', pooling=pooling)
  get_and_save_bf(config, dir_path, 'x_valid_cache',
                  'x_valid', pooling=pooling)
  get_and_save_bf(config, dir_path, 'x_test_cache',
                  'x_test', pooling=pooling)

  if mul_imgs:
    get_and_save_bf(config, dir_path,
                    'x_test_multi_obj_cache',
                    'x_test_multi_obj',
                    pooling=pooling)

  utils.thick_line()
  print('Done! Using {:.4}s'.format(time.time() - start_time))
  utils.thick_line()


if __name__ == '__main__':

  global_seed = None

  parser = argparse.ArgumentParser(
      description='Testing the model.'
  )
  parser.add_argument('-b', '--baseline', action='store_true',
                      help='Use baseline configurations.')
  parser.add_argument('-m', '--mnist', action='store_true',
                      help='Preprocess the MNIST database.')
  parser.add_argument('-c', '--cifar', action='store_true',
                      help='Preprocess the CIFAR-10 database.')
  parser.add_argument('-t1', '--tl1', action='store_true',
                      help='Save transfer learning cache data.')
  parser.add_argument('-t2', '--tl2', action='store_true',
                      help='Get transfer learning bottleneck features.')
  parser.add_argument('-si', '--show_img', action='store_true',
                      help='Get transfer learning bottleneck features.')
  parser.add_argument('-ft', '--fine_tune', action="store_true",
                      help="Fine-tuning.")

  args = parser.parse_args()

  show_img_flag = True if args.show_img else False
  mul_imgs_flag = True if cfg.NUM_MULTI_OBJECT else False

  cfg_ = cfg
  if args.baseline:
    utils.thick_line()
    print('Running baseline model.')
    cfg_ = basel_cfg
    database_name_ = basel_cfg.DATABASE_NAME
  elif args.mnist:
    utils.thick_line()
    print('Preprocess the MNIST database.')
    database_name_ = 'mnist'
  elif args.cifar:
    utils.thick_line()
    print('Preprocess the CIFAR-10 database.')
    database_name_ = 'cifar10'
  else:
    utils.thick_line()
    print('Preprocess the MNIST database.')
    database_name_ = 'mnist'

  if args.fine_tune:
    database_name_ = cfg_.FT_DATABASE_NAME

  if args.tl1:
    DataPreProcess(config=cfg_,
                   seed=global_seed,
                   data_base_name=database_name_,
                   tl_encode=True,
                   show_img=show_img_flag).pipeline()

  elif args.tl2:
    save_bottleneck_features(cfg,
                             data_base_name=database_name_,
                             mul_imgs=mul_imgs_flag)
  else:
    DataPreProcess(config=cfg_,
                   seed=global_seed,
                   data_base_name=database_name_,
                   show_img=show_img_flag).pipeline()
