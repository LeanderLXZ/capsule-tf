from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2 as cv
import random
import argparse
from os.path import join
import sklearn.utils

from models.utils import *
from config import config
from experiments.baseline_config import config as basel_cfg


class GenerateMNISTN(object):

  def __init__(self, source_data_path, save_path, cfg):
    self.source_data_path = source_data_path
    self.save_path = join(save_path, cfg.MN_DATABASE_NAME)
    self.cfg = cfg
    self.train_images = None
    self.train_labels = None
    self.test_images = None
    self.test_labels = None

  def _load_data(self):

    self.train_images = load_data_from_pkl(
        join(self.source_data_path, 'train_images.p'))
    self.train_labels = load_data_from_pkl(
        join(self.source_data_path, 'train_labels.p'))
    self.test_images = load_data_from_pkl(
        join(self.source_data_path, 'test_images.p'))
    self.test_labels = load_data_from_pkl(
        join(self.source_data_path, 'test_labels.p'))

  @staticmethod
  def _img_shear(img):
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

  @staticmethod
  def _img_rotate(image, rotate_range=(-10, 10)):
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

  @staticmethod
  def _cal_margin(image_, images, height, width):
    """cal the suitable pos of images."""
    height_available = height - image_.shape[0]
    total_width = 0
    for image_i in images:
      total_width += image_i.shape[1]
    width_available = (width - total_width) // 2
    return height_available // 2, width_available

  def _img_resize(self, image, images):
    height_, width_ = self._cal_margin(image, images, 56, 56)
    s = random.randint(0, min(height_, width_))
    dst = cv.resize(
        image, (image.shape[1] + s, image.shape[0] + s),
        interpolation=cv.INTER_CUBIC)
    return dst

  def _merge(self, images, img_size=(56, 56)):
    """_merge images."""
    img_size_m = (56, 56)
    target = Image.new('L', img_size_m)
    left = 0
    for image in images:
      h, w = self._cal_margin(image, images, *img_size_m)
      x = random.randint(0, w)
      y = random.randint(0, h)
      target.paste(
          Image.fromarray(image.astype('uint8')).convert('L'),
          box=((x + left), y, (
              x + left + image.shape[1]), (y + image.shape[0])))
      left += x + image.shape[1]

    if tuple(img_size) != img_size_m:
      target = img_resize([target],
                          img_size,
                          img_mode='L',
                          resize_filter=Image.ANTIALIAS)
    return target

  @staticmethod
  def _choose_img(images, labels, num):
    # num [0-9]
    index_array = np.argwhere(labels == num)
    r = np.random.choice(index_array.reshape((index_array.shape[0])), 1)
    return images[r]

  def _data_gen(self,
                images,
                labels,
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
    # num : numbers of every class
    images_list = []
    labels_list = []
    count = 0
    # choose image and process
    for i in range(10):
      for j in range(10):
        if (i * 10 + j) in range(num_range[0], num_range[1]+1):
          print('Generating: ', i * 10 + j)
          for _ in tqdm(range(num_in_class), ncols=100):
            img1 = self._choose_img(
                images, labels, i).reshape((28, 28))
            img2 = self._choose_img(
                images, labels, j).reshape((28, 28))

            if rotate_range:
              img1 = self._img_rotate(img1, rotate_range=rotate_range)
              img2 = self._img_rotate(img2, rotate_range=rotate_range)

            img1_sheared = self._img_shear(img1)
            img2_sheared = self._img_shear(img2)

            imgs_for_merge = [img1_sheared, img2_sheared]
            merged_img = np.array(self._merge(imgs_for_merge, img_size=img_size))

            images_list.append(merged_img)
            labels_list.append(i * 10 + j)
            count += 1

    images = np.expand_dims(np.array(images_list, dtype=int), axis=-1)
    labels = np.array(labels_list, dtype=int)

    return images, labels

  def _shuffle(self, seed=None):

    print('Shuffling images and labels...')
    self.train_images, self.train_labels = sklearn.utils.shuffle(
        self.train_images, self.train_labels, random_state=seed)
    self.test_images, self.test_labels = sklearn.utils.shuffle(
        self.test_images, self.test_labels, random_state=seed)

  @staticmethod
  def _save_csv(file_path):
    title = ['num', 'loop', 'turn', 'cross']
    for i in range(10):
      title.append(i)

    list_ = []
    csv_file = open(join(file_path, 'features.csv'), 'a', newline='')
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

  def _save_data(self):
    save_data_to_pkl(self.train_images, join(self.save_path, 'train_images.p'))
    save_data_to_pkl(self.train_labels, join(self.save_path, 'train_labels.p'))
    save_data_to_pkl(self.test_images, join(self.save_path, 'test_images.p'))
    save_data_to_pkl(self.test_labels, join(self.save_path, 'test_labels.p'))

  def pipeline(self):

    start_time = time.time()

    check_dir([self.save_path])

    self._load_data()

    self.train_images, self.train_labels = self._data_gen(
        self.train_images,
        self.train_labels,
        num_in_class=self.cfg.MN_NUM_IN_CLASS_TRAIN,
        num_range=self.cfg.MN_NUM_RANGE,
        img_size=self.cfg.MN_IMAGE_SIZE,
        rotate_range=self.cfg.MN_ROTATE_RANGE
    )
    self.test_images, self.test_labels = self._data_gen(
        self.test_images,
        self.test_labels,
        num_in_class=self.cfg.MN_NUM_IN_CLASS_TEST,
        num_range=self.cfg.MN_NUM_RANGE,
        img_size=self.cfg.MN_IMAGE_SIZE,
        rotate_range=self.cfg.MN_ROTATE_RANGE
    )

    self._shuffle()

    self._save_data()
    
    self._save_csv(self.save_path)

    print('Done! Using {:.4}s'.format(time.time() - start_time))



if __name__ == '__main__':

  source_data_path_ = '../data/source_data/mnist'
  save_path_ = '../data/source_data/'

  parser = argparse.ArgumentParser(description='Testing the model.')
  parser.add_argument('-b', '--baseline', action='store_true',
                      help='Use baseline configurations.')
  args = parser.parse_args()

  if args.baseline:
    cfg_ = basel_cfg
  else:
    cfg_ = config

  GenerateMNISTN(source_data_path=source_data_path_,
                 save_path=save_path_,
                 cfg=cfg_).pipeline()
