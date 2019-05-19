from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from os import environ

from models import utils
from experiments.pipeline_config import *
from experiments.pipeline_arch import model_arch
from main import Main


def pipeline(cfg_list, architecture, mode):

  global_start_time = time.time()

  for i, cfg in enumerate(cfg_list):
    start_time = time.time()

    utils.thick_line()
    print('Training task: {}/{}'.format(i+1, len(cfg_list)))
    model = Main(cfg, architecture, mode)
    model.train()

    utils.thick_line()
    print('Task done! Using {:.4}s'.format(time.time() - start_time))
    print('Total time {:.4}s'.format(time.time() - global_start_time))
    utils.thick_line()

  utils.thick_line()
  print('All Task done! Using {:.4}s'.format(time.time() - global_start_time))
  utils.thick_line()

if __name__ == '__main__':

  environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

  parser = argparse.ArgumentParser(
      description="Training the model."
  )
  parser.add_argument('-g', '--gpu', nargs=1,
                      choices=[0, 1], type=int, metavar='',
                      help="Run single-gpu version."
                           "Choose the GPU from: {!s}".format([0, 1]))
  parser.add_argument('-m', '--mgpu', action="store_true",
                      help="Run multi-gpu version.")
  args = parser.parse_args()

  if args.mgpu:
    utils.thick_line()
    print('Running multi-gpu version.')
    mode_ = 'multi-gpu'
  elif args.gpu:
    utils.thick_line()
    print('Running single version. Using /gpu: %d' % args.gpu[0])
    environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu[0])
    mode_ = 'single-gpu'
  else:
    utils.thick_line()
    print('Running normal version.')
    mode_ = 'normal'

  cfg_list_ = [cfg_0, cfg_1, cfg_2, cfg_3, cfg_4, cfg_5, cfg_6, cfg_7, cfg_8]

  pipeline(cfg_list_, model_arch, mode_)
