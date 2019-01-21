### pix2pix网络的API，支持单张pose stick figure的输入并生成real图，供外部调用。
import torch
import random
from PIL import Image
from models import create_model
from data import CreateDataLoader
from options.test_options import TestOptions
import torchvision.transforms as transforms


### API
# @input1: opt
def Pix2PixAPI(opt):

    # hard-code some parameters for api
    opt.name = 'pbug_pix2pix'
    opt.model = 'pix2pix'       # required!!! why? unknown.
    opt.dataroot = './datasets/pbug_full'
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1  # no visdom display
    opt.isTrain = False
    # opt.gpu_ids = '-1'
    opt.loadSize = 512
    opt.fineSize = 512
    opt.epoch = 'latest'
    opt.results_dir = './'
    
    model = create_model(opt)
    model.setup(opt)
    # 图片格式转换为张量
    # data = imgDataLoader(opt, poseStick)

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # pix2pix: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        model.set_input(data)
        model.test()

def gen_fake():
    opt = TestOptions().parse()
    Pix2PixAPI(opt)


if __name__ == '__main__':
    gen_fake()
