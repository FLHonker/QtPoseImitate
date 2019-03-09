### pix2pix网络的API，支持单张pose stick figure的输入并生成real图，供外部调用。
import os
import time
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html


### API
# @input1: opt
def Pix2PixAPI(opt):

    # hard-code some parameters for api
    opt.name = 'pbug_pix2pix_result'
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
    opt.phase = "test"
    opt.results_dir = './'
    
    model = create_model(opt)
    model.setup(opt)
    
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    # test with eval mode. This only affects layers like batchnorm and dropout.
    # pix2pix: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    model.eval()
    while True:
        data_loader = CreateDataLoader(opt)
        dataset = data_loader.load_data()
        for i, data in enumerate(dataset):
            if i >= opt.num_test:
                break
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()
            print('processing (%04d)-th image... %s' % (i, img_path))
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        time.sleep(0.1)

def gen_fake():
    opt = TestOptions().parse()
    Pix2PixAPI(opt)


if __name__ == '__main__':
    gen_fake()
