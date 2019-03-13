# for pix2pix
import os
import time
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html

# ------------------------- pytorch_pix2pix API ----------------------------
class Pix2Pix():
    def __init__(self, opt):

        # hard-code some parameters for api
        self.opt = opt
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
        opt.loadSize = 256
        opt.fineSize = 256
        opt.epoch = 'latest'
        opt.phase = "test"
        opt.results_dir = './'

        self.data_loader = CreateDataLoader(opt)
        
        self.model = create_model(opt)
        self.model.setup(opt)
        
        # create a website
        web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
        self.webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

        # test with eval mode. This only affects layers like batchnorm and dropout.
        # pix2pix: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
        self.model.eval()
        

    def gen_fake(self):
        dataset = self.data_loader.load_data()
        for i, data in enumerate(dataset):
            self.model.set_input(data)
            self.model.test()
            visuals = self.model.get_current_visuals()
            img_path = self.model.get_image_paths()
            save_images(self.webpage, visuals, img_path, aspect_ratio=self.opt.aspect_ratio, width=self.opt.display_winsize)
