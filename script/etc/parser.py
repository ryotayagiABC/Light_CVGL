# Modified version of "Coming Down to Earth: Satellite-to-Street View Synthesis for Geo-Localization" paper

import argparse
import os
import torch
import datetime

class Parser():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        #basic parameters
        parser.add_argument('--results_dir', '-o', type=str, default='./RESULTS', help='models are saved here')
        parser.add_argument('--name', type=str, default='', help='')
        parser.add_argument('--seed', type=int, default=10)
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1')
        parser.add_argument('--isTrain', default=True, action='store_true')
        parser.add_argument('--start_epoch', type=int, default=0)
        parser.add_argument('--debug', default=False, action='store_true')
        parser.add_argument('--test', default=False, action='store_true')

        #data parameters
        parser.add_argument('--CVUSA', default=False, action='store_true')
        parser.add_argument('--CVACT', default=False, action='store_true')
        parser.add_argument('--data_root_cvact', type=str, default= '')
        parser.add_argument('--data_root_cvusa', type=str, default= '')
        parser.add_argument('--train_csv', type=str, default='train-19zl.csv')
        parser.add_argument('--val_csv', type=str, default='val-19zl.csv')
        parser.add_argument('--mat_list', type=str, default= 'ACT_data.mat')
        parser.add_argument('--save_step', type=int, default=10)
        parser.add_argument('--checkpoint', type=str, default=None)
        parser.add_argument("--sat_size_h", type=float, default=256, help="satellite image size")
        parser.add_argument("--sat_size_w", type=float, default=256, help="satellite image size")
        parser.add_argument("--grd_size_h", type=float, default=128, help="satellite image size")
        parser.add_argument("--grd_size_w", type=float, default=512, help="satellite image size")
        parser.add_argument("--sa_num", type=float, default=8, help="attention num")

        #train parameters
        parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of combined training")
        parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
        parser.add_argument("--batch_size_val", type=int, default=32, help="size of the validation batches")
        parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
        parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

        #loss parameters
        parser.add_argument("--lambda_sm", type=int, default=10, help="loss weight for soft margin")
        parser.add_argument("--hard_topk_ratio", type=float, default=1.0, help="hard negative ratio")
        parser.add_argument("--hard_decay1_topk_ratio", type=float, default=0.1, help="hard negative ratio")
        parser.add_argument("--hard_decay2_topk_ratio", type=float, default=0.05, help="hard negative ratio")
        parser.add_argument("--hard_decay3_topk_ratio", type=float, default=0.01, help="hard negative ratio")

        self.initialized = True
        return parser


    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        opt, _ = parser.parse_known_args()
        self.parser = parser
        return parser.parse_args()


    def print_options(self, opt):
        # save to the disk
        dt_now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
        dt_now = str(dt_now.year) + '_' + str(dt_now.month) + '_' + str(dt_now.day) + '_' + str(dt_now.hour) + '_' + str(dt_now.minute) + '_' + str(dt_now.second)

        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        prefix ='{}_lr{}_batch{}_HN_{}_HN1decay_{}HN2decay_{}HN3decay_{}'.format(dt_now, opt.lr, opt.batch_size,  opt.hard_topk_ratio, opt.hard_decay1_topk_ratio, opt.hard_decay2_topk_ratio, opt.hard_decay3_topk_ratio)

        out_dir = os.path.join(opt.results_dir, prefix)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        file_name = os.path.join(out_dir, 'log.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
            opt_file.flush()
        return file_name


    def parse(self):
        opt = self.gather_options()
        file = self.print_options(opt)
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])
        self.opt = opt
        return self.opt, file


    def log(self, ms, log=None):
        print(ms)
        if log:
            log.write(ms + '\n')
            log.flush()
