from etc import parser
from etc.setup_helper import *
from data.custom_transforms import *
from data.cvact_utils import CVACT
from data.cvusa_utils import CVUSA
from networks import safa
from os.path import exists, join, basename, dirname
from argparse import Namespace
from tqdm import tqdm
from tensorboardX import SummaryWriter
import datetime
import pytorch_lightning as pl
import time

class LightCNN(pl.LightningModule):
    
    def __init__(self, opt, log_file):
        super().__init__()
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.model_names = []
        self.save_dir = dirname(log_file)
        posDistThr = 25
        self.posDistSqThr = posDistThr * posDistThr
        self.ret_best_acc = 0.0
        self.retrieval = safa.SAFA(sa_num=self.opt.sa_num, H1=self.opt.sat_size_h, W1=self.opt.sat_size_w, H2=self.opt.grd_size_h, W2=self.opt.grd_size_w)
        load_networks(self)        
        self.street_batches_v = []
        self.satellite_batches_v = []

        
    ######### DATALOADER #########
    def train_dataloader(self):
        composed_transforms = transforms.Compose([ToTensor()])
        if self.opt.CVACT == True:
            print('CVACT')
            train_dataset = CVACT(root=self.opt.data_root_cvact, mat_list = opt.mat_list, isTrain=opt.isTrain, transform_op=ToTensor())
        else:
            print('CVUSA')
            train_dataset = CVUSA(root=self.opt.data_root_cvusa, csv_file=opt.train_csv, transform_op=ToTensor())
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
        return train_loader
    

    def val_dataloader(self):
        if self.opt.CVACT == True:
            print('CVACT')
            val_dataset = CVACT(root=self.opt.data_root_cvact, mat_list = opt.mat_list, isTrain=False, transform_op=ToTensor())
        else:
            print('CVUSA')
            val_dataset = CVUSA(root=self.opt.data_root_cvusa, csv_file=opt.val_csv, name=opt.name, transform_op=ToTensor())
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size_val, shuffle=False, num_workers=0)
        return val_loader
    

    def test_dataloader(self):
        if self.opt.CVACT == True:
            val_dataset = CVACT(root=self.opt.data_root_cvact, mat_list = opt.mat_list, isTrain=False, transform_op=ToTensor())
        else:
            val_dataset = CVUSA(root=self.opt.data_root_cvusa, csv_file=opt.val_csv, name=opt.name, transform_op=ToTensor())
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.batch_size_val, shuffle=False, num_workers=0)
        return val_loader
    

    ######### CALLING DATA ######### 
    def set_input_cvact(self, data, utm):
        sat_ims = data['satellite']
        grd_ims = data['street']
        self.satellite = sat_ims
        self.street = grd_ims 
        self.in_batch_dis = torch.zeros(utm.shape[0], utm.shape[0]).to(self.device)
        for k in range(utm.shape[0]):
            for j in range(utm.shape[0]):
                self.in_batch_dis[k, j] = (utm[k,0] - utm[j,0])*(utm[k,0] - utm[j,0]) + (utm[k, 1] - utm[j, 1])*(utm[k, 1] - utm[j, 1])


    def set_input_cvusa(self, batch):
        sat_ims = batch['satellite']
        grd_ims = batch['street']
        self.satellite = sat_ims
        self.street = grd_ims
        

    ######### OPTIMIZER #########
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam([{'params': self.retrieval.parameters()}], lr=opt.lr, betas=(opt.b1, opt.b2))
        return self.optimizer 
        

    ######### TRAIN #########
    def on_train_epoch_start(self):
        #print
        self.start_time = time.time()
        self.street_batches_t = []
        self.satellite_batches_t = []
        self.epoch_retrieval_loss = []
        self.epoch = self.current_epoch + self.initial_epoch
        log_print('>>> LIGHTCNN Epoch {}'.format(self.epoch))        
        

    def training_step(self, batch, batch_idx):
        if opt.debug == True:
            if batch_idx > 41:
                return
        self.count += len(batch) 
        if self.opt.CVACT==True:
            data, utm = batch
            self.set_input_cvact(data, utm)
        else:
            data = batch
            self.in_batch_dis = None
            self.set_input_cvusa(data)
        self.satellite_out, self.street_out = self.retrieval(self.satellite, self.street)
        if self.epoch <= 20:
            self.ret_loss = compute_loss(self, self.satellite_out, self.street_out, self.in_batch_dis, self.posDistSqThr, loss_weight=self.opt.lambda_sm, hard_topk_ratio=self.opt.hard_topk_ratio) 
        elif self.epoch > 20 and self.epoch <=40:
            self.ret_loss = compute_loss(self, self.satellite_out, self.street_out, self.in_batch_dis, self.posDistSqThr, loss_weight=self.opt.lambda_sm, hard_topk_ratio=self.opt.hard_decay1_topk_ratio)
        elif self.epoch > 40 and self.epoch <=60:
            self.ret_loss = compute_loss(self, self.satellite_out, self.street_out, self.in_batch_dis, self.posDistSqThr, loss_weight=self.opt.lambda_sm, hard_topk_ratio=self.opt.hard_decay2_topk_ratio) 
        elif self.epoch > 60:
            self.ret_loss = compute_loss(self, self.satellite_out, self.street_out, self.in_batch_dis, self.posDistSqThr, loss_weight=self.opt.lambda_sm, hard_topk_ratio=self.opt.hard_decay3_topk_ratio)
        loss = self.ret_loss
        self.log('epoch', self.epoch, on_step=True, on_epoch=True, prog_bar=True)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.satellite_batches_t.append(self.satellite_out.cpu().data)
        self.street_batches_t.append(self.street_out.cpu().data)
        self.epoch_retrieval_loss.append(self.ret_loss.item())
        if (batch_idx + 1) % 50 == 0:
            satellite_vec = torch.cat(self.satellite_batches_t, dim=0)
            street_vec = torch.cat(self.street_batches_t, dim=0)
            dists = 2 - 2 * torch.matmul(satellite_vec, street_vec.permute(1, 0))
            tp1 = mutual_topk_acc(self, dists, topk=1)
            tp5 = mutual_topk_acc(self, dists, topk=5)
            log_print('Batch:{} loss={:.3f} samples:{} tp1={tp1[0]:.2f}/{tp1[1]:.2f} ' \
                      'tp5={tp5[0]:.2f}/{tp5[1]:.2f}'.format(batch_idx + 1, np.mean(self.epoch_retrieval_loss), len(dists), tp1=tp1, tp5=tp5))
            self.street_batches_t.clear()
            self.satellite_batches_t.clear()
            self.epoch_retrieval_loss.clear()
        return loss
    

    def on_train_epoch_end(self):
        save_networks(self, self.epoch, dirname(log_file), count=self.count, best_acc=self.ret_best_acc, last_ckpt=True)  
        if (self.epoch + 1) % opt.save_step == 0:
            if opt.debug == True:
                pass
            else:
                save_networks(self, self.epoch, dirname(log_file), count=self.count, best_acc=self.ret_best_acc)


    ######### VALIDATION #########
    def on_validation_epoch_start(self):
        print("VALIDATING...")
        self.street_batches_v = []
        self.satellite_batches_v = []
    

    def validation_step(self, batch, batch_idx):
        if self.opt.CVACT==True:
            data, utm = batch
            self.set_input_cvact(data, utm)
        else:
            data = batch
            self.set_input_cvusa(data)
        self.satellite_out_val, self.street_out_val = self.retrieval(self.satellite, self.street)
        self.satellite_batches_v.append(self.satellite_out_val.cpu().data)
        self.street_batches_v.append(self.street_out_val.cpu().data)
        

    def on_validation_epoch_end(self):
        satellite_vec = torch.cat(self.satellite_batches_v, dim=0)
        street_vec = torch.cat(self.street_batches_v, dim=0)
        dists = 2 - 2 * torch.matmul(satellite_vec, street_vec.permute(1, 0))
        tp1 = mutual_topk_acc(self, dists, topk=1)
        tp5 = mutual_topk_acc(self, dists, topk=5)
        tp10 = mutual_topk_acc(self, dists, topk=10)
        num = len(dists)
        tp1p = mutual_topk_acc(self, dists, topk=0.01 * num)
        acc = Namespace(num=len(dists), tp1=tp1, tp5=tp5, tp10=tp10, tp1p=tp1p)
        log_print('\nEvaluate Samples:{num:d}\nRecall(p2s/s2p) tp1:{tp1[0]:.2f}/{tp1[1]:.2f} ' \
                    'tp5:{tp5[0]:.2f}/{tp5[1]:.2f} tp10:{tp10[0]:.2f}/{tp10[1]:.2f} ' \
                    'tp1%:{tp1p[0]:.2f}/{tp1p[1]:.2f}'.format(self.epoch + 1, num=acc.num, tp1=acc.tp1, tp5=acc.tp5, tp10=acc.tp10, tp1p=acc.tp1p))
        tp1_p2s_acc = acc.tp1[0]
        if tp1_p2s_acc > self.ret_best_acc:
            self.ret_best_acc = tp1_p2s_acc
            if opt.debug == True:
                pass
            else:
                save_networks(self, self.epoch, dirname(log_file), count=self.count, best_acc=self.ret_best_acc, is_best=True)
            log_print('>>Save best model: epoch={} best_acc(tp1_p2s):{:.2f}'.format(self.epoch + 1, tp1_p2s_acc))
        self.log('tp1_p2s', tp1[0], on_epoch=True, prog_bar=True, logger=True)
        self.log('tp1_s2p', tp1[1], on_epoch=True, prog_bar=True, logger=True)
        self.log('tp5_p2s', tp5[0], on_epoch=True, prog_bar=True, logger=True)
        self.log('tp5_s2p', tp5[1], on_epoch=True, prog_bar=True, logger=True)
        self.log('tp10_p2s', tp10[0], on_epoch=True, prog_bar=True, logger=True)
        self.log('tp10_s2p', tp10[1], on_epoch=True, prog_bar=True, logger=True)
        self.log('tp1p_p2s', tp1p[0], on_epoch=True, prog_bar=True, logger=True)
        self.log('tp1p_s2p', tp1p[1], on_epoch=True, prog_bar=True, logger=True)
        # Progam stastics
        rss, vms = get_sys_mem()
        log_print('Memory usage: rss={:.2f}GB vms={:.2f}GB Time:{:.2f}s'.format(rss, vms, time.time() - self.start_time))
    

    ######### TEST #########
    def on_test_start(self):
        self.start_time = time.time()
        self.street_batches_v = []
        self.satellite_batches_v = []
    

    def test_step(self, batch, batch_idx):
        if self.opt.CVACT==True:
            data, utm = batch
            self.set_input_cvact(data, utm)
        else:
            data= batch
            self.set_input_cvusa(data)
        self.satellite_out_val, self.street_out_val = self.retrieval(self.satellite, self.street)
        self.satellite_batches_v.append(self.satellite_out_val.cpu().data)
        self.street_batches_v.append(self.street_out_val.cpu().data)
        

    def on_test_end(self):
        satellite_vec = torch.cat(self.satellite_batches_v, dim=0)
        street_vec = torch.cat(self.street_batches_v, dim=0)
        print('satellite_vec', satellite_vec.size())
        self.epoch = self.initial_epoch
        dists = 2 - 2 * torch.matmul(satellite_vec, street_vec.permute(1, 0))
        tp1 = mutual_topk_acc(self, dists, topk=1)
        tp5 = mutual_topk_acc(self, dists, topk=5)
        tp10 = mutual_topk_acc(self, dists, topk=10)
        num = len(dists)
        tp1p = mutual_topk_acc(self, dists, topk=0.01 * num)
        acc = Namespace(num=len(dists), tp1=tp1, tp5=tp5, tp10=tp10, tp1p=tp1p)
        log_print('\nEvaluate Samples:{num:d}\nRecall(p2s/s2p) tp1:{tp1[0]:.2f}/{tp1[1]:.2f} ' \
                    'tp5:{tp5[0]:.2f}/{tp5[1]:.2f} tp10:{tp10[0]:.2f}/{tp10[1]:.2f} ' \
                    'tp1%:{tp1p[0]:.2f}/{tp1p[1]:.2f}'.format(self.epoch + 1, num=acc.num, tp1=acc.tp1,
                                                            tp5=acc.tp5, tp10=acc.tp10, tp1p=acc.tp1p))
        
if __name__=='__main__':
    dt_now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    dt_now = str(dt_now.year) + '_' + str(dt_now.month) + '_' + str(dt_now.day) + '_' + str(dt_now.hour) + '_' + str(dt_now.minute) + '_' + str(dt_now.second)
    parse = parser.Parser()
    opt, log_file = parse.parse()
    opt.is_Train = True
    make_deterministic(opt.seed)
    log = open(log_file, 'a')
    log_print = lambda ms: parse.log(ms, log)
    log_print('Init model')
    model = LightCNN(opt, log_file)

    if opt.test == True:
        print('TESTING')
        trainer = pl.Trainer(accelerator='gpu',
                            devices = [opt.gpu_ids[0]],
                            max_epochs=opt.n_epochs,
                            strategy = 'ddp')
        trainer.test(model)
    else:
        tb_logger = pl.loggers.TensorBoardLogger(save_dir="logs/" + dt_now + '/')
        trainer = pl.Trainer(accelerator='gpu',
                            devices = opt.gpu_ids,
                            max_epochs=opt.n_epochs,
                            strategy = 'ddp',
                            logger=tb_logger,
                            val_check_interval=1.0,
                            num_sanity_val_steps=0,
                            deterministic=True,
                            #fast_dev_run=1, # runs only 1 training and 1 validation batch and the program ends
                            )
        trainer.fit(model)
    
    

