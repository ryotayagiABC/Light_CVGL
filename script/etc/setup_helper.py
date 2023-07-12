# Modified version of "Coming Down to Earth: Satellite-to-Street View Synthesis for Geo-Localization" paper

from collections import OrderedDict
import torch
import random
import numpy as np
import psutil
import os

# Check array/tensor size
mem_size_of = lambda a: a.element_size() * a.nelement()
gb = lambda bs: bs / 2. ** 30

def get_sys_mem():
    p = psutil.Process()
    pmem = p.memory_info()
    return gb(pmem.rss), gb(pmem.vms)


def load_weights(weights_dir, device, key='state_dict'):
    map_location = lambda storage, loc: storage.cuda(device.index) if torch.cuda.is_available() else storage
    weights_dict = None
    if weights_dir is not None:
        weights_dict = torch.load(weights_dir, map_location=map_location)
    return weights_dict


def lprint(ms, log=None):
    '''Print message on console and in a log file'''
    print(ms)
    if log:
        log.write(ms + '\n')
        log.flush()


def make_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 


def config2str(config):
    print_ignore = ['weights_dict', 'optimizer_dict']
    args = vars(config)
    separator = '\n'
    confstr = ''
    confstr += '------------ Configuration -------------{}'.format(separator)
    for k, v in sorted(args.items()):
        if k in print_ignore:
            if v is not None:
                confstr += '{}:{}{}'.format(k, len(v), separator)
            continue
        confstr += '{}:{}{}'.format(k, str(v), separator)
    confstr += '----------------------------------------{}'.format(separator)
    return confstr


def mutual_topk_acc(self, dists, topk=1):
    pos_dists = torch.diag(dists)
    N = len(pos_dists)
    # Distances smaller than positive pair
    dist_s2p = pos_dists.unsqueeze(1) - dists
    dist_p2s = pos_dists - dists
    acc_s2p = 100.0 * ((dist_s2p > 0).sum(1) < topk).sum().float() / N
    acc_p2s = 100.0 * ((dist_p2s > 0).sum(0) < topk).sum().float() / N
    return acc_p2s.item(), acc_s2p.item()


def compute_loss(self, sate_vecs, pano_vecs, utms_x=None, UTMthres=None, loss_weight=10, hard_topk_ratio=1.0):
    dists = 2 - 2 * torch.matmul(sate_vecs, pano_vecs.permute(1, 0))  # Pairwise matches within batch
    pos_dists = torch.diag(dists)
    N = len(pos_dists)
    diag_ids = np.arange(N)
    if self.opt.CVACT == True:
        useful_pairs = torch.ge(utms_x[:,:], UTMthres)
        useful_pairs = useful_pairs.float()
        pair_n = useful_pairs.sum()
        num_hard_triplets = int(hard_topk_ratio * (N * (N - 1))) if int(hard_topk_ratio * (N * (N - 1))) < pair_n else pair_n

        # Match from satellite to street pano
        triplet_dist_s2p = (pos_dists.unsqueeze(1) - dists) * useful_pairs
        loss_s2p = torch.log(1 + torch.exp(loss_weight * triplet_dist_s2p))
        loss_s2p[diag_ids, diag_ids] = 0
        if num_hard_triplets != pair_n:
            loss_s2p = loss_s2p.view(-1)
            loss_s2p, s2p_ids = torch.topk(loss_s2p, num_hard_triplets)
        loss_s2p = loss_s2p.sum() / num_hard_triplets

        # Match from street pano to satellite
        triplet_dist_p2s = (pos_dists - dists) * useful_pairs
        loss_p2s = torch.log(1 + torch.exp(loss_weight * triplet_dist_p2s))
        loss_p2s[diag_ids, diag_ids] = 0
        if num_hard_triplets != pair_n:
            loss_p2s = loss_p2s.view(-1)
            loss_p2s, p2s_ids = torch.topk(loss_p2s, num_hard_triplets)

    else:
        num_hard_triplets = int(hard_topk_ratio * (N * (N - 1))) if hard_topk_ratio < 1.0 else N * (N - 1)

        # Match from satellite to street pano
        triplet_dist_s2p = pos_dists.unsqueeze(1) - dists
        loss_s2p = torch.log(1 + torch.exp(loss_weight * triplet_dist_s2p))
        loss_s2p[diag_ids, diag_ids] = 0  # Ignore diagnal losses

        if hard_topk_ratio < 1.0:  # Hard negative mining
            loss_s2p = loss_s2p.view(-1)
            loss_s2p, s2p_ids = torch.topk(loss_s2p, num_hard_triplets)
        loss_s2p = loss_s2p.sum() / num_hard_triplets

        # Match from street pano to satellite
        triplet_dist_p2s = pos_dists - dists
        loss_p2s = torch.log(1 + torch.exp(loss_weight * triplet_dist_p2s))
        loss_p2s[diag_ids, diag_ids] = 0  # Ignore diagnal losses

        if hard_topk_ratio < 1.0:  # Hard negative mining
            loss_p2s = loss_p2s.view(-1)
            loss_p2s, p2s_ids = torch.topk(loss_p2s, num_hard_triplets)
    loss_p2s = loss_p2s.sum() / num_hard_triplets
    loss = (loss_s2p + loss_p2s) / 2.0
    return loss


def save_networks(self, epoch, out_dir, count, last_ckpt=False, best_acc=None, is_best=False):
    print('self.device', self.device)
    ckpt = {'last_epoch': epoch,
            'best_acc': best_acc,
            'count' : count,
            'retrieval_model_dict': self.retrieval.state_dict(),
            'optimizer_dict': self.optimizer.state_dict(),
            }

    if last_ckpt:
        print('SAVING LAST NETWORK on DEVICE {}'.format(self.device))
        ckpt_name = 'last_ckpt_{}.pth'.format(str(self.device).replace(':', ''))
    elif is_best:
        print('best_acc', best_acc)
        print('SAVING BEST NETWORK on DEVICE {}'.format(self.device))
        ckpt_name = 'best_ep{}_ckpt_{}_{}.pth'.format(epoch, np.round(best_acc,3), str(self.device).replace(':', ''))
    else:
        ckpt_name = 'ckpt_ep{}_{}.pth'.format(self.epoch + 1, str(self.device).replace(':', ''))
    ckpt_path = os.path.join(out_dir, ckpt_name)
    print('ckpt_path', ckpt_path)
    torch.save(ckpt, ckpt_path)


def load_networks(self):
    if self.opt.checkpoint is None:
        self.initial_epoch = 0
        self.count = 0
        return
    else:
        ckpt_path = self.opt.checkpoint
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        self.ret_best_acc = ckpt['best_acc']
        self.initial_epoch = ckpt['last_epoch'] + 1
        self.count = ckpt['count']

        # Load net state
        print('ckpt.keys', ckpt.keys())
        retrieval_dict = ckpt['retrieval_model_dict']
        self.retrieval.load_state_dict(retrieval_dict)
        optimizer = ckpt['optimizer_dict']
        self.optimizer = optimizer











