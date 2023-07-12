# Modified version of "Coming Down to Earth: Satellite-to-Street View Synthesis for Geo-Localization" paper

from data.custom_transforms import *
import torch
import cv2
import os

class CVUSA(torch.utils.data.Dataset):
    def __init__(self, root, csv_file, sate_size=(256, 256), pano_size=(512,128), use_polar=False, name=None, transform_op=None):
        self.root = root
        self.name = name
        self.use_polar = use_polar
        self.sate_size = pano_size if use_polar else sate_size
        self.pano_size = pano_size if use_polar else pano_size
        self.csv_path = os.path.join(root, csv_file)
        self.transform_op = transform_op

        # Load image list
        csv_path = os.path.join(root, 'splits', csv_file)
        with open(csv_path, 'r') as f:
            lines = f.readlines()
            pano_ims, sate_ims, item_ids, pano_ann = [], [], [], []
            for line in lines:
                items = line.strip().split(',')
                item_id = (items[0].split('/')[-1]).split('.')[0]
                a,b,c = items[0].split("/")
                sate_ims.append(c)
                item_ids.append(item_id)
                pano_ims.append(c.replace('.jpg', '.png'))
                pano_ann.append(items[2])
        self.pano_ims, self.sate_ims, self.pano_ann, self.item_ids = pano_ims, sate_ims, pano_ann, item_ids
        self.num = len(self.pano_ims)
        print('Load data from {}, total {}'.format(csv_path, self.num))


    @classmethod
    def load_im(self, im_path, resize=None):
        im = cv2.imread(im_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if resize:
            im = cv2.resize(im, resize, interpolation=cv2.INTER_AREA)
        im = np.array(im, dtype=np.float32)
        return im


    def __getitem__(self, index):
        # Triplet construction
        pos_id = index
        sate_path = os.path.join(self.root + 'sat/', self.sate_ims[pos_id])
        pano_path = os.path.join(self.root + 'grd/', self.pano_ims[pos_id])
        # Load and process images
        sate_im = self.load_im(sate_path)
        pano_im = self.load_im(pano_path)
        sample = {'satellite': sate_im, 'street': pano_im}
        if self.transform_op:
            sample = self.transform_op(sample)
        sample['im_path'] = (sate_path, pano_path)
        sample['item_id'] = self.item_ids[pos_id]
        return sample


    def __len__(self):
        return self.num


    def __repr__(self):
        fmt_str = 'CVUSA \n'
        fmt_str += 'Pair cvs path: {}\n'.format(self.csv_path)
        fmt_str += 'Number of data pairs: {}\n'.format(self.__len__())
        fmt_str += 'Dataset root : {}\n'.format(self.root)
        fmt_str += 'Image Transforms: {}\n'.format(self.transform_op.__repr__().replace('\n', '\n    '))
        return fmt_str


