# Modified version of "Coming Down to Earth: Satellite-to-Street View Synthesis for Geo-Localization" paper

import torch.utils.data as data
import scipy.io as sio
import numpy as np
import cv2

class CVACT(data.Dataset):
    def __init__(self, root, mat_list, isTrain=False, transform_op=None):
        self.train = isTrain
        self.transform_op = transform_op
        self.posDistThr = 25
        self.posDistSqThr = self.posDistThr * self.posDistThr
        self.root = root
        self.mat_list = mat_list

        #LOAD MAT FILE
        idx = 0
        self.all_list, self.all_list_idx = [], []
        self.all_data = sio.loadmat(self.root + self.mat_list)

        for i in range(0, len(self.all_data['panoIds'])):
            grd_id_ori = self.root + '_' + self.all_data['panoIds'][i] + '/' + self.all_data['panoIds'][i] + '_zoom_2.jpg'
            grd_id_align = self.root + 'grd/' + self.all_data['panoIds'][i] + '_grdView.jpg'
            grd_id_ori_sem = self.root + '_' + self.all_data['panoIds'][i] + '/' + self.all_data['panoIds'][i] + '_zoom_2_sem.jpg'
            grd_id_align_sem = self.root + '_' + self.all_data['panoIds'][i] + '/' + self.all_data['panoIds'][i] + '_zoom_2_aligned_sem.jpg'
            sat_id_ori = self.root + 'sat/' + self.all_data['panoIds'][i] + '_satView_polish.jpg'
            sat_id_sem = self.root + '_' + self.all_data['panoIds'][i] + '/' + self.all_data['panoIds'][i] + '_satView_sem.jpg'
            self.all_list.append([grd_id_ori, grd_id_align, grd_id_ori_sem,
                                  grd_id_align_sem, sat_id_ori, sat_id_sem,
                                  self.all_data['utm'][i][0], self.all_data['utm'][i][1]])
            self.all_list_idx.append(idx)
            idx += 1
        self.all_data_size = len(self.all_list)

        #PARTITION IMAGES INTO CELL
        self.utms_all = np.zeros([2, self.all_data_size], dtype=np.float32)
        for i in range(0, self.all_data_size):
            self.utms_all[0, i] = self.all_list[i][6]
            self.utms_all[1, i] = self.all_list[i][7]
        if self.train:
            self.data_inds = self.all_data['trainSet']['trainInd'][0][0] - 1
        else:
            self.data_inds = self.all_data['valSet']['valInd'][0][0] - 1
        self.dataNum = len(self.data_inds)
        self.dataList = []
        self.dataIdList = []
        self.dataUTM = np.zeros([2, self.dataNum], dtype=np.float32)
        for k in range(self.dataNum):
            self.dataList.append(self.all_list[self.data_inds[k][0]])
            self.dataUTM[:, k] = self.utms_all[:, self.data_inds[k][0]]
            self.dataIdList.append(k)
        self.data_list_size = len(self.dataList)
        print('Load data from {}, total {}'.format(self.mat_list, self.data_list_size))


    def load_im(self, im_path, resize=None):
        im = cv2.imread(im_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if resize:
            im = cv2.resize(im, resize, interpolation=cv2.INTER_AREA)
        im = np.array(im, dtype=np.float32)
        return im


    def __getitem__(self, index):
        sate_im = self.load_im(im_path=self.dataList[index][4])
        pano_im = self.load_im(im_path=self.dataList[index][1])
        utm = self.dataUTM[:, index]
        img_data = {'satellite': sate_im,
                     'street': pano_im}
        if self.transform_op:
            img_data = self.transform_op(img_data)
        return img_data, utm


    def __len__(self):
        return self.data_list_size
