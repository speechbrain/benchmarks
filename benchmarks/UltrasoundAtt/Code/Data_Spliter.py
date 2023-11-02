import numpy as np  # linear algebra

import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random
#import torchaudio.transforms as Ta
from scipy.io import loadmat,savemat
import sys
import pickle

import numpy as np  # linear algebra

import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import random
#import torchaudio.transforms as Ta
from scipy.io import loadmat
import sys
import seaborn as sns
from tabulate import tabulate
#import data_class_real as D_real
#import SincFiles as SincF
from torch.utils.data import Subset
from orion.client import report_objective
import torch.optim.lr_scheduler as lr_scheduler




def load_ultrasound(ULTRA_PATH):
    dic = {}
    data_dic = loadmat(ULTRA_PATH)
    try:
        dic['rf_data'] = data_dic['rf_data'].reshape((-1,))
        dic['rf_env'] = data_dic['rf_env'].reshape((-1,))
        dic['my_att'] = data_dic['my_att'][0][0]
    except:
        dic['rf_data'] = data_dic['rf_data'].reshape((-1,))
        dic['my_att'] = data_dic['my_att'].item()
        dic['rf_env'] = 0
        #return None
    #print(dic['my_att'],ULTRA_PATH)
    return dic['rf_data'] , dic['rf_env'], dic['my_att']

def collate_fn(batch):
    batch = list(filter (lambda x:x is not None, batch)) # filter out all the Nones
    return torch.utils.data.dataloader.default_collate(batch)


class UltrasoundDataset(Dataset):
    def __init__(self, path, feature_transform=None,
                 label_transform=None, train=True,
                 train_size=0.80):
        self.path = path
        self.file_list = []
        self.label_list = []
        self.feature_transform = feature_transform
        self.label_transform = label_transform
        for dirname, _, filenames in os.walk(path):
            for filename in filenames:
                if filename[-3:] != "txt":
                    self.file_list.append(os.path.join(dirname, filename))

        total_len = len(self.file_list)
        if train:
            self.file_list, self.label_list = self.file_list[:int(0.80 * total_len)], self.label_list[
                                                                                      :int(0.80 * total_len)]
        else:
            self.file_list, self.label_list = self.file_list[int(0.80 * total_len):], self.label_list[
                                                                                      int(0.80 * total_len):]

    def __getitem__(self, idx):
        try:
                rf_data, rf_env,attenuation = load_ultrasound(self.file_list[idx])
                len_wav = rf_data.shape[0]
                pddd = 4500 #9000

                #print()
                if len_wav < pddd:
                    pad = np.zeros(pddd - len_wav)
                    rf_data = np.hstack([rf_data, pad])
                    rf_env = np.hstack([rf_env, pad])
                elif len_wav > pddd:
                    rf_data = rf_data[:pddd]
                return rf_data,attenuation,self.file_list[idx]
        except:
            return [],0,''

    def __len__(self):
        return len(self.file_list)






def load_ultrasound2(ULTRA_PATH):
    dic = {}
    data_dic = loadmat(ULTRA_PATH)
    try:
        dic['rf_data'] = data_dic['rf_data'][0]
        dic['my_att'] = data_dic['my_att'][0][0]
    except:
        dic['rf_data'] = data_dic['rf_data'][0]
        dic['my_att'] = data_dic['my_att'].item()
    return dic['rf_data'] , dic['my_att']

def real_data_decomposer(ULTRA_PATH,elem_num=50,frame_num=10):
    """" Decomposes data to the standard format of the generated data with FIELD2"""
    dic = {}

    print(ULTRA_PATH.split('/'))
    for dirname, _, filenames in os.walk(ULTRA_PATH):
        if ('Uniform' in dirname.split('/')):
            for filename in filenames:
                if filename[0:4] == "Norm" and filename[-3:] !='txt' :
                    for i in range(frame_num):
                        for j in range(elem_num):
                            data_dic = loadmat(os.path.join(dirname, filename))
                            dic['rf_data'] = data_dic['RFSampleFrames'][:,j,i]#.reshape((data_dic['RFSampleFrames'].shape[0],-1))
                            #print(dic['rf_data'])
                            dic['my_att'] = 0.654
                            dic['Fs'] = 40
                            if not os.path.exists(os.path.join(dirname, 'Decomposed/')):
                                os.makedirs(os.path.join(dirname, 'Decomposed/'))
                            rf_data_mat_name = "rf_data_"+"elem_"+str(j)+"_fra_"+str(i)+'_'+ filename
                            print(os.path.join(dirname, 'Decomposed/',rf_data_mat_name))
                            savemat( os.path.join(dirname, 'Decomposed/',rf_data_mat_name) , dic)
    return None#dic['RFSampleFrames']

def collate_fn2(batch):
    batch = list(filter (lambda x:x is not None, batch)) # filter out all the Nones
    return torch.utils.data.dataloader.default_collate(batch)


class UltrasoundDataset_real(Dataset):
    def __init__(self, path, feature_transform=None,
                 label_transform=None, train=True,
                 train_size=0.80):
        self.path = path
        self.file_list = []
        self.label_list = []
        self.feature_transform = feature_transform
        self.label_transform = label_transform
        for dirname, _, filenames in os.walk(path):
            for filename in filenames:
                if filename[0:3] == "rf_":
                    self.file_list.append(os.path.join(dirname, filename))
        total_len = len(self.file_list)
        if train:
            self.file_list, self.label_list = self.file_list[:int(train_size * total_len)], self.label_list[
                                                                                      :int(train_size * total_len)]
        else:
            self.file_list, self.label_list = self.file_list[int(train_size * total_len):], self.label_list[
                                                                                      int(train_size * total_len):]

    def __getitem__(self, idx):
        try:
            rf_data,attenuation = load_ultrasound2(self.file_list[idx])
            len_wav = rf_data.shape[0]
            pddd = 4500#4000

            if len_wav < pddd:
                pad = np.zeros(pddd - len_wav)
                rf_data = np.hstack([rf_data, pad])
            elif len_wav > pddd:
                rf_data = rf_data[:pddd]
            return rf_data,attenuation,self.file_list[idx]
        except:
            return None




    def __len__(self):
        return len(self.file_list)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def Spliter_simulated_data(dataloader,save_path,mode):
    for batch, (X,y,addr_tuple) in enumerate(dataloader):
        try:
            print(addr_tuple,batch,len(dataloader))
            addr = addr_tuple[0]
            dic_temp = {}
            dic_temp['rf_data_padded'] = X 
            dic_temp['my_att'] = y 

            data_dic2 = loadmat(addr)
            #print(data_dic2)
            #dic_temp['rf_env'] = data_dic2['rf_env'].reshape((-1,))
            dic_temp['rf_data'] = data_dic2['rf_data'].reshape((-1,))
            savemat( os.path.join(save_path,mode+ '_'+addr.split('/')[-1]) , dic_temp)
            print('Done!')
        except:
            print(addr_tuple,batch,len(dataloader))
            print('Curropted!!')
            pass
    return None


def Spliter_real_data(dataloader,save_path,mode):
    
    for batch, (X,y,addr_tuple) in enumerate(dataloader):
        print(addr_tuple)
        addr = addr_tuple[0]
        dic_temp = {}
        dic_temp['rf_data_padded'] = X 
        dic_temp['my_att'] = y 

        data_dic2 = loadmat(addr)

        dic_temp['rf_data'] = data_dic2['rf_data'][0]
        savemat( os.path.join(save_path,mode+ '_'+addr.split('/')[-1]) , dic_temp)
    return None



if __name__ == "__main__":
    SEED = 12345
    BATCH_SIZE = 1
    torch.manual_seed(SEED)
    g = torch.Generator()
    g.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    PreFix_PATH = '/home/arian/PycharmProjects/'
    DATA_SET_PATH = PreFix_PATH+'DataSet/rf_data_82k/rf_data/'
    DATA_SET_PATH_REAL = PreFix_PATH+'DataSet/Real_test_data/'


    train_ds = UltrasoundDataset(DATA_SET_PATH,
                                     train=True)
    train_mask = torch.tensor(['ln1' in x for i, x in enumerate(train_ds.file_list)])
    train_indices = train_mask.nonzero().reshape(-1)
    print(train_indices)
    train_subset = Subset(train_ds, train_indices)


    test_ds = UltrasoundDataset(DATA_SET_PATH,
                                    train=False)
    test_mask = torch.tensor(['ln1' in x for i, x in enumerate(test_ds.file_list)])
    test_indices = test_mask.nonzero().reshape(-1)
    print(test_indices)
    test_subset = Subset(test_ds, test_indices)


    test_ds_real = UltrasoundDataset_real(DATA_SET_PATH_REAL,train=False,train_size = 0.1)



    train_dataloader = DataLoader(train_ds,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True,
                                      collate_fn=collate_fn,
                                      worker_init_fn=seed_worker,
                                      generator=g)

    test_dataloader = DataLoader(test_ds,
                                 batch_size=BATCH_SIZE,
                                 shuffle=True,
                                 collate_fn=collate_fn,
                                 worker_init_fn=seed_worker,
                                 generator=g)

    test_dataloader_real = DataLoader(test_ds_real,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True,
                                      collate_fn=collate_fn,
                                      worker_init_fn=seed_worker,
                                      generator=g)
    

    Spliter_simulated_data(dataloader=train_dataloader,save_path=PreFix_PATH+'DataSet/train',mode='train')
    Spliter_simulated_data(dataloader=test_dataloader,save_path=PreFix_PATH+'DataSet/valid',mode='valid')
    Spliter_real_data(dataloader=test_dataloader_real,save_path=PreFix_PATH+'DataSet/test',mode='test')