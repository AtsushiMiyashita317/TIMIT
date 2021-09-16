import argparse
import os
import pickle

import numpy as np
from numpy.ma import indices
import pandas as pd
import soundfile as sf
from scipy import signal
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler


class Timit(Dataset):
    def __init__(self, root, annotations_file, phncode_file, data_dir, 
                 nperseg=256, norverlap=None, nframe=15, 
                 signal_transform=None, spec_transform=None, frame_transform=None, target_transform=None, 
                 datasize=None, cachesize = 100):

        self.data_dir = os.path.join(root, data_dir)
        self.nperseg = nperseg
        if norverlap:
            self.noverlap = norverlap
        else:
            self.noverlap = nperseg//2
        self.nframe = nframe

        self.signal_transform = signal_transform
        self.spec_transform = spec_transform
        self.frame_transform = frame_transform
        self.target_transform = target_transform

        self.cachesize = cachesize
        self.cache_spec = [np.empty(0)] * self.cachesize
        self.cache_label = [np.empty(0)] * self.cachesize
        self.cache_range = [(0,0)] * self.cachesize
        self.cache_last = np.full(cachesize,-1)
        self.cache_time = 0

        self.datasize = datasize

        self.annotations = pd.read_csv(annotations_file)

        sign_length = self.annotations['length'].values
        spec_length = (sign_length+self.nperseg-self.noverlap-1)//(self.nperseg-self.noverlap) + 1
        item_length = spec_length - self.nframe + 1
        maxidx = np.cumsum(item_length)
        minidx = np.zeros_like(maxidx, dtype=np.int64)
        minidx[1:] = maxidx[:-1]
        self.annotations['minidx'] = minidx
        self.annotations['maxidx'] = maxidx

        with open(phncode_file,'rb') as f:
            self.phn_dict,self.phn_list,self.phn_count = pickle.load(f)
        

    def __len__(self):
        if self.datasize:
            return self.datasize
        else:
            return self.annotations['maxidx'].max()

    def __getitem__(self, idx):
        hit = -1
        for i in range(self.cachesize):
            if self.cache_range[i][0]<=idx and idx<self.cache_range[i][1]:
                hit = i
                self.cache_last[i] = self.cache_time
                self.cache_time += 1
                break
        if hit < 0:
            oldest = np.argmin(self.cache_last)

            cand = self.annotations[self.annotations['maxidx']>idx]
            
            wav_path = os.path.join(self.data_dir, cand.iat[0, 1])
            sign, sr = sf.read(wav_path)
            if self.signal_transform:
                sign = self.signal_transform(sign)    

            self.cache_spec[oldest] = np.abs(signal.stft(sign,sr,nperseg=self.nperseg,noverlap=self.noverlap)[2])

            if self.spec_transform:
                self.cache_spec[oldest] = self.spec_transform(self.cache_spec[oldest])    

            self.cache_spec[oldest] = np.transpose(self.cache_spec[oldest])

            phn_path = os.path.join(self.data_dir, cand.iat[0, 2])
            df_phn = pd.read_csv(phn_path, delimiter=' ', header=None)
            self.cache_label[oldest] = np.zeros(self.cache_spec[oldest].shape[0], dtype=np.int64)

            for i in range(len(df_phn)):
                phn = df_phn.iat[i,2]
                begin = df_phn.iat[i,0]//(self.nperseg - self.noverlap)
                end = df_phn.iat[i,1]//(self.nperseg - self.noverlap)
                self.cache_label[oldest][begin:end] = self.phn_dict[phn]       

            self.cache_range[oldest] = (cand.iat[0, 4],cand.iat[0, 5])

            hit = oldest
            self.cache_last[oldest] = self.cache_time
            self.cache_time += 1

        local_idx = idx - self.cache_range[hit][0]
        frames = self.cache_spec[hit][local_idx:local_idx + self.nframe]
        label = self.cache_label[hit][local_idx + self.nframe//2]

        if self.frame_transform:
            frames = self.frame_transform(frames)
        if self.target_transform:
            label = self.target_transform(label)

        return frames, label

class TimitSample(Dataset):
    def __init__(self, root, annotations_file, phncode_file, data_dir, 
                 nperseg=256, norverlap=None, 
                 signal_transform=None, spec_transform=None, target_transform=None):

        self.data_dir = os.path.join(root, data_dir)
        self.nperseg = nperseg
        if norverlap:
            self.noverlap = norverlap
        else:
            self.noverlap = nperseg//2

        self.signal_transform = signal_transform
        self.spec_transform = spec_transform
        self.target_transform = target_transform

        self.annotations = pd.read_csv(annotations_file)

        with open(phncode_file,'rb') as f:
            self.phn_dict,self.phn_list,self.phn_count = pickle.load(f)
        

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        wav_path = os.path.join(self.data_dir, self.annotations.iat[idx, 1])
        sign, sr = sf.read(wav_path)
        if self.signal_transform:
            sign = self.signal_transform(sign)    

        spec = np.abs(signal.stft(sign,sr,nperseg=self.nperseg,noverlap=self.noverlap)[2])

        if self.spec_transform:
            spec = self.spec_transform(spec)    

        spec = np.transpose(spec)

        phn_path = os.path.join(self.data_dir, self.annotations.iat[idx, 2])
        df_phn = pd.read_csv(phn_path, delimiter=' ', header=None)
        label = np.zeros(spec.shape[0], dtype=np.int64)

        for i in range(len(df_phn)):
            phn = df_phn.iat[i,2]
            begin = df_phn.iat[i,0]//(self.nperseg - self.noverlap)
            end = df_phn.iat[i,1]//(self.nperseg - self.noverlap)
            label[begin:end] = self.phn_dict[phn]       

        if self.target_transform:
            label = self.target_transform(label)

        return spec, label
     
class TimitMetrics(Dataset):
    def __init__(self, root, annotations_file, phncode_file, data_dir):
        self.annotations = pd.read_csv(os.path.join(root, annotations_file))

        with open(os.path.join(root, phncode_file),'rb') as f:
            self.phn_dict,self.phn_list,self.phn_count = pickle.load(f)
        
        self.data_dir = os.path.join(root, data_dir)
        self.metrics = pd.DataFrame(index=self.phn_list, 
                                    columns=['count',
                                             'min_length',
                                             'max_length',
                                             'sum_length'],
                                    dtype=np.int)
        self.metrics.fillna(0,inplace=True)
        self.metrics['min_length'] = np.inf
        self.metrics.info()

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        phn_path = os.path.join(self.data_dir, self.annotations.iat[idx, 2])
        df_phn = pd.read_csv(phn_path, delimiter=' ', header=None, names=['begin','end','code'])
        df_phn['length'] = df_phn['end'] - df_phn['begin']

        for code in self.phn_list:
            df_code = df_phn[df_phn['code']==code]
            self.metrics.at[code,'count'] += len(df_code)
            self.metrics.at[code,'min_length'] = min(self.metrics.at[code,'min_length'], df_code['length'].min())
            self.metrics.at[code,'max_length'] = max(self.metrics.at[code,'max_length'], df_code['length'].max())
            self.metrics.at[code,'sum_length'] += df_code['length'].sum()

        return 0

class TimitRow(Dataset):
    def __init__(self, root, annotations_file, data_dir, phn_info=None):
        df = pd.read_csv(os.path.join(root, annotations_file))
        df = df.sort_values('path_from_data_dir')
    
        self.df_wav = df[df['filename'].str.endswith('.wav',na=False)]
        self.df_phn = df[df['filename'].str.endswith('.PHN',na=False)]

        self.data_dir = os.path.join(root, data_dir)

        if phn_info:
            self.phn_dict,self.phn_list,self.phn_count = phn_info
        else:
            self.phn_dict = {}
            self.phn_list = []
            self.phn_count = 0

        self.annotations_new = []

    def __len__(self):
        return len(self.df_wav)

    def __getitem__(self, idx):
        wav_path = os.path.join(self.data_dir, self.df_wav.iat[idx, 5])
        sign, _ = sf.read(wav_path)

        phn_path = os.path.join(self.data_dir, self.df_phn.iat[idx, 5])
        df = pd.read_csv(phn_path, delimiter=' ', header=None)

        labels = np.zeros(sign.shape[0], dtype=np.int64)

        for i in range(len(df)):
            phn = df.iat[i,2]
            if not phn in self.phn_dict:
                self.phn_dict[phn] = self.phn_count
                self.phn_list.append(phn)
                self.phn_count += 1
            labels[df.iat[i,0]:df.iat[i,1]] = self.phn_dict[phn]

        self.annotations_new.append({'wav_path':self.df_wav.iat[idx, 5],
                                     'phn_path':self.df_phn.iat[idx, 5],
                                     'length':sign.shape[0]})
                    
        return sign,labels

class RandomFrameSamplar(BatchSampler):
    def __init__(self, maxidx, batchsize, cachesize):
        self.maxidx = maxidx
        self.minidx = np.zeros_like(maxidx, dtype=np.int64)
        self.minidx[1:] = maxidx[:-1]
        self.datasize = np.max(maxidx)
        self.rng = np.random.default_rng()
        
        self.cachesize = cachesize - 1
        self.count = 0
        self.batchsize = batchsize

    def __iter__(self):
        rng_idx = np.arange(self.maxidx.size)
        self.rng.shuffle(rng_idx)
        self.maxidx = self.maxidx[rng_idx]
        self.minidx = self.minidx[rng_idx]
        arrs = []
        for i in range(self.cachesize):
            arr = np.arange(self.minidx[i],self.maxidx[i])
            self.rng.shuffle(arr)
            arrs.append(arr)

        res = np.array([],dtype=np.int64)
        for i in range(self.maxidx.size):
            indices = np.array([],dtype=np.int64)
            for j in range(self.cachesize):
                k = (j-i)%self.cachesize
                begin = (k*arrs[j].size)//self.cachesize
                end = ((k+1)*arrs[j].size)//self.cachesize
                arr = arrs[j][begin:end]
                indices = np.concatenate([indices,arr])
            self.rng.shuffle(indices)
            indices = np.concatenate([res,indices])
            for j in range(0,indices.size,self.batchsize):
                res = indices[j:j+self.batchsize]
                if res.size == self.batchsize:
                    yield res
            k = (i+self.cachesize)%self.maxidx.size
            arr = np.arange(self.minidx[k],self.maxidx[k])
            self.rng.shuffle(arr)
            arrs[i%self.cachesize] = arr


    def __len__(self):
        return self.maxidx.size//self.cachesize

def main():
    parser = argparse.ArgumentParser(description="test class FramedTimit")
    parser.add_argument("path", type=str, help="path to the directory that has annotation files")
    args = parser.parse_args()

    train_data = Timit(args.path,'train_annotations.csv','phn.pickle','data/')
    test_data = Timit(args.path,'test_annotations.csv','phn.pickle','data/')

    train_dataloader = DataLoader(train_data, batch_size=128)
    test_dataloader = DataLoader(test_data, batch_size=128)

    print(train_data[0][0].shape)

    for batch, (X,y) in enumerate(train_dataloader):
        print(f"processing train... batch = {batch}\r",end='')

    print("processing train... Done.")


    for batch, (X,y) in enumerate(test_dataloader):
        print(f"processing test... batch = {batch}\r",end='')

    print("processing test... Done.")


def create_annotations():
    parser = argparse.ArgumentParser(description="load TIMIT and convert to npz file")
    parser.add_argument("path", type=str, help="path to the directory that has annotation files")
    args = parser.parse_args()

    path_self = os.path.dirname(os.path.abspath(__file__))

    train_data = TimitRow(args.path,'train_data.csv','data/')
    train_dataloader = DataLoader(train_data, batch_size=1)

    count = 0
    for _ in train_dataloader:
        print(f"processing train... count = {count}\r",end='')
        count += 1

    df = pd.DataFrame(train_data.annotations_new)
    """
        df['maxidx'] = df['length'].cumsum()
        df['minidx'] = df['maxidx'].shift()
        df.at[df.index[0],'minidx'] = 0
        df['minidx'] = df['minidx'].astype(np.int64)
    """
    df.to_csv(os.path.join(path_self, 'train_annotations.csv'))

    print("processing train... Done.")

    phn_info = (train_data.phn_dict,train_data.phn_list,train_data.phn_count)

    test_data = TimitRow(args.path,'test_data.csv','data/',phn_info=phn_info)
    test_dataloader = DataLoader(test_data, batch_size=1)

    count = 0
    for _ in test_dataloader:
        print(f"processing test... count = {count}\r",end='')
        count += 1

    df = pd.DataFrame(test_data.annotations_new)
    """
        df['maxidx'] = df['length'].cumsum()
        df['minidx'] = df['maxidx'].shift()
        df.at[df.index[0],'minidx'] = 0
        df['minidx'] = df['minidx'].astype(np.int64)
    """
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_annotations.csv'))

    print("processing train... Done.")

    with open("phn.pickle", "wb") as f:
        pickle.dump((test_data.phn_dict,test_data.phn_list,test_data.phn_count), f)


def metrics():
    parser = argparse.ArgumentParser(description="test class FramedTimit")
    parser.add_argument("path", type=str, help="path to the directory that has annotation files")
    args = parser.parse_args()

    train_data = TimitMetrics(args.path,'train_annotations.csv','phn.pickle','data/')
    test_data = TimitMetrics(args.path,'test_annotations.csv','phn.pickle','data/')

    train_dataloader = DataLoader(train_data, batch_size=1)
    test_dataloader = DataLoader(test_data, batch_size=1)

    for batch, x in enumerate(train_dataloader):
        print(f"train batch = {batch}\r")

    train_data.metrics.to_csv('train_metrics.csv')

    for batch, x in enumerate(test_dataloader):
        print(f"test batch = {batch}\r")

    test_data.metrics.to_csv('test_metrics.csv')


if __name__=="__main__":
    main()