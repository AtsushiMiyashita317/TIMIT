import argparse
import os
import pickle

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
from torch.utils.data import Dataset, DataLoader


phn_dict = {}
phn_list = []
phn_count = 0

class Timit(Dataset):
    def __init__(self, root, annotations_file, phncode_file, data_dir, 
                n_fft=256, n_frame=15, transform1=None, transform2=None, target_transform=None, datasize=None):

        self.annotations = pd.read_csv(annotations_file)

        with open(phncode_file,'rb') as f:
            self.phn_dict,self.phn_list,self.phn_count = pickle.load(f)
        
        self.data_dir = os.path.join(root, data_dir)
        self.n_fft = n_fft
        self.n_frame = n_frame
        self.transform1 = transform1
        self.transform2 = transform2
        self.target_transform = target_transform
        self.cache_spec = None
        self.cache_label = None
        self.cache_range = (0,0)
        self.cache_centor = None
        self.datasize = datasize

    def __len__(self):
        if self.datasize:
            return self.datasize
        else:
            return self.annotations['maxidx'].max()

    def __getitem__(self, idx):
        if not (self.cache_range[0]<=idx and idx<self.cache_range[1]):
            cand = self.annotations[self.annotations['maxidx']>idx]
            
            wav_path = os.path.join(self.data_dir, cand.iat[0, 1])
            sign, sr = sf.read(wav_path)
            self.cache_spec = signal.stft(sign,sr,nperseg=self.n_fft)[2]

            phn_path = os.path.join(self.data_dir, cand.iat[0, 2])
            df_phn = pd.read_csv(phn_path, delimiter=' ', header=None)
            self.cache_label = np.zeros(len(df_phn), dtype=np.int64)
            self.cache_centor = np.zeros(len(df_phn), dtype=np.int64)

            for i in range(len(df_phn)):
                begin = df_phn.iat[i,0]
                end = df_phn.iat[i,1]
                phn = df_phn.iat[i,2]
                self.cache_label[i] = self.phn_dict[phn]
                self.cache_centor[i] = (begin + end)//self.n_fft

            if self.transform1:
                self.cache_spec = self.transform1(self.cache_spec)    

            self.cache_range = (cand.iat[0, 5],cand.iat[0, 4])
        
        frames = np.zeros(self.cache_spec.shape[:-1]+(self.n_frame,),dtype=np.complex128)
        local_idx = idx - self.cache_range[0]
        centor = self.cache_centor[local_idx]
        lower = centor - self.n_frame//2
        upper = centor + (self.n_frame + 1)//2
        lower_sc = max(0, lower)
        upper_sc = min(self.cache_spec.shape[-1], upper)
        lower_dst = lower_sc - lower
        upper_dst = self.n_frame - (upper - upper_sc)

        frames[...,lower_dst:upper_dst] = self.cache_spec[...,lower_sc:upper_sc]
        label = self.cache_label[...,local_idx]

        if self.transform2:
            frames = self.transform2(frames)
        if self.target_transform:
            label = self.target_transform(label)

        return frames, label

    def describe(self):
        pass
        
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
    def __init__(self, root, annotations_file, data_dir):
        df = pd.read_csv(os.path.join(root, annotations_file))
        df = df.sort_values('path_from_data_dir')
    
        self.df_wav = df[df['filename'].str.endswith('.wav',na=False)]
        self.df_phn = df[df['filename'].str.endswith('.PHN',na=False)]

        self.data_dir = os.path.join(root, data_dir)

        self.annotations_new = []

    def __len__(self):
        return len(self.df_wav)

    def __getitem__(self, idx):
        wav_path = os.path.join(self.data_dir, self.df_wav.iat[idx, 5])
        sign, sr = sf.read(wav_path)

        phn_path = os.path.join(self.data_dir, self.df_phn.iat[idx, 5])
        df = pd.read_csv(phn_path, delimiter=' ', header=None)

        global phn_dict
        global phn_list
        global phn_count

        for i in range(len(df)):
            phn = df.iat[i,2]
            if not phn in phn_dict:
                phn_dict[phn] = phn_count
                phn_list.append(phn)
                phn_count += 1

        self.annotations_new.append({'wav_path':self.df_wav.iat[idx, 5],
                                     'phn_path':self.df_phn.iat[idx, 5],
                                     'length':len(df)})
        
            
        return sign


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

    print()

    for batch, (X,y) in enumerate(test_dataloader):
        print(f"processing test... batch = {batch}\r",end='')


def create_annotations():
    parser = argparse.ArgumentParser(description="load TIMIT and convert to npz file")
    parser.add_argument("path", type=str, help="path to the directory that has annotation files")
    args = parser.parse_args()

    path_self = os.path.dirname(os.path.abspath(__file__))

    train_data = TimitRow(os.path.join(args.path, 'train_data.csv'),
                          os.path.join(args.path, 'data/'))   
    test_data = TimitRow(os.path.join(args.path, 'test_data.csv'),
                         os.path.join(args.path, 'data/'))  

    train_dataloader = DataLoader(train_data, batch_size=1)
    test_dataloader = DataLoader(test_data, batch_size=1)

    count = 0
    
    for sign in train_dataloader:
        print(f"processing train... count = {count}\r",end='')
        count += 1

    df = pd.DataFrame(train_data.annotations_new)
    df['maxidx'] = df['length'].cumsum()
    df['minidx'] = df['maxidx'].shift()
    df.at[df.index[0],'minidx'] = 0
    df['minidx'] = df['minidx'].astype(np.int64)
    df.to_csv(os.path.join(path_self, 'train_annotations.csv'))

    print('\r',end='')

    for sign in test_dataloader:
        print(f"processing test... count = {count}\r",end='')
        count += 1

    df = pd.DataFrame(test_data.annotations_new)
    df['maxidx'] = df['length'].cumsum()
    df['minidx'] = df['maxidx'].shift()
    df.at[df.index[0],'minidx'] = 0
    df['minidx'] = df['minidx'].astype(np.int64)
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_annotations.csv'))

    with open("phn.pickle", "wb") as f:
        pickle.dump((phn_dict,phn_list,phn_count), f)


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