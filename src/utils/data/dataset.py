import itertools
from os import read
import numpy as np
import pandas as pd
import pickle as pkl


def create_index(sessions):
    lens = np.fromiter(map(len, sessions), dtype=np.long)
    session_idx = np.repeat(np.arange(len(sessions)), lens - 1)
    label_idx = map(lambda l: range(1, l), lens)
    label_idx = itertools.chain.from_iterable(label_idx)
    label_idx = np.fromiter(label_idx, dtype=np.long)
    idx = np.column_stack((session_idx, label_idx))
    return idx


def read_sessions(filepath):
    sessions = pd.read_csv(filepath, sep='\t', header=None, squeeze=True)
    sessions = sessions.apply(lambda x: list(map(int, x.split(',')))).values
    return sessions

def read_timestamps(filepath):
    sessions = pd.read_csv(filepath, sep='\t', header=None, squeeze=True)
    sessions = sessions.apply(lambda x: list(map(float, x.split(',')))).values
    return sessions

def read_dataset(dataset_dir):
    dataset = dataset_dir.strip().split('/')[-1]
    if dataset in ["gowalla"]:
        train_sessions  = read_sessions(dataset_dir   / 'train.txt')
        test_sessions   = read_sessions(dataset_dir   / 'test.txt')
        train_timestamp = read_timestamps(dataset_dir / 'train_timestamp.txt')
        test_timestamp  = read_timestamps(dataset_dir / 'test_timestamp.txt') 
    elif dataset in ["tmall", "nowplaying"]:
        train_dict      = pkl.load(open(dataset_dir + "train.txt", "rb"))
        test_dict       = pkl.load(open(dataset_dir + "test.txt", "rb"))
        train_sessions  = train_dict[0]
        test_sessions   = test_dict[0]
        train_timestamp = train_dict[1]
        test_timestamp  = test_dict[1]
    
    with open(dataset_dir / 'num_items.txt', 'r') as f:
            num_items = int(f.readline())
    return train_sessions, test_sessions, train_timestamp, test_timestamp, num_items
        

class AugmentedDataset:
    def __init__(self, sessions, timestamps, sort_by_length=False):
        self.sessions   = sessions
        self.timestamps = timestamps
        # self.graphs = graphs
        index = create_index(sessions)  # columns: sessionId, labelIndex

        if sort_by_length:
            # sort by labelIndex in descending order
            ind = np.argsort(index[:, 1])[::-1]
            index = index[ind]
        self.index = index

    def __getitem__(self, idx):
        #print(idx)
        sid, lidx = self.index[idx]
        seq       = self.sessions[sid][:lidx]
        label     = self.sessions[sid][lidx]
        times     = self.timestamps[sid][:lidx]#  - self.sessions[sid][0]
        temp      = times[0]
        print(times)
        # times     = [(t - temp) / 100000 for t in times]
        times     = [(t - temp) / 1000000 for t in times]
        
        # print(times)
        
        return seq, times, label #,seq

    def __len__(self):
        return len(self.index)
