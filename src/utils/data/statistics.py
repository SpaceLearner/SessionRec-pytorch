import pandas
import numpy as np
from collections import *
from itertools import combinations
from dataset import read_dataset, AugmentedDataset
from pathlib import Path
import pickle

def countCoAppear(seqs, winSize, order=4):
    
    # assert winSize >= order
    coaDict = Counter()
    for seq in seqs:
        winLen = min(len(seq), winSize)
        for i in range(len(seq)-winLen+1):
            coms = []
            for j in range(1, min(winLen+1, order+2)):
                for com in combinations(seq[i:i+winLen], j):
                    com = tuple(sorted(com))
                    coms.append(com)
            counter = Counter(coms)
            coaDict.update(counter)
    return dict(coaDict)

def filterDict(coaDict, min_num=20):
    
    keys = list(coaDict.keys())
    for key in keys:
        if coaDict[key] < min_num:
            coaDict.pop(key)
            
    return coaDict

def relabelDict(coaDict):
    keys = list(coaDict.keys())
    cnt = 0
    for key in keys:
        coaDict[key] = cnt
        cnt += 1
    return coaDict

def count_short_long(seqs):

    short_count = 0
    long_count  = 0
    total_length = 0
    dataset = AugmentedDataset(seqs)
    for i in range(len(dataset)):
        total_length += len(dataset[i][0])
        total_length += 1
        # print(dataset[i])
        if len(dataset[i][0]) <= 4:
            short_count += 1
        else:
            long_count += 1
    
    num_samples = len(dataset)

    num_items   = np.unique(np.array(seqs))
    
    return short_count, long_count, total_length, num_samples, num_items
    

if __name__ == "__main__":
    
    datafolder = "../../datasets/yoochoose1_4"
    
    train_session, test_session, num_items = read_dataset(Path(datafolder))

    num_short1, num_long1, total_length1, num_samples1, num_items1 = count_short_long(train_session)

    # print(num_short, num_long)

    num_short2, num_long2, total_length2, num_samples2, num_items2 = count_short_long(test_session)

    num_short    = num_short1    + num_short2
    num_long     = num_long1     + num_long2
    total_length = total_length1 + total_length2
    num_samples  = num_samples1  + num_samples2
    average_len  = (total_length1 + total_length2) / num_samples

    print(num_short, num_long, num_samples, num_samples1, num_samples2, average_len)
    print(num_short + num_long)
    
    # coaDict = countCoAppear(train_session, 5, 4)
    
    # coaDict = filterDict(coaDict, min_num=200)
    # coaDict = relabelDict(coaDict)
    # # print(len(coaDict))
    # pickle.dump(coaDict, open(datafolder+'/coaDict.pkl', 'wb'))
    
    # dic = pickle.load(open(datafolder+'/coaDict.pkl', 'rb'))
    
    # print(dic)
