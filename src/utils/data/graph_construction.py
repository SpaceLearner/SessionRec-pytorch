import torch as th
import numpy as np
import pandas as pd
import itertools
import pickle
from collections import *
from dataset import read_dataset, create_index
from pathlib import Path

def seq_to_ccs_graph(seq, order=1):
    
    order1 = order
    order = min(order, len(seq))
    items = th.unique(th.tensor(seq).long(), sorted=False).numpy()
    # items, indices = np.unique(seq, return_inverse=True)
    iid2nid = {iid: i for i, iid in enumerate(items)}
    
    num_nodes = len(items)
    seq_nid = [iid2nid[iid] for iid in seq]
    last_k = [iid2nid[seq[-1]]]
    combine_seqs = []
    
    def com(i, order):
        item = str(seq[i:i+order])
        # for k in range(order):
        #     item += str(seq_nid[i+k])
        return item 
    
    class combine:
        def __init__(self):
            self.dict = {}
        
        def __call__(self, *input):
            return self.forward(*input)    
        
        def forward(self, i, order):
            if str(i) not in self.dict:
                self.dict[str(i)] = {}
            if order not in self.dict[str(i)]:
                self.dict[str(i)][order] = com(i, order)
            return self.dict[str(i)][order]
        
    combine = combine()  
    
    item_dicts = [iid2nid]
    for i in range(1, order1):
        combine_seq = []
        item_dict = {}
        cnt = 0
        for j in range(len(seq_nid)-i):
            item = combine(j, i+1)
            if item not in item_dict:
                item_dict[item] = cnt
                cnt += 1
                combine_seq.append([seq[idx] for idx in range(j, j+i+1)])
        if len(item_dict) > 0:
            last_k.append(item_dict[item])
        else:
            last_k.append(0)
        combine_seqs.append(combine_seq)
                
        item_dicts.append(item_dict)
    
    graph_data = {}
    
    for k in range(order):
        if k == 0:
            counter = Counter([(seq_nid[i], seq_nid[i+1]) for i in range(len(seq)-1)]) ## original connect
        else:       
            counter = Counter([(item_dicts[k][combine(i, k+1)], item_dicts[k][combine(i+1, k+1)]) for i in range(len(seq)-k-1)])
        
        edges = counter.keys()
        
        if len(edges) > 0:
            src, dst = zip(*edges)
            weight = th.tensor(list(counter.values()))
        else:
            src, dst = [], []
            weight = th.ones(1).long()
        
        graph_data[('s'+str(k+1), 'intra'+str(k+1), 's'+str(k+1))] = (th.tensor(src).long(), th.tensor(dst).long())

    for k in range(1, order): 
       
        counter = Counter([(seq_nid[i], item_dicts[k][combine(i+1, k+1)]) for i in range(len(seq)-k-1)])
        
        edges = counter.keys()
        
        if len(edges) > 0:
            src, dst = zip(*edges)
            weight = th.tensor(list(counter.values()))
        else:
            src, dst = th.LongTensor([]), th.LongTensor([])
            weight = th.ones(1).long()

        graph_data[('s1', 'inter', 's'+str(k+1))] = (src, dst)
        
        counter = Counter([(item_dicts[k][combine(i, k+1)], seq_nid[i+k+1]) for i in range(len(seq)-k-1)])
        
        edges = counter.keys()
    
        if len(edges) > 0:
            src, dst = zip(*edges)
            weight = th.tensor(list(counter.values()))
        else:
            src, dst = th.LongTensor([]), th.LongTensor([])
            weight = th.ones(1).long()
        
        graph_data[('s'+str(k+1), 'inter', 's1')] = (src, dst)
    
    if order < order1:
        for i in range(order, order1):
            graph_data[('s'+str(i+1), 'intra'+str(i+1), 's'+str(i+1))]=(th.LongTensor([]), th.LongTensor([]))
            graph_data[('s'+str(i+1), 'inter', 's1')]=(th.LongTensor([]), th.LongTensor([]))
            graph_data[('s1', 'inter', 's'+str(i+1))]=(th.LongTensor([]), th.LongTensor([]))
    
    return graph_data, last_k, items, combine_seqs, order

def construct_graphs(sessions, index, max_order=4, dataset_dir='../../datasets/yoochoose1_64', phrase='train'):
    
    seqs = []
    for idx in index:
        sid, lidx = idx
        seq = sessions[sid][:lidx]
        seqs.append(seq)
    print(len(seqs))
    for i in range(max_order):
        print('preprocessing ' + str(i+1) + '...')
        graphs = list(map(seq_to_ccs_graph, seqs, [i+1 for _ in range(len(seqs))]))
        # graph = seq_to_ccs_graph(seq, order=i+1)
        pickle.dump(graphs, open(dataset_dir+'/'+phrase+'_graphs_order_'+str(i+1)+'.pkl', 'wb'))

if __name__ == "__main__":
    
    dataset_dir = "../../datasets/diginetica"
    
    max_order = 4
    
    train_sessions, test_sessions, num_items = read_dataset(Path(dataset_dir))
    
    train_index = create_index(train_sessions)
    test_index  = create_index(test_sessions)
    
    construct_graphs(train_sessions, train_index, max_order, dataset_dir, 'train')
    construct_graphs(test_sessions, test_index, max_order, dataset_dir, 'test')
    
    
    
    
    
        
