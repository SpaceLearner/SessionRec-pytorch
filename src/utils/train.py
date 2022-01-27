import time

import numpy as np
import torch as th
from sklearn.metrics import accuracy_score
from torch import nn, optim
from tqdm import tqdm
import wandb


# ignore weight decay for parameters in bias, batch norm and activation
def fix_weight_decay(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(map(lambda x: x in name, ['bias', 'batch_norm', 'activation'])):
            no_decay.append(param)
        else:
            decay.append(param)
    params = [{'params': decay}, {'params': no_decay, 'weight_decay': 0}]
    return params


def prepare_batch(batch, device):
    inputs, labels, embeds_ids, times, num_nodes = batch
    # inputs, labels = batch
    inputs_gpu  = [x.to(device) for x in inputs]
    labels_gpu  = labels.to(device)
    embeds_ids  = embeds_ids.to(device)
    times       = times.to(device)
    num_nodes   = num_nodes.to(device)
   
    return inputs_gpu, labels_gpu, embeds_ids, times, num_nodes
    # return inputs_gpu, 0, labels_gpu, 0


def evaluate(model, data_loader, device, cutoff=20):
    model.eval()
    mrr = 0
    hit = 0
    num_samples = 0

    with th.no_grad():
        for batch in data_loader:
            inputs, labels, embeds_ids, times, num_nodes = prepare_batch(batch, device)
            logits = model(*inputs, embeds_ids, times, num_nodes)
        
            batch_size   = logits.size(0)
            num_samples += batch_size
            topk         = logits.topk(k=cutoff)[1]
            labels       = labels.unsqueeze(-1)
            hit_ranks    = th.where(topk == labels)[1] + 1
            hit         += hit_ranks.numel()
            mrr         += hit_ranks.float().reciprocal().sum().item()
            
    return mrr / num_samples, hit / num_samples
class TrainRunner:
    def __init__(
        self,
        dataset,
        model,
        train_loader,
        test_loader,
        device,
        lr=1e-3,
        weight_decay=0,
        patience=3,
    ):
        self.dataset = dataset
        self.model = model
        if weight_decay > 0:
            params = fix_weight_decay(model)
        else:
            params = model.parameters()
        self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)
        
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.device       = device
        self.epoch        = 0
        self.batch        = 0
        self.patience     = patience

    def train(self, epochs, log_interval=100):
        max_mrr10 = max_mrr20 = 0
        max_hit10 = max_hit20 = 0
        bad_counter = 0
        t = time.time()
        mean_loss = 0

        mrr, hit = evaluate(self.model, self.test_loader, self.device)
        for epoch in tqdm(range(epochs)):
            self.model.train()
            for batch in self.train_loader:
                inputs, labels, embeds_ids, times, num_nodes = prepare_batch(batch, self.device)
                self.optimizer.zero_grad()
                scores = self.model(*inputs, embeds_ids, times, num_nodes)
                assert not th.isnan(scores).any()
                loss   = nn.functional.nll_loss(scores, labels)
                loss.backward()
                self.optimizer.step()
                
                mean_loss += loss.item() / log_interval
                
                if self.batch > 0 and self.batch % log_interval == 0:
                    print(f'Batch {self.batch}: Loss = {mean_loss:.4f}, Time Elapsed = {time.time() - t:.2f}s')
                    t = time.time()
                    mean_loss = 0
                    
                self.batch += 1
            self.scheduler.step()
            mrr10, hit10 = evaluate(self.model, self.test_loader, self.device, cutoff=10)
            mrr20, hit20 = evaluate(self.model, self.test_loader, self.device)
            
            wandb.log({"hit@20": hit20, "mrr@20": mrr20})

            print(f'Epoch {self.epoch}: MRR@10 = {mrr10 * 100:.3f}%, Hit@10 = {hit10 * 100:.3f}%, MRR@20 = {mrr20 * 100:.3f}%, Hit@20 = {hit20 * 100:.3f}%')

            if mrr20 < max_mrr20 and hit20 < max_hit20:
                bad_counter += 1
                if bad_counter == self.patience:
                    break
            else:
                bad_counter = 0
            max_mrr10 = max(max_mrr10, mrr10)
            max_hit10 = max(max_hit10, hit10)
            max_mrr20 = max(max_mrr20, mrr20)
            max_hit20 = max(max_hit20, hit20)
            
            self.epoch += 1
        return max_mrr10, max_mrr20, max_hit10, max_hit20
