import torch, time, os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from backbone import MTAN_ResNet, DMTL, Model_h
from utils import *

from create_dataset import office_dataloader
from min_norm_solvers import MinNormSolver, gradient_normalizers
import argparse
torch.set_num_threads(3)
def parse_args():
    parser = argparse.ArgumentParser(description= 'MOML for Office')
    parser.add_argument('--dataroot', default='', type=str, help='data root')
    parser.add_argument('--dataset', default='office-31', type=str, help='office-31, office-home')
    parser.add_argument('--gpu_id', default='6', help='gpu_id') 
    parser.add_argument('--model', default='MTAN', type=str, help='DMTL, MTAN')
    parser.add_argument('--MGDA', action='store_true', help='MGDA in UL')
    parser.add_argument('--batchsize', default=64, type=int, help='batchsize') 
    return parser.parse_args()

params = parse_args()
print(params)

os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_id
            
        
if params.dataset == 'office-31':
    task_num, class_num, batchsize = 3, 31, params.batchsize
elif params.dataset == 'office-home':
    task_num, class_num, batchsize = 4, 65, params.batchsize
data_loader, iter_data_loader = office_dataloader(params.dataset, batchsize=batchsize, dataroot=params.dataroot)
def build_model(model):
    if model == 'DMTL':
        model = DMTL(task_num=task_num, class_num=class_num).cuda()
    elif model == 'MTAN':
        model = MTAN_ResNet(task_num, class_num).cuda()
    return model
        
model = build_model(params.model)
weight_optimizer = optim.Adam(model.parameters(), lr=1e-4)

h = Model_h(task_num=task_num).cuda()
h.train()

h_optimizer = torch.optim.Adam(h.parameters(), lr=1e-3)

total_epoch = 200
trval_batch = max(len(data_loader[i]['trval']) for i in range(task_num))
loss_fn = nn.CrossEntropyLoss().cuda()
best_val_acc, best_test_acc, early_count = 0, 0, 0
for epoch in range(total_epoch):
    print('--- Epoch {}'.format(epoch))
    s_t = time.time()

    # iteration for all batches
    model.train()
    for batch_index in range(trval_batch):
        
        meta_model = build_model(params.model)
        meta_model.load_state_dict(model.state_dict())
                
        model_np = {}
        for n, p in meta_model.named_parameters():
            model_np[n] = p
        
        loss_train = torch.zeros(task_num).cuda()
        for task_index in range(task_num):
            try:
                train_data, train_label = iter_data_loader[task_index]['train'].next()
            except:
                iter_data_loader[task_index]['train'] = iter(data_loader[task_index]['train'])
                train_data, train_label = iter_data_loader[task_index]['train'].next()
            train_data, train_label = train_data.cuda(non_blocking=True), train_label.cuda(non_blocking=True)
            loss_train[task_index] = loss_fn(meta_model(train_data, task_index), train_label)
        loss = h(loss_train)

        meta_model.zero_grad()
        grads = torch.autograd.grad(loss, (meta_model.parameters()), create_graph=True)

        for g_index, name in enumerate(model_np.keys()):
            p = set_param(meta_model, name, mode='get')
            p_fast = p - 1e-4 * grads[g_index]
            set_param(meta_model, name, param=p_fast, mode='update')
            model_np[name] = p_fast
        del grads, model_np, train_data, train_label
        
        # update outer loop
        grads = {}
        loss_valid_data = {}
        val_loss = torch.zeros(task_num).cuda()
        for task_index in range(task_num):
            try:
                val_data, val_label = iter_data_loader[task_index]['val'].next()
            except:
                iter_data_loader[task_index]['val'] = iter(data_loader[task_index]['val'])
                val_data, val_label = iter_data_loader[task_index]['val'].next()
            val_data, val_label = val_data.cuda(non_blocking=True), val_label.cuda(non_blocking=True)
            val_loss[task_index] = loss_fn(meta_model(val_data, task_index), val_label)
            if params.MGDA:
                grads[task_index] = torch.autograd.grad(val_loss[task_index], h.parameters(), retain_graph=True)[0]
                loss_valid_data[task_index] = val_loss[task_index].item()
        del val_data, val_label
        if params.MGDA:
            gn = gradient_normalizers(grads, loss_valid_data, normalization_type='l2')
            for kn in range(task_num):
                grads[kn] = grads[kn] / gn[kn]

            sol, _ = MinNormSolver.find_min_norm_element([grads[kn] for kn in range(task_num)])
            del grads, gn, loss_valid_data
        else:
            sol = [1]*task_num
        
        loss_sum = (torch.stack([float(sol[k]) * val_loss[k] for k in range(task_num)])).sum()
        h_optimizer.zero_grad()
        loss_sum.backward()
        h_optimizer.step()
        del val_loss, loss_sum, meta_model
        
        # update inner loop
        loss_train = torch.zeros(task_num).cuda()
        for task_index in range(task_num):
            try:
                trval_data, trval_label = iter_data_loader[task_index]['trval'].next()
            except:
                iter_data_loader[task_index]['trval'] = iter(data_loader[task_index]['trval'])
                trval_data, trval_label = iter_data_loader[task_index]['trval'].next()
            trval_data, trval_label = trval_data.cuda(non_blocking=True), trval_label.cuda(non_blocking=True)
            loss_train[task_index] = loss_fn(model(trval_data, task_index), trval_label)
        del trval_data, trval_label
        loss = h(loss_train)
        weight_optimizer.zero_grad()
        loss.backward()
        weight_optimizer.step()       

    model.eval()
    with torch.no_grad(): 
        right_num = np.zeros([2, task_num])
        count = np.zeros([2, task_num])
        loss_data_count = np.zeros([2, task_num])
        for mode_index, mode in enumerate(['val', 'test']):
            for k in range(task_num):
                for test_it, test_data in enumerate(data_loader[k][mode]):
                    x_test, y_test = test_data[0].cuda(non_blocking=True), test_data[1].cuda(non_blocking=True)
                    y_pred = model(x_test, k)
                    loss_t = loss_fn(y_pred, y_test)
                    loss_data_count[mode_index, k] += loss_t.item()
                    right_num[mode_index, k] += ((torch.max(F.softmax(y_pred, dim=-1), dim=-1)[1])==y_test).sum().item()
                    count[mode_index, k] += y_test.shape[0]
        acc_avg = (right_num/count).mean(axis=-1)
        loss_data_avg = (loss_data_count/count).mean(axis=-1)
        print('val acc {} {}, loss {}'.format(right_num[0]/count[0], acc_avg[0], loss_data_count[0]))
        print('test acc {} {}, loss {}'.format(right_num[1]/count[1], acc_avg[1], loss_data_count[1]))
    e_t = time.time()
    print('-- cost time {}'.format(e_t-s_t))
    if acc_avg[0] > best_val_acc:
        best_val_acc = acc_avg[0]
        early_count = 0
        print('-- -- epoch {} ; best val {} {} ; test acc {} {}'.format(epoch, right_num[0]/count[0], acc_avg[0],
                                                                         right_num[1]/count[1], acc_avg[1]))
    else:
        early_count += 1
#     if count > 8:
#         break
    if acc_avg[1] > best_test_acc:
        best_test_acc = acc_avg[1]
        print('!! -- -- epoch {}; best test acc {} {}'.format(epoch, right_num[1]/count[1], acc_avg[1]))
    print(h.weight)
