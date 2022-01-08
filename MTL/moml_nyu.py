import torch, time, os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from backbone import DeepLabv3, MTANDeepLabv3, Model_h
from utils import *

from create_dataset import NYUv2
from min_norm_solvers import MinNormSolver, gradient_normalizers

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description= 'MOML for NYUv2')
    parser.add_argument('--dataset_path', default='', type=str, help='dataset path')
    parser.add_argument('--gpu_id', default='6', help='gpu_id') 
    parser.add_argument('--model', default='DMTL', type=str, help='DMTL, MTAN')
    parser.add_argument('--aug', action='store_true', default=False, help='data augmentation')
    parser.add_argument('--MGDA', action='store_true', help='MGDA in UL')
    return parser.parse_args()

params = parse_args()
print(params)

os.environ["CUDA_VISIBLE_DEVICES"] = params.gpu_id
            

dataset_path = params.dataset_path
if params.model == 'DMTL':
    batch_size = 8
elif params.model == 'MTAN':
    batch_size = 4

nyuv2_train_set = NYUv2(root=dataset_path, mode='train', augmentation=params.aug)
nyuv2_val_set = NYUv2(root=dataset_path, mode='val', augmentation=params.aug)
nyuv2_trval_set = NYUv2(root=dataset_path, mode='trainval', augmentation=params.aug)
nyuv2_test_set = NYUv2(root=dataset_path, mode='test', augmentation=params.aug)

nyuv2_test_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_test_set,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True)

nyuv2_train_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True)

nyuv2_trval_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_trval_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    drop_last=True)

nyuv2_val_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_val_set,
    batch_size=batch_size,
    shuffle=True, 
    num_workers=2,
    pin_memory=True)

def build_model(model):
    if model == 'DMTL':
        model = DeepLabv3().cuda()
    elif model == 'MTAN':
        model = MTANDeepLabv3().cuda()
    return model
        
model = build_model(params.model)
task_num = len(model.tasks)
weight_optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(weight_optimizer, step_size=100, gamma=0.5)

h = Model_h(task_num=task_num).cuda()
h.train()

h_optimizer = torch.optim.Adam(h.parameters(), lr=1e-4)

print('LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR | NORMAL_LOSS MEAN MED <11.25 <22.5 <30')
total_epoch = 500
trval_batch = len(nyuv2_trval_loader)
# val_batch = len(nyuv2_val_loader)
avg_cost = torch.zeros([total_epoch, 24])
for index in range(total_epoch):
    s_t = time.time()
    cost = torch.zeros(24)

    # iteration for all batches
    model.train()
    trval_dataset = iter(nyuv2_trval_loader)
    conf_mat = ConfMatrix(model.class_nb)
    for k in range(trval_batch):
        
        meta_model = build_model(params.model)
        meta_model.load_state_dict(model.state_dict())
        
        model_np = {}
        for n, p in meta_model.named_parameters():
            model_np[n] = p
        
        try:            
            train_data, train_label, train_depth, train_normal = train_dataset_iter.next()
        except:
            train_dataset_iter = iter(nyuv2_train_loader)
            train_data, train_label, train_depth, train_normal = train_dataset_iter.next()
        train_data, train_label = train_data.cuda(non_blocking=True), train_label.long().cuda(non_blocking=True)
        train_depth, train_normal = train_depth.cuda(non_blocking=True), train_normal.cuda(non_blocking=True)

        train_pred = meta_model(train_data)

        train_loss = [model_fit(train_pred[0], train_label, 'semantic'),
                      model_fit(train_pred[1], train_depth, 'depth'),
                      model_fit(train_pred[2], train_normal, 'normal')]
        loss_train = torch.zeros(3).cuda()
        for i in range(3):
            loss_train[i] = train_loss[i]
        loss = h(loss_train)

        meta_model.zero_grad()
        grads = torch.autograd.grad(loss, (meta_model.parameters()), create_graph=True)

        for g_index, name in enumerate(model_np.keys()):
            p = set_param(meta_model, name, mode='get')
            p_fast = p - 1e-4 * grads[g_index]
            set_param(meta_model, name, param=p_fast, mode='update')
            model_np[name] = p_fast
        del grads, model_np
        del train_data, train_label, train_depth, train_normal
        
        # update outer loop
        try:
            val_data, val_label, val_depth, val_normal = val_dataset_iter.next()
        except:
            val_dataset_iter = iter(nyuv2_val_loader)
            val_data, val_label, val_depth, val_normal = val_dataset_iter.next()
        val_data, val_label = val_data.cuda(non_blocking=True), val_label.long().cuda(non_blocking=True)
        val_depth, val_normal = val_depth.cuda(non_blocking=True), val_normal.cuda(non_blocking=True)
        valid_pred = meta_model(val_data)
        valid_loss = [model_fit(valid_pred[0], val_label, 'semantic'),
                      model_fit(valid_pred[1], val_depth, 'depth'),
                      model_fit(valid_pred[2], val_normal, 'normal')]
        del val_data, val_label, val_depth, val_normal
        # for MGDA
        if params.MGDA:
            grads = {}
            loss_valid_data = {}
            for kn in range(task_num):
                grads[kn] = torch.autograd.grad(valid_loss[kn], h.parameters(), retain_graph=True)[0]
                loss_valid_data[kn] = valid_loss[kn].item()
            gn = gradient_normalizers(grads, loss_valid_data, normalization_type='loss')
            for kn in range(task_num):
                grads[kn] = grads[kn] / gn[kn]

            sol, _ = MinNormSolver.find_min_norm_element([grads[kn] for kn in range(task_num)])
            del grads, gn, loss_valid_data
        else:
            sol = [1]*task_num
        loss_sum = (torch.stack([float(sol[k]) * valid_loss[k] for k in range(task_num)])).sum()
        h_optimizer.zero_grad()
        loss_sum.backward()
        h_optimizer.step()
        del valid_loss, loss_sum, meta_model
        
        # update inner loop
        trval_data, trval_label, trval_depth, trval_normal = trval_dataset.next()
        trval_data, trval_label = trval_data.cuda(non_blocking=True), trval_label.long().cuda(non_blocking=True)
        trval_depth, trval_normal = trval_depth.cuda(non_blocking=True), trval_normal.cuda(non_blocking=True)
        trval_pred = model(trval_data)
        trval_loss = [model_fit(trval_pred[0], trval_label, 'semantic'),
                      model_fit(trval_pred[1], trval_depth, 'depth'),
                      model_fit(trval_pred[2], trval_normal, 'normal')]
        loss_final = torch.zeros(3).cuda()
        for i in range(3):
            loss_final[i] = trval_loss[i]
        loss = h(loss_final)
        weight_optimizer.zero_grad()
        loss.backward()
        weight_optimizer.step()        

        # accumulate label prediction for every pixel in training images
        conf_mat.update(trval_pred[0].argmax(1).flatten(), trval_label.flatten())

        cost[0] = trval_loss[0].item()
        cost[3] = trval_loss[1].item()
        cost[4], cost[5] = depth_error(trval_pred[1], trval_depth)
        cost[6] = trval_loss[2].item()
        cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(trval_pred[2], trval_normal)
        avg_cost[index, :12] += cost[:12] / trval_batch
        
        del trval_data, trval_label, trval_depth, trval_normal, trval_pred, trval_loss, loss_final, loss

    # compute mIoU and acc
    avg_cost[index, 1], avg_cost[index, 2] = conf_mat.get_metrics()

    # evaluating test data
    model.eval()
    conf_mat = ConfMatrix(model.class_nb)
    with torch.no_grad():  # operations inside don't track history
        val_dataset = iter(nyuv2_test_loader)
        val_batch = len(nyuv2_test_loader)
        for k in range(val_batch):
            val_data, val_label, val_depth, val_normal = val_dataset.next()
            val_data, val_label = val_data.cuda(non_blocking=True), val_label.long().cuda(non_blocking=True)
            val_depth, val_normal = val_depth.cuda(non_blocking=True), val_normal.cuda(non_blocking=True)

            val_pred = model(val_data)
            val_loss = [model_fit(val_pred[0], val_label, 'semantic'),
                         model_fit(val_pred[1], val_depth, 'depth'),
                         model_fit(val_pred[2], val_normal, 'normal')]

            conf_mat.update(val_pred[0].argmax(1).flatten(), val_label.flatten())

            cost[12] = val_loss[0].item()
            cost[15] = val_loss[1].item()
            cost[16], cost[17] = depth_error(val_pred[1], val_depth)
            cost[18] = val_loss[2].item()
            cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(val_pred[2], val_normal)
            avg_cost[index, 12:] += cost[12:] / val_batch

        # compute mIoU and acc
        avg_cost[index, 13], avg_cost[index, 14] = conf_mat.get_metrics()

    scheduler.step()
    e_t = time.time()
    print('Epoch: {:04d} | TRAIN: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} ||'
        'TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} || {:.4f}'
        .format(index, avg_cost[index, 0], avg_cost[index, 1], avg_cost[index, 2], avg_cost[index, 3],
                avg_cost[index, 4], avg_cost[index, 5], avg_cost[index, 6], avg_cost[index, 7], avg_cost[index, 8],
                avg_cost[index, 9], avg_cost[index, 10], avg_cost[index, 11], avg_cost[index, 12], avg_cost[index, 13],
                avg_cost[index, 14], avg_cost[index, 15], avg_cost[index, 16], avg_cost[index, 17], avg_cost[index, 18],
                avg_cost[index, 19], avg_cost[index, 20], avg_cost[index, 21], avg_cost[index, 22], avg_cost[index, 23], e_t-s_t))
    print(h.weight)
