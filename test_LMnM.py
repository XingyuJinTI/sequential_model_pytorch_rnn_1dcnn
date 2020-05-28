import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from dataset import LandmarkList, LandmarkListTest
from torch.utils import data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import argparse
import pickle
import os
import os.path


from model import *

DEVICES = 0
torch.cuda.set_device(DEVICES)


def default_loader(path):
    with open(path, 'rb') as fp:
        lm_list = pickle.load(fp)
    fp.close()
    return lm_list


def default_list_reader(fileList):
    lmList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            lmPath=line.strip()[:-2].strip()
            label=line.strip()[-1]

            lmList.append((lmPath, int(label)))
    return lmList


def lm_m_list_reader(fileList):
    lmList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            mPath=line.strip()[:-2].strip()
            label=line.strip()[-1]
            lmPath = mPath[:-8] + "vid.txt"
            lmList.append((mPath, lmPath, int(label)))
    return lmList
#
# class LM_M_List(data.Dataset):
#     def __init__(self, root_LM, root_M, fileList, transform=None, list_reader=lm_m_list_reader, loader=default_loader):
#         self.root_LM  = root_LM
#         self.root_M  = root_M
#         self.lmList   = list_reader(fileList)
#         self.transform = transform
#         self.loader    = loader
#
#     def __getitem__(self, index):
#         mPath, lmPath, target = self.lmList[index]
#         lm = self.loader(os.path.join(self.root_LM, lmPath))
#         m = self.loader(os.path.join(self.root_M, mPath))
#         if self.transform is not None:
#             lm = F.normalize(lm, dim=0)
#         return lm, m, target, lm.shape[0]
#
#     def __len__(self):
#         return len(self.lmList)


class LM_M_ListListTest(data.Dataset):
    def __init__(self, root_LM, root_M, fileList, transform=None, list_reader=lm_m_list_reader, loader=default_loader):
        self.root_LM  = root_LM
        self.root_M  = root_M
        self.lmList   = list_reader(fileList)
        self.transform = transform
        self.loader    = loader

    def __getitem__(self, index):
        mPath, lmPath, target = self.lmList[index]
        lm = self.loader(os.path.join(self.root_LM, lmPath))
        m = self.loader(os.path.join(self.root_M, mPath))
        if self.transform is not None:
            lm = F.normalize(lm, dim=0)
        return lm, m, target, lm.shape[0], lmPath

    def __len__(self):
        return len(self.lmList)


class LM_exclM_ListTest(data.Dataset):
    def __init__(self, root, fileList, transform=None, list_reader=default_list_reader, loader=default_loader):
        self.root      = root
        self.lmList   = list_reader(fileList)
        self.transform = transform
        self.loader    = loader

    def __getitem__(self, index):
        lmPath, target = self.lmList[index]
        lm = self.loader(os.path.join(self.root, lmPath))
        if self.transform is not None:
            lm = F.normalize(lm, dim=0)
        return lm, target, lm.shape[0], lmPath

    def __len__(self):
        return len(self.lmList)


def compute_binary_accuracy(model_M, model_LM, data_loader, th_list):
    len_th_list = len(th_list)
    correct_pred, num_examples, FP, FN = [0.]*len_th_list, 0, [0]*len_th_list, [0]*len_th_list
    FP_list = []
    FN_list = []
    for _ in range(len_th_list):
        FP_list.append([])
        FN_list.append([])
    model_M.eval() # M
    model_LM.eval() # LM
    with torch.no_grad():
        for batch_lm, batch_m, labels, lengths, f_names in data_loader:

            # import pdb; pdb.set_trace()
            batch_lm = F.interpolate(batch_lm[0].unsqueeze(0).permute(0, 2, 1), size=model_LM.sample,
                                  mode='nearest').permute(0, 2, 1)
            batch_m = F.interpolate(batch_m[0].unsqueeze(0).permute(0, 2, 1), size=model_M.sample,
                                  mode='nearest').permute(0, 2, 1)
            # import pdb; pdb.set_trace()
            logits_LM = model_LM(batch_lm.cuda(), lengths)
            logits_M = model_M(batch_m.cuda(), lengths)
            num_examples += len(lengths)
            for i, th in enumerate(th_list):
                prob = th * torch.sigmoid(logits_LM) + (1-th) * torch.sigmoid(logits_M)
                predicted_labels = (prob > 0.5).long()
                # import pdb;pdb.set_trace()
                if predicted_labels.squeeze(1).cpu().long() == torch.LongTensor(labels):
                    correct_pred[i] += 1
                elif labels == 0:
                    FP[i] += 1
                    FP_list[i].append(f_names[0] + '_' + str(labels.item()) + '_' + str(
                        prob.squeeze(1).cpu().item()))
                else:
                    FN[i] += 1
                    FN_list[i].append(f_names[0] + '_' + str(labels.item()) + '_' + str(
                        prob.squeeze(1).cpu().item()))
        return [n_correct/num_examples * 100 for n_correct in correct_pred], FP, FN, FP_list, FN_list, correct_pred, num_examples


def compute_binary_accuracy_wo_M(model_LM, data_loader, th_list):
    len_th_list = len(th_list)
    correct_pred, num_examples, FP, FN = [0.]*len_th_list, 0, [0]*len_th_list, [0]*len_th_list
    FP_list = []
    FN_list = []
    for _ in range(len_th_list):
        FP_list.append([])
        FN_list.append([])
    # model_M.eval() # M
    model_LM.eval() # LM
    with torch.no_grad():
        for batch_lm, labels, lengths, f_names in data_loader:

            # import pdb; pdb.set_trace()
            batch_lm = F.interpolate(batch_lm[0].unsqueeze(0).permute(0, 2, 1), size=model_LM.sample,
                                  mode='nearest').permute(0, 2, 1)
            # import pdb; pdb.set_trace()
            logits_LM = model_LM(batch_lm.cuda(), lengths)
            num_examples += len(lengths)
            for i, th in enumerate(th_list):
                prob = torch.sigmoid(logits_LM)
                predicted_labels = (prob > 0.5).long()
                # import pdb;pdb.set_trace()
                if predicted_labels.squeeze(1).cpu().long() == torch.LongTensor(labels):
                    correct_pred[i] += 1
                elif labels == 0:
                    FP[i] += 1
                    FP_list[i].append(f_names[0] + '_' + str(labels.item()) + '_' + str(
                        prob.squeeze(1).cpu().item()))
                else:
                    FN[i] += 1
                    FN_list[i].append(f_names[0] + '_' + str(labels.item()) + '_' + str(
                        prob.squeeze(1).cpu().item()))
        return [n_correct/num_examples * 100 for n_correct in correct_pred], FP, FN, FP_list, FN_list, correct_pred, num_examples




model_M = cnn_Classifier(4, 6, 12, 1, sample=128)
model_LM = cnn_Classifier(8, 68*2, 68*2*2, 1, sample=192)
print('Model_M is loading...')
model_M.load_state_dict(torch.load("models/motion_models/cnn_L4_LM6_frame128.pt"))
print('Model_LM is loading...')
model_LM.load_state_dict(torch.load("models/landmark_models/cnn_L8_LM68_frame192.pt"))
print('Model Loading - DONE...')
# model.load_state_dict(torch.load("models/"+str(rnn)+'_L' + str(N_LAYERS_RNN) + "_GC.pt"))
model_M = model_M.cuda()
model_LM = model_LM.cuda()

loss_function = torch.nn.BCEWithLogitsLoss()
loss_function_eval_sum = torch.nn.BCEWithLogitsLoss(reduction='sum')
# optimizer = optim.Adam(model.parameters(), lr=LR)

dataset_train = LM_M_ListListTest(root_M='/datasets/mc_m_min16frame/', root_LM='/datasets/move_closer/Data_Landmark/', fileList='/datasets/mc_m_min16frame/TrainListM.txt')
dataloader_train = data.DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0)#, collate_fn=pad_collate)

dataset_test = LM_M_ListListTest(root_M='/datasets/mc_m_min16frame/', root_LM='/datasets/move_closer/Data_Landmark/', fileList='/datasets/mc_m_min16frame/TestListM.txt')
dataloader_test = data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)#, collate_fn=pad_collate)


dataset_train_wo_M = LM_exclM_ListTest(root='/datasets/move_closer/Data_Landmark/', fileList='/datasets/mc_m_min16frame/TrainListLM_wo_M.txt')
dataloader_train_wo_M = data.DataLoader(dataset_train_wo_M, batch_size=1, shuffle=False, num_workers=0)#, collate_fn=pad_collate)

dataset_test_wo_M = LM_exclM_ListTest(root='/datasets/move_closer/Data_Landmark/', fileList='/datasets/mc_m_min16frame/TestListLM_wo_M.txt')
dataloader_test_wo_M = data.DataLoader(dataset_test_wo_M, batch_size=1, shuffle=False, num_workers=0)#, collate_fn=pad_collate)


# thresholds = [x * 0.01 for x in range(30, 71)]
# thresholds = [0.5]
# alphas = [x * 0.05 for x in range(21)]
alphas = [0,0.60,1]

# train_acc, train_fp, train_fn, train_fp_list, train_fn_list = compute_binary_accuracy(model, dataloader_train, thresholds)
# test_acc, test_fp, test_fn, test_fp_list, test_fn_list = compute_binary_accuracy(model, dataloader_test, thresholds)
train_acc, train_fp, train_fn, train_fp_list, train_fn_list, train_correct_pred, train_num_examples = compute_binary_accuracy(model_M, model_LM, dataloader_train, alphas)
test_acc, test_fp, test_fn, test_fp_list, test_fn_list, test_correct_pred, test_num_examples = compute_binary_accuracy(model_M, model_LM, dataloader_test, alphas)
train_acc_wo_M, train_fp_wo_M, train_fn_wo_M, train_fp_list_wo_M, train_fn_list_wo_M, train_wo_M_correct_pred, train_wo_M_num_examples = compute_binary_accuracy_wo_M(model_LM, dataloader_train_wo_M, [.5])
test_acc_wo_M, test_fp_wo_M, test_fn_wo_M, test_fp_list_wo_M, test_fn_list_wo_M, test_wo_M_correct_pred, test_wo_M_num_examples  = compute_binary_accuracy_wo_M(model_LM, dataloader_test_wo_M, [.5])

for i in range(0, len(alphas)):
    print('\n\n-----------------Eval for alpha of {:.2f}-------------------\n\n'.format(alphas[i]))
    print('train_acc,{:.2f}%,train_fp,{},train_fn,{}, correct_pred,{},num_examples{}\nvalid_acc,{:.2f}%,valid_fp,{},valid_fn,{},correct_pred,{},num_examples,{}\n'
          .format(train_acc[i], train_fp[i], train_fn[i], train_correct_pred[i], train_num_examples, test_acc[i], test_fp[i], test_fn[i], test_correct_pred[i], test_num_examples))
    # if i == len(alphas)-1:
    print('Train FP')
    for n in train_fp_list[i]:
        print(n)
    print('\nTrain FN')
    for n in train_fn_list[i]:
        print(n)

    print('\n\n\nTest FP')
    for n in test_fp_list[i]:
        print(n)
    print('\nTest FN')
    for n in test_fn_list[i]:
        print(n)

print('\n\n\n\n-----------------Eval for those without Motion-------------------\n\n'.format(alphas[i]))
print('train_acc,{:.2f}%,train_fp,{},train_fn,{}, correct_pred,{},num_examples{}\nvalid_acc,{:.2f}%,valid_fp,{},valid_fn,{},correct_pred,{},num_examples{}\n'
      .format(train_acc_wo_M[0], train_fp_wo_M[0], train_fn_wo_M[0], train_wo_M_correct_pred[0], train_wo_M_num_examples, test_acc_wo_M[0], test_fp_wo_M[0], test_fn_wo_M[0], test_wo_M_correct_pred[0], test_wo_M_num_examples))
# if i == len(alphas)-1:
print('Train FP')
for n in train_fp_list_wo_M[0]:
    print(n)
print('\nTrain FN')
for n in train_fn_list_wo_M[0]:
    print(n)

print('\n\n\nTest FP')
for n in test_fp_list_wo_M[0]:
    print(n)
print('\nTest FN')
for n in test_fn_list_wo_M[0]:
    print(n)

