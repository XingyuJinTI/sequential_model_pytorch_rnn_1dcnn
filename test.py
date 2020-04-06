import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import LandmarkList, LandmarkListTest
from torch.utils import data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import argparse

from model import *

# rnn = 'frameGRU'
# rnn = 'sumGRU'
# rnn = 'crnn'
# rnn = 'cnn'
# rnn = 'GRU'
# rnn = 'framewise_GRU'
# rnn = 'embedGRU'
rnn = 'biGRU'
# rnn = 'LSTM'
EMBEDDING_DIM = 68*2
HIDDEN_DIM = 68*2* 2
N_LAYERS_RNN = 3
LR = 1e-4
DEVICES = 0
torch.cuda.set_device(DEVICES)


def compute_binary_accuracy(model, data_loader, th_list):
    len_th_list = len(th_list)
    correct_pred, num_examples, FP, FN = [0.]*len_th_list, 0, [0]*len_th_list, [0]*len_th_list
    FP_list = []
    FN_list = []
    for _ in range(len_th_list):
        FP_list.append([])
        FN_list.append([])
    model.eval()
    with torch.no_grad():
        if rnn == 'frameGRU':
            for batch, labels, lengths, f_names in data_loader:
                logits = model(batch.cuda(), lengths)
                out = torch.sigmoid(logits)
                new_out_list = []
                for i in range(len(lengths)):
                    new_out_list.append(out[i][:lengths[i]].mean(0, keepdim=True))
                # import pdb; pdb.set_trace()
                out = torch.cat(new_out_list, 0)
                num_examples += len(lengths)
                for i, th in enumerate(th_list):
                    predicted_labels = (out > th).long()
                    if predicted_labels.squeeze(1).cpu().long() == torch.LongTensor(labels):
                        correct_pred[i] += 1
                    elif labels == 0:
                        # print('FP: ', FP)
                        FP[i] += 1
                        FP_list[i].append(f_names[0] + '_' + str(labels.item()) + '_' + str(
                            out.squeeze(1).cpu().item()))
                    else:
                        # print('FN: ', FN)
                        FN[i] += 1
                        FN_list[i].append(f_names[0] + '_' + str(labels.item()) + '_' + str(
                            out.squeeze(1).cpu().item()))
            return [n_correct/num_examples * 100 for n_correct in correct_pred], FP, FN, FP_list, FN_list
        else:
            for batch, labels, lengths, f_names in data_loader:
                #import pdb; pdb.set_trace()
                logits = model(batch.cuda(), lengths)
                num_examples += len(lengths)
                for i, th in enumerate(th_list):
                    predicted_labels = (torch.sigmoid(logits) > th).long()
                    if predicted_labels.squeeze(1).cpu().long() == torch.LongTensor(labels):
                        correct_pred[i] += 1
                    elif labels == 0:
                        FP[i] += 1
                        FP_list[i].append(f_names[0]+'_'+str(labels.item())+'_'+str(torch.sigmoid(logits).squeeze(1).cpu().item()))
                    else:
                        FN[i] += 1
                        FN_list[i].append(f_names[0]+'_'+str(labels.item())+'_'+str(torch.sigmoid(logits).squeeze(1).cpu().item()))
            return [n_correct/num_examples * 100 for n_correct in correct_pred], FP, FN, FP_list, FN_list



if rnn == 'frameGRU':
    model = Framewise_GRU_Classifier(EMBEDDING_DIM, HIDDEN_DIM, 1, n_layer=N_LAYERS_RNN)
if rnn == 'sumGRU':
    model = sumGRU(EMBEDDING_DIM, HIDDEN_DIM, 1, n_layer=N_LAYERS_RNN)
if rnn == 'embedGRU':
    model = embed_GRU_Classifier(EMBEDDING_DIM, HIDDEN_DIM, 1, n_layer=N_LAYERS_RNN)
if rnn == 'GRU':
    model = GRU_Classifier(EMBEDDING_DIM, HIDDEN_DIM, 1, n_layer=N_LAYERS_RNN)
if rnn == 'biGRU':
    model = biGRU_Classifier(EMBEDDING_DIM, HIDDEN_DIM, 1, n_layer=N_LAYERS_RNN)
if rnn == 'LSTM':
    model = LSTM_Classifier(EMBEDDING_DIM, HIDDEN_DIM, 1, n_layer=N_LAYERS_RNN)
if rnn == 'cnn':
    model = cnn_Classifier(EMBEDDING_DIM, HIDDEN_DIM, 1)
if rnn == 'crnn':
    model = crnn_Classifier(EMBEDDING_DIM, HIDDEN_DIM, 1, n_layer=N_LAYERS_RNN)
model.load_state_dict(torch.load("models/"+str(rnn)+".pt"))
model = model.cuda()

loss_function = torch.nn.BCEWithLogitsLoss()
loss_function_eval_sum = torch.nn.BCEWithLogitsLoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=LR)

dataset_train = LandmarkListTest(root='/datasets/move_closer/Data_Landmark/', fileList='/datasets/move_closer/TrainList.txt')
dataloader_train = data.DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0)

dataset_test = LandmarkListTest(root='/datasets/move_closer/Data_Landmark/', fileList='/datasets/move_closer/TestList.txt')
dataloader_test = data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)

thresholds = [x * 0.01 for x in range(30, 71)]

train_acc, train_fp, train_fn, train_fp_list, train_fn_list = compute_binary_accuracy(model, dataloader_train, thresholds)
test_acc, test_fp, test_fn, test_fp_list, test_fn_list = compute_binary_accuracy(model, dataloader_test, thresholds)

for i in range(0, len(thresholds)):
    print('\n\n-----------------Eval for threshold of {:.2f}-------------------\n\n'.format(thresholds[i]))
    print('train_acc,{:.2f}%,train_fp,{},train_fn,{}\nvalid_acc,{:.2f}%,valid_fp,{},valid_fn,{}\n'
          .format(train_acc[i], train_fp[i], train_fn[i], test_acc[i], test_fp[i], test_fn[i]))
    # print('Train FP')
    # for n in train_fp_list[i]:
    #     print(n)
    # print('\nTrain FN')
    # for n in train_fn_list[i]:
    #     print(n)
    #
    # print('\n\n\nTest FP')
    # for n in test_fp_list[i]:
    #     print(n)
    # print('\nTest FN')
    # for n in test_fn_list[i]:
    #     print(n)

