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
#rnn = 'crnn'
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


def compute_binary_accuracy(model, data_loader):
    correct_pred, num_examples, FP, FN = 0., 0, 0, 0
    FP_list = []
    FN_list = []
    model.eval()
    with torch.no_grad():
        if rnn == 'frameGRU':
            for batch, labels, lengths in data_loader:
                logits = model(batch.cuda(), lengths)
                out = torch.sigmoid(logits)
                # if rnn == 'frameGRU':
                #     new_out_list = []
                #     new_labels_list = []
                #     for i in range(len(lengths)):
                #         new_out_list.append(out[i][:lengths[i]].sum())
                #     out = torch.cat(new_out_list, 0)
                if rnn == 'frameGRU':
                    new_labels_list = []
                    new_logits_list = []
                    new_out_list = []
                    for i in range(len(lengths)):
                        new_labels_list += [labels[i]] * lengths[i]
                        new_logits_list.append(out[i][:lengths[i]])
                        new_out_list.append(out[i][:lengths[i]].mean(0, keepdim=True))
                    # import pdb; pdb.set_trace()
                    logits_framewise = torch.cat(new_logits_list, 0)
                    labels_framewise = new_labels_list
                    out = torch.cat(new_out_list, 0)
                predicted_labels = (out > 0.5).long()
                num_examples += len(lengths)
                correct_pred += (predicted_labels.squeeze(1).cpu().long() == torch.LongTensor(labels)).sum()
            return correct_pred.float().item()/num_examples * 100, total_loss
        else:
            for batch, labels, lengths, f_names in data_loader:
                logits = model(batch.cuda(), lengths)
                predicted_labels = (torch.sigmoid(logits) > 0.5).long()
                num_examples += len(lengths)
                #import pdb; pdb.set_trace()
                if predicted_labels.squeeze(1).cpu().long() == torch.LongTensor(labels):
                    correct_pred += 1
                elif labels == 0:
                    FP += 1
                    FP_list.append(f_names[0]+'_'+str(labels.item())+'_'+str(predicted_labels.squeeze(1).cpu().long().item()))
                else:
                    FN += 1
                    FN_list.append(f_names[0]+'_'+str(labels.item())+'_'+str(predicted_labels.squeeze(1).cpu().long().item()))
            return correct_pred/num_examples * 100, FP, FN, FP_list, FN_list



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

best_test_acc = 0.

train_acc, train_fp, train_fn, train_fp_list, train_fn_list = compute_binary_accuracy(model, dataloader_train)
test_acc, test_fp, test_fn, test_fp_list, test_fn_list = compute_binary_accuracy(model, dataloader_test)
print('train_acc,{:.2f}%,train_fp,{},train_fn,{}\nvalid_acc,{:.2f}%,valid_fp,{},valid_fn,{}\n'
      .format(train_acc, train_fp, train_fn, test_acc, test_fp, test_fn))
print('Test FP')
for n in test_fp_list:
    print(n)
print('\nTest FN')
for n in test_fn_list:
    print(n)