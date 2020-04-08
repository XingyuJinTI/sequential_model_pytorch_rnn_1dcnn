import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import LandmarkList
from torch.utils import data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import argparse

from model import *

# rnn = 'frameGRU'
# to be implemented - rnn = 'frameCRNN'
# rnn = 'sumGRU'
rnn = 'crnn'
# rnn = 'cnn'
# rnn = 'GRU'
# rnn = 'embedGRU'
# rnn = 'biGRU'
# rnn = 'LSTM'
EMBEDDING_DIM = 68*2
HIDDEN_DIM = 68*2* 2
N_LAYERS_RNN = 1
MAX_EPOCH = 1000
LR = 1e-4
DEVICES = 3
SAVE_BEST_MODEL = True
Grad_Clip_For_All = False
torch.cuda.set_device(DEVICES)


def compute_binary_accuracy(model, data_loader, loss_function):
    correct_pred, num_examples, total_loss = 0, 0, 0.
    model.eval()
    with torch.no_grad():
        if rnn == 'frameGRU' or rnn == 'frameCRNN':
            for batch, labels, lengths in data_loader:
                logits = model(batch.cuda(), lengths)
                out = torch.sigmoid(logits)
                # if rnn == 'frameGRU':
                #     new_out_list = []
                #     new_labels_list = []
                #     for i in range(len(lengths)):
                #         new_out_list.append(out[i][:lengths[i]].sum())
                #     out = torch.cat(new_out_list, 0)
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
                total_loss += loss_function(logits_framewise, torch.FloatTensor(labels_framewise).unsqueeze(1).cuda()).item()
                predicted_labels = (out > 0.5).long()
                num_examples += len(lengths)
                correct_pred += (predicted_labels.squeeze(1).cpu().long() == torch.LongTensor(labels)).sum()
            return correct_pred.float().item()/num_examples * 100, total_loss
        else:
            for batch, labels, lengths in data_loader:
                logits = model(batch.cuda(), lengths)
                total_loss += loss_function(logits, torch.FloatTensor(labels).unsqueeze(1).cuda()).item()
                predicted_labels = (torch.sigmoid(logits) > 0.5).long()
                num_examples += len(lengths)
                correct_pred += (predicted_labels.squeeze(1).cpu().long() == torch.LongTensor(labels)).sum()
            return correct_pred.float().item()/num_examples * 100, total_loss


def pad_collate(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    lms, tgs, lens = zip(*batch)
    new_lms = torch.zeros((len(lms), lms[0].shape[0], lms[0].shape[1])) # batch x seq x feature(136)
    new_lms[0] = lms[0]
    for i in range(1, len(lms)):
        new_lms[i] = torch.cat((lms[i], torch.zeros((lens[0] - lens[i]),136)), 0)
    return new_lms, tgs, lens

if rnn == 'frameGRU':
    model = Framewise_GRU_Classifier(EMBEDDING_DIM, HIDDEN_DIM, 1, n_layer=N_LAYERS_RNN)
if rnn == 'frameCRNN':
    model = FrameCRNN(EMBEDDING_DIM, HIDDEN_DIM, 1, n_layer=N_LAYERS_RNN)
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
model = model.cuda()

loss_function = torch.nn.BCEWithLogitsLoss()
loss_function_eval_sum = torch.nn.BCEWithLogitsLoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=LR)

dataset_train = LandmarkList(root='/datasets/move_closer/Data_Landmark/', fileList='/datasets/move_closer/TrainList.txt')
dataloader_train = data.DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=2, collate_fn=pad_collate)
# if rnn == 'frameGRU':
#     dataloader_train = data.DataLoader(dataset_train, batch_size=8, shuffle=True, num_workers=2,
#                                        collate_fn=pad_collate)

dataset_test = LandmarkList(root='/datasets/move_closer/Data_Landmark/', fileList='/datasets/move_closer/TestList.txt')
dataloader_test = data.DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=1, collate_fn=pad_collate)

best_test_acc = 0.
for epoch in range(MAX_EPOCH):
    model.train()
    n_iter = 0
    for batch, labels, lengths in dataloader_train:
        model.zero_grad()
        out = model(batch.cuda(), lengths)  # we could do a classifcation for every output (probably better)
        if rnn == 'frameGRU':
            new_labels_list = []
            new_out_list = []
            for i in range(len(lengths)):
                new_labels_list += [labels[i]] * lengths[i]
                new_out_list.append(out[i][:lengths[i]])
            out = torch.cat(new_out_list, 0)
            labels = new_labels_list
        loss = loss_function(out, torch.FloatTensor(labels).unsqueeze(1).cuda())
        loss.backward()
        if Grad_Clip_For_All:
            nn.utils.clip_grad_value_(model.parameters(), 10)
        optimizer.step()
        n_iter += 1
    train_acc, train_loss = compute_binary_accuracy(model, dataloader_train, loss_function_eval_sum)
    test_acc, test_loss = compute_binary_accuracy(model, dataloader_test, loss_function_eval_sum)
    print('Epoch{},train_acc,{:.2f}%,train_loss,{:.8f},valid_acc,{:.2f}%,valid_loss,{:.8f}'.format(epoch, train_acc, train_loss, test_acc, test_loss))
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        if SAVE_BEST_MODEL:
            torch.save(model.state_dict(), 'models/' + rnn +
                       '_L' + str(N_LAYERS_RNN) + '_GC.pt')
        print('best epoch {}, train_acc {}, test_acc {}'.format(epoch, train_acc, test_acc))








# class LSTM_Classifier(nn.Module):
#
#     def __init__(self, embedding_dim, hidden_dim, target_size=1):
#         super(LSTM_Classifier, self).__init__()
#         self.hidden_dim = hidden_dim
#
#         # The LSTM takes word embeddings as inputs, and outputs hidden states
#         # with dimensionality hidden_dim.
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim)
#
#         # The linear layer that maps from hidden state space to tag space
#         self.lc = nn.Linear(hidden_dim, target_size)
#
#     def forward(self, landmarks, lengths):
#         # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
#         packed_input = pack_padded_sequence(landmarks, lengths, batch_first=True)
#         _, (ht, _) = self.lstm(packed_input)
#         import pdb;
#         pdb.set_trace()
#         # packed_output, (ht, ct) = self.lstm(packed_input)   # ht is the final output of each batch! ht (1, 4, 272) can be found in output[:,input_sizes-1,:]
#         # output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
#         '''
#         (Pdb) output[:,input_sizes-1,:]
# tensor([[[-0.0176,  0.1605,  0.1339,  ..., -0.0914,  0.2951, -0.0065],
#          [-0.0225,  0.1589,  0.1340,  ..., -0.0925,  0.2950, -0.0095],
#          [-0.0253,  0.1574,  0.1431,  ..., -0.0865,  0.3022, -0.0119],
#          [-0.0303,  0.1515,  0.1422,  ..., -0.1094,  0.2976, -0.0032]],
#
#         [[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
#          [-0.0165,  0.1666,  0.1344,  ..., -0.0698,  0.2945, -0.0163],
#          [-0.0235,  0.1697,  0.1479,  ..., -0.0657,  0.3001, -0.0195],
#          [-0.0235,  0.1734,  0.1515,  ..., -0.0608,  0.3029, -0.0201]],
#
#         [[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
#          [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
#          [-0.0492,  0.1666,  0.1444,  ..., -0.0749,  0.2816, -0.0188],
#          [-0.0490,  0.1542,  0.1449,  ..., -0.0865,  0.2821, -0.0205]],
#
#         [[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
#          [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
#          [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
#          [-0.0460,  0.1522,  0.1381,  ..., -0.0959,  0.2843, -0.0071]]],
#        device='cuda:2', grad_fn=<IndexBackward>)
# (Pdb) ht.shape
# torch.Size([1, 4, 272])
# (Pdb) ht
# tensor([[[-0.0176,  0.1605,  0.1339,  ..., -0.0914,  0.2951, -0.0065],
#          [-0.0165,  0.1666,  0.1344,  ..., -0.0698,  0.2945, -0.0163],
#          [-0.0492,  0.1666,  0.1444,  ..., -0.0749,  0.2816, -0.0188],
#          [-0.0460,  0.1522,  0.1381,  ..., -0.0959,  0.2843, -0.0071]]],
#        device='cuda:2', grad_fn=<CudnnRnnBackward>)
#
#         '''
#         # import pdb;
#         # pdb.set_trace()
#         logit = self.lc(ht.squeeze(0))
#         return logit