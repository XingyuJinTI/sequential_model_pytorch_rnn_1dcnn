import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import LandmarkList, LandmarkListTest
from torch.utils import data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import argparse

from model import *

# rnn = 'twoStreamCNN'
# rnn = 'frameGRU'
# rnn = 'sumGRU'
# rnn = 'crnn'
rnn = 'cnn'
# rnn = 'GRU'
# rnn = 'framewise_GRU'
# rnn = 'embedGRU'
# rnn = 'biGRU'
# rnn = 'LSTM'
LANDMARK = 3
EMBEDDING_DIM = LANDMARK
HIDDEN_DIM = LANDMARK*2
N_LAYERS_RNN = 3
N_LAYERS_CNN = 8
SIZE_CNN_SAMPLE = 128
LR = 1e-4
DEVICES = 0
torch.cuda.set_device(DEVICES)


def compute_binary_accuracy(model1, model2, data_loader, th_list):
    len_th_list = len(th_list)
    correct_pred, num_examples, FP, FN = [0.]*len_th_list, 0, [0]*len_th_list, [0]*len_th_list
    FP_list = []
    FN_list = []
    for _ in range(len_th_list):
        FP_list.append([])
        FN_list.append([])
    model1.eval()
    model2.eval()
    with torch.no_grad():
        if rnn == 'frameGRU':
            for batch, labels, lengths, f_names in data_loader:
                if LANDMARK == 3:
                    batch = batch[:,:,3:]
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
                batch1 = batch[:, :, :3] # gyro - model1
                batch2 = batch[:, :, 3:] # acceler - model2

                # import pdb; pdb.set_trace()
                batch1 = F.interpolate(batch1[0].unsqueeze(0).permute(0, 2, 1), size=SIZE_CNN_SAMPLE,
                                      mode='nearest').permute(0, 2, 1)
                batch2 = F.interpolate(batch2[0].unsqueeze(0).permute(0, 2, 1), size=SIZE_CNN_SAMPLE,
                                      mode='nearest').permute(0, 2, 1)
                # import pdb; pdb.set_trace()
                logits1 = model1(batch1.cuda(), lengths)
                logits2 = model2(batch2.cuda(), lengths)
                num_examples += len(lengths)
                for i, th in enumerate(th_list):
                    prob = th * torch.sigmoid(logits1) + (1-th) * torch.sigmoid(logits2)
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
                # if LANDMARK == 3:
                #     batch = batch[:,:,3:]
                # if rnn == 'cnn':
                #     # import pdb; pdb.set_trace()
                #     batch = F.interpolate(batch[0].unsqueeze(0).permute(0, 2, 1), size=SIZE_CNN_SAMPLE,
                #                           mode='nearest').permute(0, 2, 1)
                # # import pdb; pdb.set_trace()
                # logits = model(batch.cuda(), lengths)
                # num_examples += len(lengths)
                # for i, th in enumerate(th_list):
                #     predicted_labels = (torch.sigmoid(logits) > th).long()
                #     # import pdb;pdb.set_trace()
                #     if predicted_labels.squeeze(1).cpu().long() == torch.LongTensor(labels):
                #         correct_pred[i] += 1
                #     elif labels == 0:
                #         FP[i] += 1
                #         FP_list[i].append(f_names[0]+'_'+str(labels.item())+'_'+str(torch.sigmoid(logits).squeeze(1).cpu().item()))
                #     else:
                #         FN[i] += 1
                #         FN_list[i].append(f_names[0]+'_'+str(labels.item())+'_'+str(torch.sigmoid(logits).squeeze(1).cpu().item()))
            return [n_correct/num_examples * 100 for n_correct in correct_pred], FP, FN, FP_list, FN_list



if rnn == 'frameGRU':
    model = Framewise_GRU_Classifier(EMBEDDING_DIM, HIDDEN_DIM, 1, n_layer=N_LAYERS_RNN)
if rnn == 'sumGRU':
    model = sumGRU(EMBEDDING_DIM, HIDDEN_DIM, 1, n_layer=N_LAYERS_RNN)
    model_name = "models/"+str(rnn)+'_L' + str(N_LAYERS_RNN) + ".pt"
if rnn == 'embedGRU':
    model = embed_GRU_Classifier(EMBEDDING_DIM, HIDDEN_DIM, 1, n_layer=N_LAYERS_RNN)
    model_name = "models/" + str(rnn) + "_L3.pt"
if rnn == 'GRU':
    model = GRU_Classifier(EMBEDDING_DIM, HIDDEN_DIM, 1, n_layer=N_LAYERS_RNN)
if rnn == 'biGRU':
    model = biGRU_Classifier(EMBEDDING_DIM, HIDDEN_DIM, 1, n_layer=N_LAYERS_RNN)
    model_name = "models/"+str(rnn)+'_L' + str(N_LAYERS_RNN) + ".pt"
if rnn == 'LSTM':
    model = LSTM_Classifier(EMBEDDING_DIM, HIDDEN_DIM, 1, n_layer=N_LAYERS_RNN)
if rnn == 'cnn':
    model1 = cnn_Classifier(N_LAYERS_CNN, EMBEDDING_DIM, HIDDEN_DIM, 1)
    model2 = cnn_Classifier(N_LAYERS_CNN, EMBEDDING_DIM, HIDDEN_DIM, 1)
    # model.load_state_dict(torch.load("models/" + str(rnn) + "8.pt"))
    model_name1 = "models/cnn" + '_L' + str(N_LAYERS_CNN) +'_LM'+  str(LANDMARK)+ '_frame' + str(SIZE_CNN_SAMPLE) + "_gyro.pt"
    model_name2 = "models/cnn" + '_L' + str(N_LAYERS_CNN) +'_LM'+  str(LANDMARK)+ '_frame' + str(SIZE_CNN_SAMPLE) + "_accelero.pt"
if rnn == 'crnn':
    model = crnn_Classifier(N_LAYERS_CNN, EMBEDDING_DIM, HIDDEN_DIM, 1, n_layer=N_LAYERS_RNN)
    model_name = "models/" + str(rnn) + '_L' + str(N_LAYERS_RNN) + "_GC.pt"

print('Model1 is loading: ', model_name1)
model1.load_state_dict(torch.load(model_name1))
print('Model2 is loading: ', model_name2)
model2.load_state_dict(torch.load(model_name2))
# model.load_state_dict(torch.load("models/"+str(rnn)+'_L' + str(N_LAYERS_RNN) + "_GC.pt"))
model1 = model1.cuda()
model2 = model2.cuda()

loss_function = torch.nn.BCEWithLogitsLoss()
loss_function_eval_sum = torch.nn.BCEWithLogitsLoss(reduction='sum')
# optimizer = optim.Adam(model.parameters(), lr=LR)

dataset_train = LandmarkListTest(root='/datasets/mc_m_min16frame/', fileList='/datasets/mc_m_min16frame/TrainListM.txt')
dataloader_train = data.DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0)#, collate_fn=pad_collate)

dataset_test = LandmarkListTest(root='/datasets/mc_m_min16frame/', fileList='/datasets/mc_m_min16frame/TestListM.txt')
dataloader_test = data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)#, collate_fn=pad_collate)

# thresholds = [x * 0.01 for x in range(30, 71)]
# thresholds = [0.5]
# alphas = [x * 0.05 for x in range(20)]
alphas = [0.15]

# train_acc, train_fp, train_fn, train_fp_list, train_fn_list = compute_binary_accuracy(model, dataloader_train, thresholds)
# test_acc, test_fp, test_fn, test_fp_list, test_fn_list = compute_binary_accuracy(model, dataloader_test, thresholds)
train_acc, train_fp, train_fn, train_fp_list, train_fn_list = compute_binary_accuracy(model1, model2, dataloader_train, alphas)
test_acc, test_fp, test_fn, test_fp_list, test_fn_list = compute_binary_accuracy(model1, model2, dataloader_test, alphas)

for i in range(0, len(alphas)):
    print('\n\n-----------------Eval for alpha of {:.2f}-------------------\n\n'.format(alphas[i]))
    print('train_acc,{:.2f}%,train_fp,{},train_fn,{}\nvalid_acc,{:.2f}%,valid_fp,{},valid_fn,{}\n'
          .format(train_acc[i], train_fp[i], train_fn[i], test_acc[i], test_fp[i], test_fn[i]))
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
