import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset2 import LandmarkList
from torch.utils import data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import argparse

# parser.add_argument('--root_path', type=str, default='/home/guosheng/Liveness/Code/FaceFlashing/Data/',
#                     metavar='H',
#                     help='Dir Head')
# parser.add_argument('--trainFile', type=str, default='TrainList_4sources_13082019.txt', metavar='TRF', help='training file name')


EMBEDDING_DIM = 68*2
HIDDEN_DIM = 68*4
MAX_EPOCH = 10
DEVICES = 2
torch.cuda.set_device(DEVICES)


def pad_collate(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    lms, tgs, lens = zip(*batch)
    new_lms = torch.zeros((len(lms), lms[0].shape[0], lms[0].shape[1])) # batch x seq x feature(136)
    new_lms[0] = lms[0]
    for i in range(1, len(lms)):
        # import pdb;
        # pdb.set_trace()
        new_lms[i] = torch.cat((lms[i], torch.zeros((lens[0] - lens[i]),136)), 0)
    return new_lms, tgs, lens


class LSTM_Classifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, target_size=1):
        super(LSTM_Classifier, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2)

        # The linear layer that maps from hidden state space to tag space
        self.lc = nn.Linear(hidden_dim, target_size)

    def forward(self, landmarks, lengths):
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        packed_input = pack_padded_sequence(landmarks, lengths, batch_first=True)
        _, (ht, _) = self.lstm(packed_input)
        # import pdb;
        # pdb.set_trace()
        # packed_output, (ht, ct) = self.lstm(packed_input)   # ht is the final output of each batch! ht (1, 4, 272) can be found in output[:,input_sizes-1,:]
        # output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        '''
        (Pdb) output[:,input_sizes-1,:]
tensor([[[-0.0176,  0.1605,  0.1339,  ..., -0.0914,  0.2951, -0.0065],
         [-0.0225,  0.1589,  0.1340,  ..., -0.0925,  0.2950, -0.0095],
         [-0.0253,  0.1574,  0.1431,  ..., -0.0865,  0.3022, -0.0119],
         [-0.0303,  0.1515,  0.1422,  ..., -0.1094,  0.2976, -0.0032]],

        [[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [-0.0165,  0.1666,  0.1344,  ..., -0.0698,  0.2945, -0.0163],
         [-0.0235,  0.1697,  0.1479,  ..., -0.0657,  0.3001, -0.0195],
         [-0.0235,  0.1734,  0.1515,  ..., -0.0608,  0.3029, -0.0201]],

        [[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [-0.0492,  0.1666,  0.1444,  ..., -0.0749,  0.2816, -0.0188],
         [-0.0490,  0.1542,  0.1449,  ..., -0.0865,  0.2821, -0.0205]],

        [[ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [-0.0460,  0.1522,  0.1381,  ..., -0.0959,  0.2843, -0.0071]]],
       device='cuda:2', grad_fn=<IndexBackward>)
(Pdb) ht.shape
torch.Size([1, 4, 272])
(Pdb) ht
tensor([[[-0.0176,  0.1605,  0.1339,  ..., -0.0914,  0.2951, -0.0065],
         [-0.0165,  0.1666,  0.1344,  ..., -0.0698,  0.2945, -0.0163],
         [-0.0492,  0.1666,  0.1444,  ..., -0.0749,  0.2816, -0.0188],
         [-0.0460,  0.1522,  0.1381,  ..., -0.0959,  0.2843, -0.0071]]],
       device='cuda:2', grad_fn=<CudnnRnnBackward>)

        '''
        # import pdb;
        # pdb.set_trace()
        logit = self.lc(ht[-1])
        return logit


# inp = [torch.randn(1, 68*2) for _ in range(5)]
# print(inp)
# inp = torch.cat(inp)
# print(inp)
# model = LSTM_Classifier(EMBEDDING_DIM, HIDDEN_DIM, 1)
# out = model(inp)
# print(out)


model = LSTM_Classifier(EMBEDDING_DIM, HIDDEN_DIM, 1)
model = model.cuda()
loss_function = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# l2 = torch.nn.BCELoss()


# for i in range(2):
#     model.zero_grad()
#     inp = [torch.randn(1, 68*2) for _ in range(5)]
#     inp = torch.cat(inp)
#     out = model(inp)[-1]    # we could do a classifcation for every output (probably better)
#     print(out)
#     loss = loss_function(out,torch.Tensor(1))
#     # loss = l2(nn.Sigmoid()(out), torch.Tensor(1))
#     print(loss)
#     loss.backward()
#     optimizer.step()


dataset_train = LandmarkList(root='/datasets/move_closer/Data_Landmark/', fileList='/datasets/move_closer/TrainList.txt')
dataloader_train = data.DataLoader(dataset_train, batch_size=128, shuffle=True, num_workers=4,  collate_fn=pad_collate)

for i in range(MAX_EPOCH):
    for batch, labels, lengths in dataloader_train:
        model.zero_grad()
        out = model(batch.cuda(), lengths) # we could do a classifcation for every output (probably better)
        # import pdb;
        # pdb.set_trace()
        loss = loss_function(out, torch.FloatTensor(labels).unsqueeze(1).cuda())
        # loss = l2(nn.Sigmoid()(out), labels)
        print(loss.data)
        loss.backward()
        optimizer.step()







# Demo to prove they are the same and padding doesn't feed the whole batch contineously!
# for batch, labels, lengths in dataloader_train:
#     model.zero_grad()
#     out = model(batch.cuda(), lengths) # we could do a classifcation for every output (probably better)
#     import pdb;
#     pdb.set_trace()
#
# dataloader_train = data.DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=0,  collate_fn=pad_collate)
# for batch, labels, lengths in dataloader_train:
#     model.zero_grad()
#     out = model(batch.cuda(), lengths) # we could do a classifcation for every output (probably better)
#     import pdb;
#     pdb.set_trace()



