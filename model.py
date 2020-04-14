import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


DROPOUT = 0.5


class LSTM_Classifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, target_size=1, bidirectional=False, n_layer=1):
        super(LSTM_Classifier, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layer, dropout=DROPOUT)

        # The linear layer that maps from hidden state space to tag space
        self.lc = nn.Linear(hidden_dim, target_size)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, landmarks, lengths):
        packed_input = pack_padded_sequence(landmarks, lengths, batch_first=True)
        _, (ht, _) = self.lstm(packed_input)
        ht = self.dropout(ht[-1])
        logit = self.lc(ht)
        return logit


class embed_GRU_Classifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, target_size=1, bidirectional=False, n_layer=1):
        super(embed_GRU_Classifier, self).__init__()
        self.hidden_dim = hidden_dim

        self.embed1 = nn.Linear(embedding_dim, int(hidden_dim*2), bias=False)
        self.embed2 = nn.Linear(int(hidden_dim*2), hidden_dim, bias=False)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=n_layer, bidirectional=bidirectional, dropout=DROPOUT)

        # The linear layer that maps from hidden state space to tag space
        self.lc1 = nn.Linear(hidden_dim,int(hidden_dim/2))
        self.lc2 = nn.Linear(int(hidden_dim/2), target_size)
        self.dropout = nn.Dropout(DROPOUT)

        # super(embed_GRU_Classifier, self).__init__()
        # self.hidden_dim = hidden_dim
        #
        # self.embed1 = nn.Linear(embedding_dim, int(embedding_dim/2), bias=False)
        # self.embed2 = nn.Linear(int(embedding_dim/2), int(embedding_dim/4), bias=False)
        # # The LSTM takes word embeddings as inputs, and outputs hidden states
        # # with dimensionality hidden_dim
        # self.gru = nn.GRU(int(embedding_dim/4), int(embedding_dim/4), num_layers=n_layer, bidirectional=bidirectional, dropout=DROPOUT)
        #
        # # The linear layer that maps from hidden state space to tag space
        # self.lc1 = nn.Linear(int(embedding_dim/4),int(embedding_dim/8))
        # self.lc2 = nn.Linear(int(embedding_dim/8), target_size)
        # self.dropout = nn.Dropout(DROPOUT)

    def forward(self, landmarks, lengths):
        # import pdb; pdb.set_trace()
        landmarks = F.tanh(self.embed2(F.tanh(self.embed1(landmarks))))
        packed_input = pack_padded_sequence(landmarks, lengths, batch_first=True)
        _, ht = self.gru(packed_input)
        # import pdb; pdb.set_trace()
        ht = self.dropout(ht[-1])
        logit = self.lc2(F.tanh(self.lc1(ht)))
        return logit


class GRU_Classifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, target_size=1, bidirectional=False, n_layer=1):
        super(GRU_Classifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.grad_clipping = 10.
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layer, bidirectional=bidirectional, dropout=DROPOUT)

        # The linear layer that maps from hidden state space to tag space
        self.lc1 = nn.Linear(hidden_dim,target_size)
        # self.lc1 = nn.Linear(hidden_dim,EMBEDDING_DIM)
        # self.lc2 = nn.Linear(EMBEDDING_DIM, target_size)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, landmarks, lengths):
        packed_input = pack_padded_sequence(landmarks, lengths, batch_first=True)
        _, ht = self.gru(packed_input)
        # import pdb; pdb.set_trace()
        if ht.requires_grad:
            ht.register_hook(lambda x: x.clamp(min=-self.grad_clipping, max=self.grad_clipping))
        ht = self.dropout(ht[-1])
        logit = self.lc1(ht)    # probably a 1x1 conv is need to do linear transform
        # logit = self.lc2(F.relu(self.lc1(ht)))
        return logit


class biGRU_Classifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, target_size=1, bidirectional=True, n_layer=1):
        super(biGRU_Classifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.grad_clipping = 10.
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layer, bidirectional=bidirectional, dropout=DROPOUT)

        # The linear layer that maps from hidden state space to tag space
        self.lc1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.lc2 = nn.Linear(hidden_dim, target_size)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, landmarks, lengths):
        packed_input = pack_padded_sequence(landmarks, lengths, batch_first=True)
        _, ht = self.gru(packed_input)
        if ht.requires_grad:
            ht.register_hook(lambda x: x.clamp(min=-self.grad_clipping, max=self.grad_clipping))
        ht = self.dropout(torch.cat((ht[-2,:,:], ht[-1,:,:]), dim=1))
        logit = self.lc2(F.relu(self.lc1(ht)))
        return logit


class Framewise_GRU_Classifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, target_size=1, bidirectional=False, n_layer=1):
        super(Framewise_GRU_Classifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layer, bidirectional=bidirectional, dropout=DROPOUT)

        # The linear layer that maps from hidden state space to tag space
        # self.lc1 = nn.Linear(hidden_dim, target_size)
        self.lc1 = nn.Linear(hidden_dim, embedding_dim)
        self.lc2 = nn.Linear(embedding_dim, target_size)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, landmarks, lengths):
        packed_input = pack_padded_sequence(landmarks, lengths, batch_first=True)
        packed_output, _ = self.gru(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = output.contiguous()
        output = output.view(-1, self.hidden_dim)
        output = self.dropout(output)
        logit = self.lc1(output)    # probably a 1x1 conv is need to do linear transform
        logit = self.lc2(self.dropout(F.relu(logit)))
        return logit.view(len(lengths), -1, 1)


class sumGRU(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, target_size=1, bidirectional=False, n_layer=1):
        super(sumGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layer, bidirectional=bidirectional, dropout=DROPOUT)

        # The linear layer that maps from hidden state space to tag space
        self.lc1 = nn.Linear(hidden_dim, embedding_dim)
        self.lc2 = nn.Linear(embedding_dim, target_size)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, landmarks, lengths):
        packed_input = pack_padded_sequence(landmarks, lengths, batch_first=True)
        packed_output, _ = self.gru(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # import pdb; pdb.set_trace()
        output = self.dropout(output.sum(1))
        # logit = self.lc1(output)    # probably a 1x1 conv is need to do linear transform
        logit = self.lc2(F.relu(self.lc1(output)))
        return logit



class cnn_2d(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, target_size=1, bidirectional=False):
        super(cnn_2d, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = 2  # 2, 4, 6 ,8
        if self.n_layers >= 2:
            self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(num_features=self.hidden_dim)
            self.bn2 = nn.BatchNorm1d(num_features=self.hidden_dim)
            self.p1 = nn.MaxPool1d(kernel_size=2)
        if self.n_layers >= 4:
            self.conv3 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.conv4 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm1d(num_features=self.hidden_dim)
            self.bn4 = nn.BatchNorm1d(num_features=self.hidden_dim)
            self.p2 = nn.MaxPool1d(kernel_size=2)
        if self.n_layers >= 6:
            self.conv5 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.conv6 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.bn5 = nn.BatchNorm1d(num_features=self.hidden_dim)
            self.bn6 = nn.BatchNorm1d(num_features=self.hidden_dim)
            self.p3 = nn.MaxPool1d(kernel_size=2)
        if self.n_layers == 8:
            self.conv7 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.conv8 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.bn7 = nn.BatchNorm1d(num_features=self.hidden_dim)
            self.bn8 = nn.BatchNorm1d(num_features=self.hidden_dim)

        self.glbAvgPool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(DROPOUT)
        # The linear layer that maps from hidden state space to tag space
        self.lc1 = nn.Linear(hidden_dim, int(hidden_dim*2))
        self.lc2 = nn.Linear(int(hidden_dim*2), target_size)

    def forward(self, landmarks, lengths):
        landmarks = landmarks.permute(0, 2, 1)  # (b, seq, dim) --> (b, dim, seq)
        # Convolve on Seq for each dim to get (b, dim, seq)
        if self.n_layers == 8:
            landmarks = F.relu(self.bn8(self.conv8(F.relu(self.bn7(self.conv7(self.p3(F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(self.p2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(self.p1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(landmarks)))))))))))))))))))))))))))
        elif self.n_layers == 6:
            landmarks = self.p3(F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(self.p2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(self.p1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(landmarks)))))))))))))))))))))
        elif self.n_layers == 4:
            landmarks = self.p2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(self.p1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(landmarks))))))))))))))
        elif self.n_layers == 2:
            landmarks = self.p1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(landmarks)))))))
        else:
            print('Not specify n_layers')
        # Permute back: (b, dim, d_seq) --> (b, seq, dim)
        landmarks = landmarks.permute(0, 2, 1)
        # flat it to feed into fc: (b x seq, dim)
        landmarks = landmarks.contiguous()
        batch_size, seq_len, dim_feature = landmarks.shape
        landmarks = landmarks.view(-1, dim_feature)
        landmarks = F.tanh(self.lc1(self.dropout(landmarks)))  # (b x seq, 1)
        landmarks = self.lc2(self.dropout(landmarks))
        # unflat back to (b, seq, 1)
        landmarks = landmarks.view(batch_size, seq_len, 1)

        logit_list = []
        if self.n_layers == 8 or self.n_layers == 6:
            for i, landmark in enumerate(landmarks):
                logit_list.append(self.glbAvgPool(landmark[:int(lengths[i]/8)].unsqueeze(0).permute(0, 2, 1)).squeeze(-1))
        if self.n_layers == 4:
            for i, landmark in enumerate(landmarks):
                logit_list.append(self.glbAvgPool(landmark[:int(lengths[i]/4)].unsqueeze(0).permute(0, 2, 1)).squeeze(-1))
        if self.n_layers == 2:
            for i, landmark in enumerate(landmarks):
                logit_list.append(self.glbAvgPool(landmark[:int(lengths[i]/2)].unsqueeze(0).permute(0, 2, 1)).squeeze(-1))

        return torch.cat(logit_list)


class cnn_Classifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, target_size=1, bidirectional=False):
        super(cnn_Classifier, self).__init__()
        self.hidden_dim = hidden_dim # can change to smaller ones 64 . 32. 16
        self.n_layers = 2  # 2, 4, 6 ,8
        self.use_bn = False
        if self.n_layers >= 2:
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.hidden_dim, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=3, padding=1)
            if self.use_bn:
                self.bn1 = nn.BatchNorm2d(num_features=self.hidden_dim)
                self.bn2 = nn.BatchNorm2d(num_features=self.hidden_dim)
            self.p1 = nn.MaxPool2d(kernel_size=2)
        if self.n_layers >= 4:
            self.conv3 = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=3, padding=1)
            if self.use_bn:
                self.bn3 = nn.BatchNorm2d(num_features=self.hidden_dim)
                self.bn4 = nn.BatchNorm2d(num_features=self.hidden_dim)
            self.p2 = nn.MaxPool2d(kernel_size=2)
        if self.n_layers >= 6:
            self.conv5 = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=3, padding=1)
            self.conv6 = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=3, padding=1)
            if self.use_bn:
                self.bn5 = nn.BatchNorm2d(num_features=self.hidden_dim)
                self.bn6 = nn.BatchNorm2d(num_features=self.hidden_dim)
            self.p3 = nn.MaxPool2d(kernel_size=2)
        if self.n_layers == 8:
            self.conv7 = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=3, padding=1)
            self.conv8 = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=3, padding=1)
            if self.use_bn:
                self.bn7 = nn.BatchNorm2d(num_features=self.hidden_dim)
                self.bn8 = nn.BatchNorm2d(num_features=self.hidden_dim)

        self.dropout = nn.Dropout(DROPOUT)
        # The linear layer that maps from hidden state space to tag space
        self.lc1 = nn.Linear(hidden_dim, int(hidden_dim*2))
        self.lc2 = nn.Linear(int(hidden_dim*2), target_size)

    def forward(self, landmarks, lengths):
        landmarks = landmarks.permute(0, 2, 1)  # (b, seq, dim) --> (b, dim, seq)
        # Convolve on Seq for each dim to get (b, dim, seq)
        if self.use_bn:
            if self.n_layers == 8:
                landmarks = F.relu(self.bn8(self.conv8(F.relu(self.bn7(self.conv7(self.p3(F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(self.p2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(self.p1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(landmarks)))))))))))))))))))))))))))
            elif self.n_layers == 6:
                landmarks = self.p3(F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(self.p2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(self.p1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(landmarks)))))))))))))))))))))
            elif self.n_layers == 4:
                landmarks = self.p2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(self.p1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(landmarks))))))))))))))
            elif self.n_layers == 2:
                landmarks = self.p1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(landmarks)))))))
            else:
                print('Not specify n_layers')
        else:
            if self.n_layers == 8:
                landmarks = F.relu(self.conv8(F.relu(self.conv7(self.p3(F.relu(self.conv6(F.relu(self.conv5(self.p2(F.relu(self.conv4(F.relu(self.conv3(self.p1(F.relu(self.conv2(F.relu(self.conv1(landmarks)))))))))))))))))))
            elif self.n_layers == 6:
                landmarks = self.p3(F.relu(self.conv6(F.relu(self.conv5(self.p2(F.relu(self.conv4(F.relu(self.conv3(self.p1(F.relu(self.conv2(F.relu(self.conv1(landmarks)))))))))))))))
            elif self.n_layers == 4:
                landmarks = self.p2(F.relu(self.conv4(F.relu(self.conv3(self.p1(F.relu(self.conv2(F.relu(self.conv1(landmarks))))))))))
            elif self.n_layers == 2:
                landmarks = self.p1(F.relu(self.conv2(F.relu(self.conv1(landmarks)))))
            else:
                print('Not specify n_layers')
        # Permute back: (b, dim, d_seq) --> (b, seq, dim)
        landmarks = landmarks.permute(0, 2, 1)
        # flat it to feed into fc: (b x seq, dim)
        landmarks = landmarks.contiguous()
        batch_size, seq_len, dim_feature = landmarks.shape
        landmarks = landmarks.view(-1, dim_feature)
        landmarks = F.tanh(self.lc1(self.dropout(landmarks)))  # (b x seq, 1)
        landmarks = self.lc2(self.dropout(landmarks))
        # unflat back to (b, seq, 1)
        landmarks = landmarks.view(batch_size, seq_len, 1)

        logit_list = []
        if self.n_layers == 8 or self.n_layers == 6:
            for i, landmark in enumerate(landmarks):
                logit_list.append(self.glbAvgPool(landmark[:int(lengths[i]/8)].unsqueeze(0).permute(0, 2, 1)).squeeze(-1))
        if self.n_layers == 4:
            for i, landmark in enumerate(landmarks):
                logit_list.append(self.glbAvgPool(landmark[:int(lengths[i]/4)].unsqueeze(0).permute(0, 2, 1)).squeeze(-1))
        if self.n_layers == 2:
            for i, landmark in enumerate(landmarks):
                logit_list.append(self.glbAvgPool(landmark[:int(lengths[i]/2)].unsqueeze(0).permute(0, 2, 1)).squeeze(-1))

        return torch.cat(logit_list)


class crnn_Classifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, target_size=1, bidirectional=False, n_layer=1):
        super(crnn_Classifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = 4 # 2, 4, 6 ,8
        if self.n_layers >= 2:
            self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(num_features=self.hidden_dim)
            self.bn2 = nn.BatchNorm1d(num_features=self.hidden_dim)
            self.p1 = nn.MaxPool1d(kernel_size=2)
            self.scale_pool = 2
        if self.n_layers >= 4:
            self.conv3 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.conv4 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm1d(num_features=self.hidden_dim)
            self.bn4 = nn.BatchNorm1d(num_features=self.hidden_dim)
            self.p2 = nn.MaxPool1d(kernel_size=2)
            self.scale_pool = 4
        if self.n_layers >= 6:
            self.conv5 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.conv6 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.bn5 = nn.BatchNorm1d(num_features=self.hidden_dim)
            self.bn6 = nn.BatchNorm1d(num_features=self.hidden_dim)
            self.p3 = nn.MaxPool1d(kernel_size=2)
            self.scale_pool = 8
        if self.n_layers == 8:
            self.conv7 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.conv8 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.bn7 = nn.BatchNorm1d(num_features=self.hidden_dim)
            self.bn8 = nn.BatchNorm1d(num_features=self.hidden_dim)
            self.scale_pool = 8

        self.dropout = nn.Dropout(DROPOUT)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=n_layer, bidirectional=bidirectional, dropout=DROPOUT)
        self.grad_clipping = 10.
        # The linear layer that maps from hidden state space to tag space
        self.lc1 = nn.Linear(hidden_dim, embedding_dim)
        self.lc2 = nn.Linear(embedding_dim, target_size)

    def forward(self, landmarks, lengths):
        landmarks = landmarks.permute(0, 2, 1)  # (b, seq, dim) --> (b, dim, seq)
        # Convolve on Seq for each dim to get (b, dim, seq)
        if self.n_layers == 8:
            landmarks = F.relu(self.bn8(self.conv8(F.relu(self.bn7(self.conv7(self.p3(F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(self.p2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(self.p1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(landmarks)))))))))))))))))))))))))))
        elif self.n_layers == 6:
            landmarks = self.p3(F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(self.p2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(self.p1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(landmarks)))))))))))))))))))))
        elif self.n_layers == 4:
            landmarks = self.p2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(self.p1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(landmarks))))))))))))))
        elif self.n_layers == 2:
            landmarks = self.p1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(landmarks)))))))
        else:
            print('Not specify n_layers')

        # Permute back: (b, dim, d_seq) --> (b, seq, dim) with shorter seq
        landmarks = landmarks.permute(0, 2, 1)
        # Feed into GRU
        # import pdb; pdb.set_trace()
        # packed_input = pack_padded_sequence(self.dropout(landmarks), torch.IntTensor(lengths)/self.scale_pool, batch_first=True)
        packed_input = pack_padded_sequence(self.dropout(landmarks), tuple(int(x/self.scale_pool) for x in lengths), batch_first=True)
        _, ht = self.gru(packed_input)
        if ht.requires_grad:
            ht.register_hook(lambda x: x.clamp(min=-self.grad_clipping, max=self.grad_clipping))
        ht = self.dropout(ht[-1])
        logit = F.relu(self.lc1(ht))
        logit = self.lc2(self.dropout(logit))
        return logit

# to be implemented
class FrameCRNN(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, target_size=1, bidirectional=False, n_layer=1):
        super(FrameCRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = 2  # 2, 4, 6 ,8
        if self.n_layers >= 2:
            self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm1d(num_features=self.hidden_dim)
            self.bn2 = nn.BatchNorm1d(num_features=self.hidden_dim)
            self.p1 = nn.MaxPool1d(kernel_size=2)
            self.scale_pool = 2
        if self.n_layers >= 4:
            self.conv3 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.conv4 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm1d(num_features=self.hidden_dim)
            self.bn4 = nn.BatchNorm1d(num_features=self.hidden_dim)
            self.p2 = nn.MaxPool1d(kernel_size=2)
            self.scale_pool = 4
        if self.n_layers >= 6:
            self.conv5 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.conv6 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.bn5 = nn.BatchNorm1d(num_features=self.hidden_dim)
            self.bn6 = nn.BatchNorm1d(num_features=self.hidden_dim)
            self.p3 = nn.MaxPool1d(kernel_size=2)
            self.scale_pool = 8
        if self.n_layers == 8:
            self.conv7 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.conv8 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
            self.bn7 = nn.BatchNorm1d(num_features=self.hidden_dim)
            self.bn8 = nn.BatchNorm1d(num_features=self.hidden_dim)
            self.scale_pool = 8

        self.dropout = nn.Dropout(DROPOUT)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=n_layer, bidirectional=bidirectional, dropout=DROPOUT)

        # The linear layer that maps from hidden state space to tag space
        self.lc1 = nn.Linear(hidden_dim, embedding_dim)
        self.lc2 = nn.Linear(embedding_dim, target_size)

    def forward(self, landmarks, lengths):
        landmarks = landmarks.permute(0, 2, 1)  # (b, seq, dim) --> (b, dim, seq)
        # Convolve on Seq for each dim to get (b, dim, seq)
        if self.n_layers == 8:
            landmarks = F.relu(self.bn8(self.conv8(F.relu(self.bn7(self.conv7(self.p3(F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(self.p2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(self.p1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(landmarks)))))))))))))))))))))))))))
        elif self.n_layers == 6:
            landmarks = self.p3(F.relu(self.bn6(self.conv6(F.relu(self.bn5(self.conv5(self.p2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(self.p1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(landmarks)))))))))))))))))))))
        elif self.n_layers == 4:
            landmarks = self.p2(F.relu(self.bn4(self.conv4(F.relu(self.bn3(self.conv3(self.p1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(landmarks))))))))))))))
        elif self.n_layers == 2:
            landmarks = self.p1(F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(landmarks)))))))
        else:
            print('Not specify n_layers')

        # Permute back: (b, dim, d_seq) --> (b, seq, dim) with shorter seq
        landmarks = landmarks.permute(0, 2, 1)
        # Feed into GRU
        packed_input = pack_padded_sequence(self.dropout(landmarks), torch.IntTensor(lengths)/self.scale_pool, batch_first=True)
        _, ht = self.gru(packed_input)
        ht = self.dropout(ht[-1])
        logit = F.relu(self.lc1(ht))
        logit = self.lc2(self.dropout(logit))
        return logit





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