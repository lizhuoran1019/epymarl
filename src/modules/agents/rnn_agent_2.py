# code adapted from https://github.com/wendelinboehmer/dcg

import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNAgent2(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent2, self).__init__()
        self.args = args
        
        self.time_shape = args.time_shape
        self.non_time_shape = input_shape - args.time_shape
        
        self.rnn = nn.GRU(1, args.hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(args.hidden_dim + self.non_time_shape, args.n_actions)

        # self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        # if self.args.use_rnn:
        #     self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        # else:
        #     self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        # self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        # return self.fc1.weight.new(1, self.args.hidden_dim).zero_()
        return None

    def forward(self, inputs, hidden_state):
        time_feature = inputs[:, :self.time_shape].unsqueeze(-1)  # Add a dimension for GRU input
        
        gru_out, h = self.rnn(time_feature, hidden_state)  # h is the hidden state

        non_time_feature = inputs[:, self.time_shape:]
        
        # Concatenate the GRU output with the non-time features
        gru_out = gru_out[:, -1, :]
        combined_features = torch.cat((gru_out, non_time_feature), dim=1)
        
        q = self.fc2(combined_features)
        return q, h
        # x = F.relu(self.fc1(inputs))
        # h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        # if self.args.use_rnn:
        #     h = self.rnn(x, h_in)
        # else:
        #     h = F.relu(self.rnn(x))
        # q = self.fc2(h)
        # return q, h

