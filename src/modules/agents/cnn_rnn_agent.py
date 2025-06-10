import torch.nn as nn
import torch.nn.functional as F
import torch


class CNNRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CNNRNNAgent, self).__init__()
        self.args = args
        
        input_channels = 1 # 输入通道数

        # CNN特征提取部分
        self.cnn = nn.Sequential(
            # 输入: [batch*args, 1, time_shape]
            nn.Conv1d(input_channels, 16, kernel_size=3, padding=1),
            # nn.BatchNorm1d(16),
            # nn.ReLU(),
            # nn.Tanh(),
            # nn.MaxPool1d(kernel_size=2),  # 输出: [batch, 16, 5]
            
            # nn.Conv1d(16, 32, kernel_size=3, padding=1),
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
            # nn.MaxPool1d(2)  # 输出: [batch, 32, 2]
        )
        
        # 计算CNN输出维度
        with torch.no_grad():
            dummy = torch.randn(1, input_channels, args.time_shape)
            cnn_out = self.cnn(dummy)
            self.cnn_output_size = cnn_out.size(1) * cnn_out.size(2)  # 全连接备用
            self.gru_input_size = cnn_out.size(1)  # GRU输入特征维度
            self.gru_seq_length = cnn_out.size(2)   # GRU输入序列长度
        
        # GRU时序建模部分
        self.gru = nn.GRU(
            input_size=self.gru_input_size,  # 32
            hidden_size=args.hidden_dim,     # 64
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 分类器
        extra_features = input_shape - args.time_shape
        self.fc = nn.Sequential(
            nn.Linear(args.hidden_dim*2+extra_features, 32),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(32, args.n_actions)
        )
        
        # 初始化权重
        self.orthogonal_init()
        
    def orthogonal_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param)
                    else:
                        nn.init.constant_(param, 0)
        
    def init_hidden(self):
        # make hidden states on same device as model
        return None

    def forward(self, inputs, hidden_state=None):
        
        x_time = inputs[:, :self.args.time_shape].unsqueeze(1)  # [batch*agent, 1, time_shape]

        # CNN特征提取
        cnn_out = self.cnn(x_time)  # [batch*agent, 32, 2]
        
        # 准备GRU输入: [batch*agnet, channels, timesteps] -> [batch*agent, timesteps, channels]
        gru_input = cnn_out.permute(0, 2, 1)  # [batch*agent, 2, 32]
        
        # GRU处理
        gru_out, h = self.gru(gru_input, hidden_state)  # gru_out形状: [batch, 2, 64]
        
        # 取最后一个时间步的输出
        last_output = gru_out[:, -1, :]  # [batch, 64]
        
        # 全连接分类器
        x_non_time = inputs[:, self.args.time_shape:]
        last_output_concated = torch.cat((last_output, x_non_time), dim=1)  # [batch, 64 + (e - time_shape)]
        q = self.fc(last_output_concated)  # [batch, n_actions]
        
        # 分类
        return q, h  # 返回形状: [batch, agent, n_actions], [batch, agent, hidden_dim]