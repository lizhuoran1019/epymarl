# implementation of centralV critic with noise for mappo_noise

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class CentralVCriticRNN(nn.Module):
    def __init__(self, scheme, args):
        super(CentralVCriticRNN, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        
        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"

        # Set up network layers
        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        self.gru = nn.GRU(input_size=args.hidden_dim, hidden_size=args.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.layer_norm = nn.LayerNorm(args.hidden_dim)
        self.fc2 = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 1)
        )
        
        self.__param_init()
        
    def __param_init(self):
        for name, param in self.named_parameters():
            if 'weight_hh' in name:  # 隐含层权重
                nn.init.orthogonal_(param)
            elif 'weight_ih' in name:  # 输入层权重
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)  # 偏置置零

    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)
        inputs = inputs.permute(0, 2, 1, 3)  # [bs, n_agents, max_t, input_shape]
        inputs = inputs.reshape(bs * self.n_agents, max_t, -1)  # [bs*n_agents, max_t, input_shape]
        # x = F.tanh(self.fc1(inputs))
        x = F.relu(self.fc1(inputs))
        x, _ = self.gru(x, None)
        x = self.layer_norm(x)
        x = x.view(bs, self.n_agents, max_t, -1) # [bs, n_agents, max_t, hidden_dim]
        x = x.permute(0, 2, 1, 3)  # [bs, max_t, n_agents, hidden_dim]
        q = self.fc2(x)
        return q

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observations
        if self.args.obs_individual_obs:
            inputs.append(batch["obs"][:, ts].view(bs, max_t, -1).unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # last actions
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1))
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
                last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
                inputs.append(last_actions)

        # 添加智能体ID
        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
        
        inputs = th.cat(inputs, dim=-1)
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        # observations
        if self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"] * self.n_agents
        # last actions
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        # 智能体ID
        input_shape += self.n_agents
        return input_shape
