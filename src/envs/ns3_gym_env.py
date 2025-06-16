from envs.multiagentenv import MultiAgentEnv
from ns3gym.ns3env import Ns3Env
import numpy as np
import json

class Ns3GymEnv(MultiAgentEnv):
    
    def __init__(self, targetName, ns3Path, ns3Settings={}, port=0, simSeed=0, episode_limit=500, hist_obs_num=1, **kwargs):
        self.common_reward = kwargs.get('common_reward', True)
        np.random.seed()
        self.env = Ns3Env(ns3_path=ns3Path, target_name=targetName, port=port, startSim=True, simSeed=simSeed, simArgs=ns3Settings, ns3Args=["--quiet", "--no-build"], debug=False)
        # self.env = Ns3Env(ns3_path=ns3Path, target_name=targetName, port=5555, startSim=False, simSeed=simSeed, simArgs=ns3Settings, ns3Args=["--quiet", "--no-build"], debug=False)

        self.n_agents = self.env.action_space.shape[0]
        self.n_actions = int(self.env.action_space.high[0] - self.env.action_space.low[0] + 1)

        self.__action_one_hot_matrix = np.eye(self.n_actions)
        self.__last_action = np.zeros((self.n_agents, self.n_actions), dtype=np.int16)

        self.episode_limit = episode_limit
        self._episode_count = 0

        self._total_steps = 0
        self._episode_steps = 0

        self.hist_obs_num = hist_obs_num  # 历史观测数量

        self.last_info = {}

        self.__init_hist_obs()

    def __init_hist_obs(self):
        self.hist_obs = np.zeros((self.n_agents, self.hist_obs_num, self.__get_one_obs_size()), dtype=np.float32)

    def __update_hist_obs(self, obs):
        """ 更新历史观测 """
        if self.hist_obs_num > 1:
            self.hist_obs = np.roll(self.hist_obs, shift=-1, axis=1)
        self.hist_obs[:, -1, :] = obs

    def step(self, actions):
        """ Returns reward, terminated, info """
        actions_int = [int(a) for a in actions]
        self.env.step(actions_int)
        obs_dict, reward, terminated, info = self.env.get_state()

        self._total_steps += 1
        self._episode_steps += 1

        # if self._episode_steps >= self.episode_limit:
        #     done = True
        truncated = False if self._episode_steps < self.episode_limit else True

        if terminated or truncated:
            self._episode_count += 1

        # 处理info字典
        if isinstance(info, str):
            info = json.loads(info)
            
        # if terminated: # FIXME: 这里正常应该判断truncated
        #     info['episode_limit'] = True
        # else:
        #     info['episode_limit'] = False

        # 读取当前的info
        pu_rx_count = info['pu_rx_count']
        pu_tx_count = info['pu_tx_count']
        su_rx_count = info['su_rx_count']
        su_tx_count = info['su_tx_count']

        # 计算丢包率
        pu_loss_rate = 1- pu_rx_count / pu_tx_count if pu_tx_count > 0 else 0
        su_loss_rate = 1 - su_rx_count / su_tx_count if su_tx_count > 0 else 0
        loss_rate = 1 - (pu_rx_count + su_rx_count) / (pu_tx_count + su_tx_count) if (pu_tx_count + su_tx_count) > 0 else 0
        info['pu_loss_rate'] = pu_loss_rate
        info['su_loss_rate'] = su_loss_rate
        info['loss_rate'] = loss_rate

        # 计算奖励
        if self.common_reward:
            # 如果使用全局奖励
            reward = self.__calc_global_reward(obs_dict, actions_int, info)
        else:
            # 如果使用个体奖励
            reward = self.__calc_individual_reward(obs_dict, actions_int, info)

        # 更新历史信息
        self.last_info = info.copy()  # 保存当前info
        self.__update_hist_obs(self.__get_now_obs())
        self.__last_action = self.__action_one_hot_matrix[np.array(actions_int)]

        return self.get_obs(), reward, terminated, truncated, info

    def __calc_global_reward(self, obs_dict, actions_int, info):
        """ Calculate the reward based on the current observation and actions """
        main_link_rx_count_delta = info['main_link_rx_count'] - self.last_info.get('main_link_rx_count', 0)
        sub_link_rx_count_delta = info['sub_link_rx_count'] - self.last_info.get('sub_link_rx_count', 0)
        rx_good = obs_dict['rx_good']
        tx_good = np.roll(rx_good, shift=-1, axis=0)  # 假设tx_good是rx_good的前一个状态
        tx_good[3] = main_link_rx_count_delta
        tx_good[6] = sub_link_rx_count_delta
        tx = [act>0 for act in actions_int]
        tx_error = np.logical_and(tx, np.logical_not(tx_good))  # tx错误

        reward = (
            main_link_rx_count_delta * 2.0
            + sub_link_rx_count_delta * 1.0
            # + np.sum(tx_good * 0.1)
            - np.sum(tx_error * 0.2)
        )

        # reward /= 3.5

        return reward

    def __calc_individual_reward(self, obs_dict, actions_int, info):
        """ Calculate the reward based on the current observation and actions """
        main_link_rx_count_delta = info['main_link_rx_count'] - self.last_info.get('main_link_rx_count', 0)
        sub_link_rx_count_delta = info['sub_link_rx_count'] - self.last_info.get('sub_link_rx_count', 0)
        rx_good = obs_dict['rx_good']
        tx_good = np.roll(rx_good, shift=-1, axis=0)  # 假设tx_good是rx_good的前一个状态
        tx_good[3] = main_link_rx_count_delta
        tx_good[6] = sub_link_rx_count_delta
        tx = [act>0 for act in actions_int]
        tx_error = np.logical_and(tx, np.logical_not(tx_good))  # tx错误

        # reward = (
        #     main_link_rx_count_delta * 2.0
        #     + sub_link_rx_count_delta * 1.0
        #     + tx_good * [0.3,0.3,0.3,0.3,0.1,0.1,0.1]
        #     - tx_error * 0.2
        # )
        reward = (
            tx_good * [0.3, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1]
            - tx_error * 0.2
        )
        reward[:4] += main_link_rx_count_delta * 2.0  # 主链路奖励
        reward[4:] += sub_link_rx_count_delta * 1.0 # 子链路奖励

        # reward /= 3.5
        # reward = reward.reshape(-1, 1)  # 3维 -> 4维

        return reward

    def __get_now_obs(self):
        """ Returns all agent observations in a list """
        obs_dict,_,_,_ = self.env.get_state()
        obs_dict_copy = obs_dict.copy()
        # obs_dict_copy.pop("rx_good")  # 删除rx_good
        obs_dict_copy.pop("left_energy_frac")  # 删除left_energy
        for key in obs_dict_copy.keys():
            obs_dict_copy[key] = obs_dict_copy[key].reshape(self.n_agents, -1, order="F")
            if key == "channel_energy": # normalize channel_energy
                obs_dict_copy[key] = obs_dict_copy[key]/200
            #     mean = obs_dict_copy[key].mean(axis=1, keepdims=True)
            #     std = obs_dict_copy[key].std(axis=1, keepdims=True)
            #     obs_dict_copy[key] = (obs_dict_copy[key] - mean) / (std + 1e-8)  # 防止除以0
        obs = np.concatenate([value for value in obs_dict_copy.values()], axis=1)
        # 添加归一化的时间步
        obs = np.column_stack((obs, np.full((self.n_agents, 1), self._episode_steps / self.episode_limit)))

        return obs

    def __get_one_obs_size(self):
        """ Returns the shape of the observation """
        obs_size = 0
        for key in self.env.observation_space.spaces.keys():
            if key == "left_energy_frac":
                continue
            obs_size += self.env.observation_space.spaces[key].shape[1]
        obs_size += 1 # 添加归一化的时间步

        return obs_size

    # def __get_now_obs(self):
    #     obs_dict,_,_,_ = self.env.get_state()
    #     obs_dict_copy = obs_dict.copy()
    #     obs_dict_copy.pop("rx_good")  # 删除rx_good
    #     obs_dict_copy.pop("channel_energy")  # 删除channel_energy
    #     obs_dict_copy.pop("left_energy_frac")  # 删除left_energy
    #     for key in obs_dict_copy.keys():
    #         obs_dict_copy[key] = obs_dict_copy[key].reshape(self.n_agents, -1, order="F")
    #     obs = np.concatenate([value for value in obs_dict_copy.values()], axis=1)
    #     obs = obs.reshape(1, -1)
    #     obs = obs.repeat(self.n_agents,axis=0)

    #     # obs_with_action = np.concatenate((obs, self.__last_action), axis=1)
    #     # state = obs_with_action.flatten()
    #     # state = obs.flatten()
    #     return obs

    # def __get_one_obs_size(self):
    #     """ Returns the shape of the state"""
    #     obs_size = 0
    #     for key in self.env.observation_space.spaces.keys():
    #         if key == "left_energy_frac" or key == "channel_energy" or key == "rx_good":
    #             continue
    #         obs_size += self.env.observation_space.spaces[key].shape[1]
    #     obs_size *= self.n_agents
    #     # obs_size += self.n_agents * self.n_actions
    #     return obs_size

    def get_obs(self):
        obs = self.hist_obs.reshape(self.n_agents, -1, order="F")
        return obs

    def get_obs_size(self):
        return self.__get_one_obs_size() * self.hist_obs_num

    def get_state(self):
        obs_dict,_,_,_ = self.env.get_state()
        obs_dict_copy = obs_dict.copy()
        obs_dict_copy.pop("rx_good")  # 删除rx_good
        obs_dict_copy.pop("channel_energy")  # 删除channel_energy
        obs_dict_copy.pop("left_energy_frac")  # 删除left_energy
        for key in obs_dict_copy.keys():
            obs_dict_copy[key] = obs_dict_copy[key].reshape(self.n_agents, -1, order="F")
        obs = np.concatenate([value for value in obs_dict_copy.values()], axis=1)
        # obs_with_action = np.concatenate((obs, self.__last_action), axis=1)
        # state = obs_with_action.flatten()
        state = obs.flatten()
        state = np.append(state, self._episode_steps / self.episode_limit)  # 添加归一化的时间步
        return state

    def get_state_size(self):
        """ Returns the shape of the state"""
        state_size = 0
        for key in self.env.observation_space.spaces.keys():
            if key == "left_energy_frac" or key == "channel_energy" or key == "rx_good":
                continue
            state_size += self.env.observation_space.spaces[key].shape[1]
        state_size *= self.n_agents
        # state_size += self.n_agents * self.n_actions
        state_size += 1 # 添加归一化的时间步
        return state_size

    def get_avail_actions(self):
        avail_actions = np.empty((self.n_agents, self.n_actions))
        for agent_id in range(self.n_agents):
            avail_actions[agent_id] = self.get_avail_agent_actions(agent_id)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        obs_dict,_,_,_ = self.env.get_state()
        avail_actions = [0] * self.n_actions
        if agent_id < 4:
            if obs_dict['queue_size'][agent_id] > 0:
                avail_actions = [1] * self.n_actions
            else:
                avail_actions[0] = 1
        else: 
            if obs_dict['queue_size'][agent_id] > 0:
                avail_actions = [1] * self.n_actions
            else:
                avail_actions[0] = 1
        return avail_actions

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions

    def reset(self):
        """ Returns initial observations and states"""
        self.env.reset()
        self.__init_hist_obs()
        self.last_info = {}
        self.__last_action = np.zeros((self.n_agents, self.n_actions), dtype=np.int16)
        self._episode_steps = 0
        return self.get_obs(), self.get_state()

    def render(self):
        raise NotImplementedError

    def close(self):
        self.env.close()

    def seed(self):
        raise NotImplementedError

    def save_replay(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_stats(self):
        return None
