import numpy as np


class DecayThenFlatSchedule():

    def __init__(self,
                 start,
                 finish,
                 time_length,
                 decay="exp"):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length / np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- T / self.exp_scaling)))
    pass

class CosineAnnealingSchedule():
    """
    余弦退火调度器，类似于PyTorch的CosineAnnealingLR
    """
    def __init__(
        self,
        start,
        finish,
        time_length,
        restart_period=None
    ):
        """
        参数:
            start: 初始值
            finish: 最小值
            time_length: 总的调度时间长度
            restart_period: 重启周期，如果不为None，则会每隔restart_period重新开始
        """
        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.restart_period = restart_period if restart_period is not None else time_length
        self.amplitude = start - finish  # 振幅

    def eval(self, T):
        """
        在时刻T评估当前值
        """
        # 如果设置了重启周期，计算在当前周期内的位置
        if self.restart_period is not None:
            T = T % self.restart_period
            
        # 确保T在有效范围内
        T = min(T, self.time_length)
        
        # 计算余弦退火值
        cosine_factor = 0.5 * (1 + np.cos(np.pi * T / self.time_length))
        current_value = self.finish + self.amplitude * cosine_factor
        
        return current_value

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    # 测试 CosineAnnealingSchedule
    schedule = CosineAnnealingSchedule(start=1.0, finish=0.1, time_length=100000, restart_period=100100)
    T_values = np.linspace(0, 400000, 1000)  # 从0到400000的时间点
    values = [schedule.eval(T) for T
        in T_values]
    plt.plot(T_values, values)
    plt.title("Cosine Annealing Schedule")
    plt.xlabel("Time (T)")
    plt.ylabel("Value")
    plt.grid()
    plt.show()