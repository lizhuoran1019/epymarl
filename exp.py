import subprocess  # 导入subprocess模块用于创建子进程
import time

avail_list = [6,7,8,5,4]
a2_list = [0.4,0.5,0.6,0.7]
avail_gpu = 8

for avail in avail_list:
    processes = []
    for a2 in a2_list:
        env_name = f"ns3_avail_{avail}_a2_{a2}"
        # 开启一个子进程执行命令
        cmd = f"python src/main-ns3.py --config=mappo_noise --env-config={env_name} with save_model=True t_max=2001000"
        for i in range(avail_gpu):
            cmd_gpu = f"CUDA_VISIBLE_DEVICES={i} {cmd} > /dev/null 2>&1"
            print(f"执行命令: {cmd_gpu}")
            proc = subprocess.Popen(cmd_gpu, shell=True)
            processes.append(proc)
    
    # 等待当前avail值的所有子进程结束
    print(f"等待avail={avail}的所有进程完成...")
    for proc in processes:
        proc.wait()
    print(f"avail={avail}的所有进程已完成，开始下一组...")


