# Panda 强化学习姿态控制

[English](README.md) | [简体中文](README.zh-CN.md)

## 项目介绍

Franka Panda 机械臂在规定时间内跟踪三维直线，同时调整关节构型，避开静态球形障碍物。传统雅可比控制器负责工具位置跟踪，人工势场（APF）或 PPO 提供附加的姿态指令。

所有方法共用跟踪器、电机、约束和碰撞检测。这里的**姿态指关节构型**，不控制工具朝向。任务在 PyBullet 中运行，采用固定底座、固定手指和四秒参考轨迹。

## 演示

**跟踪器失败，APF 与 PPO 成功**：原集合场景 `0024`，PPO seed `550901`。

![跟踪器、APF 和 PPO 对比：APF 与 PPO 完成任务](media/gifs/tracking-apf-ppo-success.gif)

**APF 成功，PPO 失败**：原集合场景 `0032`，PPO seed `550901` 约在 0.496 秒终止。

![失败案例对比：APF 完成任务，PPO 提前终止](media/gifs/apf-success-ppo-failure.gif)

动画来自已有视频，按保存的实际关节状态重建。红色为障碍物，绿色为工具点；失败回合在终止后保持冻结画面。[完整分辨率视频](media/videos)。

## 方法概述

- **位置跟踪**：五次时间函数实现平滑起停，阻尼最小二乘雅可比控制器跟踪参考工具位置。
- **姿态调整**：仅跟踪时使用 `u = 0`。APF 根据真实表面距离和关节限位生成七关节速度指令；PPO 根据91维运动与几何观测预测同样的七维指令。
- **统一执行**：姿态指令限制在 ±0.2 rad/s，经近似零空间投影后叠加到主任务。姿态更新为48 Hz，电机执行和安全检查为240 Hz。阻尼使投影无法完全消除对位置任务的影响。

PPO 学习的是姿态部分，位置控制器仍负责主任务。完整成功要求运行满四秒、全程跟踪误差不超过1 cm，且无碰撞带或关节约束失败。

## 结果

两个评价集合分别报告。PPO 使用三个**由 validation 选定的 best 检查点**，没有根据这些结果重新选模型。

| 方法 | 原集合 · 131场景 | Stage12 新增补充集合 · 100场景 |
|---|---:|---:|
| 仅位置跟踪 | 106/131 | 93/100 |
| 固定 APF | 123/131 | 99/100 |
| validation 调优 APF | 未评价 | 100/100 |
| PPO · seed 550901 | 129/131 | 98/100 |
| PPO · seed 551901 | 127/131 | 98/100 |
| PPO · seed 552901 | 131/131 | 99/100 |

![原集合与新增补充集合的成功率](stage12/analysis/success_original_supplementary.png)

PPO 在原集合中比固定 APF 完成更多场景；调优 APF 在新增集合中成功数最高。其参数仅在原58场景 validation 上选择，并在新增场景生成前冻结。

新增集合的几何比例不同，且使用 APF 家族的可行性见证筛选。它属于已开发任务族的补充评价，并非完全未受开发影响的基准。部分 PPO 种子的指令变化和关节速度代价更大，也仍有碰撞带失败；进入 +0.1 mm 接触带不一定意味着负穿透。结果不构成统计显著性、通用安全或实机性能保证。

逐回合结果：[原集合](results/reference/episodes_all1048.csv) · [新增集合](stage12/results/episodes_new600.csv)。三个 `last` 检查点保留供稳定性比较，原集合成功数分别为125、129、125/131。[详细指标](stage12/analysis/README.md)。

## 安装

使用 Linux x86_64、Python 3.11，并确保支持 `venv`。CPU 即可运行，无需 CUDA。安装脚本创建 `.venv` 并安装锁定依赖，PyTorch 来自官方 CPU wheel 源。

```bash
git clone https://github.com/linkhow/panda-rl-posture-control.git
cd panda-rl-posture-control
bash scripts/install_cpu.sh python3.11
```

[环境说明](docs/environment_zh.md)。

## 使用

先只读核对已有结果，再复现一个成功案例和一个预期失败案例：

```bash
bash run.sh check_results
bash run.sh demo --case learning_success --method PPO_best_550901 --output demo_success
bash run.sh demo --case apf_success_learning_failure --method PPO_best_550901 --output demo_failure
```

按固定配置从头训练，或在原集合中评价六个预训练检查点与两个基线：

```bash
# 三个 seed，各1,024,000交互步；只使用 validation 选择 best。
bash run.sh train --all-seeds --output training_run

# 八方法 × 131场景；结果写入新目录，并与保存的参考结果比较。
bash run.sh evaluate --workers 4 --output evaluation_original131
```

输出保存在 `outputs/`，已有名称会拒绝覆盖。训练和完整评价需要时间与磁盘空间，可用 `--help` 查看选项。新增 APF 实验见[保留参考证据的复现说明](docs/stage12_reproduction_zh.md)。

## 简要目录结构

```text
me5418/        场景、跟踪器、APF 与 Gymnasium 环境
configs/       冻结的物理、控制与 PPO 配置
datasets/      原场景参数与 train/validation/test 划分
models/        三个 best 和三个 last PPO 检查点
tools/         演示、训练、评价和结果核对入口
stage12/       调优 APF、新增场景及保存的结果
media/         GIF、视频和图表
docs/          安装、复现和技术说明
run.sh         统一命令入口
```

## 参考资料

- [PyBullet](https://github.com/bulletphysics/bullet3)：仿真、碰撞查询和自带 Panda 模型。
- [Gymnasium](https://gymnasium.farama.org/)：环境接口。
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) 与 [PyTorch](https://pytorch.org/)：学习实现。
- [Schulman 等，*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347)。
- [第三方来源与助手辅助实现说明](docs/third_party_and_assistance_zh.md)。
