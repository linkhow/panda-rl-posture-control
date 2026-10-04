# 从头训练入口

`tools/train.py` 直接使用本仓库的 `me5418` 环境、控制器和 `configs/stage09/C.json`，不依赖原档案、旧阶段预算账本、历史中间模型或个人附件。新训练从随机策略和新优化器开始；不会加载已保存的正式模型继续训练。

安装独立 CPU 环境的方法见 [安装说明](environment_zh.md)。下面命令均在 clone 后的仓库根目录运行。`--output` 是 `outputs/` 下的新目录名；重复使用已存在目录会报错，请保留旧输出并更换名称。

## 先做短训练检查

```bash
bash run.sh train --smoke --output train_smoke_001
```

默认使用独立检查种子 559901 和四个 worker，采集 2,048 个交互步，完成一个 rollout 的 PPO 更新。它先评价未训练策略，再在更新后评价全部 58 个 validation 场景，验证采样、有限数值、策略参数确实改变、优化器保存与模型重新加载。若希望采用原短试跑的长度：

```bash
bash run.sh train --smoke --steps 8192 --output train_smoke_8192
```

`--steps` 仅在 `--smoke` 下有效，必须为 2,048 的正整数倍。检查模型、验证分数和日志都标记 `smoke_only: true`，不能用来替换正式结果或宣称完成正式训练。最短检查仍需两次完整 validation，所以验证耗时可能比学习更长。

## 单个正式种子

```bash
bash run.sh train --seed 550901 --output train_seed550901_001
```

另外两个正式种子也可分别启动：

```bash
bash run.sh train --seed 551901 --output train_seed551901_001
bash run.sh train --seed 552901 --output train_seed552901_001
```

每个命令固定训练 **1,024,000 个交互步**。四个环境各采集 512 步后一起更新，总计每个 rollout 2,048 步；每个正式种子正好 500 个 rollout。交互步数包括四个环境的合计步数，不是物理仿真的 240 Hz 步数。

## 三个正式种子顺序训练

```bash
bash run.sh train --all-seeds --output train_all_three_001
```

此命令顺序执行 550901、551901、552901，各自从头初始化，共计 **3,072,000 个交互步**。不会根据成绩提前结束、换种子或延长预算。本次仓库整理只运行短检查，未自动重跑这三个百万步实验。

## 保持哪些设置

| 项目 | 冻结值 |
| --- | --- |
| 输入与动作 | 91 维固定缩放观测；7 维归一化动作，再映射到每关节 ±0.2 rad/s 姿态指令 |
| 任务执行 | 240 Hz 物理与安全检测；每五个物理步更新姿态策略，即 48 Hz；每回合固定四秒 |
| 训练数据 | train 151 个参数场景，far/near_link/tight_layout 分别 72/64/15 个 |
| 训练采样 | 三类概率 20%/40%/40%；类别内均匀、有放回；不使用可行性见证 |
| worker | 四个独立 CPU worker；种子为基础种子加 0、1、2、3；每进程一个 Torch 线程 |
| PPO | learning rate 0.0003；gamma 0.995；GAE lambda 0.95；clip range 0.2；batch size 256；每次 10 个 epoch |
| 网络 | actor 与 critic 各有 64、64 两层；初始 Gaussian std 为 1；无 VecNormalize |
| 保存间隔 | 每 51,200 个交互步保存策略、优化器及可获得 RNG 状态 |
| 选模间隔 | 每 102,400 个交互步，在 PPO 更新完成后确定性评价全部 58 个 validation 场景 |

精确观测、奖励、主跟踪、仿真与安全数值来自封存核心及配置。入口只重写运行编排、路径处理和日志，不修改这些模块。

## validation 怎样选 best

未训练模型在第 0 步保存并评价，用于检查基线，**没有 best 资格**。正式 best 只从第 102,400、204,800、……、1,024,000 步十个时点中选择：

1. 先比较全部 58 个场景的完整成功数。
2. 成功数精确相同，再比较平均折扣回报。
3. 两项精确相同，保留更早时点。

validation 场景使用原封存 `load_pool` 的 manifest 顺序。策略采用 Gaussian 均值再剪裁到动作范围，与原 validation 实现一致。每次验证使用独立仿真客户端，不重建或重播种训练 worker；入口实际检查 Python、NumPy、Torch 全局 RNG 和每个 worker 私有 RNG 在验证前后不变。

`last` 是完成固定预算后的模型；其角色不会根据 test 分数变化。训练过程没有 test 回合，也禁止打开 test split 与 `private_witnesses`。公开 manifest 本身包含 train、validation、test 的公共场景参数，因此读取 manifest 会解析三类参数；环境随后只保留所选 train 或 validation 的参数字段。这不能写成“训练进程从未读入任何 test 参数”。完整离线可行性见证不在发布仓库，也不参与观测、奖励、采样或动作。

## 输出怎样检查

以单种子命令为例，新文件位于 `outputs/train_seed550901_001/seed550901/`：

- `actual_execution.json`：真实种子、worker 种子、数据顺序、配置校验值、导入模块来源与输出角色。
- `training_progress.jsonl`、`sampling_audit.jsonl`：训练回合和各 worker 的采样记录；后者在 `training/env0/` 至 `env3/` 中。
- `ppo_update_diagnostics.jsonl`、`action_distribution.jsonl`：更新次数、有限性和原始动作剪裁比例。
- `validation/step_*/summary.json`、`validation/history.json`：固定时点评价结果。
- `validation_rng_audit.jsonl`：验证未改变训练 RNG 的实际检查。
- `checkpoints/untrained.zip`、`step_*.zip`、`best.zip`、`last.zip`：新模型与优化器；`.rng.pt` 保存可获得的随机状态。
- `checkpoints/best_selection.json`、`selection_process.jsonl`、`final_lock.json`：best 选择过程和最终模型角色。
- `reload_check.json`：真实训练观测上的保存/重新加载动作差值，要求为零，并检查优化器状态存在、策略参数已更新。
- `run.json` 和上一级 `training_summary.json`：完成步数、更新次数、耗时与完整性。
- `tensorboard/`：可使用 `.venv/bin/tensorboard --logdir outputs/train_seed550901_001/seed550901/tensorboard` 查看训练曲线。

若运行出错，入口保留 `interruption.json`，并尽可能保存 `interrupted.zip`，不删除失败证据。保存点没有完整的 Bullet 当前回合状态，本入口不提供同轨迹无缝续训；需要新的训练输出目录重新启动。不要把策略和优化器可加载等同于完整仿真状态可恢复。

## 资源和可复现边界

默认 CPU，不要求 CUDA Toolkit 或 GPU。需要能运行四个物理环境和一个验证客户端的 CPU、内存与磁盘；入口采用原保护阈值：父进程及 worker 合计实时 RSS 不超过 12 GB、可用内存不少于 512 MiB、空闲磁盘不少于 2 GiB。Linux 上在每个 rollout 边界检查内存，时间与磁盘同时检查；这不是严格实时资源保证。

一次新命令共享九小时保护上限，三 seed 命令也是合计九小时；不引用历史阶段已经使用的预算。达到保护阈值、数值非有限或环境错误时保留证据并停止，不按成功率停止。

原档案短试跑记录学习速率约 515 个交互步/秒，每次 58 个 validation 约 42.8 秒，原估计三次正式训练约 2.05 小时，含 50% 余量及收尾约 3.19 小时。这些是原机器上的估计，不能承诺新机器耗时。新环境、线程调度与浮点库差异也可能影响训练随机轨迹。已保存模型的固定评估复现与新训练逐字节产出相同模型，是不同层次的要求；后者尚未跨机器验证。

后续研究请使用独立分支、独立配置和新的输出目录，并保留原六个正式模型及结果。已使用的 test 是回归集合；新的泛化结论需要新的未见评价数据，不能用 test 调参再把结果当首次测试。
