# 第三方来源、项目代码与助手辅助

本项目组合已有仿真和学习工具，实现特定的 Panda 定时位置跟踪与辅助姿态避障任务。DLS、近似零空间投影、APF 和 PPO 均不是本项目提出的新算法理论。

## 第三方复用

| 来源 | 本项目实际用途 | 上游入口 |
| --- | --- | --- |
| PyBullet / `pybullet_data` | 动力学、电机、碰撞距离、运动学、Panda URDF/网格、TinyRenderer | [Bullet 官方仓库](https://github.com/bulletphysics/bullet3) |
| Gymnasium | 标准环境接口、空间、reset/step 与终止/截断约定 | [Env 官方文档](https://gymnasium.farama.org/api/env/) |
| Stable-Baselines3 | PPO、ActorCriticPolicy、向量环境及学习/保存/加载 | [PPO 官方文档](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) |
| PyTorch | 策略网络、自动微分及优化器运行 | [PyTorch 官方文档](https://docs.pytorch.org/docs/stable/index.html) |
| NumPy、Matplotlib、TensorBoard、Pillow | 数值计算、图表、训练日志与图像输出 | [NumPy](https://numpy.org/)、[Matplotlib](https://matplotlib.org/)、[TensorBoard](https://www.tensorflow.org/tensorboard)、[Pillow](https://python-pillow.github.io/) |
| 旧系统 OpenCV 4.6.0 | 原代表视频编码与解码核对 | [OpenCV 官方仓库](https://github.com/opencv/opencv) |

上游文档说明库的接口与算法来源；本项目实际版本和安装来源由保存的环境记录及当前安装锁确定，不以网页最新版本代替。仓库不再分发第三方库源码或 Panda 网格，Panda 资产由安装的 `pybullet_data` 提供。原视频为既有媒体产物，当前核心训练/评估没有 OpenCV 依赖。

## 项目专用代码与内部复用

| 代码 | 项目特定内容 |
| --- | --- |
| `scene.py` | 固定场景、名称映射、工具点、检测与自碰撞过滤 |
| `trajectory.py`、`tracking.py` | 时间参数化、位置跟踪、DLS 与近似投影、命令限制和电机执行 |
| `posture.py`、`posture_run.py` | 七维姿态接口、真实表面 APF 与开发运行 |
| `scenarios.py`、`evaluation.py` | 场景筛选和统一任务、安全、指标执行 |
| `arm_env.py`、`diagnostic_env.py`、`sampling.py` | 观测/奖励、91 维表示、类别采样及日志 |
| `final_evaluation.py` | 冻结策略确定性适配、原始剪裁与完整外层计时 |
| 当前公开入口 | 路径适配、发布文件校验、演示、评估、分析与从头训练编排 |

原阶段之间直接复用了项目代码；这次复制冻结核心保留其来源，没有把复制代码叫作新的独立算法实现。场景参数、split、六冻结模型和原正式结果来自同一项目档案，复制/提取位置与 SHA 记录在发布清单中。公开参数文件供当前方法运行；完整离线见证、历史搜索与大量逐步日志仍保留在原档案。

## 助手辅助与贡献表述

已有来源清单和阶段报告明确记录：助手在用户提出的项目目标、任务契约和约束下，辅助实施项目专用场景、控制、环境、训练编排、评价、审计、结果整理和文稿。本次整理还使用助手检查历史证据，分离个人资料，编写公开中文总结与独立复现入口，并执行可获得环境中的验证。实际完成与尚未完成的验证以复现记录为准。

文件不能证明各位成员的实际设计、编码、运行、分析和写作分工。因此不宣称某个人独立完成全部代码，也不虚构组员贡献或原创比例。第三方算法实现、项目专用代码和助手辅助分别如实列出。

本次没有自行选择项目开源许可证；依赖的来源说明也不等于给项目授予额外许可。后续若决定公开或授权复用，应由项目权利人确认相应条件。

## 来源追溯

原 `results/final/source_reuse_inventory.md`、Stage11 来源说明及 Stage06–10 实施报告用于核对历史来源；其中夹杂个人资料的完整文稿没有上传。公开技术摘要依据和原相对路径/SHA 见[技术来源清单](project_summary_sources.json)，本仓库最终上传文件由发布清单逐项确定。

## Stage12新增交付辅助

Codex辅助实施有限APF验证搜索、freeze后新场景与600评价、风险回归、新环境安装验证、模板相关分析、双语报告和Release打包。原正式PPO模型未重训；工程2048步smoke独立输出。学生职责/原创比例未推定。报告字体DejaVu/Bitstream与Droid/Apache许可文件随assets/fonts保留；ReportLab、pypdf和Matplotlib用于报告构建，项目runtime锁依赖不因报告构建变更。同学kimzclandi仓库仅参考，不作为发布目标或本项目成绩来源。
