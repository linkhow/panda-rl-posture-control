# 运行与复现

发布仓库包含独立运行所需的核心源码、参数场景、固定 split、配置和六个封存模型。它可以重放代表案例、评价全部固定模型，并从头训练。历史完整可行性见证和全部逐步轨迹保留在原实验档案；这些文件不参与运行时动作计算，也不是安装本仓库的前提。

以下命令在 clone 后的仓库根目录执行，采用独立 Python 3.11 CPU 环境。具体版本、安装来源与验证边界见 [安装说明](environment_zh.md)。先安装：

```bash
bash scripts/install_cpu.sh python3.11
```

如果机器上的 Python 3.11 命令不同，将末尾参数替换为该解释器路径。脚本只创建此仓库的 `.venv`；已存在时会拒绝覆盖。`run.sh` 使用 `.venv/bin/python` 并清除 `PYTHONPATH`、`PYTHONHOME`，避免 ROS 或其他工程的导入路径混入本项目。

## 最短检查

```bash
bash run.sh check_results
bash run.sh quick_check --output quick_001
```

`check_results` 只使用 Python 标准库，检查冻结文件 SHA256、保存结果的 1,048 行、八种方法各自的固定 131 个场景顺序和成功数，不运行物理仿真或训练。保存结果为 tracking 106、APF 123、三个 best 129/127/131、三个 last 125/129/125。

`quick_check` 再检查实际安装版本与 Panda URDF 校验值，加载六个模型，并重新运行原先的五个代表案例。每个案例使用真实电机控制、240 Hz 物理和安全检测；输出与参考的所有 `states.csv` 字段及确定性 summary 对照。若差异超出容差会报错并保留输出。运行还记录实际 `me5418` 导入路径，并故意探测原档案和 `private_witnesses` 读取，确认访问守卫会拒绝它们。

输出在 `outputs/quick_001/`，主要看 `quick_check.json`、`models_inspection.json`、`demonstrations_summary.json` 和各案例的 `comparison.json`。六个模型角色保持不变：三个 best 是主结果，三个 last 是预先规定的补充结果。

## 代表成功和失败案例

```bash
bash run.sh demo --inspect-models --output models_001
bash run.sh demo --suite --output demos_001
bash run.sh demo --case apf_success_learning_failure --method PPO_best_550901 --output failure_001
```

五个固定案例为：

| 案例 | 场景 ID 后四位 | 重放方法 | 用途 |
| --- | --- | --- | --- |
| `learning_success` | 0024 | `PPO_best_550901` | 学习策略成功示例 |
| `apf_success_learning_failure` | 0032 | `APF`、`PPO_best_550901` | 同一场景中 APF 成功、best 失败 |
| `best_last_difference` | 0044 | `PPO_best_550901`、`PPO_last_550901` | 同一 seed 的 best/last 行为差异 |

单个 `--case` 必须搭配表中的 `--method`。场景参数与公开固定数据逐项校验，禁止通过改场景、删失败或调阈值获得“通过”。预生成的少量演示视频用于观察；评价分数和差异检查来自重新执行的物理数据。

## 全部固定模型评价：131 × 8

```bash
bash run.sh evaluate --workers 4 --output fixed131_001
```

八种方法固定为 tracking、APF、三个 best、三个 last。每种方法执行相同顺序的全部 131 个 test 场景，共 1,048 回合；失败回合不重试、不删除，不按成绩重新选模型。默认 `--workers 1`；可选择 1 至 4 个独立 DIRECT 进程。每进程只使用一个 Torch 线程。并行加速会改变 CPU 竞争和墙钟时间，不能把这次并行计时替换为原报告的性能证据，也不能据此宣称严格实时。

入口先保存 `evaluation_plan.json`，包含场景顺序、参数校验、模型角色、模型/config 校验、环境及运行目的。每种方法在 `evaluation/<方法>/progress.jsonl` 写入进度。每回合独立保存 `states.csv`、`actions48.csv`、`summary.json`、安全/碰撞数据及 `COMPLETE.json` 文件哈希，最后生成：

- `episodes_all1048.csv`：新运行结果，列与原结果一致。
- `comparison.json`：逐回合比较全部确定性物理、状态统计和策略动作统计列。
- `execution_complete.json`：实际回合数、成功数、完整性、环境、模块来源和访问审计。
- `evaluation/<方法>/method_summary.json`：每种方法的实际汇总。

时间、失败状态、失败原因、碰撞接触带/负距离/明确穿透、关节违规、跟踪误差、速度、平滑度和动作剪裁统计均保留。墙钟计时七列不作为确定性对照项。离散字段与缺失值要求相同，首次失败时间容差 `1e-12 s`，其余连续量容差 `1e-7`。五个代表案例另外比较每个保存状态的全部字段；公开仓库没有上传原 1,048 回合的全部逐步轨迹，因此完整评价以原保存的每回合 CSV 为对照，不声称逐步比较了所有原历史轨迹。

全评估输出约 2 GB，运行过程至少保留 2 GiB 空闲磁盘。耗时依机器变化。若任何基础设施错误或数值差异出现，入口保留 `run_failure.json` 或方法内的 `infrastructure_failure.json`，不会修改封存模型、参考结果或原实验档案。

这批 test 已使用过，本次属于发布副本复现验证与回归，**不是新的未见测试，也不用于选模**。

## 对新结果做分析

```bash
bash run.sh analyze --input outputs/fixed131_001/episodes_all1048.csv --output analysis_001
```

生成 `analysis.json`、`success_counts.csv` 和 `success_counts.png`，包含八种方法的完整成功数、失败原因统计，以及每个 best 相对 APF 的成功得失和共同完整成功场景上的配对指标。默认不传 `--input` 时分析保存的原结果。配对均值只取双方都完整成功、且相应距离字段未被查询范围截断的场景，保留有效配对数量；不能用不同成功子集的均值直接宣称改进。逐回合差异对照由完整评价入口保存的 `comparison.json` 提供。

## 从头训练

```bash
bash run.sh train --smoke --output train_smoke_001
bash run.sh train --seed 550901 --output train_seed550901_001
bash run.sh train --all-seeds --output train_all_three_001
```

第一条只做 2,048 步检查，确认训练采样、PPO 更新、模型保存和重新加载可用，并评价更新前后全部 58 个 validation 场景。后两条固定为每 seed 1,024,000 步、三个 seed 合计 3,072,000 步；本次整理没有自动重跑正式训练。训练细节、validation 选模规则、输出和资源限制见 [训练说明](training_zh.md)。短训练的分数不能当作正式结果。

## 输出和版本边界

所有 `--output` 都是 `outputs/` 下的新名字；重复目录会拒绝覆盖，请保留证据并更换名称。`outputs/` 不上传，封存参考位于 `results/reference/`，模型位于 `models/`。访问守卫拒绝原档案和 `private_witnesses` 的文件读取；训练还拒绝 test split。公共 manifest 包含三类 split 的场景参数，训练环境读取后只保留所选 train/validation 参数字段，这与冻结核心一致；不能声称训练进程从未解析过 test 参数。

核心源码的计算行为保持封存版本；新的入口只改路径、运行编排、独立性审计与输出记录。检查环境接受 Python 3.11 的不同补丁号并记录差异；PyTorch 必须是同一基版本，可使用 CPU 构建。数值复现仍必须通过实际对照，版本可安装或模型可加载不等于已验证物理结果。当前机器、新环境、发布副本及远程 clone 的实际验证状态见 [复现验证记录](verification_zh.md)；跨机器逐字节训练一致性没有验证。
