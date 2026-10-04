# Panda 强化学习辅助姿态控制 · ME5418 Group44

固定底座 Panda 用4秒短直线跟踪工具位置，并在单静态球附近调整关节构型。DLS跟踪器、阻尼近似零空间、240 Hz物理/安全检查和48 Hz辅助接口由所有方法共享。工具朝向不受控，手指位置保持0.02 m。

## 当前结果与结论

|方法|原已开封test131|新增补充100|
|---|---:|---:|
|tracking|106/131|93/100|
|原固定APF|123/131|99/100|
|验证集调优APF|未补跑原test|100/100|
|PPO best550901|129/131|98/100|
|PPO best551901|127/131|98/100|
|PPO best552901|131/131|99/100|

原1048回合与六模型保留：三个best在原集合超过固定APF；last125/129/125仅次级，不按test换模型。新增600回合中三个PPO成功数均低于调优APF。本轮补齐公平调参和可复现交付，没有追加正式PPO训练。

9格APF参数仅58validation选，全部58/58，按事前RMSE平分规则选择d0=0.06 m、fmax=2 rad²/(m·s)，522回合424.69 s。冻结后生成100新场景，每场景实际电机见证及重放通过、最大状态/命令差0，精确/近似重复和旧重叠0。新集合19接受模板组，far41/near49/tight10，99个见证为固定APF、1个为gentleAPF，筛选偏差强；属已知任务族的补充泛化评价，不能把原/新分数变化当性能提升或独立研究。

## 最短运行与完整课程入口

Linux x86_64、Python3.11；实际新环境3.11.16。私有仓库需要相应账户权限。

```bash
git clone https://github.com/linkhow/panda-rl-posture-control.git
cd panda-rl-posture-control
bash scripts/install_cpu.sh python3.11
bash delivery/stage12/run_full.sh quick_check --output quick_001
bash delivery/stage12/run_full.sh regression --output outputs/risk_001.json
bash delivery/stage12/run_full.sh validate --seed 550901 --n 3 --output val_001
# 短学习接口检查：2048交互，不代表正式3,072,000步重训
bash delivery/stage12/run_full.sh train --smoke --output smoke_001
# 从头正式训练入口；需要明确计算预算，按原validation规则选best
bash delivery/stage12/run_full.sh train --all-seeds --output train_001
# 原8方法×131，写新目录，不更新历史结果
bash delivery/stage12/run_full.sh evaluate --workers 4 --output fixed131_001
```

新`.venv`与输出目录独立；既有输出拒绝覆盖。模型加载还可用 `bash run.sh demo --help`，同一入口支持成功与失败代表。新环境安装、六模型、3成功/2预期失败全状态差0、9风险回归、2048步参数更新/模型与优化器重载差0均已实测；证据见[完整包验证](delivery/stage12/validation/README.md)。验证沿用同一台机器的Python解释器/标准库，但没有继承原环境sitepackages，依赖重新安装。

## 复现本轮APF搜索与新场景

当前Stage12协议与证据拒绝覆盖，先创建保留参考证据的独立副本：

```bash
.venv/bin/python -B stage12/prepare_reproduction.py --destination ../stage12_reproduction_001
cd ../stage12_reproduction_001
bash scripts/install_cpu.sh python3.11
for phase in plan tune freeze generate evaluate audit; do
  bash delivery/stage12/run_full.sh apf "$phase"
done
.venv/bin/python -B stage12/analyze.py
```

原core与49项冻结输入仍严格校验。新副本将原Stage12保留在`replication_reference/`，仅清空副本内的新增生成项；不关闭旧校验或改写旧哈希。预算为搜索/生成/比较各1800 s上限；本轮实际分别424.69/204.07/512.63 s，共1323科学运动回合，包含所有失败与见证重放。计时均非严格实时证明。

## 报告、结果和交付范围

- [英文正式候选](docs/reports/stage12/ME5418_Group44_Final_EN.pdf)、[中文核对版](docs/reports/stage12/ME5418_Group44_Final_ZH.pdf)：各10页含参考文献，共享数字/表/图/公式，保留JSON、Markdown和生成器。[报告重建说明](docs/reports/stage12/README.md)。
- [Stage12协议](stage12/configs/protocol.json)、[APF冻结](stage12/configs/tuned_apf_freeze.json)、[新600结果](stage12/results/episodes_new600.csv)、[新实验审计](stage12/results/integrity_audit.json)。
- [分析入口与指标](stage12/analysis/README.md)：每seed/类别/模板、逐ID得失、失败、共同完整成功指标和整模板bootstrap。原20组、新19组，seed不扩大场景分母；没有显著性/通用安全结论。
- [完整课程包说明](delivery/stage12/README.md)、[复现与下载](docs/stage12_reproduction_zh.md)、[原实验冻结输入](provenance/release_integrity.json)、[模型来源/校验](models_index.json)。
- [来源与助手辅助](docs/third_party_and_assistance_zh.md)：Codex辅助实施/执行/分析/报告/打包；不虚构个人职责、独立掌握或原创比例。课程无已提供模板，具体AI/原创规则和最终验收仍待课程确认。

普通clone含完整代码、锁依赖、340旧+100新参数场景、六模型、小结果/见证索引、正式报告和必要媒体。Release另给完整代码压缩包、本轮全部搜索/候选/见证/replay/600原始记录、旧1048+340见证+原模型历史证据包。大资产附SHA256SUMS；见[Release v1.1.0-stage12](https://github.com/linkhow/panda-rl-posture-control/releases/tag/v1.1.0-stage12)。环境、缓存、凭据、个人聊天及私人学习/求职记录不上传。

Stage11轻量演示包在原项目历史中保留，缺少完整训练入口，不能替代本次完整Stage12课程包。当前推荐入口是本README及`delivery/stage12/run_full.sh`；原运行入口`run.sh`继续可用。跨机器、实机、完整GUI、严格实时与输入/奖励/投影消融尚未验证。
