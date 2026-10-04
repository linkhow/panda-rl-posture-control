# Stage12 复现与Release资产

当前推荐入口：README及delivery/stage12/run_full.sh。源码根目录用相对路径；运行先严格核对49项原冻结输入，原core、数据/模型权重、config与原1048结果不修改。新APF/生成/比较是独立stage12版本与协议；历史Stage09/10不被新代码冒充。

## 三种交付物

|交付物|可以完成|额外说明|
|---|---|---|
|普通Git clone|训练/简单验证/六模型加载/代表成功失败/原1048复现/新有限APF+生成+比较/小结果与双语报告|无环境及完整原始轨迹；新实验复跑先用prepare_reproduction|
|me5418-stage12-complete.tar.gz|同上，无需.git；完整课程代码/配置/数据/模型/报告|区别历史Stage11轻量演示；依赖按锁文件实际安装|
|新experiment-evidence.tar.gz|全部522调参、288候选账本、失败尝试、101见证+100重放、600评价原始证据|解压与complete同目录层级补outputs/stage12；私有见证不作为policyinput|
|original-frozen-evidence.tar.gz|原1048十文件提交、340原见证、原六模型、冻结源码/协议和结果|独立history根，只读审计；保留原bytes/metadata路径，不依赖它运行新portable代码|

## 下载与校验

私有仓库需要你自己的GitHub登录。不要发送token。

```bash
mkdir stage12_assets
 gh release download v1.1.0-stage12 --repo linkhow/panda-rl-posture-control --dir stage12_assets
cd stage12_assets
sha256sum -c SHA256SUMS
# 查看archive成员，确认没有绝对路径或..、软链接，然后解压
tar -tzf me5418-stage12-complete.tar.gz | head
tar -xzf me5418-stage12-complete.tar.gz
tar -xzf me5418-stage12-experiment-evidence.tar.gz
cd me5418-stage12-complete
bash scripts/install_cpu.sh python3.11
bash delivery/stage12/run_full.sh quick_check --output downloaded_quick_001
.venv/bin/python -B stage12/experiment.py audit
```

SHA256SUMS覆盖Release各asset及独立manifest/index（不对它自身做循环校验）。package内PACKAGE_MANIFEST_CODE/EVIDENCE/HISTORY逐文件SHA包括原始失败。原历史包不放env、cache、request文本或个人聊天。原source/日志中路径记录保留用于证据溯源，运行依赖使用complete包的相对根和pybullet_data。

## 全新目录和环境实测

独立目录/新venv创建、从官方CPUtorch索引和PyPI安装锁定版本、pipcheck无冲突。Python解释器/标准库复用同主机已存在3.11.16；没有继承原环境sitepackages，所有runtime包从新venv导入。六model load、3完整成功+2预期失败代表、全状态最大diff0，9风险回归、简单validation3及2048步train smoke均有日志与SHA。短训练不改变科学模型和原训练预算。本次不是跨硬件或实机验证。

本次1323新增科学运动回合：522validation search+101witness+100replay+600evaluation；工程smoke/代表/regression另计。各方法失败不重试，输出完整保存。实际墙钟搜索424.689s、生成204.068s、比较512.630s；每阶段1800s cap。provider48Hz与outer240Hz计时边界见报告和协议，均值不能宣称硬实时。

## 复跑纪律

prepare_reproduction --destination必须新目录且在源码外。reference evidence留在replication_reference/，49旧冻结输入复制SHA保持一致，仅清副本新stage12输出。先plan再tune/freeze/generate/evaluate/audit；有限grid、tie规则和预算先存；scene生成受freeze gate约束。继续研究改变参数/任务应新增协议版本，不能在已冻结发布副本改source/hash来刷新结论。
