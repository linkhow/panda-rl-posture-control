# 安装与环境边界

默认使用CPU与PyBullet DIRECT（不创建窗口）。准确历史环境为Python 3.11.16、PyBullet 3.2.7、Gymnasium 1.3.0、Stable-Baselines3 2.9.0、NumPy 2.4.6、Matplotlib 3.11.2、TensorBoard 2.21.0、PyTorch 2.13.0+cu126。历史安装URL与SHA保存在environment/original_install_sources.json，不要求安装CUDA Toolkit。

发布路线使用同一PyTorch基版本的官方CPU wheel 2.13.0+cpu，其SHA为`6746dbcbeb526eb61330b76b41ff1b4eb848951103a892eeb080dfa2b264667b`。其它依赖按历史版本锁定。原锁的packaging本地file URI替换为已安装记录中的26.3；剔除CUDA、NVIDIA与Triton依赖，安装源不使用机器本地URI。新CPU安装来源另存environment/cpu_install_sources.json。

需要Linux x86_64和可用的Python 3.11解释器（本次验证3.11.16）。先检查`python3.11 --version`，然后在仓库根运行：

```bash
bash scripts/install_cpu.sh python3.11
bash run.sh check_results
```

如果Python 3.11位于其它环境，可把解释器路径作为install_cpu.sh的第一个参数；脚本只用它新建仓库自己的.venv，不向原环境安装任何包。若本机只有Conda且没有Python3.11，可在独立目录创建解释器：`conda create --prefix ./python311-bootstrap python=3.11.16 pip`，然后把`./python311-bootstrap/bin/python`传给安装脚本；Conda下载这一步未另行实测，不代表已经验证全部平台的安装。

运行入口run.sh只清除当前子进程的PYTHONPATH/PYTHONHOME并禁用user site，避免ROS混入；不安装或修改现有环境的激活钩子。使用外部独立环境时可设`PYTHON_BIN=/path/to/python bash run.sh ...`。模型和数据校验仍执行，不能因缺文件跳过。

本次曾在新venv安装后直接运行pip check，因shell继承ROS路径显示launch-ros缺少PyYAML。清除外部Python路径后pip check通过，ROS元数据不再可见；没有为此修改ROS或补装无关包。CPU模型与原cu126环境同机轨迹是否一致，以docs/verification_zh.md中的实测为准。

Python解释器来自本机已有3.11.16，venv包全部通过网络重新安装，未启用system-site-packages。代码从发布副本导入；不把这些验证写成跨机器验证。GUI、其它系统/架构、网络源未来可用性和正式训练位级一致性尚未验证。
