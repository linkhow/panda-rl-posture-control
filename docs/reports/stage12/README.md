# 双语正式报告与可编辑源

英文为正式提交候选；中文为内容核对和学习版本。每份10页含参考文献。数字、表格、公式和图从同一shared content/data对象生成，两语言的caption/正文对应；numeric parity和PDF逐页QA分别记录，不把成功输出当视觉验收。

- `build_report.py`：可编辑模板，Python3.11语法兼容；从实际原1048、新600及调参/生成审计提取数字。
- `report_content.json`：可编辑双语正文、表、图和公式对象；`report_en.md/report_zh.md`是排版前文本副本。
- `assets/`：共享公式PNG及获许可字体/许可文件，避免构建机字体路径依赖；共享分析图在stage12/analysis。
- `report_numeric_parity.json`：来源SHA、相同表/公式/图对象、PDF页数/链接/SHA。
- `pdf_visual_qa.json`：最终逐页公式/字体/表格/图例/分页/链接检查记录。

```bash
# 只建立新的报告环境，不向原训练环境安装
python3.11 -m venv .report-venv
.report-venv/bin/python -m pip install -r docs/reports/stage12/requirements.report.txt
# 保持当前可编辑正文，不重新从模板覆盖编辑
.report-venv/bin/python docs/reports/stage12/build_report.py --content docs/reports/stage12/report_content.json
# 需要从最新分析模板刷新所有正文/数字时，省略--content
```

首次实际构建使用单独bundled Python3.12.14/reportlab4.4.9/pypdf6.10.0；公式用CPU Python3.11.16/Matplotlib3.11.2渲染。运行项目的新环境验证与PDF构建验证范围不同。PDF源不需要LaTeX；字体复制来源/许可证见assets/fonts。改正文或公式后需重渲染检查全部页，不自动沿用旧QA。

没有课程模板；不写组员分工、个人经历或个人掌握。Codex辅助工程、分析和报告如实披露。proposal曾承诺≥50%原创，但无课程计数口径或真实贡献依据，本轮不声称达到比例。具体课程AI/原创规则和最终验收需课程确认。
