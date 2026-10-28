# MechKG 主模型代码、数据与权重上传包

[English](UPLOAD_BUNDLE_README.md) | **简体中文**

整理日期：2026-10-26。最终主模型：`mk_ns_nodrop`，包含其 **5 折模型**。
这是主模型独立复现包；未收录历史模型、消融或基线权重、网站文件、缓存、远程连接配置和密钥。
原工作区中的代码、原始数据及完整训练 checkpoint 均未改动。

## 上传文件

`upload_packages/` 中有 3 个可分别上传的 ZIP：

| 文件 | 内容 |
|---|---|
| `MechKG_code.zip` | `code/`、主模型已有结果 `results/`、说明、模型配置与文件校验清单 |
| `MechKG_data.zip` | 全部必要输入数据 `data/`，含完整表达矩阵、知识图谱、划分、文本/序列特征 |
| `MechKG_models_5fold.zip` | `models/`：仅 5 个 `.pth` 权重及模型配置、来源记录 |

将 3 个 ZIP **解压到同一个空目录**，得到下述结构；不需要原项目其他目录。
如上传平台允许多个附件，直接上传三个 ZIP，另附本 README 即可。

```text
README.md
MODEL_CARD.md
PROVENANCE.json             # 文件在原项目中的来源；不包含机器绝对路径
FILE_MANIFEST.json          # 大小、SHA-256；不包含压缩包本身
SHA256SUMS.txt
code/                      # 核心代码及独立训练/推理/校验入口
data/                      # 见 data/README.md
models/                    # fold_1_model.pth ... fold_5_model.pth
results/                   # 已有主模型结果，未经重新训练或推理
```

## 环境和运行

在解压目录打开终端。环境版本记录在 `code/environment_record.json`；本地检查使用 Python 3.13。
使用已有匹配的 PyTorch 环境，或按硬件准备 PyTorch 后安装 `code/requirements.txt` 中的依赖。
表达矩阵读取和 PCA 需要较多内存与 CPU 时间；本包保留原始矩阵全部行列，避免改变模型输入。

```bash
python -m pip install -r code/requirements.txt

# 校验所有文件、五折模型结构、划分和已保存预测，不训练、不推理
python code/check_release.py

# 只检查 SHA-256 和文件个数：无需安装第三方库
python code/check_release.py --hash-only

# 加载现有 5 折模型，逐折评估标准测试集和已冻结 hard 测试集
python code/predict_main.py --protocol both --device cuda

# 单折及通道贡献导出（输出到单独新目录）
python code/predict_main.py --fold 1 --protocol standard --explain --output runs/explain_fold1

# 从头训练，结果写入 runs/retrained_main/，不会覆盖上传的权重
python code/train_main.py --device cuda
```

没有 CUDA 可将 `--device cuda` 改为 `--device cpu`；CPU 上完整训练和推理会耗时较长。
入口自动使用包内路径，并设置 `MPLBACKEND=Agg`；环境变量 `PYTHONIOENCODING=utf-8` 可在启动 Python 前设置。
请通过 `train_main.py` / `predict_main.py` 运行；`main.py`、`prediction.py` 保留作为原代码模块。
预测默认分别输出五折结果，不将集成结果作为主结果。

## 数据和模型口径

- 配置见 `models/main_model_config.json`：HGT 编码器、线性证据读出、关闭协同通道、关闭通道失活、无实体残差、无全局分支。
- 使用 `kg_v2_xgft_coding_nosim.csv`，已移除不正确的序列相似边；保留孤立节点表。
- 表达输入：PCA 64 维 + 可用标记，缺失填 0；配对序列特征 6 维。
- 训练每批遮蔽监督正例，动态负样本 1:3。每折消息图只加入该折训练正例，不加入验证/测试靶向边。
- 五折共享同一独立测试集，不是 5 次完全独立实验。结果报告单模型均值和样本标准差。
- `results/main_model/results/kfold/kfold_results.json` 是原始结果，其中 `std` 用总体标准差；论文口径的样本标准差见 `results/main_model_summary.json`。
- 原始结果中的 `AUPR` 对应 sklearn 的 average precision（AP）。

## 整理和验证范围

`models/*.pth` 为推理便携版：全部 `model_state_dict` 张量与原始权重逐一完全相同，
移除优化器状态、将 NumPy 元数据转成 Python 原生类型，可 `torch.load(..., weights_only=True)` 加载。
原始 MD5、训练轮次、指标及转换记录见 `models/model_manifest.json`。
这些权重用于推理；从头训练由原训练器完成。如需从原训练中断点续训，请保留原项目的完整 checkpoint。

`code/models.py`、`mechkg.py`、`data_loader.py` 等模型核心保持原样；路径集中改为包内路径；
训练器中端点匹配采样器改为调用 `sampling.py`，函数实现从原模块原样提取。
本次整理执行了 CPU 文件/结构/划分/已有预测检查，未启动训练或模型推理。
新增的训练、推理入口尚未执行完整训练或前向验证；跨平台数值复现仍需在运行环境确认，尤其是 randomized PCA 和线性代数库版本。
`PROVENANCE.json` 记录输入与已有结果来源，`VERIFICATION.json` 记录本次检查。

外部来源数据及预计算嵌入沿用原项目来源；此包未新增或变更第三方数据的授权，未擅自指定开源许可证。
