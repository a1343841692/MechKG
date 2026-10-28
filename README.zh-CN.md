# MechKG

[English](README.md) | **简体中文**

miRNA–mRNA 靶向预测与机制通道贡献分析。这里收录最终主模型 **`mk_ns_nodrop`** 的代码、数据说明和已有结果。
上传的模型仅包含该主模型的 **5 折权重**。

## 下载数据与模型

完整数据、五折权重和原始代码上传包位于 [Release v1.0.1](https://github.com/a1343841692/MechKG/releases/tag/v1.0.1)。
仓库及附件均为私有，需要有仓库访问权限的 GitHub 账号登录。

| Release 附件 | 内容 | 大小 |
|---|---|---:|
| `MechKG_code.zip` | 原始整理版代码、已有主模型结果与说明 | 约 14 MB |
| `MechKG_data.zip` | 必要数据、完整表达矩阵、知识图谱、固定划分与特征 | 约 440 MB |
| `MechKG_models_5fold.zip` | 仅主模型的 5 个便携 checkpoint | 约 49 MB |
| `MechKG_arm_annotation_flags.zip` | 137 条独立臂注释疑似标记、序列/CLASH 证据及既有评分诊断 | 约 2.4 MB |
| `ARCHIVE_SHA256SUMS.txt` | 三个 ZIP 的 SHA-256 | — |

**使用此仓库的代码：**

```bash
gh repo clone a1343841692/MechKG
cd MechKG
gh release download v1.0.1 --repo a1343841692/MechKG --pattern 'MechKG_data.zip' --pattern 'MechKG_models_5fold.zip' --dir downloads
python -m zipfile -e downloads/MechKG_data.zip .
python -m zipfile -e downloads/MechKG_models_5fold.zip .
python -m pip install -r code/requirements.txt
python code/check_release.py
```

也可下载三个 ZIP 并解压至同一个空目录；完整说明见 [原始上传包说明](docs/UPLOAD_BUNDLE_README.zh-CN.md)。
请勿将代码 ZIP 覆盖到 Git 克隆目录：仓库 README 和校验清单针对 GitHub 版本做了调整；
两种方式请分别使用自己的校验清单。

## 臂注释疑似标记

新增[臂注释数据集](annotations/arm_annotation_flags/README.zh-CN.md)：137 条疑似记录，涉及 78 个 miRNA，其中测试集 27 条。这是独立审计的待核查标记，不是已确认错误；CLASH 为附加证据而非筛选条件。[下载 ZIP](https://github.com/a1343841692/MechKG/releases/download/v1.0.1/MechKG_arm_annotation_flags.zip)，其校验值见 `MechKG_arm_annotation_flags.sha256`；原三个核心上传包及其校验值不变。

## 运行

```bash
# 逐折评估现有权重；输出写入 runs/predictions/
python code/predict_main.py --protocol both --device cuda

# 单折通道贡献导出
python code/predict_main.py --fold 1 --explain --output runs/explain_fold1

# 按最终配置从头训练，不覆盖发布权重
python code/train_main.py --device cuda
```

没有 CUDA 时可使用 `--device cpu`。环境版本记录在 `code/environment_record.json`。
表达数据的读取和 PCA 需要较多 CPU 时间与内存，完整训练也会耗时。

## 模型与数据

- 最终图谱：`kg_v2_xgft_coding_nosim.csv`，保留孤立节点表；无序列相似边。
- HGT 编码器与线性证据读出；关闭协同通道、通道失活、实体残差和全局分支。
- 表达 PCA 64 维加可用标记；6 维配对序列特征；动态负样本 1:3。
- 固定五折划分，五折共享同一独立测试集；消息图只含各折训练正例。
- 报告单模型均值与样本标准差；不将五折集成作为主结果。
- 模型参数与原 checkpoint 逐张量相同，便携版权重移除了优化器状态。

| 评估协议 | 单模型 AUC（均值 ± 样本 SD） | 单模型 AP（均值 ± 样本 SD） |
|---|---:|---:|
| 标准独立测试 | 0.9201 ± 0.0026 | 0.9244 ± 0.0024 |
| 端点度匹配 hard 测试 | 0.6969 ± 0.0091 | 0.6934 ± 0.0152 |

上述数值来自已有逐折预测，整理及上传过程中没有重新训练或执行模型前向推理。
已做 CPU 文件校验、权重结构严格加载、划分互斥及已保存预测指标核验。
新增训练/推理入口尚未执行完整训练或前向验证，跨平台数值复现仍需运行确认。

更多信息：[模型说明](MODEL_CARD.md)、[数据说明](data/README.md)、
[最终配置](models/main_model_config.json)、[权重来源](models/model_manifest.json)、
[原文件来源](PROVENANCE.json)、[整理检查记录](VERIFICATION.json)。

第三方数据来源与授权沿用原项目记录；本仓库未指定新的开源许可证。
