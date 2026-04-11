# SASRec - MovieLens 1M 序列推荐项目

这是一个基于 **SASRec (Self-Attentive Sequential Recommendation)** 的电影序列推荐项目，数据集使用 **MovieLens 1M**。

项目会按用户的历史观影序列建模，预测下一部最可能观看的电影，并支持基于 `Recall@K`、`NDCG@K` 的离线评估。

## 项目结构

```text
sasrec/
├── main.py           # 训练入口、评估入口、日志输出
├── data.py           # 数据读取、序列构造、负采样、推荐候选生成
├── Net.py            # SASRec 模型结构
├── SasrecData.py     # Dataset 与 collate_fn
├── ml-1m/            # MovieLens 1M 数据集
│   ├── ratings.dat
│   ├── users.dat
│   ├── movies.dat
│   └── README
├── sasrec_model.pth  # 训练保存的模型权重
└── sasrec_eval.log   # 评估日志
```

## 数据说明

本项目使用 MovieLens 1M 的三份原始文件：

- `ratings.dat`：用户-电影评分与时间戳
- `users.dat`：用户属性信息
- `movies.dat`：电影标题、年份、类型

### 数据处理流程

1. 读取 `ratings.dat`，得到每个用户的交互记录。
2. 按时间戳排序，构造用户的观影序列。
3. 对电影 ID 做连续编码，`0` 保留给 padding。
4. 生成训练样本：
   - `item_seq`：历史序列
   - `pos_item`：下一个真实点击/观看物品
   - `neg_items`：负样本
5. 生成验证样本：每个用户保留最后一步做评估。

## 模型说明

`Net.py` 中实现了一个简化版 SASRec，包括：

- `Embedding_layer`：物品嵌入层
- `Self_Attention`：自注意力模块，包含因果 mask 和 padding mask
- `FFN`：前馈网络
- `Sasrec_Net`：整体推荐网络

模型输入为用户历史序列和 `mask`，输出当前序列表示，并与正负样本做匹配打分。

## 运行环境

建议使用 Python 3.11+，并安装以下依赖：

- `numpy`
- `torch`
- `scikit-learn`
- `tqdm`

安装命令：

```bash
pip install numpy torch scikit-learn tqdm
```

## 运行方式

请在 `sasrec` 目录下运行：

```bash
python main.py
```

注意：`data.py` 中使用了相对路径 `./ml-1m/*.dat`，因此必须从 `sasrec` 目录启动，否则会找不到数据文件。

## 训练与评估

### 当前 `main.py` 的行为

- 会读取数据并构造样本
- 如果存在 `sasrec_model.pth`，会优先加载已有模型
- 最后进行一次评估，并将结果写入 `sasrec_eval.log`

### 如果开启训练

`main.py` 中的训练循环目前是注释状态。你可以取消注释训练部分后：

- 使用 `BCEWithLogitsLoss` 风格的二分类目标
- 每隔若干 epoch 做一次验证
- 保存模型到 `sasrec_model.pth`

## 评估指标

当前评估输出：

- `Recall@K`
- `NDCG@K`

默认 `K=10`。

## 运行结果文件

- `sasrec_model.pth`：模型参数
- `sasrec_eval.log`：评估日志

## 常见注意事项

1. **路径问题**：请始终在 `sasrec` 目录下运行脚本。
2. **模型加载**：跨设备加载时建议保留 `map_location=device`。
3. **padding mask**：推荐确保序列 mask 基于 `seq != 0` 生成，避免把 padding 当成真实历史。
4. **测试集候选集**：评估时只在“目标物品 + 负样本”中排序，属于标准离线评测写法。

## 结果记录

训练/评估日志会通过 `log_message()` 输出到控制台并追加写入 `sasrec_eval.log`。

## 未来可改进的方向

- 将训练流程从脚本式改成可配置参数模式
- 支持命令行传参，例如 `max_len`、`lr`、`neg_sample_num`
- 添加更完整的评估脚本与结果可视化
- 将数据预处理与训练流程拆分为独立模块

## 备注

本项目基于 MovieLens 1M 数据集进行序列推荐实验，适合用于学习 SASRec 的数据构造、mask 处理、负采样和离线评估流程。

