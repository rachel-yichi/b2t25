## Brain-to-Text 分阶段训练 & 推理流程（RNN → T5 P2T → LLM 清理）

下述内容聚焦原理与流程，假设在仓库根目录 `/root/local-nvme/nejm-brain-to-text` 工作，路径可用相对路径。

### 数据与流向
- 神经 trial：`input_features`（T×512，20 ms 采样）、音素标签、文本标签，按 session 划分 train/val/test HDF5。
- 处理链：神经 → RNN 得音素 logits/序列 → P2T（音素编码器+T5）生成文本 → 可选 LLM 轻量校对 → 提交文本。

### 阶段 1：RNN 音素解码（神经→音素）
- 结构：日特定线性层对齐分布 → 5 层单向 GRU → 41 类 CTC 头（含 blank）。
- 训练：CTC 损失，cosine/linear LR，数据增强（噪声、截断、高斯平滑）。输出最佳 checkpoint 和 `args.yaml` 供后续加载。

### 阶段 2：音素→文本（P2T）（选取三个版本，small，base，large）
- 输入：音素序列（训练可用真值或 RNN 预测；推理用 RNN CTC 贪心/beam 去重复 blank 后的 ID 序列）。
- 音素编码器：Embedding + 小型 MLP 残差，将离散音素映射到与 T5 隐空间同维；参数量小、易训练。
- T5 文本解码器：利用预训练语言先验，用冻结策略“冻结 decoder/LM 头，保留 shared embedding 可训”，减少过拟合与训练算力；
- 训练目标：音素编码输出作为 T5 encoder 输入，T5 decoder 生成文本子词，交叉熵损失（DataParallel 时对 per-device loss 取均值再反传）。输出 checkpoint 含音素编码器、T5 权重、tokenizer、验证损失。

### 阶段 3：推理（RNN + P2T）
- CTC 解码：对 RNN logits 做前缀 beam，得到音素候选，去重复/去 blank。
- 文本生成：音素候选 → 音素编码器 → T5 解码（beam）。将 CTC logprob 与 T5 序列分数相加，取最佳；添加 n-best，保留若干候选及分数（ctc/t5/total）。
- 输出：主 CSV（id,text），

### 阶段 4：可选 LLM 轻量校对（ablation study）
- 目的：对阶段 3 文本做极保守修正（错字/重复），不改词序、不增删信息。
- 提示：强调“仅修显著错字/重复，不确定则原样返回”，降低 WER 回退风险。
- 归一化：对 LLM 输出统一小写、去标点、清理多余空格，保持与评测一致。

