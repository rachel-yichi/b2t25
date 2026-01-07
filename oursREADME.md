# Brain-to-Text Pipeline（RNN → T5 → LLM）

面向语音神经解码的全流程方案：先用 RNN 进行神经→音素的 CTC 解码，再用音素编码器接预训练 T5 完成音素→文本，最后可选用外部 LLM 做轻量校对。

## 数据
- 神经数据放于 `data/hdf5_data_final/<session>/data_{train,val,test}.hdf5`，每 trial 含 512 维神经特征（20 ms）、音素序列与文本标签。
- 预训练 RNN 基线权重可置于 `data/t15_pretrained_rnn_baseline`，含 `checkpoint/best_checkpoint` 与 `args.yaml`。

## 目录总览
- `model_training/`: 训练与推理主体代码（RNN、P2T、推理、LLM 校对）。
- `language_model/`: 可选 n-gram/OPT 语言模型（本方案默认不依赖）。
- `hiend_pipeline/`: 分阶段脚本与文档。
- `data/`: 数据与预训练权重放置目录。

## 方法流程
1. **神经→音素（RNN + CTC）**  
   - 日特定线性层对齐不同 session 的分布，5 层 GRU 输出 41 类（含 blank）音素 logits。  
   - CTC 损失，训练时做噪声、截断和平滑增强。输出最佳 checkpoint 供后续使用。

2. **音素→文本（P2T：音素编码器 + T5）**  
   - 将离散音素 ID 输入音素编码器（Embedding + 小型残差 MLP，可选重参数化），映射到 T5 隐空间。  
   - 预训练 T5 提供强文本先验；常用冻结策略为“冻结 decoder/LM 头，保留 shared embedding 可训”，减少过拟合和计算。  
   - 训练目标：音素编码输出作为 T5 encoder 输入，T5 decoder 生成文本子词，使用交叉熵损失（多卡时对各卡 loss 取均值）。

3. **推理与 n-best**  
   - RNN logits 先做 CTC 前缀 beam（或贪心）得到音素候选，去重复/去 blank。  
   - 每条音素候选经音素编码器 + T5 解码（beam）；可将 CTC logprob 与 T5 序列分数相加选择最佳。  
   - 主输出为单一文本，可选保存 n-best（包含 ctc/t5/总分）以便后处理。

4. **可选 LLM 轻量校对**  
   - 将 P2T 文本送入外部 Chat Completion 接口，提示语强调“仅修明显错字/重复，不确定则原样返回”。  
   - 支持多 API 并行调用并做统一归一化（小写、去标点、清理空格）。在严格 WER 场景下建议先小规模验证增益。

## 资源与建议
- 显存：T5-base 及以上需较大显存；可用 t5-small、缩短序列、减小 batch 或半精度降低占用。全冻结 T5 不能显著减少正向计算。  
- 冻结策略：推荐仅冻结 decoder（或 encoder），保留 shared/音素编码器可训；全冻结通常会显著降性能。  
- 评估：若目标是低 WER，先在验证集比较 “仅 P2T” vs “P2T+LLM” 再决定是否启用校对。  
- 依赖：推荐使用仓库提供的 conda 环境脚本 `setup.sh`（训练/推理）和可选 `setup_lm.sh`（外部 n-gram/OPT）。  
- 数据完整性：确保 `data/hdf5_data_final` 下各 session 均有对应 split 文件；缺失会导致阶段 2/推理跳过该 session。  
