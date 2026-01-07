"""
过渡方案：使用预训练文本模型 T5 作为解码器/语言先验，前面接一个音素嵌入层作为 encoder 输入。
流程：音素序列 -> 音素嵌入 -> T5 encoder -> T5 decoder 生成文本（BPE）。
不依赖 WFST/Redis，利用 T5 的文本先验，音素侧从头训练嵌入。

示例：
python model_training/hiend_pipeline/stage2_phoneme2text_t5.py \
  --model_path /root/local-nvme/nejm-brain-to-text/data/t15_pretrained_rnn_baseline \
  --data_dir /root/local-nvme/nejm-brain-to-text/data/hdf5_data_final \
  --csv_path /root/local-nvme/nejm-brain-to-text/data/t15_copyTaskData_description.csv \
  --save_dir /root/local-nvme/nejm-brain-to-text/model_training/trained_models/p2t_t5 \
  --gpus 0 --epochs 3 --batch_size 16 --max_src_len 400 --max_tgt_len 128 \
  --t5_model t5-small --lr 3e-4
"""

import argparse
import os
import sys
import pathlib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_model_helpers import (
    load_h5py_file,
    LOGIT_TO_PHONEME,
    runSingleDecodingStep,
    rearrange_speech_logits_pt,
    remove_punctuation,
)
from transformers import T5ForConditionalGeneration, T5Tokenizer
import numpy as np

PHONEME_TO_ID = {p: i for i, p in enumerate(LOGIT_TO_PHONEME)}
BLANK_ID = 0


def normalize_text(txt: str) -> str:
    """
    对句子做轻量归一化：解码 bytes、去标点、多余空格、小写。
    与评测保持一致，避免训练/推理文本分布不齐。
    """
    if isinstance(txt, (bytes, bytearray)):
        txt = txt.decode()
    txt = str(txt)
    txt = remove_punctuation(txt)
    return txt


class PhonemeTextDataset(Dataset):
    def __init__(self, data, tokenizer, max_src_len=400, max_tgt_len=128, use_rnn_pred=False, rnn=None, model_args=None, device=None):
        self.phonemes = []
        self.text = []
        for sess, d in data.items():
            session_idx = None
            if model_args is not None:
                session_idx = model_args["dataset"]["sessions"].index(sess)
            for i in range(len(d["seq_class_ids"])):
                if use_rnn_pred and rnn is not None and model_args is not None:
                    neural = np.expand_dims(d["neural_features"][i], axis=0)
                    neural_tensor = torch.tensor(neural, device=device, dtype=torch.bfloat16)
                    logits = runSingleDecodingStep(neural_tensor, session_idx, rnn, model_args, device)
                    logits = rearrange_speech_logits_pt(logits)[0]
                    pred_seq = np.argmax(logits, axis=-1)
                    pred_seq = [int(p) for p in pred_seq if p != 0]
                    pred_seq = [pred_seq[j] for j in range(len(pred_seq)) if j == 0 or pred_seq[j] != pred_seq[j - 1]]
                    ph = np.array(pred_seq, dtype=np.int64)
                else:
                    ph = d["seq_class_ids"][i][: d["seq_len"][i]]
                txt = normalize_text(d["sentence_label"][i])
                if len(ph) == 0 or len(txt.strip()) == 0:
                    continue
                self.phonemes.append(np.array(ph, dtype=np.int64)[:max_src_len])
                self.text.append(txt.strip())
        self.tokenizer = tokenizer
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.phonemes)

    def __getitem__(self, idx):
        ph = self.phonemes[idx]
        tgt = self.tokenizer(
            self.text[idx],
            max_length=self.max_tgt_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        return torch.tensor(ph, dtype=torch.long), tgt


def collate_fn(batch, pad_id):
    src_seqs, tgt_seqs = zip(*batch)
    src_lens = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)
    max_src = max(src_lens).item()
    src_pad = torch.full((len(batch), max_src), BLANK_ID, dtype=torch.long)
    for i, s in enumerate(src_seqs):
        src_pad[i, : len(s)] = s
    tgt_pad = torch.stack(tgt_seqs)
    return src_pad, src_lens, tgt_pad


class PhonemeEncoder(nn.Module):
    """
    音素嵌入 + 残差 MLP，可选重参数化噪声（mu + sigma * eps）来正则化。
    """

    def __init__(self, vocab_size, d_model, dropout=0.1, hidden_mult=3, reparam=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=BLANK_ID)
        hidden_dim = int(d_model * hidden_mult)
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model)
        )
        self.reparam = reparam
        if self.reparam:
            self.logvar = nn.Linear(d_model, d_model)

    def forward(self, src):
        x = self.embed(src)
        h = x + self.proj(x)
        if self.reparam and self.training:
            logvar = self.logvar(h)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            h = h + eps * std
        return h


def main():
    parser = argparse.ArgumentParser(description="Phoneme -> text using T5 decoder")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="trained_models/p2t_t5")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU ids; multiple will use DataParallel")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_src_len", type=int, default=400)
    parser.add_argument("--max_tgt_len", type=int, default=128)
    parser.add_argument("--t5_model", type=str, default="t5-base")
    parser.add_argument("--cache_dir", type=str, default="/root/local-nvme/hf_cache")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--freeze_decoder", action="store_true", help="Freeze T5 decoder/LM head")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze T5 encoder layers (keep shared embed trainable)")
    parser.add_argument("--freeze_all_t5", action="store_true", help="Freeze encoder+decoder+LM head; only train phoneme encoder and shared embedding")
    parser.add_argument("--use_rnn_pred", action="store_true", help="Use RNN greedy phoneme predictions instead of ground-truth phonemes")
    parser.add_argument("--reparam", type=bool, default=False, help="Enable reparameterization noise in phoneme encoder to regularize")
    args = parser.parse_args()

    gpu_ids = [int(x) for x in str(args.gpus).split(",") if x != ""]
    use_cuda = torch.cuda.is_available() and len(gpu_ids) > 0 and gpu_ids[0] >= 0
    device = torch.device(f"cuda:{gpu_ids[0]}") if use_cuda else torch.device("cpu")

    # tokenizer and T5
    tokenizer = T5Tokenizer.from_pretrained(args.t5_model, cache_dir=args.cache_dir)
    model_t5 = T5ForConditionalGeneration.from_pretrained(args.t5_model, cache_dir=args.cache_dir)
    if args.freeze_all_t5:
        for p in model_t5.encoder.parameters():
            p.requires_grad = False
        for p in model_t5.decoder.parameters():
            p.requires_grad = False
        for p in model_t5.lm_head.parameters():
            p.requires_grad = False
        for p in model_t5.shared.parameters():
            p.requires_grad = True  # allow shared embed to adapt to phoneme encoder
    else:
        if args.freeze_decoder:
            for p in model_t5.decoder.parameters():
                p.requires_grad = False
            for p in model_t5.lm_head.parameters():
                p.requires_grad = False
            for p in model_t5.shared.parameters():
                p.requires_grad = True  # shared embedding used by encoder/decoder; keep trainable for phoneme encoder alignment
        if args.freeze_encoder:
            for p in model_t5.encoder.parameters():
                p.requires_grad = False
            # keep shared embedding trainable for alignment
            for p in model_t5.shared.parameters():
                p.requires_grad = True

    phoneme_encoder = PhonemeEncoder(
        len(LOGIT_TO_PHONEME),
        model_t5.config.d_model,
        dropout=args.dropout,
        hidden_mult=2,
        reparam=args.reparam,
    )

    model_t5 = model_t5.to(device)
    phoneme_encoder = phoneme_encoder.to(device)
    if use_cuda and len(gpu_ids) > 1:
        model_t5 = nn.DataParallel(model_t5, device_ids=gpu_ids)
        phoneme_encoder = nn.DataParallel(phoneme_encoder, device_ids=gpu_ids)

    # optional RNN for predicted phonemes
    rnn = None
    model_args = OmegaConf.load(os.path.join(args.model_path, "checkpoint", "args.yaml"))
    if args.use_rnn_pred:
        from rnn_model import GRUDecoder  # local import to avoid cycle
        rnn = GRUDecoder(
            neural_dim=model_args["model"]["n_input_features"],
            n_units=model_args["model"]["n_units"],
            n_days=len(model_args["dataset"]["sessions"]),
            n_classes=model_args["dataset"]["n_classes"],
            rnn_dropout=model_args["model"]["rnn_dropout"],
            input_dropout=model_args["model"]["input_network"]["input_layer_dropout"],
            n_layers=model_args["model"]["n_layers"],
            patch_size=model_args["model"]["patch_size"],
            patch_stride=model_args["model"]["patch_stride"],
        )
        ckpt = torch.load(os.path.join(args.model_path, "checkpoint", "best_checkpoint"), map_location=device, weights_only=False)
        sd_clean = {}
        for k, v in ckpt["model_state_dict"].items():
            nk = k.replace("module.", "").replace("_orig_mod.", "")
            sd_clean[nk] = v
        rnn.load_state_dict(sd_clean, strict=False)
        rnn.to(device).eval()

    # load data
    b2txt_csv_df = pd.read_csv(args.csv_path)
    train_data = {}
    val_data = {}
    for session in model_args["dataset"]["sessions"]:
        train_file = os.path.join(args.data_dir, session, "data_train.hdf5")
        val_file = os.path.join(args.data_dir, session, "data_val.hdf5")
        if os.path.exists(train_file):
            train_data[session] = load_h5py_file(train_file, b2txt_csv_df)
        if os.path.exists(val_file):
            val_data[session] = load_h5py_file(val_file, b2txt_csv_df)

    train_ds = PhonemeTextDataset(
        train_data,
        tokenizer,
        args.max_src_len,
        args.max_tgt_len,
        use_rnn_pred=args.use_rnn_pred,
        rnn=rnn,
        model_args=model_args,
        device=device,
    )
    val_ds = PhonemeTextDataset(
        val_data,
        tokenizer,
        args.max_src_len,
        args.max_tgt_len,
        use_rnn_pred=args.use_rnn_pred,
        rnn=rnn,
        model_args=model_args,
        device=device,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    params = list(phoneme_encoder.parameters()) + [p for p in model_t5.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_path = None

    for epoch in range(args.epochs):
        model_t5.train()
        phoneme_encoder.train()
        pbar = tqdm(train_loader, desc=f"Train epoch {epoch}")
        total_loss = 0
        for src, src_lens, tgt in pbar:
            src = src.to(device)
            tgt = tgt.to(device)
            attention_mask = (src != BLANK_ID).long()
            inputs_embeds = phoneme_encoder(src)
            outputs = model_t5(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=tgt,
            )
            # label smoothing handled inside transformer loss if supported; otherwise manual CE could be used
            loss = outputs.loss
            if loss.dim() > 0:
                loss = loss.mean()  # DataParallel may return per-device losses
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        avg_train = total_loss / len(train_loader)
        print(f"Epoch {epoch} train loss {avg_train:.4f}")

        model_t5.eval()
        phoneme_encoder.eval()
        val_loss = 0
        with torch.no_grad():
            for src, src_lens, tgt in val_loader:
                src = src.to(device)
                tgt = tgt.to(device)
                attention_mask = (src != BLANK_ID).long()
                inputs_embeds = phoneme_encoder(src)
                outputs = model_t5(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=tgt,
                )
                loss_val = outputs.loss
                if loss_val.dim() > 0:
                    loss_val = loss_val.mean()
                val_loss += loss_val.item()
        val_loss /= max(1, len(val_loader))
        print(f"Epoch {epoch} val loss {val_loss:.4f}, ppl ~{np.exp(val_loss):.2f}")

        # save best
        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, "p2t_t5.pt")
            # unwrap DataParallel for saving
            save_t5 = model_t5.module if isinstance(model_t5, nn.DataParallel) else model_t5
            save_encoder = phoneme_encoder.module if isinstance(phoneme_encoder, nn.DataParallel) else phoneme_encoder
            torch.save(
                {
                    "phoneme_encoder": save_encoder.state_dict(),
                    "t5_model": save_t5.state_dict(),
                    "tokenizer": args.t5_model,
                    "val_loss": best_val,
                },
                save_path,
            )
            best_path = save_path
            print(f"Saved best to {save_path} (val loss {best_val:.4f})")

    if best_path is None:
        print("No model saved (no val data?)")
    else:
        print(f"Best checkpoint: {best_path}, val loss {best_val:.4f}")


if __name__ == "__main__":
    main()
