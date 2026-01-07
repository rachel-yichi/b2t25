"""
阶段 2 推理脚本：使用训练好的 RNN + 音素嵌入 + 预训练 T5（已在 stage2_phoneme2text_t5.py 微调），对指定 split 生成提交 CSV。
无需 WFST/Redis。

示例（请替换路径）：
python model_training/hiend_pipeline/stage2_phoneme2text_t5_infer.py \
  --rnn_model_path /root/local-nvme/nejm-brain-to-text/model_training/trained_models/max_rnn_run1 \
  --p2t_ckpt /root/local-nvme/nejm-brain-to-text/model_training/trained_models/p2t_t5/p2t_t5.pt \
  --data_dir /root/local-nvme/nejm-brain-to-text/data/hdf5_data_final \
  --csv_path /root/local-nvme/nejm-brain-to-text/data/t15_copyTaskData_description.csv \
  --eval_type test --gpu 0 --max_gen_len 64 --beam_size 4
"""

import argparse
import os
import sys
import pathlib
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from transformers import T5ForConditionalGeneration, T5Tokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rnn_model import GRUDecoder
from evaluate_model_helpers import (
    load_h5py_file,
    runSingleDecodingStep,
    rearrange_speech_logits_pt,
    LOGIT_TO_PHONEME,
    remove_punctuation,
)

BLANK_ID = 0


def normalize_text(txt: str) -> str:
    """
    推理输出/标签归一化：解码 bytes、去标点、多余空格、小写，和训练/评测一致。
    """
    if isinstance(txt, (bytes, bytearray)):
        txt = txt.decode()
    txt = str(txt)
    return remove_punctuation(txt)


def ctc_prefix_beam_search(log_probs, beam_width=10, topk=1, blank_id=0):
    """
    轻量 CTC 前缀 beam search，返回 topk (score, prefix)。
    log_probs: [T, V] (np.float64)
    """
    T, V = log_probs.shape
    beams = {(): (0.0, -np.inf)}  # prefix -> (p_blank, p_non_blank)
    for t in range(T):
        next_beams = {}
        for prefix, (pb, pnb) in beams.items():
            for v in range(V):
                p = log_probs[t, v]
                if v == blank_id:
                    nb_pb, nb_pnb = next_beams.get(prefix, (-np.inf, -np.inf))
                    nb_pb = np.logaddexp(nb_pb, pb + p)
                    nb_pb = np.logaddexp(nb_pb, pnb + p)
                    next_beams[prefix] = (nb_pb, nb_pnb)
                else:
                    last = prefix[-1] if len(prefix) > 0 else None
                    new_prefix = prefix + (v,)
                    nb_pb, nb_pnb = next_beams.get(new_prefix, (-np.inf, -np.inf))
                    if v == last:
                        nb_pnb = np.logaddexp(nb_pnb, pb + p)
                        nb_pnb = np.logaddexp(nb_pnb, pnb + p)
                        next_beams[new_prefix] = (nb_pb, nb_pnb)
                        nb_pb2, nb_pnb2 = next_beams.get(prefix, (-np.inf, -np.inf))
                        nb_pnb2 = np.logaddexp(nb_pnb2, pnb + p)
                        next_beams[prefix] = (nb_pb2, nb_pnb2)
                    else:
                        nb_pnb = np.logaddexp(nb_pnb, pb + p)
                        nb_pnb = np.logaddexp(nb_pnb, pnb + p)
                        next_beams[new_prefix] = (nb_pb, nb_pnb)
        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]),
                reverse=True,
            )[:beam_width]
        )
    scored = []
    for prefix, (pb, pnb) in beams.items():
        scored.append((np.logaddexp(pb, pnb), prefix))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:topk]


class PhonemeEncoder(torch.nn.Module):
    def __init__(self, vocab_size, d_model, dropout=0.1, hidden_mult=2, reparam=False):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, d_model, padding_idx=BLANK_ID)
        hidden_dim = int(d_model * hidden_mult)
        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, d_model)
        )
        self.reparam = reparam
        if self.reparam:
            self.logvar = torch.nn.Linear(d_model, d_model)

    def forward(self, src):
        x = self.embed(src)
        h = x + self.proj(x)
        if self.reparam and self.training:
            logvar = self.logvar(h)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            h = h + eps * std
        return h


def load_rnn(rnn_model_path, device):
    args_path = os.path.join(rnn_model_path, "checkpoint", "args.yaml")
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"args.yaml not found at {args_path}")
    args = OmegaConf.load(args_path)
    model = GRUDecoder(
        neural_dim=args["model"]["n_input_features"],
        n_units=args["model"]["n_units"],
        n_days=len(args["dataset"]["sessions"]),
        n_classes=args["dataset"]["n_classes"],
        rnn_dropout=args["model"]["rnn_dropout"],
        input_dropout=args["model"]["input_network"]["input_layer_dropout"],
        n_layers=args["model"]["n_layers"],
        patch_size=args["model"]["patch_size"],
        patch_stride=args["model"]["patch_stride"],
    )
    ckpt = torch.load(os.path.join(rnn_model_path, "checkpoint", "best_checkpoint"), map_location=device, weights_only=False)
    sd_clean = {}
    for k, v in ckpt["model_state_dict"].items():
        nk = k.replace("module.", "").replace("_orig_mod.", "")
        sd_clean[nk] = v
    model.load_state_dict(sd_clean, strict=False)
    model.to(device).eval()
    return model, args


def load_p2t(p2t_ckpt, device, cache_dir=None):
    ckpt = torch.load(p2t_ckpt, map_location=device)
    tokenizer_name = ckpt.get("tokenizer", "t5-base")
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    t5 = T5ForConditionalGeneration.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    t5.load_state_dict(ckpt["t5_model"])
    reparam = any("logvar" in k for k in ckpt["phoneme_encoder"].keys())
    phoneme_encoder = PhonemeEncoder(
        len(LOGIT_TO_PHONEME),
        t5.config.d_model,
        dropout=0.1,
        hidden_mult=2,
        reparam=reparam,
    )
    phoneme_encoder.load_state_dict(ckpt["phoneme_encoder"])
    t5.to(device).eval()
    phoneme_encoder.to(device).eval()
    return phoneme_encoder, t5, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Phoneme->Text T5 inference")
    parser.add_argument("--rnn_model_path", type=str, required=True)
    parser.add_argument("--p2t_ckpt", type=str, required=True, help="stage2_phoneme2text_t5.py 输出的 p2t_t5.pt")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--eval_type", type=str, choices=["val", "test"], default="test")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default=None, help="HF cache dir for tokenizer/model")
    parser.add_argument("--max_gen_len", type=int, default=64)
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--nbest_size", type=int, default=1, help="输出前N条候选（结合CTC与T5分数）；1表示只输出最佳")
    parser.add_argument("--nbest_output", type=str, default=None, help="nbest CSV 保存路径（默认与主输出同目录）")
    parser.add_argument("--ctc_beam_width", type=int, default=10, help="CTC前缀beam宽度（RNN音素解码）")
    parser.add_argument("--ctc_topk", type=int, default=1, help="返回前k条CTC音素候选，默认取最佳1条")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() and args.gpu >= 0 else torch.device("cpu")

    rnn, rnn_args = load_rnn(args.rnn_model_path, device)
    phoneme_encoder, t5, tokenizer = load_p2t(args.p2t_ckpt, device, cache_dir=args.cache_dir)

    b2txt_csv_df = pd.read_csv(args.csv_path)
    test_data = {}
    total = 0
    for session in rnn_args["dataset"]["sessions"]:
        h5_file = os.path.join(args.data_dir, session, f"data_{args.eval_type}.hdf5")
        if os.path.exists(h5_file):
            d = load_h5py_file(h5_file, b2txt_csv_df)
            test_data[session] = d
            total += len(d["neural_features"])
    if total == 0:
        raise RuntimeError("No eval data found; check data_dir/eval_type.")
    print(f"Total trials: {total}")

    results = {"id": [], "text": []}
    nbest_rows = [] if args.nbest_size > 1 else None
    idx = 0
    for session, data in test_data.items():
        session_idx = rnn_args["dataset"]["sessions"].index(session)
        for i in range(len(data["neural_features"])):
            neural_input = np.expand_dims(data["neural_features"][i], axis=0)
            neural_tensor = torch.tensor(neural_input, device=device, dtype=torch.bfloat16)
            logits = runSingleDecodingStep(neural_tensor, session_idx, rnn, rnn_args, device)
            logits = rearrange_speech_logits_pt(logits)[0]  # [T, V]
            # CTC prefix beam 搜索音素
            max_logits = np.max(logits, axis=-1, keepdims=True)
            log_probs = logits - max_logits - np.log(np.sum(np.exp(logits - max_logits), axis=-1, keepdims=True))
            candidates = ctc_prefix_beam_search(
                log_probs.astype(np.float64),
                beam_width=args.ctc_beam_width,
                topk=args.ctc_topk,
                blank_id=BLANK_ID,
            )
            best_text = ""
            best_score = -np.inf

            if len(candidates) == 0:
                text = ""
                if nbest_rows is not None:
                    nbest_rows.append(
                        {
                            "id": idx,
                            "rank": 1,
                            "text": "",
                            "score_ctc": -np.inf,
                            "score_t5": 0.0,
                            "score_total": -np.inf,
                        }
                    )
                best_text = ""
            else:
                for cand_score, cand_prefix in candidates:
                    pred_seq = []
                    for p in cand_prefix:
                        if p == BLANK_ID:
                            continue
                        if len(pred_seq) == 0 or p != pred_seq[-1]:
                            pred_seq.append(p)

                    if len(pred_seq) == 0:
                        continue

                    src = torch.tensor(pred_seq, device=device).unsqueeze(0)
                    attn = (src != BLANK_ID).long()
                    inputs_embeds = phoneme_encoder(src)
                    with torch.no_grad():
                        generated = t5.generate(
                            inputs_embeds=inputs_embeds,
                            attention_mask=attn,
                            max_length=args.max_gen_len,
                            num_beams=args.beam_size,
                            num_return_sequences=min(args.nbest_size, args.beam_size),
                            output_scores=True,
                            return_dict_in_generate=True,
                        )

                    for j in range(generated.sequences.size(0)):
                        t5_score = generated.sequences_scores[j].item()
                        total_score = cand_score + t5_score
                        text_candidate = tokenizer.decode(generated.sequences[j], skip_special_tokens=True)
                        text_candidate = normalize_text(text_candidate)

                        if total_score > best_score:
                            best_score = total_score
                            best_text = text_candidate

                        if nbest_rows is not None:
                            nbest_rows.append(
                                {
                                    "id": idx,
                                    "rank": None,  # rank later
                                    "text": text_candidate,
                                    "score_ctc": cand_score,
                                    "score_t5": t5_score,
                                    "score_total": total_score,
                                }
                            )

                if nbest_rows is not None:
                    # keep top nbest_size for this sample
                    sample_rows = [r for r in nbest_rows if r["id"] == idx]
                    sample_rows.sort(key=lambda r: r["score_total"], reverse=True)
                    sample_rows = sample_rows[: args.nbest_size]
                    for rank, r in enumerate(sample_rows, 1):
                        r["rank"] = rank
                    # remove older entries for this id then extend with ranked
                    nbest_rows = [r for r in nbest_rows if r["id"] != idx] + sample_rows

            text = normalize_text(best_text)
            results["id"].append(idx)
            results["text"].append(text)
            idx += 1

    out_path = os.path.join(args.rnn_model_path, f"p2t_t5_{args.eval_type}_predicted_sentences.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

    if nbest_rows is not None:
        nbest_path = args.nbest_output or os.path.join(
            args.rnn_model_path, f"p2t_t5_{args.eval_type}_nbest.csv"
        )
        pd.DataFrame(nbest_rows).to_csv(nbest_path, index=False)
        print(f"Saved n-best to {nbest_path}")


if __name__ == "__main__":
    main()
