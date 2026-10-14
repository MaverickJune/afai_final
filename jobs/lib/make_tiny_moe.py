"""Create a tiny random-init Qwen3-MoE HF checkpoint (same architecture family as Qwen3-30B-A3B)
plus a small SFT parquet with a `messages` column.

usage: python make_tiny_moe.py OUT_DIR TOKENIZER_DIR
"""

import shutil
import sys

import pandas as pd
import torch
from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM

out, tok_dir = sys.argv[1], sys.argv[2]
torch.manual_seed(0)
cfg = Qwen3MoeConfig(
    vocab_size=151936,
    hidden_size=512,
    intermediate_size=1536,
    moe_intermediate_size=256,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=64,
    num_experts=8,
    num_experts_per_tok=2,
    norm_topk_prob=True,
    decoder_sparse_step=1,
    mlp_only_layers=[],
    max_position_embeddings=4096,
    rope_theta=1000000.0,
    rms_norm_eps=1e-6,
    tie_word_embeddings=False,
    torch_dtype="bfloat16",
)
model = Qwen3MoeForCausalLM(cfg).to(torch.bfloat16)
model.save_pretrained(f"{out}/hf_tiny_moe", safe_serialization=True)
AutoTokenizer.from_pretrained(tok_dir).save_pretrained(f"{out}/hf_tiny_moe")
for f in ("generation_config.json",):
    shutil.copy(f"{tok_dir}/{f}", f"{out}/hf_tiny_moe/{f}")
print({k: v.shape for k, v in model.state_dict().items() if "layers.0." in k})

rows = []
for i in range(64):
    a, b = i + 3, 2 * i + 7
    rows.append(
        {
            "messages": [
                {"role": "user", "content": f"What is {a} plus {b}?"},
                {"role": "assistant", "content": f"{a} plus {b} equals {a + b}."},
            ]
        }
    )
pd.DataFrame(rows).to_parquet(f"{out}/sft_tiny.parquet")
print("wrote", len(rows), "SFT rows")
