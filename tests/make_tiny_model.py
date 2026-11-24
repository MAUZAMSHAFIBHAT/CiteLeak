"""Build a tiny RANDOM-WEIGHT chat model for offline smoke tests (no download).

Its answers are gibberish; it only exists to exercise the full code path
(chat template -> generate -> batched ablation log-probs -> analysis).
Never use its numbers as results.
"""
import sys
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from citeleak.data import build_world  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "tests/tiny-model")
w = build_world(n_background=300, n_targets=20)
corpus = [d["text"] for d in w["docs"]]
tk = Tokenizer(models.BPE(unk_token="<unk>"))
tk.pre_tokenizer = pre_tokenizers.ByteLevel()
tk.train_from_iterator(corpus * 3, trainers.BpeTrainer(
    vocab_size=2000, special_tokens=["<unk>", "<pad>", "<|im_start|>", "<|im_end|>"]))
tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>",
                              eos_token="<|im_end|>")
tok.chat_template = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
                     "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                  eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
model = Qwen2ForCausalLM(cfg)
model.generation_config.eos_token_id = tok.eos_token_id
out.mkdir(parents=True, exist_ok=True)
tok.save_pretrained(out)
model.save_pretrained(out)
print("saved tiny model to", out)
