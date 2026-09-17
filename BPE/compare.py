import time
from transformers import AutoTokenizer
import tiktoken
from mybpe import  SimpleBPE

my_tokenizer = SimpleBPE()
my_tokenizer.load("bpe_model")

gpt2_tokenizer = tiktoken.get_encoding("gpt2")

qwen2_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

test_text = """
The Transformer model architecture has revolutionized Natural Language Processing.
手动实现一个 Byte-level BPE 分词器是理解大语言模型（LLM）的第一步！
def self_attention(Q, K, V):
    scores = Q @ K.T / (d_k ** 0.5)
    return softmax(scores) @ V
"""

print(f"=== 原始文本长度： {len(test_text.encode('utf-8'))} 字节 ===")

t0 = time.time()
my_tokens = my_tokenizer.encode(test_text)
my_time = (time.time() - t0) * 1000
my_compression_rate = len(test_text.encode('utf-8'))/len(my_tokens)
print(f"""SimpleBPE
词表大小：{len(my_tokenizer.vocab)}
编码后的Tokens:{len(my_tokens)}
耗时：{my_time:.2f} ms
压缩比：{my_compression_rate:.2f}""")

print("================================")
t1 = time.time()
gpt2_tokens = gpt2_tokenizer.encode(test_text)
gpt2_time = (time.time() - t1) * 1000
gpt2_compression_rate = len(test_text.encode('utf-8'))/len(gpt2_tokens)
print(f"""GPT2
词表大小：{gpt2_tokenizer.n_vocab}
编码后的Tokens:{len(gpt2_tokens)}
耗时：{gpt2_time:.2f} ms
压缩比：{gpt2_compression_rate:.2f}""")


print("================================")
t2 = time.time()
qwen2_tokens = qwen2_tokenizer.encode(test_text)
qwen2_time = (time.time() - t2) * 1000
qwen2_compression_rate = len(test_text.encode('utf-8'))/len(qwen2_tokens)
print(f"""Qwen2
词表大小：{len(qwen2_tokenizer.vocab)}
编码后的Tokens:{len(qwen2_tokens)}
耗时：{qwen2_time:.2f} ms
压缩比：{qwen2_compression_rate:.2f}""")