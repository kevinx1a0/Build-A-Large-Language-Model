import os
import re
from collections import defaultdict
from tabnanny import verbose
from tqdm import tqdm

class SimpleBPE:
    def __init__(self):
        self.merges = {}
        self.vocab = {i:bytes([i]) for i in range(256)}

    def train(self, text_path, num_merges):
        # -----------  Step 1: 读数据 ------------
        print("1. 正在读取训练语料")
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
            context = f.read()
            print(f"语料加载完成，字符总数：{len(text)}")

        # -----------  Step 2: 字节级初始化词表 ------------
        print("2. 正在进行字节级（Byte-level）初始化")
        tokens = list(text.encode("utf-8"))

        # -----------  Step 3: BPE 训练循环 ---------------
        print("3. 开始迭代合并最高频字节对")
        for i in tqdm(range(num_merges),desc="Training BPE"):
            stats = self.get_stats(tokens)
            if not stats:
                break

            best_pair = self.get_best_pair(stats)
            new_id = 256 + i

            tokens = self.merge(tokens, best_pair, new_id)

            self.merges[best_pair] = new_id

            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            if verbose:
             tqdm.write(f"第 {i + 1} 次合并: 最高频对 {best_pair} -> 新 ID {new_id} (对应字符: {self.vocab[new_id]})")

        return tokens


    def get_stats(self,tokens):
        stats = defaultdict(int)

        for i in range(len(tokens)-1):
            pair = (tokens[i], tokens[i+1])
            stats[pair] += 1
        return stats

    def get_best_pair(self,stats):
        return max(stats, key= stats.get)

    def merge(self,tokens, best_pair, new_id):
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def decode(self, ids):
        part_bytes = [self.vocab[idx] for idx in ids]
        tokens_bytes = b"".join(part_bytes)
        return tokens_bytes.decode("utf-8", errors="ignore")

    def encode(self, text):
        tokens = list(text.encode("utf-8"))

        if len(tokens) < 2:
            return tokens
        while True:
            stats = self.get_stats(tokens)
            if not stats:
                break
            pair_to_merge = None
            min_id = float("inf")

            for pair in stats:
                if pair in self.merges:
                    if self.merges[pair] < min_id:
                        min_id = self.merges[pair]
                        pair_to_merge = pair

            if pair_to_merge is None:
                break

            new_id = self.merges[pair_to_merge]
            tokens = self.merge(tokens, pair_to_merge, new_id)
        return tokens

if __name__ == "__main__":
    bpe = SimpleBPE()
    tokens = bpe.train("train.txt", 1000)

    test_text = "HELLO, BPE tokenizer, 这是一个测试,gugugaga,awrkjwnefskdasdfgdfgrete3453^^$%@#$"
    encode_ids = bpe.encode(test_text)
    decode_text = bpe.decode(encode_ids)

    raw_bytes = test_text.encode("utf-8")
    num_bytes = len(raw_bytes)

    num_tokens = len(encode_ids)

    compression_rate = num_bytes / num_tokens
    print(compression_rate)
    assert test_text == decode_text, "解析文本不一致"
    # stats = bpe.get_stats(tokens)
    # max = bpe.get_best_pair(stats)
    # print(max)
    # print(bpe.decode(tokens))

