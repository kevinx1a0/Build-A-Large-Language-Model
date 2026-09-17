import json
import os
import re
from collections import defaultdict
from tabnanny import verbose
from tqdm import tqdm


class SimpleBPE:
    # merges 合并所用的新的字典，如 {(192, 48) : 256}
    # vocab 词表，如{153：b'x'}
    def __init__(self):
        self.merges = {}
        self.vocab = {i:bytes([i]) for i in range(256)}

    # 输入tokens：Byte序列，输出stats：就是汇总后的字节对，如 (32, 119) : 4076
    def get_stats(self,tokens):
        stats = defaultdict(int)

        for i in range(len(tokens)-1):
            pair = (tokens[i], tokens[i+1])
            stats[pair] += 1
        return stats

    # 找最大的汇总后的字节对，返回这个字节对
    def get_best_pair(self,stats):
        return max(stats, key= stats.get)

    # 合并函数
    # 输入字节序列，然后通过扫描用合并后的词代替原词，输出合并了一轮后的tokens
    # 要传入最优先的字节对与此字节对的identifier
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

    # 训练函数：输入：原始的训练语料，输出：迭代固定次数后的Byte序列
    # num_merges:训练次数
    # Step 1: 先读训练语料
    # Step 2: 将所读语料全部转为Byte
    # Step 3: 一轮一轮迭代
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

            self.merges[best_pair] = new_id

            tokens = self.merge(tokens, best_pair, new_id)

            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            if verbose:
             tqdm.write(f"第 {i + 1} 次合并: 最高频对 {best_pair} -> 新 ID {new_id} (对应字符: {self.vocab[new_id]})")

        return tokens

    # 解码：就是把byte翻译成字，输入一个byte列表，输出正常文本
    def decode(self, ids):
        part_bytes = [self.vocab[idx] for idx in ids]
        tokens_bytes = b"".join(part_bytes)
        return tokens_bytes.decode("utf-8", errors="ignore")

    # 编码：文本转字符，
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

    def save(self, model_prefix:str):
        merges_file = f"{model_prefix}.model"
        with open(merges_file, "w", encoding="utf-8") as f:
            f.write("#version: 1.0\n")
            for(p0, p1), idx in self.merges.items():
                f.write(f"{p0} {p1}\n")

        vocab_file = f"{model_prefix}.json"
        string_vocab = {idx: list(b) if isinstance(b, bytes) else b for idx, b in self.vocab.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(string_vocab, f, ensure_ascii=False, indent=2)

        print(f"模型已成功保存至: {vocab_file} 和 {merges_file}")

    def load(self, model_prefix: str):
        vocab_file = f"{model_prefix}.json"
        merges_file = f"{model_prefix}.model"

        if not os.path.exists(vocab_file) or not os.path.exists(merges_file):
            raise FileNotFoundError(f"找不到指定的模型文件：{model_prefix}")

        self.merges = {}
        with open(merges_file, "r", encoding="utf-8") as f:
            idx = 256
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p0, p1 = map(int, line.split())
                self.merges[(p0, p1)] = idx
                idx += 1

        with open(vocab_file, "r", encoding="utf-8") as f:
            loaded_vocab = json.load(f)

            self.vocab = {
                int(k) : bytes(v) if isinstance(v, list) else v
                for k, v in loaded_vocab.items()
            }
        print(f"已成功加载模型: {model_prefix}")

if __name__ == "__main__":
    bpe = SimpleBPE()

    # tokens = bpe.train("train.txt", 1000)

    # bpe.save("bpe_model")
    bpe.load("bpe_model")

    test_text ="HELLO, BPE tokenizer, 这是一个测试,gugugaga,awrkjwnefskdasdfgdfgrete3453^^$%@#$"
    raw_bytes = test_text.encode("utf-8")
    encode_bytes = bpe.encode(test_text)
    decode_afternode = bpe.decode(encode_bytes)

    print("压缩率：", len(raw_bytes)/len(encode_bytes))
    print(encode_bytes)
    print(decode_afternode)

    # test_text = "HELLO, BPE tokenizer, 这是一个测试,gugugaga,awrkjwnefskdasdfgdfgrete3453^^$%@#$"
    # encode_ids = bpe.encode(test_text)
    # decode_text = bpe.decode(encode_ids)
    #
    # raw_bytes = test_text.encode("utf-8")
    # num_bytes = len(raw_bytes)
    #
    # num_tokens = len(encode_ids)
    #
    # compression_rate = num_bytes / num_tokens
    # print(compression_rate)
    # assert test_text == decode_afternode, "解析文本不一致"
    # stats = bpe.get_stats(tokens)
    # max = bpe.get_best_pair(stats)
    # print(max)
    # print(bpe.decode(tokens))

