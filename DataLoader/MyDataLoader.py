import numpy as np

class NativeDataLoader:
    # shuffle：每个epoch开始前是否打乱样本顺序？
    def __init__(self, dataset, batch_size, block_size, shuffle=True):
        self.dataset = np.array(dataset, dtype=np.int64)
        self.batch_size = batch_size
        self.block_size = block_size
        self.shuffle = shuffle

        self.indices = np.arange(len(self.dataset) - self.block_size)

        self.current_idx = 0
        self.reset()

    def reset(self):
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self.current_idx + self.batch_size > len(self.indices):
            raise StopIteration

        batch_start_indices = self.indices[self.current_idx : self.current_idx + self.batch_size]
        x = np.array([self.dataset[i : i+self.block_size] for i in batch_start_indices])
        y = np.array([self.dataset[i + 1 : i + 1 + self.block_size] for i in batch_start_indices])

        self.current_idx += self.batch_size

        return x, y

    def __len__(self):
        return len(self.indices) // self.batch_size

if __name__ == "__main__":
    tokens = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

    # 设置 参数
    batch_size = 2
    block_size = 4

    loader = NativeDataLoader(tokens, batch_size=batch_size, block_size=block_size, shuffle=False)
    for step, (x, y) in enumerate(loader):
        print(f"\n--- Batch {step + 1} ---")
        print("x (Input IDs):\n", x)
        print("y (Target IDs):\n", y)