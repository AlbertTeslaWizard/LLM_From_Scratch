import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    """使用滑动窗口从文本中创建输入/目标序列对。"""
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 将整个文本编码为 token ID 序列
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "文本 token 数必须大于 max_length"

        # 滑动窗口切分
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# ------------------------------------------------------------
# DataLoader 创建函数
# ------------------------------------------------------------
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128,
                         shuffle=True, drop_last=True, num_workers=0):
    """根据给定文本创建 DataLoader，返回批次数据。"""
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader


# ------------------------------------------------------------
# 主程序：演示数据集和 DataLoader 的使用
# ------------------------------------------------------------
if __name__ == "__main__":
    # 读取文本（只做一次）
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    tokenizer = tiktoken.get_encoding("gpt2")
    enc_text = tokenizer.encode(raw_text)
    enc_sample = enc_text[50:]   # 从第50个 token 开始取一段作为示例

    context_size = 4
    x = enc_sample[:context_size]
    y = enc_sample[1:context_size + 1]
    print(f"x: {x}")
    print(f"y: {y}")

    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desire = enc_sample[i]
        print(f"[{context}] ----> {desire}")

    for i in range(1, context_size + 1):
        context = tokenizer.decode(enc_sample[:i])
        desire = tokenizer.decode([enc_sample[i]])
        print(f"{context} ----> {desire}")

    print("\n--- stride=1, batch_size=1 ---")
    dataloader1 = create_dataloader_v1(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
    )
    data_iter = iter(dataloader1)
    first_batch = next(data_iter)
    second_batch = next(data_iter)
    print(first_batch)
    print(second_batch)

    print("\n--- stride=4, batch_size=8 ---")
    dataloader2 = create_dataloader_v1(
        raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
    )
    data_iter = iter(dataloader2)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)