import os
import urllib.request

# 可选模型：
# "gpt2-small-124M.pth"
# "gpt2-medium-355M.pth"
# "gpt2-large-774M.pth"
# "gpt2-xl-1558M.pth"

file_name = "gpt2-small-124M.pth"

url = f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{file_name}"

def download_file(url, save_path):
    if os.path.exists(save_path):
        size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"文件已存在，无需重复下载：{save_path} ({size_mb:.2f} MB)")
        return

    print(f"开始下载：{url}")
    print(f"保存到：{save_path}")

    try:
        urllib.request.urlretrieve(url, save_path)
        size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"下载完成：{save_path} ({size_mb:.2f} MB)")
    except Exception as e:
        print("下载失败：", e)

if __name__ == "__main__":
    download_file(url, file_name)