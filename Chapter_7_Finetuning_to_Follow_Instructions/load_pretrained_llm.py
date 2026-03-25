import torch
import tiktoken
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GPTModel import GPTModel
from prepare_dataset import download_and_load_file, format_input
from generate_text import text_to_token_ids, token_ids_to_text
from top_k_sampling import generate

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Use PyTorch 2.9 or newer for stable mps results
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    CHOOSE_MODEL = "gpt2-medium (355M)"

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model = GPTModel(BASE_CONFIG)
    state_dict = torch.load('../models/gpt2-medium-355M.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )
    data = download_and_load_file("instruction-data.json", url)
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = int(len(data) - train_portion - test_portion)

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion+test_portion:]

    torch.manual_seed(123)
    input_text = format_input(val_data[0])
    print(input_text)

    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=35,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256,
    )
    
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text):].strip()
    print(response_text)