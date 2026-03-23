import torch
import torch.nn as nn
import tiktoken
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GPT_CONFIG_124M
from GPTModel import GPTModel
from generate_text import text_to_token_ids, token_ids_to_text, generate_text_simple


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tokenizer = tiktoken.get_encoding("gpt2")
    
    BASE_CONFIG = GPT_CONFIG_124M.copy()
    BASE_CONFIG.update({"context_length": 1024})
    BASE_CONFIG.update({"qkv_bias": True})

    state_dict = torch.load(
        '../models/gpt2-small-124M.pth',
        map_location='cpu',
        weights_only=True
    )

    model = GPTModel(BASE_CONFIG)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    text_1 = "Every effort moves you"
    input_ids = text_to_token_ids(text_1, tokenizer).to(device)

    token_ids = generate_text_simple(
        model=model,
        idx=input_ids,
        max_new_tokens=15,
        context_size=BASE_CONFIG["context_length"]
    )

    print(token_ids_to_text(token_ids, tokenizer))

    text_2 = (
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        " 'You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award.'"
    )

    input_ids = text_to_token_ids(text_2, tokenizer).to(device)
    token_ids = generate_text_simple(
        model=model,
        idx=input_ids,
        max_new_tokens=23,
        context_size=BASE_CONFIG["context_length"]
    )

    print(token_ids_to_text(token_ids, tokenizer))
