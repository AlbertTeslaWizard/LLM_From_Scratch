import torch
import tiktoken

from GPTModel import GPTModel
from config import GPT_CONFIG_124M
from generate_text import text_to_token_ids, token_ids_to_text
from top_k_sampling import generate

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update({"context_length": 1024})
    NEW_CONFIG.update({"qkv_bias": True})

    state_dict = torch.load(
        'models/gpt2-small-124M.pth',
        map_location='cpu',
        weights_only=True
    )

    print("Loaded parameter count:", len(state_dict))
    print(state_dict["tok_emb.weight"])
    print("Some parameter keys:", list(state_dict.keys())[:10])

    gpt = GPTModel(NEW_CONFIG)
    gpt.load_state_dict(state_dict)
    gpt.eval()
    gpt.to(device)
    print(gpt)

    tokenizer = tiktoken.get_encoding("gpt2")

    torch.manual_seed(123)

    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=25,
        context_size=NEW_CONFIG["context_length"],
        top_k=50,
        temperature=1.5
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))