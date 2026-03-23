import torch
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

    for param in model.parameters():
        param.requires_grad = False
    
    torch.manual_seed(123)
    num_classes = 2
    model.out_head = torch.nn.Linear(
        in_features=BASE_CONFIG["emb_dim"],
        out_features=num_classes
    )

    model.to(device)

    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True
    
    
    inputs = tokenizer.encode("Do you have time")
    inputs = torch.tensor(inputs).unsqueeze(0).to(device)
    print("Inputs:", inputs)
    print("Inputs dimensions:", inputs.shape) # shape: (batch_size, num_tokens)

    
    with torch.no_grad():
        outputs = model(inputs)

    print("Outputs:\n", outputs)
    print("Outputs dimensions:", outputs.shape) # shape: (batch_size, num_tokens, num_classes)

    print(outputs[:, -1, : ])