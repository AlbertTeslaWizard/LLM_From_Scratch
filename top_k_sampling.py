import torch
import torch.nn as nn
import tiktoken
from config import GPT_CONFIG_124M
from GPTModel import GPTModel
from generate_text import text_to_token_ids, token_ids_to_text, generate_text_simple
from sliding_window import create_dataloader_v1
from training_validation_set_losses import calc_loss_batch, calc_loss_loader
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # New (not in book): numerical stability tip to get equivalent results on mps device
            # subtract rowwise max before softmax
            logits = logits - logits.max(dim=-1, keepdim=True).values
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

if __name__ == '__main__':
    torch.manual_seed(123)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    next_token_logits = torch.tensor(
        [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
    )

    top_k = 3
    top_logits, top_pos = torch.topk(next_token_logits, top_k)

    print("Top logits:", top_logits)
    print("Top positions:", top_pos)

    new_logits = torch.where(
        condition = next_token_logits < top_logits[-1],
        input = torch.tensor(torch.tensor(float("-inf"))),
        other = next_token_logits
    )

    print(new_logits)
    
    topk_probas = torch.softmax(new_logits, dim=0)
    print(topk_probas)


    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load("models/model_after_10_epochs.pth", map_location="cpu"))
    model.eval()

    inference_device = torch.device("cpu")
    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(inference_device),
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=25,
        temperature=1.4
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))