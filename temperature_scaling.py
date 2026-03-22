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

def print_sampled_tokens(probas):
    torch.manual_seed(123) # Manual seed for reproducibility
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1000)]
    sampled_ids = torch.bincount(torch.tensor(sample), minlength=len(probas))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")

def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)


if __name__ == '__main__':
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")
    
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load("models/model_after_10_epochs.pth", map_location="cpu"))
    model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    vocab = { 
        "closer": 0,
        "every": 1, 
        "effort": 2, 
        "forward": 3,
        "inches": 4,
        "moves": 5, 
        "pizza": 6,
        "toward": 7,
        "you": 8,
    }  

    inverse_vocab = {v: k for k, v in vocab.items()}

    # Suppose input is "every effort moves you", and the LLM
    # returns the following logits for the next token:
    next_token_logits = torch.tensor(
        [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
    )

    probas = torch.softmax(next_token_logits, dim=0)
    next_token_id = torch.argmax(probas).item()

    # The next generated token is then as follows:
    print(inverse_vocab[next_token_id])

    next_token_id = torch.multinomial(probas, num_samples=1).item()
    print(inverse_vocab[next_token_id])
    print_sampled_tokens(probas)

    
    # Temperature values
    temperatures = [1, 0.1, 5]  # Original, higher confidence, and lower confidence

    # Calculate scaled probabilities
    scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]

    # Plotting
    x = torch.arange(len(vocab))
    bar_width = 0.15

    fig, ax = plt.subplots(figsize=(5, 3))
    for i, T in enumerate(temperatures):
        rects = ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f'Temperature = {T}')

    ax.set_ylabel('Probability')
    ax.set_xticks(x)
    ax.set_xticklabels(vocab.keys(), rotation=90)
    ax.legend()

    plt.tight_layout()
    plt.savefig("temperature-plot.png")
    plt.show()