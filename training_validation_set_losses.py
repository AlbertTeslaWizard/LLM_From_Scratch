import torch
import torch.nn as nn
import tiktoken
from config import GPT_CONFIG_124M
from GPTModel import GPTModel
from generate_text import text_to_token_ids, token_ids_to_text
from sliding_window import create_dataloader_v1

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if(len(data_loader) == 0):
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches

if __name__ == '__main__':
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")

    model = GPTModel(GPT_CONFIG_124M)
    model.eval() # disable dropout

    file_path = 'the-verdict.txt'
    with open(file_path ,'r', encoding='utf-8') as file:
        text_data = file.read()
    
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))

    print("Characters:", total_characters)
    print("Tokens:", total_tokens)

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:
        print("Not enough tokens for the training loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "increase the `training_ratio`")

    if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]:
        print("Not enough tokens for the validation loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "decrease the `training_ratio`")
    
    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)

    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)

    train_tokens = 0
    for input_batch, target_batch in train_loader:
        train_tokens += input_batch.numel()

    val_tokens = 0
    for input_batch, target_batch in val_loader:
        val_tokens += input_batch.numel()

    print("Training tokens:", train_tokens)
    print("Validation tokens:", val_tokens)
    print("All tokens:", train_tokens + val_tokens)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using {device} device.")
    model.to(device)

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)
