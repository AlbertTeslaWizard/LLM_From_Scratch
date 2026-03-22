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
from pathlib import Path

def save_checkpoint(checkpoint_path, model, optimizer, next_epoch, global_step,
                    tokens_seen, train_losses, val_losses, track_tokens_seen):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "next_epoch": next_epoch,
            "global_step": global_step,
            "tokens_seen": tokens_seen,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "track_tokens_seen": track_tokens_seen,
        },
        tmp_path,
    )
    tmp_path.replace(checkpoint_path)


def try_save_checkpoint(checkpoint_path, model, optimizer, next_epoch, global_step,
                        tokens_seen, train_losses, val_losses, track_tokens_seen):
    try:
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            next_epoch,
            global_step,
            tokens_seen,
            train_losses,
            val_losses,
            track_tokens_seen,
        )
        return True
    except KeyboardInterrupt:
        return False
    except Exception as e:
        print(f"Checkpoint save failed: {e}")
        return False


def load_checkpoint(checkpoint_path, model, optimizer, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint
    except Exception as e:
        corrupt_path = checkpoint_path.with_name(checkpoint_path.stem + ".corrupt" + checkpoint_path.suffix)
        try:
            checkpoint_path.replace(corrupt_path)
            print(f"Checkpoint load failed: {e}. Moved corrupt checkpoint to {corrupt_path}.")
        except Exception:
            print(f"Checkpoint load failed: {e}. Could not move corrupt checkpoint.")
        return None


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer,
                       start_epoch=0, global_step=-1, tokens_seen=0,
                       train_losses=None, val_losses=None, track_tokens_seen=None,
                       checkpoint_path=None):
    train_losses = [] if train_losses is None else train_losses
    val_losses = [] if val_losses is None else val_losses
    track_tokens_seen = [] if track_tokens_seen is None else track_tokens_seen

    interrupted = False
    epoch = start_epoch

    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()

            for input_batch, target_batch in train_loader:
                optimizer.zero_grad()
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward()
                optimizer.step()
                tokens_seen += input_batch.numel()
                global_step += 1

                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader, device, eval_iter)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Ep {epoch+1} (Step {global_step:06d}): "
                          f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            generate_and_print_sample(
                model, tokenizer, device, start_context
            )

            if checkpoint_path is not None:
                checkpoint_saved = try_save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    next_epoch=epoch + 1,
                    global_step=global_step,
                    tokens_seen=tokens_seen,
                    train_losses=train_losses,
                    val_losses=val_losses,
                    track_tokens_seen=track_tokens_seen,
                )
                if not checkpoint_saved:
                    interrupted = True
                    print("Training interrupted. Checkpoint save was interrupted.")
                    return train_losses, val_losses, track_tokens_seen, interrupted
    except KeyboardInterrupt:
        interrupted = True
        checkpoint_saved = False
        if checkpoint_path is not None:
            checkpoint_saved = try_save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                next_epoch=epoch,
                global_step=global_step,
                tokens_seen=tokens_seen,
                train_losses=train_losses,
                val_losses=val_losses,
                track_tokens_seen=track_tokens_seen,
            )
        if checkpoint_saved:
            print("Training interrupted. Checkpoint saved.")
        else:
            print("Training interrupted. Checkpoint not updated.")

    return train_losses, val_losses, track_tokens_seen, interrupted

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.png")
    plt.show()

if __name__ == '__main__':
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")

    model = GPTModel(GPT_CONFIG_124M)
    file_path = 'the-verdict.txt'
    with open(file_path ,'r', encoding='utf-8') as file:
        text_data = file.read()

    total_tokens = len(tokenizer.encode(text_data))

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

    if total_tokens * train_ratio < GPT_CONFIG_124M["context_length"]:
        print("Not enough tokens for the training loader. "
              "Try to lower the `GPT_CONFIG_124M['context_length']` or "
              "increase the `training_ratio`")

    if total_tokens * (1 - train_ratio) < GPT_CONFIG_124M["context_length"]:
        print("Not enough tokens for the validation loader. "
              "Try to lower the `GPT_CONFIG_124M['context_length']` or "
              "decrease the `training_ratio`")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using {device} device.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    checkpoint_path = Path("models/model_and_optimizer.pth")

    start_epoch = 0
    global_step = -1
    tokens_seen = 0
    train_losses, val_losses, tracked_tokens = [], [], []

    if checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path, model, optimizer, device)
        if checkpoint is not None:
            start_epoch = checkpoint.get("next_epoch", 0)
            global_step = checkpoint.get("global_step", -1)
            tokens_seen = checkpoint.get("tokens_seen", 0)
            train_losses = checkpoint.get("train_losses", [])
            val_losses = checkpoint.get("val_losses", [])
            tracked_tokens = checkpoint.get("track_tokens_seen", [])
            print(f"Resumed from checkpoint at epoch {start_epoch}.")
        else:
            print("Starting training from scratch.")

    num_epochs = 10
    train_losses, val_losses, tracked_tokens, interrupted = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer,
        start_epoch=start_epoch, global_step=global_step, tokens_seen=tokens_seen,
        train_losses=train_losses, val_losses=val_losses,
        track_tokens_seen=tracked_tokens, checkpoint_path=checkpoint_path
    )

    if not interrupted:
        save_path = Path("models/model_after_10_epochs.pth")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print("Model saved.")

    if len(train_losses) > 0:
        epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
        plot_losses(epochs_tensor, tracked_tokens, train_losses, val_losses)