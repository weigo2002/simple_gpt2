import tiktoken
import torch
from tinycss2 import tokenizer

from data import create_datalaoder
from evaluation import calc_loss_batch, evaluate_model
from helper import generate_and_print_sample
from model import GPTModel


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
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
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): Train loss {train_loss:.3f}, f Val loss {val_loss:.3f}")
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen

def prepare_data(file_path, train_ratio):
    with open(file_path, 'r') as f:
        text_data = f.read()

    split_idx = int(len(text_data) * train_ratio)
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    return train_data, val_data

if __name__ == '__main__':
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 256,  # 1
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,  # 2
        "qkv_bias": False
    }

    torch.manual_seed(123)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    tokenizer = tiktoken.get_encoding("gpt2")

    train_data, val_data = prepare_data("the-verdict.txt", train_ratio=0.9)

    train_loader = create_datalaoder(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    val_loader = create_datalaoder(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )


    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    num_epochs = 10
    train_losses, val_losses, track_tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device, num_epochs,
        eval_freq=5, eval_iter=5, start_context="Every effort moves you", tokenizer=tokenizer
    )