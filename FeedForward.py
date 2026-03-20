import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from config import GPT_CONFIG_124M

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
             (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], cfg["emb_dim"] * 4),
            GELU(),
            nn.Linear(cfg["emb_dim"] * 4, cfg["emb_dim"])
        )
    
    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    gelu, relu = GELU(), nn.ReLU()

    # Some sample data
    x = torch.linspace(-3, 3, 100)
    y_gelu, y_relu = gelu(x), relu(x)

    plt.figure(figsize=(8, 3))
    for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
        plt.subplot(1, 2, i)
        plt.plot(x, y)
        plt.title(f"{label} activation function")
        plt.xlabel("x")
        plt.ylabel(f"{label}(x)")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("activation.png")

    ffn = FeedForward(GPT_CONFIG_124M)
    x = torch.rand(2, 3, 768) 
    out = ffn(x)
    print(out)