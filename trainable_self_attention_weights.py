import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vec = attn_weights @ values
        return context_vec
    
if __name__ == '__main__':
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your     (x^1)
        [0.55, 0.87, 0.66], # journey  (x^2)
        [0.57, 0.85, 0.64], # starts   (x^3)
        [0.22, 0.58, 0.33], # with     (x^4)
        [0.77, 0.25, 0.10], # one      (x^5)
        [0.05, 0.80, 0.55]] # step     (x^6)
    )

    x_2 = inputs[1] # second input element
    d_in = inputs.shape[1] # the input embedding size, d=3
    d_out = 2 # the output embedding size, d=2

    torch.manual_seed(123)

    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    query_2 = x_2 @ W_query
    key_2 = x_2 @ W_key
    value_2 = x_2 @ W_value
    print(query_2)

    keys = inputs @ W_key
    values = inputs @ W_value
    print(f"keys: {keys} values: {values}")

    attention_score_2 = query_2 @ keys.T
    d_k = keys.shape[-1]
    print(attention_score_2)
    attention_weights_2 = torch.softmax(attention_score_2 / (d_k ** 0.5), dim=-1)
    print(attention_weights_2)

    context_vector_2 = attention_weights_2 @ values
    print(context_vector_2)

    torch.manual_seed(123)
    sa_v1 = SelfAttention_v1(d_in, d_out)
    print(sa_v1(inputs))

    torch.manual_seed(789)
    sa_v2 = SelfAttention_v2(d_in, d_out)
    print(sa_v2(inputs))

    torch.manual_seed(123)
    sa_v1 = SelfAttention_v1(d_in, d_out)
    torch.manual_seed(123)
    sa_v2 = SelfAttention_v2(d_in, d_out)

    with torch.no_grad():
        sa_v1.W_query.copy_(sa_v2.W_query.weight.T)
        sa_v1.W_key.copy_(sa_v2.W_key.weight.T)
        sa_v1.W_value.copy_(sa_v2.W_value.weight.T)

    out1 = sa_v1(inputs)
    out2 = sa_v2(inputs)

    print(out1)
    print(out2)