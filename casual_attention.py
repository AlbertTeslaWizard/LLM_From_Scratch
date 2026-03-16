import torch
from trainable_self_attention_weights import SelfAttention_v2

if __name__ == '__main__':
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your     (x^1)
        [0.55, 0.87, 0.66], # journey  (x^2)
        [0.57, 0.85, 0.64], # starts   (x^3)
        [0.22, 0.58, 0.33], # with     (x^4)
        [0.77, 0.25, 0.10], # one      (x^5)
        [0.05, 0.80, 0.55]] # step     (x^6)
    )

    torch.manual_seed(789)
    d_in = inputs.shape[1] # the input embedding size, d=3
    d_out = 2 # the output embedding size, d=2
    sa_v2 = SelfAttention_v2(d_in, d_out)

    queries = sa_v2.W_query(inputs)
    keys = sa_v2.W_key(inputs)
    values = sa_v2.W_value(inputs)

    attention_scores = queries @ keys.T
    attention_weights = torch.softmax(queries @ keys.T / keys.shape[-1] ** 0.5, dim = -1)
    context_vectors = attention_weights @ values
    print(attention_weights)

    context_length = attention_scores.shape[0]
    mask_simple = torch.tril(torch.ones(context_length, context_length))
    print(mask_simple)

    masked_simple = attention_weights * mask_simple
    print(masked_simple)
    row_sums = masked_simple.sum(dim = -1, keepdim = True)
    masked_simple_norm = masked_simple / row_sums
    print(masked_simple_norm)
