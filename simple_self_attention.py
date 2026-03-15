import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1]
res = 0
for i in range(len(query)):
    res += inputs[0][i] * query[i]

print(res)
print(torch.dot(inputs[0], query))

attention_scores = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attention_scores[i] = torch.dot(x_i, query)

print(attention_scores)

attention_weights = attention_scores / attention_scores.sum()
print(attention_weights)

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim = 0)

print(softmax_naive(attention_scores))
print(softmax_naive(attention_scores).sum())

attention_weights_2 = torch.softmax(attention_scores, dim = 0)
print(attention_weights_2)
print(attention_weights_2.sum())

context_vector_2 = torch.zeros(query.shape)

for i, x_i in enumerate(inputs):
    context_vector_2 += attention_weights_2[i] * x_i

print(context_vector_2)

all_attention_scores = torch.zeros(inputs.shape[0], inputs.shape[0])
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        all_attention_scores[i, j] = torch.dot(x_i, x_j)

print(all_attention_scores)
print(inputs @ inputs.T)

all_attention_weights = torch.softmax(all_attention_scores, dim = -1)
print(all_attention_weights)

all_context_vectors = all_attention_weights @ inputs
print(all_context_vectors)