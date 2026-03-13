from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

for i, token_id in enumerate(integers):
    token_piece = tokenizer.decode([token_id])
    print(f"{i:2d}. (ID: {token_id:5d}) -> '{token_piece}'")

tokens = tokenizer.decode(integers)
print(tokens)

text = 'Akwirw ier'
integers = tokenizer.encode(text)
print(integers)

tokens = tokenizer.decode(integers)
print(tokens)

for i in integers:
    temp = tokenizer.decode([i])
    print(temp)
