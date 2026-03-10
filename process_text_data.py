import re

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
#print("Total number of character:", len(raw_text))
#print(raw_text[:99])

text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)
result2 = re.split(r'([,.]|\s)', text)
print(result)
print(result2)

item = [x for x in result2 if x.strip()]
print(item)