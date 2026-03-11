import re

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
print("Total number of character:", len(raw_text))
print(raw_text[:99])

text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)
result2 = re.split(r'([,.]|\s)', text)
print(result)
print(result2)

item = [x for x in result2 if x.strip()]
print(item)

text = "Hello, world. Is this-- a test?"
result_temp = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result_temp_process = [x for x in result_temp if x.strip()]
print(result_temp_process)

res = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
res_process = [x for x in res if x.strip()]
print(res_process[:20])
print(len(res_process))