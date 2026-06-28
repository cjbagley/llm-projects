from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
text = "The way you do anything is the way you do everything."
gpt2_tokens = tokenizer.encode(text)
for token in gpt2_tokens:
    toktext = tokenizer.decode([token])
    print(f'{token:4} is "{toktext}"')
