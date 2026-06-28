text = "The way you do anything is the way you do everything."

char_vocab = sorted(set(text))

# Lookup dicts
char_to_idx = {char: idx for idx, char in enumerate(char_vocab)}
idx_to_char = dict(enumerate(char_vocab))

encoded_text = [char_to_idx[c] for c in text]

print(encoded_text)
