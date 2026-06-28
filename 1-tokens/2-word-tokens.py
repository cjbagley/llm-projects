text = "The way you do anything is the way you do everything."

word_vocab = sorted(set(text.split()))

# Lookup dicts
word_to_idx = {word: idx for idx, word in enumerate(word_vocab)}
idx_to_word = dict(enumerate(word_vocab))

encoded_text = [word_to_idx[w] for w in text.split()]

print(encoded_text)
