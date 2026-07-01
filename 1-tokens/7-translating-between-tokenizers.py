import numpy as np
import requests
import tiktoken
from transformers import AutoTokenizer

# GPT-4’s tokenizer
tokenizer_4 = tiktoken.get_encoding("cl100k_base")

# GPT-2’s tokenizer
tokenizer_2 = AutoTokenizer.from_pretrained("gpt2")


print(f"GPT-2 has a vocab size of {tokenizer_2.vocab_size:7}")
print(f"GPT-4 has a vocab size of {tokenizer_4.n_vocab:7}")
# GPT-2 has a vocab size of   50257
# GPT-4 has a vocab size of  100277

baseurl = "https://www.gutenberg.org/cache/epub/"

bookurls = [
    # code       title
    ["84", "Frankenstein"],
    ["64317", "GreatGatsby"],
    ["11", "AliceWonderland"],
    ["1513", "RomeoJuliet"],
    ["76", "HuckFinn"],
    ["219", "HeartDarkness"],
    ["2591", "GrimmsTales"],
    ["2148", "EdgarAllenPoe"],
    ["36", "WarOfTheWorlds"],
    ["829", "GulliversTravels"],
]

for code, title in bookurls:
    url = baseurl + code + "/pg" + code + ".txt"
    text = requests.get(url).text

    gpt2_toks = tokenizer_2.encode(text)
    gpt4_toks = tokenizer_4.encode(text)

    # average token length (still decoding one-by-one, per-token stats)
    gpt2_tok_lengths = [len(tokenizer_2.decode([t]).encode("utf-8")) for t in gpt2_toks]
    gpt4_tok_lengths = [len(tokenizer_4.decode([t]).encode("utf-8")) for t in gpt4_toks]

    avarage_token_lenth_gpt2 = np.mean(gpt2_tok_lengths)
    avarage_token_lenth_gpt4 = np.mean(gpt4_tok_lengths)

    # bytes of the RECONSTRUCTED TEXT (decode all at once -> should match original)
    reconstructed_bytes_gpt2 = len(tokenizer_2.decode(gpt2_toks).encode("utf-8"))
    reconstructed_bytes_gpt4 = len(tokenizer_4.decode(gpt4_toks).encode("utf-8"))

    # bytes to STORE THE TOKEN STREAM ITSELF, as fixed-width integer IDs
    gpt2_id_bytes = 2  # uint16: vocab 50257 fits under 65536
    gpt4_id_bytes = 4  # uint32: vocab 100277 exceeds uint16 range
    encoded_stream_bytes_gpt2 = len(gpt2_toks) * gpt2_id_bytes
    encoded_stream_bytes_gpt4 = len(gpt4_toks) * gpt4_id_bytes

    print(f"Book: {title}")
    print(f"Original text size:                  {len(text.encode('utf-8'))} bytes")
    print(f"GPT-2 token count:                   {len(gpt2_toks)}")
    print(f"GPT-2 avarage token length:          {avarage_token_lenth_gpt2:.3f}")
    print(f"GPT-2 reconstructed text bytes:      {reconstructed_bytes_gpt2}")
    print(f"GPT-2 encoded token stream bytes:    {encoded_stream_bytes_gpt2}")
    print(f"GPT-4 token count:                   {len(gpt4_toks)}")
    print(f"GPT-4 avarage token length:          {avarage_token_lenth_gpt4:.3f}")
    print(f"GPT-4 reconstructed text bytes:      {reconstructed_bytes_gpt4}")
    print(f"GPT-4 encoded token stream bytes:    {encoded_stream_bytes_gpt4}")
