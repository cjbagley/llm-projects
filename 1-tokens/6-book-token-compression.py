import requests
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

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

print("  Book title     |  Chars  |  Words  |  Tokens |  Compression")
print("-----------------+---------+---------+---------+---------")
for code, title in bookurls:
    url = baseurl + code + "/pg" + code + ".txt"
    text = requests.get(url).text

    char_count = len(text)
    word_count = len(text.split())
    token_count = len(tokenizer.encode(text))
    compression_tokens_to_chars = (token_count * 100) / char_count
    compression_tokens_to_words = (token_count * 100) / word_count

    print(
        f"{title} | {char_count} | {word_count} | {token_count} | {compression_tokens_to_chars:.0f} / {compression_tokens_to_words:.0f}"
    )

#   Book title     |  Chars  |  Words  |  Tokens |  Compression
# -----------------+---------+---------+---------+---------
# Frankenstein | 446582 | 78106 | 114213 | 26 / 146
# GreatGatsby | 296899 | 51262 | 89263 | 30 / 174
# AliceWonderland | 167712 | 29569 | 52948 | 32 / 179
# RomeoJuliet | 167469 | 29005 | 56180 | 34 / 194
# HuckFinn | 602752 | 114130 | 193373 | 32 / 169
# HeartDarkness | 232924 | 40957 | 63988 | 27 / 156
# GrimmsTales | 549776 | 104156 | 158433 | 29 / 152
# EdgarAllenPoe | 632171 | 98105 | 197302 | 31 / 201
# WarOfTheWorlds | 363440 | 63118 | 96422 | 27 / 153
# GulliversTravels | 611781 | 108140 | 158340 | 26 / 146

# The conclusion of Project 5 seems to be that tokenization compresses text relative to the number of characters, by a factor of 3-4.
# That certainly is a valid conclusion based on the data used in this project, but the conclusion does not necessarily generalize to other languages.
