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

print("  Book title     |  Chars  |  Words  |  Tokens")
print("-----------------+---------+---------+---------")
for code, title in bookurls:
    url = baseurl + code + "/pg" + code + ".txt"
    text = requests.get(url).text

    char_count = len(text)
    word_count = len(text.split())
    token_count = len(tokenizer.encode(text))

    print(f"{title} | {char_count} | {word_count} | {token_count}")
