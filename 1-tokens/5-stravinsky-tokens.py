import requests
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# https://en.wikipedia.org/wiki/Pulcinella_(ballet)
txt = "Pulcinella is a 21-section ballet by Igor Stravinsky with arias for soprano, tenor and bass vocal soloists, and two sung trios. It is based on the 18th-century play Quatre Polichinelles semblables, or Four similar Pulcinellas, revolving around a stock character from commedia dell'arte. The work premiered at the Paris Opera on 15 May 1920 under the baton of Ernest Ansermet. The central dancer, Léonide Massine, created both the libretto and the choreography, while Pablo Picasso designed the costumes and sets. The ballet was commissioned by Sergei Diaghilev, impresario of the Ballets Russes. A complete performance takes 35–40 minutes. Stravinsky revised the score in 1965."

word_count = len(txt.split())
tokenized_txt = tokenizer.encode(txt)
unique_tokens = set(tokenized_txt)

reconstructed = tokenizer.decode(tokenized_txt)

print(reconstructed == txt)
