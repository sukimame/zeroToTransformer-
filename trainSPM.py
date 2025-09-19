import sentencepiece as spm
import re
import codecs

f = open("JP2EN.tsv", encoding="utf-8")

text = f.read()
text = text.split("\n")[:-1]

jp_corpus = []
en_corpus = []

for i in text:
    s = i.split("\t")
    print(s)
    jp_corpus.append(s[1])
    en_corpus.append(s[3])

print(*(jp_corpus+en_corpus), sep="\n", file=codecs.open("corpus.txt", "w", "utf-8"))


print(*jp_corpus, sep="\n", file=codecs.open("jp_corpus.txt", "w", "utf-8"))
print(*en_corpus, sep="\n", file=codecs.open("en_corpus.txt", "w", "utf-8"))

# 学習の実行
spm.SentencePieceTrainer.Train(
   '--input=corpus.txt --model_prefix=sentencepiece --vocab_size=8000 --character_coverage=0.9995'
)