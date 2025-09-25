from transformer import Model

f1 = open("jp_corpus.txt", "r", encoding="utf-8")
f2 = open("en_corpus.txt", "r", encoding="utf-8")
jp_corpus = f1.read()
jp_corpus = jp_corpus.split("\n")
en_corpus = f2.read()
en_corpus = en_corpus.split("\n")

model = Model()
print(model.forward(jp_corpus[1], en_corpus[1]).shape)

print(jp_corpus[1])


