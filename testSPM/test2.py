import sentencepiece as spm

# 学習の実行
spm.SentencePieceTrainer.Train(
   '--input=wiki.txt --model_prefix=sentencepiece --vocab_size=8000 --character_coverage=0.9995'
)

# モデルの作成
sp = spm.SentencePieceProcessor()
sp.Load("sentencepiece.model")

f = open("wiki.txt", encoding="utf-8")
corpus = f.read()
corpus = corpus.split("\n")

# テキストを語彙列に分割
print(sp.EncodeAsPieces(corpus[0]))
print(sp.EncodeAsIds(corpus[0]), "\n")

print(sp.EncodeAsPieces(corpus[1]))
print(sp.EncodeAsIds(corpus[1]), "\n")

print(sp.GetPieceSize())
