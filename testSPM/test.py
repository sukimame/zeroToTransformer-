import codecs
import re
import requests
from bs4 import BeautifulSoup
import sentencepiece as spm

# URLとキーワードリスト
url = "https://ja.wikipedia.org/wiki/"
keyword_list = [
    "中島敦", "太宰治", "国木田独歩",
    "江戸川乱歩", "谷崎潤一郎", "宮沢賢治",
    "与謝野晶子", "芥川龍之介", "樋口一葉",
    "中原中也", "尾崎紅葉", "梶井基次郎",
    "夢野久作", "森鷗外", "織田作之助"   
]
headers = {"User-Agent": "Mozilla/5.0"}

# Wikipediaの小説家の記事のダウンロード
corpus = []
for keyword in keyword_list:
    response = requests.get(url + keyword, headers=headers)
   
    soup = BeautifulSoup(response.text, 'lxml')
    for p_tag in soup.find_all('p'):
        # 空白の削除
        text = "".join(p_tag.text.strip().split(" "))

        # 空行は無処理
        if len(text) == 0:
            continue

        # 注釈の削除 (例: [注釈1], [注釈1], [1])
        text = re.sub(r"\[注釈[0-9]+\]", "", text)
        text = re.sub(r"\[注[0-9]+\]", "", text)
        text = re.sub(r"\[[0-9]+\]", "", text)

        # 行の追加
        corpus.append(text)

# ファイルの保存       
print(*corpus, sep="\n", file=codecs.open("wiki.txt", "w", "utf-8"))

# 学習の実行
spm.SentencePieceTrainer.Train(
   '--input=wiki.txt --model_prefix=sentencepiece --vocab_size=8000 --character_coverage=0.9995'
)

# モデルの作成
sp = spm.SentencePieceProcessor()
sp.Load("sentencepiece.model")

# テキストを語彙列に分割
print(corpus[0])
print(sp.EncodeAsPieces(corpus[0]))
print(sp.EncodeAsIds(corpus[0]))

print(sp.GetPieceSize())