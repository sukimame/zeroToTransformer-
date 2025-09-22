#set page("a4")
#set text(font:"BIZ UDGothic")

#let title = "TF実装に当たってのレポート"
#show title: set align(center)
#show title: set text(weight: "black", size: 1.5em)

#let name = "23266053 志田光"
#show name: set align(right)
#show name: set text(weight: "black", size: 1.0em)


#title
#name

= トランスフォーマー

= 日英コーパス
https://tatoeba.org/ja/downloads

= 逆伝播
== scaled dot-product attention

#figure(image("スクリーンショット 2025-09-19 183208.png", width:30%))


= 行列積の微分

#let dydw = $mat((diff y_11)/(diff W), (diff y_12)/(diff W), 
                 (diff y_13)/(diff W), (diff y_14)/(diff W);
                 (diff y_21)/(diff W), (diff y_22)/(diff W), 
                 (diff y_23)/(diff W), (diff y_24)/(diff W))$

$ X in RR^(2*3), W in RR^(3*4), Y = X W $と設定し、
$ (d Y) / (d W) $を計算する。この時、
$ (d Y)/ (d W) = dydw $
となり、さらに
$ (diff y_11)/(diff W) = mat((diff y_11)/(diff W_11),(diff y_11)/(diff W_12),    
                             (diff y_11)/(diff W_13), (diff y_11)/(diff W_14);
                             (diff y_11)/(diff W_21), (diff y_11)/(diff W_22), 
                             (diff y_11)/(diff W_23), (diff y_11)/(diff W_24);
                             (diff y_11)/(diff W_31), (diff y_11)/(diff W_32), 
                             (diff y_11)/(diff W_33), (diff y_11)/(diff W_34)) 
$

となる。よって、$ (d Y)/ (d W) in RR^(2*4*3*4) $となり、4階テンソルとなることが分かる。

$ G = (d L)/(d Y) in RR^(2*4) $としたとき、

$ (d L)/(d W) = (d L)/(d Y) (d Y)/(d W) $で求められるはず。だが、これだと行列積が可能なサイズでない。よって、Gを転置するしかない？

あだマール積なら完璧!（微分が同形の行列の場合。今回は4階テンソルを2次元までの行列として扱うことでうまくいった。）
これを中身で見てみると、

$ (d L)/(d Y) circle.stroked.tiny (d Y)/(d W) =  
mat(
(diff L)/(diff y_11), (diff L)/(diff y_12), (diff L)/(diff y_13), (diff L)/(diff y_14);
(diff L)/(diff y_21), (diff L)/(diff y_22), (diff L)/(diff y_23), (diff L)/(diff y_24))
dydw 
= 
mat(
(diff L)/(diff y_11) (diff y_11)/(diff W), (diff L)/(diff y_12) (diff y_12)/(diff W),
(diff L)/(diff y_13) (diff y_13)/(diff W), (diff L)/(diff y_14) (diff y_14)/(diff W);
(diff L)/(diff y_21) (diff y_21)/(diff W), (diff L)/(diff y_22) (diff y_22)/(diff W),
(diff L)/(diff y_23) (diff y_23)/(diff W), (diff L)/(diff y_24) (diff y_24)/(diff W))
$

合成関数の微分の定理（2変数関数を2変数でそれぞれ偏微分）では同じ変数での偏微分結果をすべて足し合わせて求める。

つまり、

$ (diff L)/(diff w_11) = 
&(diff L)/(diff y_11) (diff y_11)/(diff w_11) + 
(diff L)/(diff y_12) (diff y_12)/(diff w_11) + 
(diff L)/(diff y_13) (diff y_13)/(diff w_11) + 
(diff L)/(diff y_14) (diff y_14)/(diff w_11)\
&+(diff L)/(diff y_21) (diff y_21)/(diff w_11) + 
(diff L)/(diff y_22) (diff y_22)/(diff w_11) +
(diff L)/(diff y_23) (diff y_23)/(diff w_11) +
(diff L)/(diff y_24) (diff y_24)/(diff w_11)
$

このように、$(d L)/(d Y) circle.stroked.tiny (d Y)/(d W)$の各要素である各y要素をWで偏微分した結果を足し合わせると、Lをwの各要素で偏微分した結果が得られる。

すなわち、$ G = (d L)/(d Y), K = (d Y)/(d W)$とおくと、
$W ← W - eta sum_(i, j) (G circle.stroked.tiny K)_(i, j) $
となる。

$ y_(i, j) = sum_(k=1)^3 (w_(k, i) x_(j, k)) $

$ (d Y)/(d W) = dydw = mat(
  mat(x_11, 0, 0, 0;
      x_12, 0, 0, 0;
      x_13, 0, 0, 0
  ) 
  mat(0, x_11, 0, 0;
      0, x_12, 0, 0;
      0, x_13, 0, 0
  )
  mat(0, 0, x_11, 0;
      0, 0, x_12, 0;
      0, 0, x_13, 0
  )
  mat(0, 0, 0, x_11;
      0, 0, 0, x_12;
      0, 0, 0, x_13
  );
  mat(x_21, 0, 0, 0;
      x_22, 0, 0, 0;
      x_23, 0, 0, 0
  ) 
  mat(0, x_21, 0, 0;
      0, x_22, 0, 0;
      0, x_23, 0, 0
  )
  mat(0, 0, x_21, 0;
      0, 0, x_22, 0;
      0, 0, x_23, 0
  )
  mat(0, 0, 0, x_21;
      0, 0, 0, x_22;
      0, 0, 0, x_23
  );
)
 $
この時、$(d L)/(d Y)$とのあだマール積を求め、各要素の行列を足し合わせて0の成分を無視すると、
$ 
(d L)/(d Y) circle.stroked.tiny (d Y)/(d W) = mat(
(diff L)/(diff y_11)x_11 + (diff L)/(diff y_21) x_21, (diff L)/(diff y_12)x_11 + (diff L)/(diff y_22) x_21, dots.h, (diff L)/(diff y_14)x_11 + (diff L)/(diff y_24) x_21;
(diff L)/(diff y_11)x_12 + (diff L)/(diff y_21) x_22, dots.h, dots.down, dots.v;
(diff L)/(diff y_11)x_13 + (diff L)/(diff y_21) x_23, (diff L)/(diff y_12)x_13 + (diff L)/(diff y_22) x_23, dots.h, (diff L)/(diff y_14)x_13 + (diff L)/(diff y_24) x_23;
)
$
この行列は、$X^T G$の演算結果と同じことが分かる。すなわち、

$ (d L)/(d W) = (d L)/(d Y) circle.stroked.tiny (d Y)/(d W) = X^T G $

同じようにして、
$ (d L)/(d X) = (d L)/(d Y) circle.stroked.tiny (d Y)/(d X) = G W^T $

4階テンソルとのあだマール積は不可能。よって、3次元以降の行列らをスカラー倍する操作を別に定義する必要性あり。

