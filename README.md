# python で naive_beys の学習

## きっかけ

- [前回gunma.web](https://gunmaweb.connpass.com/event/66392/) の「機械学習で奇妙な冒険」のウケがよかったので<br>
試しに python で実装しました。
- 参考にしたドキュメントは[こちら](https://qiita.com/katryo/items/6a2266ffafb7efa9a46c)です。

## 実行環境

- OS
  - centos7
- python
  - 3.6.2

## 必要なライブラリ

- pip install mecab-python3

## 使い方

- 学習データ
  - data/jojo.dat
  - キャラとセリフを足したければここを修正してください。

- 学習method
  - set_category('カテゴリー 登場人物など')
  - set_word('言葉 セリフなど')
  - learn

- 結果を判定する method
  - classify

- テストコード を参考にすれば行けると思います。
  - test_naive_bayes.py
