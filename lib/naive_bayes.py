
import math
import sys
import MeCab

class NaiveBayes:

  def __init__(self):
    self._vocabularies = set()
    self._word_count = {}
    self._category_count = {}
    self._category = ''
    self._word = ''
  
  @property
  def word_count(self):
    return self._word_count

  @property
  def vocabularies(self):
    return self._vocabularies

  @property
  def category(self):
    return self._category

  @property
  def category_count(self):
    return self._category_count

  @property
  def word(self):
    return self._word

  @category.setter
  def category(self, v):
    self._category = v

  @word.setter
  def word(self, v):
    self._word = v

  def learn(self):
    words = self.wakati_word(self.word)
    for w in words:
      self.count_up_word(w)
    self.count_up_category(self.category)

  def count_up_category(self, category):
    self._category_count.setdefault(category, 0)
    self._category_count[category] += 1

  def count_up_word(self, word):
    self._word_count.setdefault(self.category, {})
    self._word_count[self.category].setdefault(word, 0)
    self._word_count[self.category][word] += 1
    self._vocabularies.add(word)

  def wakati_word(self, word):
    tagger = MeCab.Tagger()
    node = tagger.parseToNode(word)
    ret = []
    while node:
      words = node.feature.split(',')
      if words[0] in ["名詞", "動詞", "形容詞", "副詞", "連体詞", "感動詞", "助詞", "助動詞", "記号"]:
        if not words[6] == '*':
          ret.append(words[6])
          ret.append(words[7])
      node = node.next
    return tuple(ret)

  def classify(self, word):
    best_category = None
    max_prob_before = -sys.maxsize
    words = self.wakati_word(word)
    for category in self.category_count.keys():
      self.category = category
      prob = self.score_words(words)
      if prob > max_prob_before:
        max_prob_before = prob
        best_category = category
    return best_category
    
  def score_words(self, words):
    score = math.log(self.prior_prob_category(self.category) )
    for word in words:
      score += math.log(self.word_prob_word(word) )
    return score

  def prior_prob_category(self, category):
    categories_count_sum = sum(self.category_count.values())
    categories_count = self.category_count[category]
    return categories_count / categories_count_sum

  def word_prob_word(self, word):
    numerator = self.word_count_sum(word) + 1
    denominator = sum(self.word_count[self.category].values()) + len(self.vocabularies)
    return numerator / denominator

  def word_count_sum(self, word):
    self._word_count[self.category].setdefault(word, 0)
    return self.word_count[self.category][word]


