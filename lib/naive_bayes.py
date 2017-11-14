
import math
import sys
import MeCab

class NaiveBayes:

  def __init__(self):
    self.vocabularies = set()
    self.word_count = {}
    self.category_count = {}
    self.category = ''
    self.word = ''

  def set_category(self,category):
    self.category = category

  def set_word(self,word):
    self.word = word

  def learn(self):
    words = self.get_wakati_by_word(self.word)
    for w in words:
      self.count_up_by_wakati_word(w)
    self.count_up_by_category(self.category)

  def count_up_by_category(self,category):
    self.category_count.setdefault(category,0)
    self.category_count[category] += 1

  def count_up_by_wakati_word(self,word):
    self.word_count.setdefault(self.category,{})
    self.word_count[self.category].setdefault(word,0)
    self.word_count[self.category][word] += 1
    self.vocabularies.add(word)

  def get_wakati_by_word(self,word):
    tagger = MeCab.Tagger()
    node = tagger.parseToNode(word)
    ret = []
    while node:
      words = node.feature.split(',')
      if not words[6] == '*':
        ret.append(words[6])
      node = node.next
    return tuple(ret)

  def classify(self,word):
    best_category = None
    max_prob_before = -sys.maxsize
    words = self.get_wakati_by_word(word)
    for category in self.category_count.keys():
      self.set_category(category)
      prob = self.get_score_by_words(words)
      if prob > max_prob_before:
        max_prob_before = prob
        best_category = category
    return best_category
    
  def get_score_by_words(self,words):
    score = math.log(self.get_prior_prob_by_category(self.category) )
    for word in words:
      score += math.log(self.get_word_prob_by_word(word) )
    return score

  def get_prior_prob_by_category(self,category):
    categories_count_sum = sum(self.category_count.values())
    categories_count = self.category_count[category]
    return categories_count / categories_count_sum

  def get_word_prob_by_word(self,word):
    numerator = self.get_word_count_by_word(word) + 1
    denominator = sum(self.word_count[self.category].values()) + len(self.vocabularies)
    return numerator / denominator

  def get_word_count_by_word(self,word):
    self.word_count[self.category].setdefault(word,0)
    return self.word_count[self.category][word]


