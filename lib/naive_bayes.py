
import math
import sys
import MeCab

class NaiveBayes:

  def __init__(self):
    self.vocabularies = set()
    self.word_count = {}
    self.category_count = {}

  def train(self,category,word):
    words = self.word_to_mecab(word)
    for w in words:
      self.word_count_up(word=w, category=category)
    self.count_up_by_category(category)

  def count_up_by_category(self,category):
    self.category_count.setdefault(category,0)
    self.category_count[category] += 1

  def word_count_up(self,**refs):
    self.word_count.setdefault(refs['category'],{})
    self.word_count[refs['category']].setdefault(refs['word'],0)
    self.word_count[refs['category']][refs['word']] += 1
    self.vocabularies.add(refs['word'])

  def word_to_mecab(self,word):
    tagger = MeCab.Tagger()
    node = tagger.parseToNode(word)
    ret = []
    while node:
      elem = node.feature.split(",")[6]
      if not elem == '*':
        ret.append(elem)
      node = node.next
    return tuple(ret)

  def classify(self,word):
    best_category = None
    max_prob_before = -sys.maxsize
    words = self.word_to_mecab(word)
    for category in self.category_count.keys():
      prob = self.get_score(words=words, category=category)
      if prob > max_prob_before:
        max_prob_before = prob
        best_category = category
    return best_category
    
  def get_score(self,**refs):
    score = math.log(self.get_prior_prob_by_category(refs['category']))
    for word in refs['words']:
      score += math.log(self.get_word_prob(word=word, category=refs['category']))
    return score

  def get_prior_prob_by_category(self,category):
    categories_count_sum = sum(self.category_count.values())
    categories_count = self.category_count[category]
    return categories_count / categories_count_sum

  def get_word_prob(self,**refs):
    numerator = self.get_word_count(word=refs['word'], category=refs['category']) + 1
    denominator = sum(self.word_count[refs['category']].values()) + len(self.vocabularies)
    return numerator / denominator

  def get_word_count(self,**refs):
    self.word_count[refs['category']].setdefault(refs['word'],0)
    return self.word_count[refs['category']][refs['word']]


