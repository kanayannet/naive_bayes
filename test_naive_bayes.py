
import unittest
import sys
sys.path.append('./lib')
from naive_bayes import NaiveBayes


class NaiveBayesTest(unittest.TestCase):
  
  def setUp(self):
    self.naive_bayes = NaiveBayes()
    with open('./data/jojo.dat','r') as file:
      for rec in file:
        name,serif = rec.strip().split("\t")
        self.naive_bayes.set_category(name)
        self.naive_bayes.set_word(serif)
        self.naive_bayes.learn()

  def test_result(self):
    ret = self.naive_bayes.classify('君が発表するまで勉強会を止めない！')
    self.assertEqual(ret,'ジョナサン')

    ret = self.naive_bayes.classify('おれはエンジニアをやめるぞ！ジョジョーーーーッ！！')
    self.assertEqual(ret,'ディオ')

    ret = self.naive_bayes.classify('おまえは今まで参加した勉強会をおぼえているのか？')
    self.assertEqual(ret,'ディオ')

    ret = self.naive_bayes.classify('てめーは仕様を複雑にした')
    self.assertEqual(ret,'承太郎')

if __name__ == '__main__':
  unittest.main()

exit
