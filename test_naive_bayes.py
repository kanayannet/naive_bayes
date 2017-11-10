
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
        self.naive_bayes.train(name,serif)

  def test_result(self):
    ret = self.naive_bayes.classify('君が発表するまで勉強会を止めない！')
    self.assertEqual(ret,'ジョナサン')

if __name__ == '__main__':
  unittest.main()

exit
