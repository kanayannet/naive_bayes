
import sys
sys.path.append('./lib')
from naive_bayes import NaiveBayes

  
naive_bayes = NaiveBayes()
with open('./data/jojo.dat','r') as file:
  for rec in file:
    name, serif = rec.strip().split("\t")
    naive_bayes.category = name
    naive_bayes.word = serif
    naive_bayes.learn()

res = naive_bayes.classify(sys.argv[1])
print(res)

exit
