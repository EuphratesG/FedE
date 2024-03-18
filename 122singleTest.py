from nltk.corpus import wordnet as wn
results = wn.synsets('skype com')  #得到与bank所有的同义词
for i in range(len(results)):
    print(results[i].definition())