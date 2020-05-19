# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 18:06:47 2020

@author: Raghad Alshaikh, Ghaidaa Aflah, Nada Alamouadi
"""

from urllib import request 
from bs4 import BeautifulSoup as bs
import re
import nltk 
from nltk.corpus import stopwords
import heapq
from nltk.stem.isri import ISRIStemmer

# ============================ 
# ====== Load input ======
#Store the article URL
url = "https://ar.wikipedia.org/wiki/%D8%A7%D9%84%D9%83%D9%88%D9%86"
allparagraphContent = ""
#Access the article URL
htmlArticle = request.urlopen(url)
#Get the article code including all the HTML tags
Soup = bs(htmlArticle, 'html.parser')
paragraphContents = Soup.findAll('p')
#Get the text 
for paragraphContent in paragraphContents:
    allparagraphContent += paragraphContent.text
# ============================     
# ====== Clean input by removing all the HTML tags information======  
allparagraphContent_Cleaned = re.sub(r'\[0-9]*\]',' ',allparagraphContent)
allparagraphContent_Cleaned = re.sub(r'\s+',' ',allparagraphContent)
allparagraphContent_Cleaned = re.sub(r'\[^a-zA-Z]',' ',allparagraphContent)
allparagraphContent_Cleaned = re.sub(r'\s+',' ',allparagraphContent)
# ============================     
# ====== PreProcessing: Sentence Splitting======  
sentences_tokens = nltk.sent_tokenize(allparagraphContent_Cleaned)
# ============================     
# ====== PreProcessing: Tokenization======  
words_tokens = nltk.word_tokenize(allparagraphContent_Cleaned)
# ============================     
# ====== PreProcessing: Arabic StopWords List======  
stopwords_list = stopwords.words('arabic')
# ============================     
# ====== PreProcessing: Arabic Stemming======  
st = ISRIStemmer()
words_stemm = [st.stem(word) for word in words_tokens]
# ============================     
# ====== Calculate Each Term Frequency======  
word_frequencies = {}
for word in words_stemm:
    if word not in stopwords_list:
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1
#Get MAX
maximum_frequency_word = max(word_frequencies.values())
#Normalize
for word in word_frequencies.keys():
    word_frequencies[word] = (word_frequencies[word]/maximum_frequency_word)
# ============================     
# ====== Calculate Each Sentence Frequency from the Term Frequencies======     
#Calculate Sentence scores based on each word within sentence
sentences_scores = {}
wordsCounter = 0.0
for sentence in sentences_tokens:
    for word in nltk.word_tokenize(sentence):
        word = st.stem(word)
        if word in word_frequencies.keys():
            wordsCounter += 1;
            if sentence not in sentences_scores.keys():
                sentences_scores[sentence] = word_frequencies[word]
            else:
                sentences_scores[sentence] += word_frequencies[word]
    #======Normalize======
    sentences_scores[sentence] = sentences_scores[sentence]/wordsCounter
    #Reset Counter
    wordsCounter = 0
# ============================     
# ====== Print Summary======  
#Get summary with only highest top 3 
print("Number of tokens: ",len(words_tokens))
print("Number of sentences: ",len(sentences_tokens))    
summary = heapq.nlargest(3, sentences_scores, key=sentences_scores.get)
print("\nBag of Words based summary:\n")
print(summary)

    


    