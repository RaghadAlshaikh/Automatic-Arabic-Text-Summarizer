# -- coding: utf-8 --
"""
Created on Sat Feb 22 23:53:51 2020

@author: Raghad Alshaikh, Ghaidaa Aflah, Nada Alamouadi
"""


import gensim
import re
import numpy as np
from nltk.stem.isri import ISRIStemmer
from urllib import request 
from bs4 import BeautifulSoup as bs
import nltk 
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from sklearn import cluster
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans


# ============================   
# ====== N-Grams Models ======
#Get the pretrained model
t_model = gensim.models.Word2Vec.load('full_grams_sg_100_wiki/full_grams_sg_100_wiki.mdl')
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
# ====== PreProcessing: Aracic Stemmer====== 
st = ISRIStemmer()
# ============================     
# ====== PreProcessing: Tokenization======  
words_tokens = nltk.word_tokenize(allparagraphContent_Cleaned)
# remove out-of-vocabulary words
# ====== remove out-of-vocabulary words====== 
words_tokens = [word for word in words_tokens if st.stem(word) in t_model.wv.vocab]
# ====== PreProcessing: Sentence splitting======  
sentences_tokens = nltk.sent_tokenize(allparagraphContent_Cleaned)
print("sentences number: ",len(sentences_tokens))
print("\nOriginal input sentences: ")
for sen in sentences_tokens:
    print("\n", sen)
# ====== PreProcessing: StopWords defined======  
stopwords_list = stopwords.words('arabic')
# ====== PreProcessing: Stemming======  
words_stemm = [st.stem(word) for word in words_tokens]
words_stemm = [st.stem(word) for word in nltk.word_tokenize(sen) if st.stem(word) not in stopwords_list]

# ====== From word vectors to sentence vector using Sum&Average======  
# ====== Purpose: Get the sentences vectors to process======  
senCount=0;
sentenceVec = {}
tsne = TSNE(n_components=2)
for sen in sentences_tokens:
    sen = [st.stem(word) for word in nltk.word_tokenize(sen) if st.stem(word) in t_model.wv.vocab]
    if len(sen) >= 1:
        sentenceVec[senCount]=np.mean(t_model.wv[sen], axis=0)
        senCount+=1
 
    
NUM_CLUSTERS=2
kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
newSenVec = []
for i in sentenceVec:
    newSenVec.append(sentenceVec[i])
#print(len(newSenVec))
print("\n\n")
#=======================
#=======================
#Calculate the clusters No.
#n_clusters = int(np.ceil(len(newSenVec)**0.5))
n_clusters=NUM_CLUSTERS
#Create the KMEANS 
kmeans2 = KMeans(n_clusters=n_clusters, random_state=0)
#=======================
#Fit the sentences to each cluster
#=======================
kmeans2 = kmeans2.fit(newSenVec)
avg2 = []
closest2 = []
for j in range(n_clusters):
    idx = np.where(kmeans2.labels_ == j)[0]
    avg2.append(np.mean(idx))
closest, _ = pairwise_distances_argmin_min(kmeans2.cluster_centers_,newSenVec)
ordering = sorted(range(n_clusters), key=lambda k: avg2[k])
summary2 = '\n\n'.join([sentences_tokens[closest[idx]] for idx in ordering])

#=======================
#Fit the sentences to each cluster
#=======================
kmeans.fit(newSenVec)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
n_clusters1 = n_clusters
avg1 = []
for j in range(n_clusters1):
    idx1 = np.where(kmeans.labels_ == j)[0]
    avg1.append(np.mean(idx1))
ordering = sorted(range(n_clusters1), key=lambda k: avg1[k])
#Print Summary based on first appearance 
print('Summary based on the first appearance of each cluster')  
appearedLabesl = [] 
index = 0
for i in labels:
    if i not in appearedLabesl:
        print(sentences_tokens[index], "\n")
    appearedLabesl.append(i)
    index = index+1
#Print Summary based on the closet to the center of each cluster
print("Summary based on the closet to the center of each cluster")
print(summary2)
