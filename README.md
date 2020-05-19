# Arabic Text Summarization:  Text Highlighter for Important Information in Arabic Text

Arabic is the fifth most widely spoken language worldwide; it is one of the languages
that have not been affected by changes through centuries as it is the language of Quran.
In these recent years, automatic text summarization showed a continuous and
remarkable development due to its importance and applications. However, the number
of research studies investigating Arabic text summarization is relatively small
comparing to other languages. In this project, we propose to develop a text highlighter
tool that allows the user to receive summarized feedback on the given Arabic text using
EASC (Essex Arabic Summaries Corpus). The proposed system will take a single
Arabic document text as input and returns the top-ranked sentences highlighted as its
final result. It is implemented based on three different approaches, the first approach
that achieved an accuracy of 59% is the Bag of Words (BoW), the second approach
that achieved an accuracy of 60% is the Term Frequency - Inverse Document
Frequency (TF-IDF), and the third approach that achieved an accuracy of 56% for the
first appearance-based and 48% for the closest to the center-based is Word2Vec: SkipGram model with the K-Means clustering.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

**The libraries used for the Bag of Words (BoW) summarizer:**

* To access the single article URL ling
```
from urllib import request 
```

* To extract the web information (article content)
```
from bs4 import BeautifulSoup as bs
```

* To Clean the web information (article content)
```
import re
```

* To import the sentence and words tokenizer
```
import nltk 
```

* To import the Arabic stopwords list
```
from nltk.corpus import stopwords
```

* To import the Arabic stemmer
```
from nltk.corpus import stopwords
```

**The libraries used for Term Frequency-Inverse Document Frequency (TF-IDF) summarizer:** 

* To import the sentence and words tokenizer
```
import nltk 
```

**The libraries used for the Word2Vec summarizer:** 

Install the pre-trained model for the 100 vector length skip-gram (wikipedia) and the utility file
```
github.com/bakrianoo/aravec
install: full_grams_sg_100_wiki
```

* To access the single article URL ling
```
from urllib import request 
```

* To extract the web information (article content)
```
from bs4 import BeautifulSoup as bs
```

* To Clean the web information (article content)
```
import re
```

* To import the sentence and words tokenizer
```
import nltk 
```

* To import the Arabic stopwords list
```
from nltk.corpus import stopwords
```

* To import and use the KMEANS cluster
```
from sklearn.manifold import TSNE
from sklearn import cluster
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
```

## Deployment

The python files are integrated as a CGI scripts with the web files that were developed using HTML5\CSS3\JavaScript\PHP and deployed except for the Word2Vec model on the following link:
http://arabic.highlight.heliohost.org/


## Built With

* [Anaconda Spyder](https://www.spyder-ide.org/) - The Python environment used
* [heliohostRicky](heliohost.org) - The webhost server
* [VisualStudioCode](https://code.visualstudio.com/) - The web code editor

## Authors

* **Raghad Alshaikh** - [RaghadAlshaikh](https://github.com/RaghadAlshaikh)
* **Ghaidaa Aflah**   - [GhaidaaAflah](www.linkedin.com/in/ghaida-aflah-7a241a17a)
* **Nada Alamouadi**  - [NadaAlamouadi](nood5925@gmail.comg)

Supervised By: **Dr.Amal Almansour** - [AmalAlmansour](aalmansour@kau.edu.sa)

## Acknowledgments

* The Arabic Word2Vec Pre-trained model: github.com/bakrianoo/aravec
* The EASC dataset: sourceforge.net/projects/easc-corpus/
* etc
