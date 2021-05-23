import nltk
content='''Stock (also capital stock) of a corporation, 
is all of the shares into which ownership of the corporation
 is divided. In American English, the shares are collectively
 known as "stock". A single share of the stock represents 
 fractional ownership of the corporation in proportion to the 
 total number of shares. This typically entitles the stockholder
 to that fraction of the company's earnings, proceeds from 
 liquidation of assets (after discharge of all senior claims 
                        such as secured and unsecured debt), 
 or voting power, often dividing these up in proportion to 
 the amount of money each stockholder has invested. Not all 
 stock is necessarily equal, as certain classes of stock may
 be issued for example without voting rights, with enhanced 
 voting rights, or with a certain priority to receive profits 
 or liquidation proceeds before or after other classes of 
 shareholders.'''
 
# tokenization 
#tokenization of sentences
sentences=nltk.sent_tokenize(content)
print(len(sentences))

# tokenization of sentence into words
words=nltk.word_tokenize(content)
print(len(words)) # a total of 158 words present in the dataset

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
ps=PorterStemmer() 
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

# NLTK corpus readers. The modules in this package provide functions that can be used to read corpus files in a variety of formats.
 # These functions can be used to read both the corpus files that are distributed in the NLTK corpus package,
 # and corpus files that are part of external corpora.
sw=stopwords.words('english')
print(len(sw))

## data cleansing ##

import re  # regular expression
corpus=[]
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]   
    review = ' '.join(review)
    corpus.append(review)


# stemming and lematization
#stemming: which helps to find the root word in the sentences

for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words) 
    
#  lematization code     
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)


# feature engineering techniques

# bag of words
from sklearn.feature_extraction.text import CountVectorizer 
# here there is a problem in bag of words
#if same word repeated in the all the sentences it will 1 so as other as 0
#to over come this td-and idf is used
cv=CountVectorizer()
X=cv.fit_transform(corpus).toarray()

from sklearn.feature_extraction.text import TfidfVectorizer
tv=TfidfVectorizer()
Y=tv.fit_transform(corpus).toarray()
# now  you dont have a same value for same words in the all the sentences
# log fucntion is applied to the text data 

# both there techniques doesnt not share any similar information 
# example: kind and queen they simantic 
# boy and girl belongs to gender they share similar information 





