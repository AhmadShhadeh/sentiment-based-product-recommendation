# import libraties
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
#--- HTML Tag Removal
import re 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import nltk



class Recommendation:
    
    def __init__(self):
        nltk.data.path.append('./nltk_data/')
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        self.data = pickle.load(open('data.pkl','rb'))
        self.user_final_rating = pickle.load(open('user_final_rating.pkl','rb'))
        self.model = pickle.load(open('logistic_regression.pkl','rb'))
        self.raw_data = pd.read_csv("sample30.csv")
        self.data = pd.concat([self.raw_data[['id','name','brand','categories','manufacturer']],self.data], axis=1)
        
        
    def getTopProducts(self, user):
        items = self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index
        tfs=pd.read_pickle('tfidf.pkl')
        temp=self.data[self.data.id.isin(items)]
        X = tfs.transform(temp['Review'].values.astype(str))
        temp=temp[['id']]
        temp['prediction'] = self.model.predict(X)
        temp['prediction'] = temp['prediction'].map({'Postive':1,'Negative':0})
        temp=temp.groupby('id').sum()
        temp['positive_percent']=temp.apply(lambda x: x['prediction']/sum(x), axis=1)
        final_list=temp.sort_values('positive_percent', ascending=False).iloc[:5,:].index
        return self.data[self.data.id.isin(final_list)][['id', 'brand',
                              'categories', 'manufacturer', 'name']].drop_duplicates().to_html(index=False)

    def getTopProductsNew(self, user):
        items = self.user_final_rating.loc[user].sort_values(ascending=False)[0:20].index
        tfs=pd.read_pickle('tfidf.pkl')
        #mdl=pd.read_pickle('final_lr.pkl')
        #features = pickle.load(open('features.pkl','rb'))
        #vectorizer = TfidfVectorizer(vocabulary = features)
        temp=self.data[self.data.id.isin(items)]
        X = tfs.transform(temp['Review'].values.astype(str))
        temp=temp[['id']]
        temp['prediction'] = self.model.predict(X)
        temp['prediction'] = temp['prediction'].map({'Postive':1,'Negative':0})
        temp=temp.groupby('id').sum()
        temp['positive_percent']=temp.apply(lambda x: x['prediction']/sum(x), axis=1)
        final_list=temp.sort_values('positive_percent', ascending=False).iloc[:5,:].index
        return self.data[self.data.id.isin(final_list)][['id', 'brand',
                              'categories', 'manufacturer', 'name']].drop_duplicates().to_json(orient="table")

    def getUsers(self):
        s= np.array(self.user_final_rating.index).tolist()
        #print(s)
        return ''.join(e+',' for e in s) 

    def nltk_tag_to_wordnet_tag(self,nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def lemmatize_sentence(self,sentence):
        #tokenize the sentence and find the POS tag for each token
        snow = SnowballStemmer('english') 
        lemmatizer = nltk.stem.WordNetLemmatizer()
        wordnet_lemmatizer = WordNetLemmatizer()
        nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
        #tuple of (token, wordnet_tag)
        wordnet_tagged = map(lambda x: (x[0], self.nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            if tag is None:
                #if there is no available tag, append the token as is
                lemmatized_sentence.append(snow.stem(word)) # stem the word if no lemma is obtaines
            else:
                #else use the tag to lemmatize the token
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        return " ".join(lemmatized_sentence)

    def analyiseSentiment(self,text):
            tfs=pd.read_pickle('tfidf.pkl')
            mdl=pd.read_pickle('final_lr.pkl')
            #preprocess text

            # remove html
            p = re.compile('<.*?>')#Find this kind of pattern
            text=p.sub('',text)

            #--- Punctuation removal
            p = re.compile(r'[?|!|\'|"|#|.|,|)|(|\|/|~|%|*]')#Find this kind of pattern
            text=p.sub('',text)

            stop = stopwords.words('english') #All the stopwords in English language

            text=self.lemmatize_sentence(text)

            sent_T=tfs.transform([text])

            return mdl.predict(sent_T)[0];