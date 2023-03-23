#Sentiment Analysis
#Rajat Jain

import numpy as np
import pandas as pd

# importing streamlit framework
import streamlit as st
import requests 
from streamlit_lottie import st_lottie

#loading the dataset and preprocessing it
df = pd.read_csv(r'Review.csv')
df.dropna(inplace=True)

X_train = df['Review'].tolist()
Y_train = df['Sentiment'].tolist()


#Initiaizing the web app

st.set_page_config(page_title="Sentiment Analysis", layout='wide')

#funtion for loading the lottie file
def lottie_url(url):
        r= requests.get(url)
        if r.status_code != 200:
                return None
        return r.json()

col1, col2 = st.columns(2)

with col1:
        st.title("Sentiment Analysis")
        st.subheader("This is a web app used for the sentiment analysis.")
        st.text("Given a review, it can be automatically classified in ") 
        st.text("categories.")
        st.text("These categories can be user defined (positive, negative)")
        st.text("It is a special case of text mining generally focused on")
        st.text("identifying opinion polarity,")
        st.text("So, enter the review and try to find the sentiment...")


with col2:
        lot = lottie_url("https://assets7.lottiefiles.com/private_files/lf30_lyeigshx.json")
        st_lottie(lot,height=300)

#Preprocessing of the text
#Using nltk libraries
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
    
nltk.download('stopwords') #dowloading stopwords
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')

st.write("---")

#Taking the input from the user in web app
X_test = []
st.markdown("<h5 style='text-align: left; color: #FF4B4B;'>Enter review to analyse</h5>", unsafe_allow_html=True)
input = st.text_input(' ')
X_test.append(input) 



#creating the function which do the text processing

def getCleanedText(text):
        #converting the text into lower case
        text = text.lower()
        
        #converting the text into tokens
        tokens = tokenizer.tokenize(text)
        
        #removing stopwords from the sentence and make the list containing the modified or tokenized string
        new_tokens = [token for token in tokens if token not in stop_words]
        
        #stemming
        ps = PorterStemmer()
        stem_tokens= [ps.stem(tokens) for tokens in new_tokens]
        
    #     #lemmetization if needed
    #     lemmatizer = WordNetLemmatizer()
    #     lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stem_tokens
        
        return " ".join(stem_tokens)
        
        

#calling the funtion on the data set as well as the testing statement       
X_clean = [getCleanedText(i) for i in X_train ]
Xt_clean = [getCleanedText(i) for i in X_test ]


#Vectorization 
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features=2500)
X = cv.fit_transform(X_clean).toarray()


Xt_vect = cv.transform(Xt_clean).toarray()


#Applying the model using the scikit learn library
# Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB

mn = MultinomialNB()
mn.fit(X, Y_train)
MultinomialNB()
y_pred = mn.predict(Xt_vect)

btn = st.button('Enter')

#writing the funtionalities of the button
if btn:
        if(y_pred[0] == 1.0):
                st.markdown("<div style='text-align: center; color: #FF4B4B;'>The review is positive</div>", unsafe_allow_html=True)
                thumb = lottie_url("https://assets8.lottiefiles.com/temp/lf20_NK6qfT.json")
                st_lottie(thumb,height=100)

        else:
                st.markdown("<div style='text-align: center; color: #FF4B4B;'>The review is negative</div>", unsafe_allow_html=True)
                thumbd = lottie_url("https://assets3.lottiefiles.com/packages/lf20_tc0vLd.json")
                st_lottie(thumbd,height=100)


st.write("---")


