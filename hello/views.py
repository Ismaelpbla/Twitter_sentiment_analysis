from django.shortcuts import render
from django.http import HttpResponse
#from urllib3 import HTTPResponse

from .models import Greeting

import pandas as pd
import pickle

import os

# Create your views here.
def index(request):
    # return HttpResponse('Hello from Python!')
    return render(request, "index.html")

def pipeline(request):

    def signs_tweets(tweet):
        import re

        signos = re.compile("(\.)|(\;)|(\:)|(\!)|(\?)|(\¿)|(\@)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
        return signos.sub('', tweet.lower())

    def remove_links(df):
        return " ".join(['{link}' if ('http') in word else word for word in df.split()])

    def remove_stopwords(df):
        from nltk.corpus import stopwords   
        spanish_stopwords = stopwords.words('spanish')
        return " ".join([word for word in df.split() if word not in spanish_stopwords])

    def spanish_stemmer(x):
        from nltk.stem.snowball import SnowballStemmer
        stemmer = SnowballStemmer('spanish')
        return " ".join([stemmer.stem(word) for word in x.split()])

    tweet = str(request.GET.get('tweet', 'Hello'))

    text = pd.Series(tweet)
    test_clean = pd.DataFrame(text, columns=['content'])
    test_clean['content_clean'] = test_clean['content'].apply(signs_tweets)
    test_clean['content_clean'] = test_clean['content_clean'].apply(remove_links)
    test_clean['content_clean'] = test_clean['content_clean'].apply(remove_stopwords)
    test_clean['content_clean'] = test_clean['content_clean'].apply(spanish_stemmer)
    
    with open('hello/finished_model.model', "rb") as archivo_entrada:
        pipeline_importada = pickle.load(archivo_entrada)

    predictions = pipeline_importada.predict_proba(test_clean['content_clean'])

    print(predictions)
    #test_clean['Polarity'] = pd.Series(predictions)
    #response =  test_clean['Polarity']

    return HttpResponse([predictions, tweet, test_clean['content_clean'][0]])
    