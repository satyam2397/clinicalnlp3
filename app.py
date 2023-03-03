# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 17:30:53 2023

@author: LM835TP
"""
# Importing necessary libraries
import numpy as np
from flask import Flask, request, make_response
import json
import pickle
from flask_cors import cross_origin
import pickle,shutil, os
import re, itertools, string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
nltk.download('popular')

# Declaring the flask app
app = Flask(__name__)

#------------------------------removing special characters---------------------------------------#
def rem_spcl_char (text):
    x = re.sub(r'[^a-zA-Z\d\s]', u'', text, flags=re.UNICODE)
    return x    
#-----------------------------Lemmatize with POS Tag------------------------------------------#

# Define function to lemmatize each word with its POS tag --> POS - Part-of-Speech
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
# 1. POS - Part-of-Speech --> Noun
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

# 2. Init Lemmatizer
lemmatizer = WordNetLemmatizer()

# 3. Lemmatize a Sentence with the appropriate POS tag
def lemmatize_col(sentence):
    word_list = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sentence)]
    lemmatized_output = ' '.join([w for w in word_list])
    return lemmatized_output
#    return word_list
#-----------------------------removing stop words------------------------------------------#
sw_nltk = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

others_sw = ['please', 'get', 'problem','suggest', 'advise','help','take', 'also', 'sir', 'hello', 'hi' ,'hey', 'help']
extended_sw = sw_nltk + others_sw

def remove_sw (text):
    words = [word for word in text.split() if word.lower() not in extended_sw]
    new_text = " ".join([w for w in words])
    return new_text
#-------------------------------------------------------------------------------------------#

# geting and sending response to dialogflow
@app.route('/webhook', methods=['POST'])
@cross_origin()
def webhook():

    req = request.get_json(silent=True, force=True)
    res = processRequest(req)
    res = json.dumps(res, indent=4)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r

# processing the request from dialogflow
def processRequest(req):

    result = req.get("queryResult")
    
    #Getting the intent which has fullfilment enabled
    intent = result.get("intent").get('displayName')
    
    if (intent=='Default Welcome Intent - yes - problem'):
    
        queryText = result.get("queryText")
        queryText = queryText.strip().lower()
        queryText = queryText.translate(str.maketrans('', '', string.punctuation))
        queryText = remove_sw(queryText)
        queryText = rem_spcl_char(queryText)
        queryText = lemmatize_col(queryText)
        
        #-------------------------------------------------------------------------------------------#
        vectorizar = pickle.load(open('/tfidf/vectorizer.pkl', 'rb'))
        queryText_tfidf = vectorizar.transform([queryText])
        #-------------------------------------------------------------------------------------------#
    
        my_path = "/models_nlp/"
        file_list = os.listdir(my_path)
    
        res_final = pd.DataFrame()
        for model in file_list:
 #           print(model)
            try:
             log_classifier = pickle.load(open(model, 'rb'))
             prediction_prob = log_classifier.predict_proba(queryText_tfidf)
             if prediction_prob[0][1] >= 0.1:
                 res = pd.DataFrame(prediction_prob, columns = ['prob0', 'prob1'])
                 res['disease'] = model[0:-4]
                 res_final = res_final.append(res)
            except:
                print('fail')  
                    
            res_final = res_final.sort_values(['prob1'])
        fulfillmentText = ', '.join(res_final['disease'].tolist())
    return {"fulfillmentText": fulfillmentText}

#print("app_works")
if __name__ == '__main__':
    app.run()