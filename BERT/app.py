#Import libraries
from io import BytesIO
import requests
from itertools import product
from gsheetsdb import connect
import pandas as pd
import streamlit as st
from pyngrok import ngrok
import numpy as np
import gspread
import gspread_dataframe as gd
from google.oauth2 import service_account
from google.auth.transport.requests import AuthorizedSession
import tensorflow
import numpy as np
import pandas as pd
import df2gspread as d2g
import matplotlib.pyplot as plt
import re
from bert_serving.client import BertClient
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
def main(question_orig):
  #%tensorflow_version 1.x
  url = 'https://docs.google.com/spreadsheets/d/1oDEjBsdmlr6bBwUVy0UVJen80xIZLmdPbypQPmEgh3w'
  data = pd.read_csv(str(url) + '/export?gid=0&format=csv')
  data = data.dropna()
  bc = BertClient() 
  def retrieveAndPrintAnswer(question_embedding, sentence_embeddings, AQdf, sentences):
    max_sim = -1
    index_sim = -1
    for index, faq_embedding in enumerate(sentence_embeddings): 
      #using cosine similarity
      sim = cosine_similarity(faq_embedding, question_embedding)[0][0] 
      print(index, sim, sentences[index])
      if sim>max_sim:
        max_sim=sim
        index_sim = index
    #print("Questions: ", question)
    #print("Retrieved: ",AQdf.iloc[index_sim,0])
    return AQdf.iloc[index_sim,1]

  def clean_sentences(sentence, stopwords=False):
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^a-z0-9\s]','',sentence) #removing alphanumeric charaters

    if stopwords:
      sentence = remove_stopwords(sentence) #remove stopwords
    return sentence
  def get_cleaned_sentences(df, stopwords=False):
    sents = data[['Question']]
    cleaned_sentences=[]

    for index,row in data.iterrows():
      cleaned =clean_sentences(row['Question'],stopwords)
      cleaned_sentences.append(cleaned)
    return cleaned_sentences
  question = clean_sentences(question_orig, stopwords=False)
  cleaned_sentences = get_cleaned_sentences(data,stopwords=False)
  sent_bertphase_embeddings=[]
  for sent in cleaned_sentences:
    sent_bertphase_embeddings.append(bc.encode([sent]))
  question_embedding = bc.encode([question])
  return retrieveAndPrintAnswer(question_embedding,sent_bertphase_embeddings, data ,cleaned_sentences)
  
# creating data frame
df = pd.DataFrame(data=[[30, 31, 32, 33, 37, 45, 49, 55, 60, 67, 70, 42, 78, 38, 44, 49, 50, 53], 
                        [0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1], 
                        [0,0,0,1,0,1,0,0,0,1,1,1,1,1,0,0,0,1],
                        [1,0,1,0,0,0,1,1,1,1,1,0,0,0,1,1,1,1]]).T
df.columns=['Age', 'Gender', 'Diabetes', 'Hypertension']
columns=['Age', 'Gender', 'Diabetes', 'Hypertension']
# apply product method
dfl = list(product(df['Age'], df['Gender'], df['Diabetes'], df['Hypertension']))
df = pd.DataFrame(dfl, columns = columns)
df['Totalscore'] = (df['Age']/10)+df['Diabetes']+df['Hypertension']
x = df.drop('Totalscore', axis =1)
y = df['Totalscore']

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
clf = LinearRegression()
clf.fit(x, y)


st.title('Assure')
st.subheader('Bajaj Finserve Health')
col1, col2, col3 = st.beta_columns([1,6,1])
with col1:
  st.write("")
with col2:
  st.image(plt.imread('/content/bhealth-purple-white.png'))
with col3:
  st.write("")
st.markdown('Team Data Another Day')
st.markdown('A chatbot that not just talks but connects to you!')
@st.cache(allow_output_mutation=True)
def get_data():
    return []
question_orig = st.text_input("Hi! I'm Assure!, Please enter your question!", 'hi')
st.write(main(question_orig))

credentials = service_account.Credentials.from_service_account_file(
    '/content/jsonFileFromGoogle.json')

scoped_credentials = credentials.with_scopes(
        ['https://spreadsheets.google.com/feeds',
         'https://www.googleapis.com/auth/drive']
        )

gc = gspread.Client(auth=scoped_credentials)
gc.session = AuthorizedSession(scoped_credentials)
sheet = gc.open_by_key('1Dx-2_WE1cWcDf1K3NeeNRx21drf0L7trxfbr9KRynqc')

get_data().append({"Question": question_orig, "Answer": main(question_orig)})

ws = gc.open_by_key("1Dx-2_WE1cWcDf1K3NeeNRx21drf0L7trxfbr9KRynqc").worksheet("Sheet1")
gd.set_with_dataframe(ws, pd.DataFrame(get_data()))

if main(question_orig) == "Sure, I will recommend you a plan, please tell me the following details":
  age = st.text_input('Please enter your Age',"")
  age = np.array([int(age)])
  gender  = st.text_input('Please enter your Gender as M or F',"")
  if gender == 'M':
    gender = np.array([0])
  else:
    gender = np.array([1])
  diab = st.text_input('Do you have Diabetes? enter as Yes or No', "")
  if diab == 'No' or diab =='no':
    diab = np.array([0])
  else:
    diab = np.array([1])
  hype = st.text_input('Do you have Hypertension? enter as Yes or No', "")
  if hype == 'No' or diab =='no':
    hype = np.array([0])
  else:
    hype = np.array([1])
  test = pd.DataFrame()
  test['Age'] = age
  test['Gen'] = gender
  test['Diab'] = diab
  test['Hype']  = hype
  pred = []
  pred.append(round(clf.predict(test)[0]))
  if pred[0]<4:
    plan = 'Health Basic'
  if pred[0]>=4 and pred[0] <8:
    plan = 'Health Plus'
  if pred[0]>8:
    plan = 'Health Advanced'
  st.write('The best plan for you is:', plan)

st.text('Response History')
st.write(pd.DataFrame(get_data()))
st.text('Thank you!')
