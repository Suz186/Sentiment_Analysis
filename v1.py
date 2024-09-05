#Using VADER (Valence Aware Dictionary Reasoner) _ Bag of Words Approach

#Read data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


plt.style.use('ggplot')

import nltk

df=pd.read_csv("/Users/snehathomas/DataSciencePr/Sentiment_Analysis/Reviews.csv")

print(df.shape)
df = df.head(500)

#Streamlit App

st.title('Sentiment Analysis')

ax=df['Score'].value_counts().sort_index() \
    .plot(kind='bar', title ='Count of Reviews by Stars', figsize=(10,5))
ax.set_xlabel("Review Stars")

st.pyplot(plt)

