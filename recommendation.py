import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gensim import corpora, models, similarities
from gensim.models import TfidfModel
import re
from underthesea import word_tokenize
from surprise import SVD, dump

# Load data

df = pd.read_csv('Products_ThoiTrangNam_small.csv')
df1 = pd.read_csv('Products_ThoiTrangNam_rating_filtered.csv')
df2 = pd.read_csv('Products_ThoiTrangNam_raw.csv')

# load tfidf model
tfidf = TfidfModel.load('products.tfidf')

# load dictionary and corpus
dictionary = corpora.Dictionary.load('products.dict')
corpus = corpora.MmCorpus('products.mm')

# calculate similarity
index = similarities.MatrixSimilarity(tfidf[corpus])

# Load SVD algorith
_, algorithm = dump.load('SVD')

# define function to get similar products
def recommend_products(product_id, num_recommendations):
    # get the similarity of the product with all other products
    sim = index[tfidf[corpus[product_id]]]
    # sort the similarity scores in descending order
    sim_sorted = sorted(enumerate(sim), key=lambda x: x[1], reverse=True)
    # get the top n+1 most similar products (including the current product)
    sim_sorted = sim_sorted[:num_recommendations+1]
    # get the product ids and remove the current product
    product_ids = [x[0] for x in sim_sorted if x[0] != product_id]
    # get the similarity scores
    sim_scores = [x[1] for x in sim_sorted if x[0] != product_id]
    # get the product names
    product_names = [df['product_name'].iloc[x] for x in product_ids]
    # create a dataframe to store the results
    df_results = pd.DataFrame({'product_id': product_ids, 'product_name': product_names, 'similarity': sim_scores})
    return df_results

# get recommended products for entering text
def recommend_products_text(text, num_recommendations):
    # tokenize the text
    text_wt = word_tokenize(text, format="text")
    # load stop words
    file = open('vietnamese-stopwords.txt', 'r', encoding="utf8")
    stopwords_lst = file.read().split('\n')
    file.close()
    # remove stop words
    text_wt = ' '.join([word for word in text_wt.split() if word not in stopwords_lst])
    # remove special characters
    text_wt = re.sub(r'[^\w\s]', '', text_wt)
    # remove numbers
    text_wt = re.sub(r'\d+', '', text_wt)
    # split the text into words
    text_gem = [text_wt.split()]
    # create a corpus for the text
    text_corpus = [dictionary.doc2bow(text) for text in text_gem]
    # get the similarity of the text with all products
    sim = index[tfidf[text_corpus[0]]]
    # sort the similarity scores in descending order
    sim_sorted = sorted(enumerate(sim), key=lambda x: x[1], reverse=True)
    # get the top n most similar products
    sim_sorted = sim_sorted[:num_recommendations]
    # get the product ids
    product_ids = [x[0] for x in sim_sorted]
    # get the similarity scores
    sim_scores = [x[1] for x in sim_sorted]
    # get the product names
    product_names = [df['product_name'].iloc[x] for x in product_ids]
    # create a dataframe to store the results
    df_results = pd.DataFrame({'product_id': product_ids, 'product_name': product_names, 'similarity': sim_scores})
    return df_results

# create a function to recommend top 5 products for a user in a table
def recommend_products(user_id, n=5):
    user_recs = []
    for product_id in df1['product_id'].unique():
        user_recs.append((product_id, algorithm.predict(user_id, product_id).est))
    user_recs = sorted(user_recs, key=lambda x: x[1], reverse=True)
    return user_recs[:n]

# Title
st.title("Product Recommendation System")
menu = ["Home", "Content-based Filtering", "Collaborative Filtering"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Home':    
    st.subheader("Homepage")
elif choice == 'Content-based Filtering':  
    # GUI for content-based filtering
    st.subheader("## Content-based Filtering")
    # display the number of recommendations
    num_recommendations = st.slider("Number of recommendations", 1, 10, 5)
    # get recommended products for selecting a product
    product_id = st.selectbox("Select a product:", df['product_name'])
    if product_id:
        product_id = df[df['product_name'] == product_id].index[0]
        df_results = recommend_products(product_id, num_recommendations)
        st.write(df_results)

    # display pictures of recommended products
    st.write("### Recommended products:")
    for i in range(num_recommendations):
        st.write(df_results['product_name'].iloc[i])
        st.image(df['image'].iloc[df_results['product_id'].iloc[i]], width=200)

    # get recommended products for entering text
    text = st.text_input("Enter text:")
    if text:
        df_results = recommend_products_text(text, num_recommendations)
        st.write(df_results)

    # display pictures of recommended products
    st.write("### Recommended products:")
    for i in range(num_recommendations):
        st.write(df_results['product_name'].iloc[i])
        st.image(df['image'].iloc[df_results['product_id'].iloc[i]], width=200)

elif choice == 'Collaborative Filtering':  
    st.subheader("Collaborative Filtering")

    # Select user
    user_name = st.selectbox("Select a user:", df1['user'].unique())
    user_id = df1[df1['user'] == user_name].index[0]

    # get top 5 recommended products for the user
    user_recs = recommend_products(user_id)
    st.write("### Recommended products:")
    for i in range(5):
        st.write(df2['product_name'].iloc[user_recs[i][0]])
        # display pictures of recommended products, if there is a link to the image
        product_id = user_recs[i][0]
        matching_row = df2.loc[df2['product_id'] == product_id]

        if not matching_row.empty and matching_row['image'].values[0]:
            st.image(matching_row['image'].values[0], width=200)
        else:
            st.text('***No image available***')
        
        

