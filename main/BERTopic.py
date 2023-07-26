import logging
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
import spacy
from bertopic import BERTopic
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Set the environment variables
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

# Ignore warning messages
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# to ensure reproducibility
# BERTopic uses HDBSCAN for clustering. HDBSCAN is a density-based clustering
# algorithm, and its results may not be fully reproducible even with a fixed random seed.
np.random.seed(42)
hdbscan_params = {'min_cluster_size': 10, 'metric': 'euclidean', 'cluster_selection_method': 'eom'}
spacy.cli.download("en_core_web_sm")
nlp = spacy.load('en_core_web_sm')

# loading the data file
df = pd.read_csv('/Users/busracaliskan/Desktop/Thesis Data/Twitter Jan Mar.csv')

# Copy dataset just in case
df_bert = df.copy()

# PREPARING DATA SET

# dropping the rows that contain NA values
df_dropped = df_bert.dropna()
df_dropped.shape

# To check if there are duplicate values in the "content" column
duplicates = df_dropped.duplicated(subset='content')

# Remove the duplicate rows
df_unique = df_dropped.drop_duplicates(subset='content')

# Check the DataFrame shape without duplicates
print('df_unique shape:', df_unique.shape)

# Sort dataset by retweet count
sorted_df = df_unique.sort_values(by='retweet_count', ascending=False)

# Reset the index from the beginning
sorted_df.reset_index(drop=True, inplace=True)

# Select the first 30000 tweets for analysis
new_df = sorted_df.head(30000)
print('new_df shape:', new_df.shape)

# Convert the content to lowercase
new_df.loc[:, 'content'] = new_df['content'].str.lower()

# Remove hashtags
new_df.loc[:, 'content'] = new_df['content'].apply(lambda x: re.sub(r'#\w+', '', x))

# Remove mentions
new_df.loc[:, 'content'] = new_df['content'].apply(lambda x: re.sub(r"@\w+", "", x))

# Remove URLs and HTTPs
url_pattern = r"http[s]?:\/\/\S+"
new_df.loc[:, 'content'] = new_df['content'].str.replace(url_pattern, "", regex=True)


# Remove punctuation from the 'content' column
def remove_punctuation(text):
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    return cleaned_text


new_df.loc[:, 'content'] = new_df['content'].apply(remove_punctuation)

# Remove numbers from the 'content' column
new_df.loc[:, 'content'] = new_df['content'].str.replace('\d+', '', regex=True)


# Remove emojis from text
def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # Emojis
                               u"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
                               u"\U0001F680-\U0001F6FF"  # Transport & Map Symbols
                               u"\U0001F1E0-\U0001F1FF"  # Flags (iOS)
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


new_df.loc[:, 'content'] = new_df['content'].apply(remove_emojis)

# Tokenize contents and remove stopwords from the 'content' column
stop_words = set(stopwords.words('english'))
new_df.loc[:, 'content'] = new_df['content'].apply(
    lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))

# Extra words to remove
new_df['content'] = new_df['content'].apply(lambda x: re.sub(r'\b(chatgpt|chat gpt|gpt|amp)\b', '', x))


# Lemmatization
def lemmatizer(text, nlp, length_list):
    sent = []
    doc = nlp(text)
    for word in doc:
        sent.append(word.lemma_)
    length_list.append(len(sent))
    return " ".join(sent)


length_list = []  # List to store the length of each lemmatized text
new_df.loc[:, "lemmatized"] = new_df["content"].apply(lambda x: lemmatizer(x, nlp, length_list))

texts = new_df["lemmatized"]

# RUNNING THE MODEL (BERTopic)

# Start the timer to calculate actual running time
start_time = time.time()

# Set the language to English and use the hdbscan_params dictionary
topic_model = BERTopic(language="english", umap_model=42, hdbscan_model=hdbscan_params, calculate_probabilities=True,
                       verbose=True)

# Configure the logging module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create the BERTopics
# Add the logging statements to the code
logging.info('Starting BERTopic fitting and transformation...')
topic_model.fit(texts)
logging.info('Starting BERTopic fitting and transformation...1')
topics, probabilities = topic_model.transform(texts)
logging.info('BERTopic fitting and transformation completed.')

# Stop the timer
end_time = time.time()

# Calculate the running time
running_time = end_time - start_time

print("Running time:", running_time)

# Save model
# topic_model.save("bert_model")

# Load model
# bert_model = BERTopic.load("bert_model")


print(topic_model.get_topic_info())

# Get the topic information
topic_info = topic_model.get_topic_info()

print(topic_info.head())
print(topic_info.columns)

# visualize topics with inter-topic distance map
fig = topic_model.visualize_topics()
fig.write_html("/Users/busracaliskan/IdeaProjects/Thesis_BERTopic/visualization.html")

# Visualize topic hierarchy to make informed decisions about topic reduction and the number of topics to retain
# based on the structure and relationships observed in the dendrogram.
fig = topic_model.visualize_hierarchy(top_n_topics=50)
# Save the figure as an HTML file
fig.write_html("/Users/busracaliskan/IdeaProjects/Thesis_BERTopic/topic_hierarchy.html")

# Perform hierarchical topic reduction

# reduced_topics = topic_model.reduce_topics(texts, nr_topics="auto")  # to perform automated topic number reduction

reduced_topics = topic_model.reduce_topics(texts, nr_topics=50)  # force to reduce topic number to 50

print(reduced_topics.get_topic_info())

# Get the topic information
topic_info = reduced_topics.get_topic_info()

# Setting display to see all  columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)

# Print top 10 topics with keywords and probabilities (we'll print head 11 since -1 shows outliers)
top_10_topics = topic_info.head(11)

# Print top 10 topics with all columns
print(top_10_topics)
