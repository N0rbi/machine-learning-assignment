import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import gensim
from gensim.models import Word2Vec

model = gensim.models.KeyedVectors.load_word2vec_format('./data/embedding/GoogleNews-vectors-negative300.txt', binary=False)

def w2v_centroid(row):
    text = row["Description"]
    # f = text.replace("\n", " ")
    word_representations = np.array([model[word] for word in text if word not in stopwords.words('english')])

    return np.mean(word_representations, axis=0)

df = pd.read_csv("data/train/train.csv")

df = df[["Description"]]

print(df.apply(w2v_centroid, axis=1))
