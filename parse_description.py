import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import gensim
from gensim.models import Word2Vec
import enchant
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')
# model = gensim.models.KeyedVectors.load_word2vec_format('w2v_tiny.txt', binary=False)

model = gensim.models.KeyedVectors.load_word2vec_format('./data/embedding/GoogleNews-vectors-negative300.txt', binary=False)

meta = {
"wc": 0,
"numerical": 0,
"non_english": 0,
"stopword": 0,
"not_in_w2v": 0
}

def w2v_centroid(row):
    text = row["Description"]
    text = str(text).lower()
    meta["wc"] += len(text)
    tokenizer = RegexpTokenizer(r'\b[^\d\W]+\b')
    text_tokenized = tokenizer.tokenize(text)
    meta["numerical"] += len(text) - len(text_tokenized)
    en_dict = enchant.Dict("en_US")
    text_en = [word for word in text_tokenized if en_dict.check(word)]
    meta["non_english"] += len(text_tokenized) - len(text_en)
    text_non_sw = [word for word in text_en if word not in stopwords.words('english')]
    meta["stopword"] += len(text_en) - len(text_non_sw)
    word_representations = []
    for word in text_non_sw:
        try:
            word_representations.push(model[word])
        except:
            meta["not_in_w2v"] += 1

    if len(word_representations) == 0:
        word_representations = np.zeros(shape=(1,300))
    word_representations = np.array(word_representations)

    return np.mean(word_representations, axis=0)

df = pd.read_csv("data/train/train.csv")

df = df[["Description"]]

# print(df.apply(w2v_centroid, axis=1))
df =  df.head(10)
sentence_centroids = df.apply(w2v_centroid, axis=1)
pca = PCA(n_components=1)
sentence_centroids_1d = pca.fit_transform(np.array([s for s in sentence_centroids]))

df = pd.read_csv("data/train/train.csv")
df = df[["AdoptionSpeed"]]
df["SentenceCentroids"] = pd.Series(sentence_centroids_1d.reshape(sentence_centroids_1d.shape[0]))

sns.stripplot(data=df, x="AdoptionSpeed", y="SentenceCentroids")
plt.savefig("description.pdf")
plt.savefig("description.png")
print(meta)
