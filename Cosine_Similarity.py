from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from nltk.corpus import stopwords
import numpy as np
import numpy.linalg as LA

import pandas as pd

data=pd.read_csv('dummy-data-2.csv')

docs=data['Name'].tolist()

cx = lambda a, b : round(np.inner(a, b)/(LA.norm(a)*LA.norm(b)), 3)
def generate_cosine_similarity(docs):
    stopWords = stopwords.words('english')    
    vectorizer = CountVectorizer(stop_words = stopWords)
    transformer = TfidfTransformer()
    vect_docs = vectorizer.fit_transform(docs).toarray()
    vect_docs = transformer.fit_transform(vect_docs).toarray()
    sim_df=[]
    for i,x in enumerate(vect_docs):
        for j,y in enumerate(vect_docs):
            cosine = cx(x, y)
            sim_df.append([i,j,cosine])
            #print(cosine)
    return pd.DataFrame(sim_df,columns=['ProdId1','ProdId2','similarity'])

sim_df=generate_cosine_similarity(docs)

