# Usage: python -m product_impacts.ml_pipeline.generate_embeddings

import pandas as pd
import numpy as np
import pickle
import math
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

### enter path here
data_dir = '../../future_of_food/openfoodfacts/all/'

### edit this function to filter
def get_products(data_dir):
    
    products = pd.read_csv(f'{data_dir}openfoodfacts_lang.csv', low_memory=False)
    products = products.fillna('')
    products = products[(products['ingredients_text_language']=='und') & (products['product_name_language']=='en')]
    
    return products

if __name__ == '__main__':
    
    products = get_products(data_dir)
    products['text'] = products['product_name_en'].str.lower() + ' - ' + products['ingredients_text_en'].str.lower()
    model = SentenceTransformer('all-mpnet-base-v2')
    prefix = 'eng' ### enter this if splitting embeddings into folders, else blank string
    i = 41 ### enter this if starting subscript needs to be higher, else 0 
    N = math.ceil(len(products)/10000)
    
    for chunk in np.array_split(products, N):
        
        print(i)
        print(chunk.shape)
        
        queries = chunk['text'].values.tolist()
        query_embeddings = model.encode(queries, convert_to_tensor=True)
        np.save(f'{data_dir}embeddings/{prefix}embeddings_{i}.npy', np.array(query_embeddings))
        np.save(f'{data_dir}embeddings/{prefix}ids_{i}', np.array(chunk['product_id'].values))
        print('embeddings done')

        # also generating tsne dims
        # pca = PCA(n_components=0.80)
        # X_pca = pca.fit_transform(np.array(query_embeddings))
        # tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000)
        # tsne_results = tsne.fit_transform(X_pca)
        # np.save(f'{data_dir}embeddings/{prefix}tsne_results_{i}.npy', np.array(tsne_results))
        # print('tsne done')
        
        i += 1