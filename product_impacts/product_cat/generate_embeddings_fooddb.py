# Usage: python -m product_impacts.product_cat.generate_embeddings_fooddb

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pickle

if __name__ == '__main__':
    
    products1 = pd.read_csv('../../SFS/all_fooddb_data_extracts/Extract_2019/foodDB_Raw/products.csv', usecols= [
        'product_id', 'product_list_name', 'product_name', 'ingredients_text', 'url', 'energy_per_100'])
    products2 = pd.read_csv('../../SFS/all_fooddb_data_extracts/Extract_2021/foodDB_dat/products.csv', usecols= [
        'product_id', 'product_list_name', 'product_name', 'ingredients_text', 'url', 'energy_per_100'])
    products3 = pd.read_csv('../../SFS/all_fooddb_data_extracts/Extract_2022/May_2022_Extract/products.csv',  usecols= [
        'foodDB_product_id', 'product_list_name', 'product_name', 'ingredients_list', 'product_url', 'Energy']).rename(columns={
        'foodDB_product_id': 'product_id',
        'ingredients_list': 'ingredients_text',
        'product_url': 'url',
        'Energy': 'energy_per_100'})
    products = pd.concat([products1, products2, products3], axis=0, ignore_index=True)
    
    products = products.drop_duplicates(['product_id']).reset_index(drop=True)
    products.loc[(products['product_name'].notnull()) 
                 & (products['product_name'].str.len()>products['product_list_name'].str.len()),
                 'product_list_name'] = products[
        (products['product_name'].notnull()) 
        & (products['product_name'].str.len()>products['product_list_name'].str.len())]['product_name']
    products = products.fillna('')
    products['text'] = products['product_list_name'].str.lower() + ' - ' + products['ingredients_text'].str.lower()
    
    model = SentenceTransformer('all-mpnet-base-v2')
    queries = products['text'].values.tolist()
    query_embeddings = model.encode(queries, convert_to_tensor=True)
    pickle.dump(query_embeddings, open('../../SFS/bert/all_embeddings_all3.pkl', 'wb'))
    np.save('../../SFS/bert/all_embeddings_all3.npy', np.array(query_embeddings))
    np.save('../../SFS/bert/all_ids_all3', np.array(products['product_id'].values))
    