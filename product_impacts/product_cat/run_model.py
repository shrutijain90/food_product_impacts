# Usage: python -m product_impacts.product_cat.run_model

import os
import os.path
import pandas as pd
from skimpy import skim
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, mean_squared_error, r2_score, roc_auc_score, balanced_accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_classif
from hiclass import LocalClassifierPerParentNode, LocalClassifierPerLevel

import numpy as np
import pickle
import json

def get_store(row):
    store = 'null'
    if isinstance(row['url'], str):
        if 'http' in row['url']:
            store = row['url']
    if isinstance(row['product_name'], str):
        if 'http' in row['product_name']:
            store = row['product_name']
    if isinstance(row['energy_per_100'], str):
        if 'http' in row['energy_per_100']:
            store = row['energy_per_100']
    store_list = ['waitrose', 'morrisons', 'tesco', 'sainsbury', 'cook', 'aldi', 'iceland', 'ocado'] 
    for s in store_list:
        if s in store:
            if s == 'tesco':
                if 'tesco.ie' in store:
                    store = 'tesco ireland'
                else:
                    store = s
            else:
                store = s
    return store
    
def get_ndns_cats(fname):
    
    ndns = pd.read_csv(fname, encoding='unicode_escape')
    ndns.loc[ndns['subfoodgroupdesc_edited'].notnull(), 'subfoodgroupdesc'] = ndns[ndns['subfoodgroupdesc_edited'].notnull()]['subfoodgroupdesc_edited']
    ndns = ndns.drop(['subfoodgroupdesc_edited'], axis=1)
    ndns = ndns.fillna('')
    
    return ndns

def get_products():
    
    products1 = pd.read_csv('../../SFS/all_fooddb_data_extracts/Extract_2019/foodDB_Raw/products.csv', usecols= [
        'product_id', 'product_list_name', 'product_name', 'ingredients_text', 'url', 'energy_per_100'])
    products1['extract_year'] = 2019
    products2 = pd.read_csv('../../SFS/all_fooddb_data_extracts/Extract_2021/foodDB_dat/products.csv', usecols= [
        'product_id', 'product_list_name', 'product_name', 'ingredients_text', 'url', 'energy_per_100'])
    products2['extract_year'] = 2021
    products3 = pd.read_csv('../../SFS/all_fooddb_data_extracts/Extract_2022/May_2022_Extract/products.csv',  usecols= [
        'foodDB_product_id', 'product_list_name', 'product_name', 'ingredients_list', 'product_url', 'Energy']).rename(columns={
        'foodDB_product_id': 'product_id',
        'ingredients_list': 'ingredients_text',
        'product_url': 'url',
        'Energy': 'energy_per_100'})
    products3['extract_year'] = 2022
    products = pd.concat([products1, products2, products3], axis=0, ignore_index=True)
    
    products['store'] = products.apply(lambda row: get_store(row), axis=1)
    products = products.drop_duplicates(['product_id']).reset_index(drop=True)
    products.loc[(products['product_name'].notnull()) 
                 & (products['product_name'].str.len()>products['product_list_name'].str.len()),
                 'product_list_name'] = products[
        (products['product_name'].notnull()) 
        & (products['product_name'].str.len()>products['product_list_name'].str.len())]['product_name']
    products = products.fillna('')
    
    products = products[['product_id', 'product_list_name', 'product_name', 'url', 'ingredients_text', 'store', 'extract_year']]
    
    return products

def get_ndns_matches(ndns, products, pred_fname=None):
    
    # original categories from Mike
    ndns_matches = pd.read_csv('../../SFS/NDNS UK/labels/matches.10.05.22.csv')
    
    # to match perfectly with matches data
    ndns_matches.loc[ndns_matches['subfoodgroupcode']=='61R', 'mainfoodgroupcode'] = 45
    ndns_matches.loc[ndns_matches['subfoodgroupcode']=='61R', 'mainfoodgroupdesc'] = 'FRUIT JUICE'
    ndns_matches.loc[ndns_matches['subfoodgroupcode']=='61R', 'subfoodgroupdesc'] = 'SMOOTHIES'
    
    # there are 3 rows where there are no matches, drop them (or replace those with foodname options)
    # after that, drop foodname column and drop duplicate rows

    #ndns_matches.loc[ndns_matches['matches'].isna(), 'matches'] = ndns_matches[(ndns_matches['matches'].isna())]['foodname']
    ndns_matches = ndns_matches.dropna()
    ndns_matches = ndns_matches.drop('foodname', axis=1)
    ndns_matches = ndns_matches.drop_duplicates()
    
    # some products have been categorized into multiple categories. keep only one? 
    ndns_matches = ndns_matches.sort_values(by=['mainfoodgroupcode', 'subfoodgroupcode', 'matches']).reset_index(drop=True)
    ndns_matches = ndns_matches.drop_duplicates(subset='matches').reset_index(drop=True)
    
    # merging matches and products data
    # some matches are mapped to multiple products, e.g. when 2 products across stores have the same name. some of them have ingredients listed and some don't
    # not worrying about which one is kept right now, as the data for prediction can be of any kind
    # only 5 product names don't match, probably because of string names, left with 17814 products, 13 parent categories, 53 main food groups, 117 subfood groups
    labelled_data = ndns_matches[['mainfoodgroupcode', 'subfoodgroupcode', 'matches'
                                 ]].merge(ndns.drop(['detaileddesc'], axis=1), how='left')
    labelled_data = labelled_data.merge(products, left_on='matches', right_on='product_name', how='left')
    labelled_data = labelled_data.drop_duplicates(subset=['matches']).reset_index(drop=True)
    labelled_data = labelled_data[['product_id', 'product_name', 'product_list_name', 'ingredients_text', 'store', 'parentcategory',
                                   'mainfoodgroupcode', 'mainfoodgroupdesc', 'subfoodgroupcode', 'subfoodgroupdesc']]
    
    
    ###################
    # some corrections done after first round of predictions
    corr = pd.read_csv('../../SFS/NDNS UK/labels/predictions_corr_manual_Shruti.csv')
    remap_dict = {
        'OTHER FRUIT NOT CANNED': 'Other fruit not canned', 
        'CITRUS FRUIT NOT CANNED': 'Citrus fruit not canned',
        'SALAD AND OTHER RAW': 'Salad and other raw', 
        'NUTS AND SEEDS': 'Nuts and seeds', 
        'FRUIT JUICE': 'Fruit juice',
        'CHOCOLATE CONFECTIONERY': 'Chocolate confectionery', 
        'CANNED FRUIT IN JUICE': 'Canned fruit in juice',
        'CANNED FRUIT IN SYRUP': 'Canned fruit in syrup',
        'SAVOURY SAUCES PICKLES GRAVIES & CONDIMENTS': 'Savoury sauces pickles gravies & condiments', 
        'WINE': 'Wine',
        'CIDER AND PERRY': 'Cider and perry', 
        'BEERS AND LAGERS': 'Beers and lagers', 
        'SPIRITS': 'Spirits', 
        'NOT FOOD': 'not food',
        'FROMAGE FRAIS AND OTHER DAIRY DESSERTS (MANUFACTURED)': 'Fromage frais and other dairy desserts (manufactured)',
        'FRUIT PIES': 'Fruit pies', 
        'PRESERVES': 'Preserves', 
        'LIQUEURS': 'Liqueurs', 
        'WHOLE MILK': 'Whole milk', 
        'OTHER MILK': 'Other milk'
    }

    corr = corr[corr['Corrected sub category'].notnull()][['product_id', 'Corrected sub category']].rename(
        columns={'Corrected sub category': 'subfoodgroupdesc'})
    corr = corr.replace({"subfoodgroupdesc": remap_dict})
    non_food_products = corr[corr['subfoodgroupdesc']=='not food']['product_id'].unique()
    corr = corr.merge(ndns.drop(['detaileddesc'], axis=1)).drop_duplicates(subset=['product_id'])
    corr = corr.merge(products)
    corr = corr[['product_id', 'product_name', 'product_list_name', 'ingredients_text', 'store', 'parentcategory',
                 'mainfoodgroupcode', 'mainfoodgroupdesc', 'subfoodgroupcode', 'subfoodgroupdesc']]
    
    # if some of the existing matches have been corrected, delete original rows from labelled data and then insert the corrected labels 
    labelled_data = labelled_data[~labelled_data['product_id'].isin(corr['product_id'].values)].reset_index(drop=True)
    labelled_data = pd.concat([labelled_data, corr], ignore_index=True, axis=0)
    
    
    ###################
    # categories from labelling exercise
    fnames = ['../../SFS/NDNS UK/labels/Copy of sample_7_EH.xlsx',
              '../../SFS/NDNS UK/labels/sample_2_SJ.xlsx',
              '../../SFS/NDNS UK/labels/WorkForShruti.xlsx']
    
    for fname in fnames:
        df = pd.read_excel(fname)
        not_food_add = df[df['subfoodgroupdesc']=='not food']['product_id'].unique()
        non_food_products = list(set(non_food_products).union(set(not_food_add)))
        df.loc[(df['mainfoodgroupdesc']=='Soft drinks, not diet') 
                & (df['subfoodgroupdesc'].isin(['Soft drinks, not diet, not low calorie concentrated',
                                                 'Soft drinks not low calorie concentrated'])),
               'subfoodgroupdesc'] = 'Soft drinks not low calorie concentrated'

        df.loc[(df['mainfoodgroupdesc']=='Soft drinks, not diet') 
                & (df['subfoodgroupdesc'].isin(['Soft drinks, not diet, not low calorie carbonated',
                                                 'Soft drinks not low calorie carbonated'])),
               'subfoodgroupdesc'] = 'Soft drinks not low calorie carbonated'

        df.loc[(df['mainfoodgroupdesc']=='Soft drinks, not diet') 
                & (df['subfoodgroupdesc'].isin(['Soft drinks, not diet, not low calorie, ready to drink, still',
                                                 'Soft drinks not low calorie, ready to drink, still'])),
               'subfoodgroupdesc'] = 'Soft drinks not low calorie, ready to drink, still'

        df.loc[(df['mainfoodgroupdesc']=='Soft drinks, diet') 
                & (df['subfoodgroupdesc'].isin(['Soft drinks, diet, not low calorie concentrated',
                                                 'Soft drinks not low calorie concentrated'])),
               'subfoodgroupdesc'] = 'Soft drinks low calorie concentrated'

        df.loc[(df['mainfoodgroupdesc']=='Soft drinks, diet') 
                & (df['subfoodgroupdesc'].isin(['Soft drinks, diet, not low calorie carbonated',
                                                 'Soft drinks not low calorie carbonated'])),
               'subfoodgroupdesc'] = 'Soft drinks low calorie carbonated'

        df.loc[(df['mainfoodgroupdesc']=='Soft drinks, diet') 
                & (df['subfoodgroupdesc'].isin(['Soft drinks, diet, not low calorie, ready to drink, still',
                                                 'Soft drinks not low calorie, ready to drink, still'])),
               'subfoodgroupdesc'] = 'Soft drinks low calorie, ready to drink, still'

        df = df[df['subfoodgroupdesc'].notnull()]

        df = df[['product_id', 'parentcategory', 'mainfoodgroupdesc', 'subfoodgroupdesc'
                  ]].merge(ndns.drop(['detaileddesc'], axis=1)).drop_duplicates(subset=['product_id'])
        
        df = df[['product_id', 'parentcategory', 'mainfoodgroupdesc', 'subfoodgroupdesc', 'mainfoodgroupcode','subfoodgroupcode']].merge(products)
        df = df[['product_id', 'product_name', 'product_list_name', 'ingredients_text', 'store', 'parentcategory',
                 'mainfoodgroupcode', 'mainfoodgroupdesc', 'subfoodgroupcode', 'subfoodgroupdesc']]
        labelled_data = pd.concat([labelled_data, df], ignore_index=True, axis=0).drop_duplicates(
            subset=['product_id']).reset_index(drop=True)
    
    
    ###################
    # intake 24 categorizations
    with open('../../SFS/NDNS UK/labels/intake24/mapping_GPT_FM_string_duplication.json') as f:
        intake24 = json.load(f)
    df = pd.json_normalize(intake24)
    df = df.melt(ignore_index=False).reset_index(drop=True)
    intake24 = pd.DataFrame(df.value.tolist(), index= df.index)
    intake24['NDB_category'] = df['variable']
    intake24 = intake24.melt(id_vars='NDB_category', value_vars = [i for i in range(intake24.shape[1]-1)],
                      ignore_index=False).reset_index(drop=True)
    intake24 = intake24[intake24['value'].notnull()]
    intake24 = intake24[['value', 'NDB_category']].rename(columns={'value': 'product_list_name'})
    
    intake_mapping = pd.read_excel('../../SFS/NDNS UK/labels/intake24/NDNS_food_code_mapping.xlsx')
    intake_mapping = intake_mapping.rename(columns={'Unnamed: 0': 'NDB_category'})
    intake24 = intake24.merge(intake_mapping[['NDB_category', 'NDB food code', 'MainFoodGroupCode', 
                                              'MainFoodGroupDesc', 'SubFoodGroupCode', 'SubFoodGroupDesc']])
    intake24 = intake24[intake24['SubFoodGroupDesc'].notnull()]
    intake24 = intake24.drop_duplicates(subset=['product_list_name'])
    intake24 = intake24.merge(products)
    intake24 = intake24.merge(ndns, left_on='SubFoodGroupCode', right_on='subfoodgroupcode')
    intake24 = intake24[['product_id', 'product_name', 'product_list_name', 'ingredients_text', 'store', 'parentcategory',
                         'mainfoodgroupcode', 'mainfoodgroupdesc', 'subfoodgroupcode', 'subfoodgroupdesc']]
    labelled_data = pd.concat([labelled_data, intake24], ignore_index=True, axis=0).drop_duplicates(
        subset=['product_id']).reset_index(drop=True)
    
    
    ###################
    # categories from Savka
    folder_cats = '../../SFS/NDNS UK/labels/Savka NDNS Categorisation'
    for cat_fname in os.listdir('../../SFS/NDNS UK/labels/Savka NDNS Categorisation'):
        df = pd.read_csv(f"{folder_cats}/{cat_fname}")
        if len(df['NDNS_Code'].values[0].split(','))==1:
            df = df[['NDNS_Code', 'id']].merge(ndns.drop(['detaileddesc'], axis=1), 
                                               left_on='NDNS_Code', right_on='subfoodgroupcode', how='left').drop('NDNS_Code', axis=1)
            df = df.merge(products, left_on='id', right_on='product_id', how='left').drop('id', axis=1)
            df = df[['product_id', 'product_name', 'product_list_name', 'ingredients_text', 'store', 'parentcategory',
                     'mainfoodgroupcode', 'mainfoodgroupdesc', 'subfoodgroupcode', 'subfoodgroupdesc']]
            labelled_data = pd.concat([labelled_data, df], ignore_index=True, axis=0).drop_duplicates(
                subset=['product_id']).reset_index(drop=True)
        else:
            if df['NDNS_Code'].values[0]=='5, 6':
                df['NDNS_Code'] = 5
            if df['NDNS_Code'].values[0]=='8B, 8D':
                df['NDNS_Code'] = 8
            if df['NDNS_Code'].values[0]=='41B, 41R':
                df['NDNS_Code'] = 41
            if df['NDNS_Code'].values[0]=='37C, 37I, 37K, 37L':    
                df_soup = df[df['product_name'].str.lower().str.contains('soup')].reset_index(drop=True)
                df_soup['NDNS_Code'] = '50C'
                df_soup = df_soup[['NDNS_Code', 'id']].merge(ndns.drop(['detaileddesc'], axis=1), 
                                                             left_on='NDNS_Code', right_on='subfoodgroupcode', how='left').drop('NDNS_Code', axis=1)
                df_soup = df_soup.merge(products, left_on='id', right_on='product_id', how='left').drop('id', axis=1)
                df_soup = df_soup[['product_id', 'product_name', 'product_list_name', 'ingredients_text', 'store', 'parentcategory',
                                   'mainfoodgroupcode', 'mainfoodgroupdesc', 'subfoodgroupcode', 'subfoodgroupdesc']]
                labelled_data = pd.concat([labelled_data, df_soup], ignore_index=True, axis=0).drop_duplicates(
                    subset=['product_id']).reset_index(drop=True)
                df = df[~df['product_name'].str.lower().str.contains('soup')].reset_index(drop=True)
                df['NDNS_Code'] = 37
            if df['NDNS_Code'].values[0]=='2, 3, 4, 59':
                df['NDNS_Code'] = 2
            
            df = df[['NDNS_Code', 'id']].merge(ndns.drop(['detaileddesc', 'subfoodgroupcode', 'subfoodgroupdesc'], axis=1).drop_duplicates(), 
                                               left_on='NDNS_Code', right_on='mainfoodgroupcode', how='left').drop('NDNS_Code', axis=1)
            df['subfoodgroupcode'] = np.NaN
            df['subfoodgroupdesc'] = np.NaN
            df = df.merge(products, left_on='id', right_on='product_id', how='left').drop('id', axis=1)
            df = df[['product_id', 'product_name', 'product_list_name', 'ingredients_text', 'store', 'parentcategory',
                     'mainfoodgroupcode', 'mainfoodgroupdesc', 'subfoodgroupcode', 'subfoodgroupdesc']]
            labelled_data = pd.concat([labelled_data, df], ignore_index=True, axis=0)
            
            dups = labelled_data[labelled_data.duplicated(subset=['product_id'], keep=False)].reset_index(drop=True)
            dups1_tokeep = dups[dups['subfoodgroupcode'].notnull()].drop_duplicates(subset=['product_id'])
            dups2_tokeep = dups[(dups['subfoodgroupcode'].isna()) 
                                & (~dups['product_id'].isin(dups1_tokeep['product_id'].unique()))
                               ].drop_duplicates(subset=['product_id'])
            
            labelled_data = labelled_data.drop_duplicates(subset=['product_id'], keep=False).reset_index(drop=True)
            labelled_data = pd.concat([labelled_data, dups1_tokeep, dups2_tokeep], ignore_index=True, axis=0)
    
    
    ###################
    # if predicting iteratively
    if pred_fname:
        pred = pd.read_csv(pred_fname)
        pred = pred[(pred['cos_sim_pred']>0.5) & (pred['subfoodgroup_lab'].isna())]
        pred = pred.rename(columns={'subfoodgroup_pred': 'subfoodgroupdesc'})
        pred = pred.merge(ndns.drop(['detaileddesc'], axis=1)).drop_duplicates(subset=['product_id'])
        pred = pred.merge(products)
        pred = pred[['product_id', 'product_name', 'product_list_name', 'ingredients_text', 'store', 'parentcategory',
                     'mainfoodgroupcode', 'mainfoodgroupdesc', 'subfoodgroupcode', 'subfoodgroupdesc']]
        labelled_data = pd.concat([labelled_data, pred], ignore_index=True, axis=0).drop_duplicates(
            subset=['product_id']).reset_index(drop=True)
        
    # removing duplicated product names + ingredients (basically products with same exact feature vectors)
    labelled_data['product_list_name_lower'] = labelled_data['product_list_name'].str.lower()
    labelled_data['ingredients_text_lower'] = labelled_data['ingredients_text'].str.lower()
    dups = labelled_data[labelled_data.duplicated(
        subset=['product_list_name_lower', 'ingredients_text_lower'], keep=False)].reset_index(drop=True)
    dups1_tokeep = dups[dups['subfoodgroupcode'].notnull()].drop_duplicates(subset=['product_list_name_lower', 'ingredients_text_lower'])
    dups2_tokeep = dups[(dups['subfoodgroupcode'].isna()) 
                        & (~(dups['product_list_name_lower'] + dups['ingredients_text_lower']).isin(
                            (dups1_tokeep['product_list_name_lower'] + dups1_tokeep['ingredients_text_lower']).unique()))
                       ].drop_duplicates(subset=['product_list_name_lower', 'ingredients_text_lower'])

    labelled_data = labelled_data.drop_duplicates(
        subset=['product_list_name_lower', 'ingredients_text_lower'], keep=False).reset_index(drop=True)
    labelled_data = pd.concat([labelled_data, dups1_tokeep, dups2_tokeep], ignore_index=True, axis=0)
    labelled_data = labelled_data.drop(['product_list_name_lower', 'ingredients_text_lower'], axis=1)
    
    return labelled_data, non_food_products

def get_tsne(emb, fname):
    
    if os.path.exists(fname):
        tsne_results = np.load(fname)
    else:      
        pca = PCA(n_components=0.80)
        X_pca = pca.fit_transform(emb)
        tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000)
        tsne_results = tsne.fit_transform(X_pca)
        np.save(fname, np.array(tsne_results))
    
    return tsne_results

def non_hi_model(train, features, X_cols, y_cols):
    
    mod = LinearDiscriminantAnalysis()
    X_train = train[X_cols]
    X_pred = features[X_cols]
    y_train0 = train[y_cols[0]]
    y_train1 = train[y_cols[1]]
    y_train2 = train[y_cols[2]]
    
    clf0 = mod
    clf0.fit(X_train, y_train0)
    y_train_pred0 = clf0.predict(X_train)
    y_pred0 = clf0.predict(X_pred)
    y_pred0_proba = clf0.predict_proba(X_pred)
    print(f'Lev 0 accuracy: {accuracy_score(y_train0, y_train_pred0)}')
    print(f'Lev 0 balanced accuracy: {balanced_accuracy_score(y_train0, y_train_pred0)}')

    clf1 = mod
    clf1.fit(X_train, y_train1)
    y_train_pred1 = clf1.predict(X_train)
    y_pred1 = clf1.predict(X_pred)
    y_pred1_proba = clf1.predict_proba(X_pred)
    print(f'Lev 1 accuracy: {accuracy_score(y_train1, y_train_pred1)}')
    print(f'Lev 1 balanced accuracy: {balanced_accuracy_score(y_train1, y_train_pred1)}')

    clf2 = mod
    clf2.fit(X_train, y_train2)
    y_train_pred2 = clf2.predict(X_train)
    y_pred2 = clf2.predict(X_pred)
    y_pred2_proba = clf2.predict_proba(X_pred)
    print(f'Lev 2 accuracy: {accuracy_score(y_train2, y_train_pred2)}')
    print(f'Lev 2 balanced accuracy: {balanced_accuracy_score(y_train2, y_train_pred2)}')

    print(f'Number of unique level 0 categories in train: {len(np.unique(y_train_pred0))}')
    print(f'Number of unique level 1 categories in train: {len(np.unique(y_train_pred1))}')
    print(f'Number of unique level 2 categories in train: {len(np.unique(y_train_pred2))}')

    print(f'Number of unique level 0 categories in pred: {len(np.unique(y_pred0))}')
    print(f'Number of unique level 1 categories in pred: {len(np.unique(y_pred1))}')
    print(f'Number of unique level 2 categories in pred: {len(np.unique(y_pred2))}')
    
    predicted_data = features
    predicted_data['parentcategory_pred'] = pd.Series(y_pred0, index = predicted_data.index)
    predicted_data['mainfoodgroup_pred'] = pd.Series(y_pred1, index = predicted_data.index)
    predicted_data['subfoodgroup_pred'] = pd.Series(y_pred2, index = predicted_data.index)
    # probabilities
    predicted_data['parentcategory_proba'] = y_pred0_proba.max(axis=1)
    predicted_data['maincategory_proba'] = y_pred1_proba.max(axis=1)
    predicted_data['subcategory_proba'] = y_pred2_proba.max(axis=1)
    
    return predicted_data


def hi_model(train, features, X_cols, y_cols):
    
    mod = LinearDiscriminantAnalysis()
    X_train = train[X_cols]
    X_pred = features[X_cols]
    y_train = train[y_cols]
    y_train0 = train[y_cols[0]]
    y_train1 = train[y_cols[1]]
    y_train2 = train[y_cols[2]]
    
    clf = LocalClassifierPerLevel(local_classifier=mod)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_pred = clf.predict(X_pred)
    
    y_train_pred0 = y_train_pred[:,0]
    y_pred0 = y_pred[:,0]
    y_train_pred1 = y_train_pred[:,1]
    y_pred1 = y_pred[:,1]
    y_train_pred2 = y_train_pred[:,2]
    y_pred2 = y_pred[:,2]
    
    print(f'Lev 0 accuracy: {accuracy_score(y_train0, y_train_pred0)}')
    print(f'Lev 0 balanced accuracy: {balanced_accuracy_score(y_train0, y_train_pred0)}')
    print(f'Lev 1 accuracy: {accuracy_score(y_train1, y_train_pred1)}')
    print(f'Lev 1 balanced accuracy: {balanced_accuracy_score(y_train1, y_train_pred1)}')
    print(f'Lev 2 accuracy: {accuracy_score(y_train2, y_train_pred2)}')
    print(f'Lev 2 balanced accuracy: {balanced_accuracy_score(y_train2, y_train_pred2)}')

    print(f'Number of unique level 0 categories in train: {len(np.unique(y_train_pred0))}')
    print(f'Number of unique level 1 categories in train: {len(np.unique(y_train_pred1))}')
    print(f'Number of unique level 2 categories in train: {len(np.unique(y_train_pred2))}')

    print(f'Number of unique level 0 categories in pred: {len(np.unique(y_pred0))}')
    print(f'Number of unique level 1 categories in pred: {len(np.unique(y_pred1))}')
    print(f'Number of unique level 2 categories in pred: {len(np.unique(y_pred2))}')
    
    predicted_data = features
    predicted_data['parentcategory_pred'] = pd.Series(y_pred0, index = predicted_data.index)
    predicted_data['mainfoodgroup_pred'] = pd.Series(y_pred1, index = predicted_data.index)
    predicted_data['subfoodgroup_pred'] = pd.Series(y_pred2, index = predicted_data.index)
    
    return predicted_data

def get_dist_scores(predicted_data, ndns, X_cols):
    
    def _get_text(row, level=2):
        text = row['parentcategory']
        if row['parentcategory'].lower()!=row['mainfoodgroupdesc'].lower():
            text = text + ' - ' + row['mainfoodgroupdesc']
        if level==2:
            if row['mainfoodgroupdesc'].lower()!=row['subfoodgroupdesc'].lower():
                text = text + ' - ' + row['subfoodgroupdesc']
            text = text + ' - ' + row['detaileddesc']
        return text
    
    def _cos_sim(row, ndns, X_cols):
        
        X = row[X_cols].values.reshape(1, -1)
        desc = row['detaileddesc']
        Y = ndns[ndns['detaileddesc']==desc][X_cols].values
        return cosine_similarity(X, Y)[0][0]
    
    ndns = ndns.drop_duplicates(subset=['parentcategory', 'mainfoodgroupdesc', 'subfoodgroupdesc', 'detaileddesc']).reset_index(drop=True)
    ndns['text'] = ndns.apply(lambda row: _get_text(row), axis=1)
    corpus = ndns['text'].values.tolist()
    model = SentenceTransformer('all-mpnet-base-v2')
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    corpus_embeddings = np.array(corpus_embeddings)
    corpus_df = pd.DataFrame(corpus_embeddings)
    ndns = pd.concat([ndns, corpus_df], axis=1)
    
    predicted_data = predicted_data.merge(ndns[['subfoodgroupdesc', 'detaileddesc']], left_on=['subfoodgroup_pred'],
                                          right_on=['subfoodgroupdesc'], how='left').drop(['subfoodgroupdesc'], axis=1)
    
    predicted_data['cos_sim_pred'] = predicted_data.apply(lambda row: _cos_sim(row, ndns, X_cols), axis=1)
    predicted_data = predicted_data.drop('detaileddesc', axis=1)
    
    predicted_data_sub = predicted_data[predicted_data['parentcategory_lab'].notnull()].reset_index(drop=True).merge(
        ndns[['subfoodgroupdesc', 'detaileddesc']], left_on=['subfoodgroup_lab'], 
        right_on=['subfoodgroupdesc'], how='left').drop(['subfoodgroupdesc'], axis=1)
    
    predicted_data_sub['cos_sim_lab'] = predicted_data_sub.apply(lambda row: _cos_sim(row, ndns, X_cols), axis=1)
    predicted_data_sub = predicted_data_sub.drop('detaileddesc', axis=1)
    predicted_data = predicted_data.merge(predicted_data_sub, how='left')
    return predicted_data    


if __name__ == '__main__':
    
    ndns = get_ndns_cats('../../SFS/NDNS UK/ndns_edited.csv')
    products = get_products()
    labelled_data, non_food_products = get_ndns_matches(
        ndns, products, 
        # pred_fname='../../SFS/NDNS UK/predictions/predictions_all_LDA_v1.csv'
    )
    query_embeddings = np.load('../../SFS/bert/all_embeddings_all3.npy')
    tsne_results = get_tsne(query_embeddings, '../../SFS/bert/tsne_results_all3.npy')
    product_ids = np.load('../../SFS/bert/all_ids_all3.npy')
    
    features = pd.DataFrame(data=query_embeddings)
    id_col = 'product_id'
    X_cols = features.columns.tolist()
    y_cols = ['parentcategory', 'mainfoodgroupdesc', 'subfoodgroupdesc']
    
    features['product_id'] = pd.Series(product_ids, index=features.index)
    features[['tsne_0', 'tsne_1']] = tsne_results
    # features = features[~features['product_id'].isin(non_food_products)].reset_index(drop=True)
    labelled_data = labelled_data.merge(features)
    
    # predicted_data = hi_model(labelled_data, features, X_cols, y_cols)
    predicted_data = non_hi_model(labelled_data, features, X_cols, y_cols)
    cols = ['product_id', 'product_name', 'product_list_name', 'store', 
            'parentcategory_lab', 'mainfoodgroup_lab', 'subfoodgroup_lab',
            'parentcategory_pred', 
            'mainfoodgroup_pred', 'subfoodgroup_pred',
            'parentcategory_proba', 'maincategory_proba', 'subcategory_proba',
            'tsne_0', 'tsne_1', 'cos_sim_pred', 'cos_sim_lab']
    
    predicted_data = predicted_data.merge(products[['product_id', 'product_name', 'product_list_name', 'store']], how='left')
    predicted_data = predicted_data.merge(labelled_data[['product_id', 'product_name', 'product_list_name', 'store', 
                                                         'parentcategory', 'mainfoodgroupdesc', 'subfoodgroupdesc']], how='left')
    predicted_data = predicted_data.rename(columns={'parentcategory': 'parentcategory_lab', 
                                                    'mainfoodgroupdesc': 'mainfoodgroup_lab', 
                                                    'subfoodgroupdesc': 'subfoodgroup_lab'})
    
    predicted_data = get_dist_scores(predicted_data, ndns, X_cols)

    predicted_data[cols].to_csv('../../SFS/NDNS UK/predictions/predictions_all_LDA_HI12_v1.csv', index=False)
    