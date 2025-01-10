# Usage: python -m product_impacts.ml_pipeline.make_predictions

import os
import os.path
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from product_impacts.ml_pipeline.train_models import get_labels

### enter path here
data_dir = '../../SFS/openfoodfacts/all/'

def get_pred_data(data_dir, prefix):
    
    pred = pd.read_csv(f'{data_dir}openfoodfacts_lang.csv', low_memory=False)
    pred = pred.fillna('')
    
    num_files = len([name for name in os.listdir(f'{data_dir}embeddings/{prefix}')if 'ids' in name])
    pred_df_list = []
    
    for i in range(num_files):
        pred_emb = np.load(f'{data_dir}embeddings/{prefix}embeddings_{i}.npy')
        # pred_tsne = np.load(f'{data_dir}embeddings/{prefix}tsne_results_{i}.npy')
        pred_product_ids = np.load(f'{data_dir}embeddings/{prefix}ids_{i}.npy')

        pred_features = pd.DataFrame(data=pred_emb)
        # pred_features[['tsne_0', 'tsne_1']] = pred_tsne
        pred_features['product_id'] = pd.Series(pred_product_ids, index=pred_features.index)
        pred_df_list.append(pred_features)
    
    pred_features = pd.concat(pred_df_list, axis=0, ignore_index=True)
    pred_features = pred.merge(pred_features)
    
    return pred_features

def get_predictions(stored_model, X, model_type):
    
    if model_type=='nn':
        model, label_encoder = stored_model['model'], stored_model['label_encoder']
        y = model.predict(X)
        predictions = np.argmax(y, axis=1)
        predictions = label_encoder.inverse_transform(predictions)
        predicted_probabilities = np.max(y, axis=1)
    else:
        predictions = stored_model.predict(X)
        predicted_probabilities = stored_model.predict_proba(X).max(axis=1)
    
    return predictions, predicted_probabilities
    

def lev_0_model(pred, X_cols, model_type):

    stored_model = joblib.load(f'../../SFS/NDNS UK/trained_models/{model_type}_lev0.joblib')
    X_pred = pred[X_cols].to_numpy()
    
    y_pred, y_pred_probabilities = get_predictions(stored_model, X_pred, model_type)
    pred['parentcategory'] = pd.Series(y_pred, index = pred.index)
    pred['parentcategory_prob'] = pd.Series(y_pred_probabilities, index = pred.index)
    
    return pred


def lev_2_model(train, pred, X_cols, model_type, category):

    stored_model = joblib.load(f'../../SFS/NDNS UK/trained_models/{model_type}_{category}.joblib')
    X_pred = pred[X_cols].to_numpy()
    
    y_pred2, y_pred2_probabilities = get_predictions(stored_model, X_pred, model_type)
    
    y_pred1 = pd.DataFrame(y_pred2, columns=['subfoodgroup']).merge(
        train[['mainfoodgroup', 'subfoodgroup']].drop_duplicates(), how='left')['mainfoodgroup'].values
    pred['mainfoodgroup'] = pd.Series(y_pred1, index = pred.index)
    pred['subfoodgroup'] = pd.Series(y_pred2, index = pred.index)
    pred['subfoodgroup_prob'] = pd.Series(y_pred2_probabilities, index = pred.index)
    
    return pred
    

if __name__ == '__main__':
    
    model_type = 'rf' ### enter model type here
    prefix = 'non_eng/' ### enter this if embeddings are split into folders, else blank string
    
    train, X_cols = get_labels()
    pred = get_pred_data(data_dir, prefix)

    print('Level 0')
    pred = lev_0_model(pred, X_cols, model_type)
    
    df_list = []
    
    for category in train['parentcategory'].unique().tolist():
        
        print(category)
        
        pred_df = pred[pred['parentcategory']==category]
        
        if category in ['Savoury Snacks', 'Alcoholic Beverages', 'Nuts and Seeds', 'Not Food', 'Dietary Supplements', 
                        'Artificial Sweeteners', 'Commercial Toddlers Foods and Drinks']:
            pred_df['mainfoodgroup'] = pred_df['parentcategory']
            pred_df['subfoodgroup'] = pred_df['parentcategory']
        else:
            pred_df = lev_2_model(train, pred_df, X_cols, model_type, category)
        
        df_list.append(pred_df)
        
    pred = pd.concat(df_list, axis=0, ignore_index=True)
    # pred = pred.drop([col for col in pred.columns.to_list() if isinstance(col, int)], axis=1)
    pred[['product_id', 'product_name', 'ingredients_text', 'parentcategory', 'mainfoodgroup', 'subfoodgroup', 'parentcategory_prob', 'subfoodgroup_prob']].to_csv(f'{data_dir}predictions/{prefix}predictions_{model_type}.csv', index=False)
        
        
    
    
    
    
    
    