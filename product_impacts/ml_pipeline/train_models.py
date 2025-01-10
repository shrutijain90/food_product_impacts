# Usage: python -m product_impacts.ml_pipeline.train_models

import os
import glob
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from keras.utils import to_categorical

def get_labels():
    
    food_db = pd.read_csv('../../SFS/NDNS UK/predictions/all_predictions_15Oct2024.csv')
    food_db = food_db.drop(['parentcategory_lab', 'mainfoodgroup_lab', 'subfoodgroup_lab'], axis=1).rename(
        columns={'parentcategory_pred': 'parentcategory',
                 'mainfoodgroup_pred': 'mainfoodgroup',
                 'subfoodgroup_pred': 'subfoodgroup'})
    food_db['product_list_name_lower'] = food_db['product_list_name'].str.lower()
    food_db['ingredients_text_lower'] = food_db['ingredients_text'].str.lower()
    food_db = food_db.drop_duplicates(subset=['product_list_name_lower', 'ingredients_text_lower']).reset_index(drop=True)
    food_db = food_db.drop(['product_list_name_lower', 'ingredients_text_lower'], axis=1)
    
    food_db_emb = np.load('../../SFS/bert/all_embeddings_all3.npy')
    food_db_product_ids = np.load('../../SFS/bert/all_ids_all3.npy')
    
    food_db_features = pd.DataFrame(data=food_db_emb)
    X_cols = food_db_features.columns.tolist()
    
    food_db_features['product_id'] = pd.Series(food_db_product_ids, index=food_db_features.index)
    food_db_features = food_db.merge(food_db_features)
    
    return food_db_features, X_cols

class NN_Pipeline(object):
    
    def __init__(self, X, y, class_weights):
        
        self.X = X
        self.y = y
        self.class_weights = class_weights
        self.bestparams = None
        self.results_best = None
        self.bestmodel = None
        self.paramgrid = {
                       'dense_layers':[1, 2, 3],
                       'batch_size': [16, 32, 64, 128],
                       'epochs': [50, 100],
                       'dropout': [0.2, 0.4],
                       'dense_neurons': [512, 256, 128, 64, 32]
                      }
        
    def nn_model(self, X_train, y_train, params,  X_val=None, y_val=None, verbose=1):
        
        if X_val is None:
            validation_data = None
        else:
            validation_data = (X_val, y_val)
        
        n_features, n_outputs = X_train.shape[1], y_train.shape[1]
        
        dense_layers = params['dense_layers']
        batch_size = params['batch_size']
        epochs = params['epochs']
        dropout = params['dropout']
        dense_neurons = params['dense_neurons']

        model = Sequential()
                
        # Input layer
        model.add(Input(shape=(n_features,)))

        # Hidden layers
        for n in range(dense_layers):
            fac = 2**n
            dense_neurons = int(dense_neurons/fac)
            model.add(Dense(dense_neurons, activation='relu', kernel_regularizer=l2(0.001)))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))
        
        # Output layer with softmax activation for multiclass classification
        model.add(Dense(n_outputs, activation='softmax'))

        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate = 0.001,
            decay_steps=20000,
            end_learning_rate=0.0000125,
            power=1
        )
        
        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                      loss='categorical_crossentropy',  
                      metrics=['accuracy'])


        history = model.fit(X_train, y_train, 
                            validation_data=validation_data, 
                            epochs=epochs, 
                            batch_size=batch_size,
                            class_weight=self.class_weights,
                            verbose=verbose)
        
        # getting other performance metrics to add to history
        res = {}
        y_train_pred = model.predict(X_train)
        y_train = np.argmax(y_train, axis=1)
        y_train_pred = np.argmax(y_train_pred, axis=1)
        res['accuracy'] = [accuracy_score(y_train, y_train_pred)]
        res['balanced_accuracy'] = [balanced_accuracy_score(y_train, y_train_pred)]
        res['precision'] = [precision_score(y_train, y_train_pred, average='weighted')]
        res['recall'] = [recall_score(y_train, y_train_pred, average='weighted')]
        res['f1'] = [f1_score(y_train, y_train_pred, average='weighted')]
        res['mcc'] = [matthews_corrcoef(y_train, y_train_pred)]

        if X_val is not None:
            y_val_pred = model.predict(X_val)
            y_val = np.argmax(y_val, axis=1)
            y_val_pred = np.argmax(y_val_pred, axis=1)
            res['val_accuracy'] = [accuracy_score(y_val, y_val_pred)]
            res['val_balanced_accuracy'] = [balanced_accuracy_score(y_val, y_val_pred)]
            res['val_precision'] = [precision_score(y_val, y_val_pred, average='weighted')]
            res['val_recall'] = [recall_score(y_val, y_val_pred, average='weighted')]
            res['val_f1'] = [f1_score(y_val, y_val_pred, average='weighted')]
            res['val_mcc'] = [matthews_corrcoef(y_val, y_val_pred)]

        return model, history, res
        
    def tune_hyperparam(self, res_fname, p=None):
        
        if p is None:
            p = self.paramgrid
        else: 
            self.paramgrid = p
        
        # get train and val sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        keys, values = zip(*p.items())
        param_combinations = [dict(zip(keys, v)) for v in it.product(*values)]

        i=1
        results_list = []

        for params in param_combinations:
            print(f'{i} of {len(param_combinations)}')
            model, history, _ = self.nn_model(X_train, y_train, params, X_val, y_val)
            df_res = pd.DataFrame(history.history)
            df_res['epoch'] = df_res.index + 1
            df_res = df_res[df_res['epoch']==df_res['epoch'].max()].reset_index(drop=True)
            df_res = df_res.drop('epoch', axis=1)
            for key in params.keys():
                df_res[key] = params[key]
            results_list.append(df_res)
            i = i+1
        
        results = pd.concat(results_list, axis=0)
        results = results.reset_index(drop=True)
        resutls.to_csv(res_fname, index=False)
        
        return results
    
    def get_best_params(self, res_fname, p=None, sel_by='loss', prefit=True):
        
        if p is None:
            p = self.paramgrid
        else:
            self.paramgrid = p
        
        if len(glob.glob(res_fname))==0 or not prefit:
            results = self.tune_hyperparam(res_fname, p=p)
        else:
            results = pd.read_csv(res_fname)
        
        if sel_by=='loss':
            best_p = results.iloc[results['val_loss'].idxmin()] 
        else:
            best_p = results.iloc[results[f'val_{sel_by}'].idxmax()] 
        
        bestparams = {}
        results_best = results
        
        for key in p.keys():
            bestparams[key] = best_p[key]
            results_best = results_best[results_best[key]==best_p[key]]
        
        self.bestparams = bestparams
        self.results_best = results_best
            
        return bestparams, results_best
    
    def get_trained_model(self, X_train=None, y_train=None, p=None, X_val=None, y_val=None):
        
        if p is None:
            if self.bestparams is None:
                print('provide model parameters')
            else:
                p = self.bestparams

        if X_train is None:
            X_train, y_train = self.X, self.y
                
        model, history, _ = self.nn_model(X_train, y_train, p, X_val=X_val, y_val=y_val)
        return model, history


class RF_Pipeline(object):
    
    def __init__(self, X, y):
        
        self.X = X
        self.y = y
        self.bestparams = None
        self.results_best = None
        self.bestmodel = None
        self.paramgrid = {
                       'n_estimators':[200, 400, 500],
                       'min_samples_leaf': [2, 4, 10],
                       'max_depth': [10, 20, None],
                       'class_weight': ['balanced']
                      }
        
    def rf_model(self, X_train, y_train, params,  X_val=None, y_val=None, verbose=1):
        
        if X_val is None:
            validation_data = None
        else:
            validation_data = (X_val, y_val)
        

        model = RandomForestClassifier(n_estimators=params['n_estimators'], 
                                       min_samples_leaf=params['min_samples_leaf'],
                                       max_depth=params['max_depth'],
                                       class_weight=params['class_weight']
                                       )

        res = {}
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        res['accuracy'] = [accuracy_score(y_train, y_train_pred)]
        res['balanced_accuracy'] = [balanced_accuracy_score(y_train, y_train_pred)]
        res['precision'] = [precision_score(y_train, y_train_pred, average='weighted')]
        res['recall'] = [recall_score(y_train, y_train_pred, average='weighted')]
        res['f1'] = [f1_score(y_train, y_train_pred, average='weighted')]
        res['mcc'] = [matthews_corrcoef(y_train, y_train_pred)]

        if X_val is not None:
            y_val_pred = model.predict(X_val)
            res['val_accuracy'] = [accuracy_score(y_val, y_val_pred)]
            res['val_balanced_accuracy'] = [balanced_accuracy_score(y_val, y_val_pred)]
            res['val_precision'] = [precision_score(y_val, y_val_pred, average='weighted')]
            res['val_recall'] = [recall_score(y_val, y_val_pred, average='weighted')]
            res['val_f1'] = [f1_score(y_val, y_val_pred, average='weighted')]
            res['val_mcc'] = [matthews_corrcoef(y_val, y_val_pred)]

        return model, res
        
    def tune_hyperparam(self, res_fname, p=None):
        
        if p is None:
            p = self.paramgrid
        else: 
            self.paramgrid = p
        
        # get train and val sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        keys, values = zip(*p.items())
        param_combinations = [dict(zip(keys, v)) for v in it.product(*values)]

        i=1
        results_list = []

        for params in param_combinations:
            print(f'{i} of {len(param_combinations)}')
            model, res = self.rf_model(X_train, y_train, params, X_val, y_val)
            df_res = pd.DataFrame(res)
            for key in params.keys():
                df_res[key] = params[key]
            results_list.append(df_res)
            i = i+1
        
        results = pd.concat(results_list, axis=0)
        results = results.reset_index(drop=True)
        resutls.to_csv(res_fname, index=False)
        
        return results
    
    def get_best_params(self, res_fname, p=None, sel_by='accuracy', prefit=True):
        
        if p is None:
            p = self.paramgrid
        else:
            self.paramgrid = p
        
        if len(glob.glob(res_fname))==0 or not prefit:
            results = self.tune_hyperparam(res_fname, p=p)
        else:
            results = pd.read_csv(res_fname)
        
        best_p = results.iloc[results[f'val_{sel_by}'].idxmax()] 
        
        bestparams = {}
        results_best = results
        
        for key in p.keys():
            bestparams[key] = best_p[key]
            results_best = results_best[results_best[key]==best_p[key]]
        
        self.bestparams = bestparams
        self.results_best = results_best
            
        return bestparams, results_best
    
    def get_trained_model(self, X_train=None, y_train=None, p=None, X_val=None, y_val=None):
        
        if p is None:
            if self.bestparams is None:
                print('provide model parameters')
            else:
                p = self.bestparams

        if X_train is None:
            X_train, y_train = self.X, self.y
        
        model, res = self.rf_model(X_train, y_train, p, X_val=X_val, y_val=y_val)
        return model, res
            


def train_nn(train, X_cols, ycol, nn_params, fname):

    X = train[X_cols]
    y = train[ycol]
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    class_weights = {i: weight for i, weight in enumerate(class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y))}
    y = to_categorical(y)

    nnpip = NN_Pipeline(X, y, class_weights)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    _, history, res = nnpip.nn_model(X_train, y_train, nn_params, X_val, y_val)
    df_res = pd.DataFrame(history.history)
    df_res['epoch'] = df_res.index + 1
    df_res = df_res[df_res['epoch']==df_res['epoch'].max()].reset_index(drop=True)
    df_res = df_res.drop('epoch', axis=1).rename(columns={'accuracy': 'accuracy_last_epoch', 'val_accuracy': 'val_accuracy_last_epoch'})
    for key in res.keys():
        df_res[key] = res[key]
    for key in nn_params.keys():
        df_res[key] = nn_params[key]

    # model, _, _ = nnpip.nn_model(X, y, nn_params)
    # model_dict = {'model': model, 'label_encoder': label_encoder}
    # joblib.dump(model_dict, fname) 

    return df_res

def train_rf(train, X_cols, ycol, rf_params, fname):

    X = train[X_cols]
    y = train[ycol]

    rfpip = RF_Pipeline(X,y)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    _, res = rfpip.rf_model(X_train, y_train, rf_params, X_val, y_val)
    df_res = pd.DataFrame(res)
    for key in rf_params.keys():
        df_res[key] = rf_params[key]

    # model, _ = rfpip.rf_model(X, y, rf_params)
    # joblib.dump(model, fname) 

    return df_res
    


if __name__ == '__main__':
    
    
    train, X_cols = get_labels()
    nn_res = []
    rf_res = []

    nn_params = {
        'dense_layers': 3, 
        'batch_size': 128,
        'epochs': 100,
        'dropout': 0.2,
        'dense_neurons': 512
    }

    rf_params = {
        'n_estimators': 200, 
        'min_samples_leaf': 4,
        'max_depth': None,
        'class_weight': 'balanced'
    }

    print('NN for level 0')
    df_res = train_nn(train, X_cols, 'parentcategory', nn_params, '../../SFS/NDNS UK/trained_models/nn_lev0.joblib') 
    df_res['category'] = 'lev 0'
    nn_res.append(df_res)
    print('RF for level 0\n')
    df_res = train_rf(train, X_cols, 'parentcategory', rf_params, '../../SFS/NDNS UK/trained_models/rf_lev0.joblib')
    df_res['category'] = 'lev 0'
    rf_res.append(df_res)
    
    for category in train['parentcategory'].unique().tolist():
        
        print(category)
        
        train_df = train[train['parentcategory']==category]
        
        if category not in ['Savoury Snacks', 'Alcoholic Beverages', 'Nuts and Seeds', 'Not Food', 'Dietary Supplements', 
                        'Artificial Sweeteners', 'Commercial Toddlers Foods and Drinks']:

            rf_params = {
                'n_estimators': 500, 
                'min_samples_leaf': 10,
                'max_depth': 20,
                'class_weight': 'balanced'
            }

            if category in ['Cereals and Cereal Products', 'Meat and Meat Products', 'Miscellaneous', 'Sugar, Preserves and Confectionery']:
                nn_params = {
                    'dense_layers': 2, 
                    'batch_size': 64,
                    'epochs': 100,
                    'dropout': 0.2,
                    'dense_neurons': 256
                }
            if category in ['Non-Alcoholic Beverages']:
                nn_params = {
                    'dense_layers': 1, 
                    'batch_size': 32,
                    'epochs': 50,
                    'dropout': 0.4,
                    'dense_neurons': 128
                }

            if category in ['Milk and Milk Products']:
                nn_params = {
                    'dense_layers': 2, 
                    'batch_size': 64,
                    'epochs': 100,
                    'dropout': 0.4,
                    'dense_neurons': 256
                }

            if category in ['Vegetables, Potatoes', 'Fish and Fish Dishes', 'Fruit']:
                nn_params = {
                    'dense_layers': 1, 
                    'batch_size': 32,
                    'epochs': 50,
                    'dropout': 0.4,
                    'dense_neurons': 64
                }

            if category in ['Fat Spreads', 'Eggs and Egg Dishes']:
                nn_params = {
                    'dense_layers': 1, 
                    'batch_size': 16,
                    'epochs': 50,
                    'dropout': 0.4,
                    'dense_neurons': 32
                }

            print('NN')
            df_res = train_nn(train_df, X_cols, 'subfoodgroup', nn_params, f'../../SFS/NDNS UK/trained_models/nn_{category}.joblib')
            df_res['category'] = category
            nn_res.append(df_res)
            print('RF\n')
            df_res = train_rf(train_df, X_cols, 'subfoodgroup', rf_params, f'../../SFS/NDNS UK/trained_models/rf_{category}.joblib')
            df_res['category'] = category
            rf_res.append(df_res)


    nn_res = pd.concat(nn_res, axis=0, ignore_index=True)
    rf_res = pd.concat(rf_res, axis=0, ignore_index=True)
    nn_res.to_csv('../../SFS/NDNS UK/trained_models/NN_performance.csv', index=False)
    rf_res.to_csv('../../SFS/NDNS UK/trained_models/RF_performance.csv', index=False)
        