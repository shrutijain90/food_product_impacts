# Usage: python -m product_impacts.impacts_pipe_io.get_processing_factors

import pandas as pd
import numpy as np

if __name__ == '__main__':

    # dataframe for recording processing factors for oil and sugar crops
    oil_sugar = pd.DataFrame({'Item': ['Sugar & Sweeteners', 
                                       'Groundnut oil', 'Oil of linseed', 
                                       'Oil of hempseed', 'Sunflower-seed oil, crude', 'Safflower-seed oil, crude', 
                                       'Oil of sesame seed', 'Cottonseed oil', 'Mustard seed oil, crude', 
                                       'Rapeseed or canola oil, crude', 'Oil of castor beans', 'Castor oil, hydrogenated', 
                                       'Olive oil', 'Coconut oil', 'Palm oil', 'Oil of palm kernel'
                                       ]})
    oil_sugar['crop_factor'] = 0
    oil_sugar['supply'] = 0

    # sua 
    sua = pd.read_csv('../../data/FAOSTAT_A-S_E/SUA_Crops_Livestock_E_All_Data_(Normalized)/SUA_Crops_Livestock_E_All_Data_(Normalized).csv',
                      encoding='latin1', low_memory=False)
    
    sua = sua[(sua['Element'].isin(['Feed', 'Food supply quantity (tonnes)', 'Loss', 'Other uses (non-food)',
                                    'Residuals', 'Seed', 'Processed', 'Tourist consumption']))
        & (sua['Year'].isin([2018, 2019, 2020, 2021, 2022])) 
    ].groupby(['Area', 'Area Code (M49)', 'Element', 'Item'])[['Value']].mean().reset_index().pivot(
        index=['Area', 'Area Code (M49)', 'Item'], columns='Element', values='Value').reset_index()
    
    sua = sua[sua['Item'].isin(['Groundnuts, excluding shelled', 'Groundnuts, shelled', 'Prepared groundnuts',
                                'Linseed', 'Hempseed', 'Sunflower seed', 'Safflower seed', 'Poppy seed', 'Sesame seed',
                                'Palm oil', 'Oil of palm kernel', 'Oil palm fruit',
                                'Soya bean oil', 'Cake of  soya beans', 'Groundnut oil', 'Cake of groundnuts', 'Oil of linseed', 'Cake of  linseed', 
                                'Oil of hempseed', 'Cake of hempseed', 'Sunflower-seed oil, crude', 'Cake of sunflower seed', 'Safflower-seed oil, crude', 
                                'Cake of safflowerseed', 'Oil of sesame seed', 'Cake of sesame seed',
                                'Cottonseed oil', 'Cake of cottonseed', 'Mustard seed oil, crude', 'Cake of mustard seed', 'Rapeseed or canola oil, crude',
                                'Cake of rapeseed', 'Oil of castor beans', 'Castor oil, hydrogenated', 'Castor oil seeds',
                                'Olive oil', 'Olives preserved', 'Olives', 'Coconut oil'])]
    
    sua = sua.rename(columns={'Food supply quantity (tonnes)': 'Food', 'Loss': 'Losses', 'Processed': 'Processing'})
    sua = sua.fillna(0)
    sua['Domestic supply quantity'] = sua['Feed'] + sua['Food'] + sua['Losses'] + sua['Seed'] + sua['Processing'] + sua['Other uses (non-food)'] \
    + sua['Tourist consumption'] + sua['Residuals']
    
    # processing factors
    oil_sugar.loc[oil_sugar['Item']=='Groundnut oil', 'crop_factor'] = (sua[(sua['Area']=='World') & (sua['Item']=='Groundnut oil')][
        'Domestic supply quantity'].values[0] + sua[(sua['Area']=='World') 
        & (sua['Item']=='Cake of groundnuts')]['Domestic supply quantity'].values[0]) / sua[(sua['Area']=='World') & (sua['Item']=='Groundnut oil')]['Domestic supply quantity'].values[0]
    oil_sugar.loc[oil_sugar['Item']=='Groundnut oil', 'supply'] = sua[(sua['Area']=='World') & (sua['Item']=='Groundnut oil')]['Domestic supply quantity'].values[0]
    
    oil_sugar.loc[oil_sugar['Item']=='Oil of linseed', 'crop_factor'] = (sua[(sua['Area']=='World') & (sua['Item']=='Oil of linseed')][
        'Domestic supply quantity'].values[0] + sua[(sua['Area']=='World') 
        & (sua['Item']=='Cake of  linseed')]['Domestic supply quantity'].values[0]) / sua[(sua['Area']=='World') & (sua['Item']=='Oil of linseed')]['Domestic supply quantity'].values[0]
    oil_sugar.loc[oil_sugar['Item']=='Oil of linseed', 'supply'] = sua[(sua['Area']=='World') & (sua['Item']=='Oil of linseed')]['Domestic supply quantity'].values[0]
    
    oil_sugar.loc[oil_sugar['Item']=='Oil of hempseed', 'crop_factor'] = (sua[(sua['Area']=='World') & (sua['Item']=='Oil of hempseed')][
        'Domestic supply quantity'].values[0] + sua[(sua['Area']=='World') 
        & (sua['Item']=='Cake of hempseed')]['Domestic supply quantity'].values[0]) / sua[(sua['Area']=='World') & (sua['Item']=='Oil of hempseed')]['Domestic supply quantity'].values[0]
    oil_sugar.loc[oil_sugar['Item']=='Oil of hempseed', 'supply'] = sua[(sua['Area']=='World') & (sua['Item']=='Oil of hempseed')]['Domestic supply quantity'].values[0]
    
    oil_sugar.loc[oil_sugar['Item']=='Sunflower-seed oil, crude', 'crop_factor'] = (sua[(sua['Area']=='World') & (sua['Item']=='Sunflower-seed oil, crude')][
        'Domestic supply quantity'].values[0] + sua[(sua['Area']=='World') & (sua['Item']=='Cake of sunflower seed')][
                            'Domestic supply quantity'].values[0]) / sua[(sua['Area']=='World') & (sua['Item']=='Sunflower-seed oil, crude')]['Domestic supply quantity'].values[0]
    oil_sugar.loc[oil_sugar['Item']=='Sunflower-seed oil, crude', 'supply'] = sua[(sua['Area']=='World') & (sua['Item']=='Sunflower-seed oil, crude')]['Domestic supply quantity'].values[0]
    
    oil_sugar.loc[oil_sugar['Item']=='Safflower-seed oil, crude', 'crop_factor'] = (sua[(sua['Area']=='World') & (sua['Item']=='Safflower-seed oil, crude')][
        'Domestic supply quantity'].values[0] + sua[(sua['Area']=='World') & (sua['Item']=='Cake of safflowerseed')][
                            'Domestic supply quantity'].values[0]) / sua[(sua['Area']=='World') & (sua['Item']=='Safflower-seed oil, crude')]['Domestic supply quantity'].values[0]
    oil_sugar.loc[oil_sugar['Item']=='Safflower-seed oil, crude', 'supply'] = sua[(sua['Area']=='World') & (sua['Item']=='Safflower-seed oil, crude')]['Domestic supply quantity'].values[0]
    
    oil_sugar.loc[oil_sugar['Item']=='Oil of sesame seed', 'crop_factor'] = (sua[(sua['Area']=='World') & (sua['Item']=='Oil of sesame seed')][
        'Domestic supply quantity'].values[0] + sua[(sua['Area']=='World') & (sua['Item']=='Cake of sesame seed')][
                                    'Domestic supply quantity'].values[0]) / sua[(sua['Area']=='World') & (sua['Item']=='Oil of sesame seed')]['Domestic supply quantity'].values[0]
    oil_sugar.loc[oil_sugar['Item']=='Oil of sesame seed', 'supply'] = sua[(sua['Area']=='World') & (sua['Item']=='Oil of sesame seed')]['Domestic supply quantity'].values[0]
    
    oil_sugar.loc[oil_sugar['Item']=='Cottonseed oil', 'crop_factor'] = (sua[(sua['Area']=='World') & (sua['Item']=='Cottonseed oil')][
        'Domestic supply quantity'].values[0] + sua[(sua['Area']=='World') & (sua['Item']=='Cake of cottonseed')][
                                    'Domestic supply quantity'].values[0]) / sua[(sua['Area']=='World') & (sua['Item']=='Cottonseed oil')]['Domestic supply quantity'].values[0]
    oil_sugar.loc[oil_sugar['Item']=='Cottonseed oil', 'supply'] = sua[(sua['Area']=='World') & (sua['Item']=='Cottonseed oil')]['Domestic supply quantity'].values[0]
    
    oil_sugar.loc[oil_sugar['Item']=='Mustard seed oil, crude', 'crop_factor'] = (sua[(sua['Area']=='World') & (sua['Item']=='Mustard seed oil, crude')][
        'Domestic supply quantity'].values[0] + sua[(sua['Area']=='World') & (sua['Item']=='Cake of mustard seed')][
                                    'Domestic supply quantity'].values[0]) / sua[(sua['Area']=='World') & (sua['Item']=='Mustard seed oil, crude')]['Domestic supply quantity'].values[0]
    oil_sugar.loc[oil_sugar['Item']=='Mustard seed oil, crude', 'supply'] = sua[(sua['Area']=='World') & (sua['Item']=='Mustard seed oil, crude')]['Domestic supply quantity'].values[0]
    
    oil_sugar.loc[oil_sugar['Item']=='Rapeseed or canola oil, crude', 'crop_factor'] = (sua[(sua['Area']=='World') & (sua['Item']=='Rapeseed or canola oil, crude')][
        'Domestic supply quantity'].values[0] + sua[(sua['Area']=='World') & (sua['Item']=='Cake of rapeseed')][
                                    'Domestic supply quantity'].values[0]) / sua[(sua['Area']=='World') & (sua['Item']=='Rapeseed or canola oil, crude')][
                                    'Domestic supply quantity'].values[0]
    oil_sugar.loc[oil_sugar['Item']=='Rapeseed or canola oil, crude', 'supply'] = sua[(sua['Area']=='World') & (sua['Item']=='Rapeseed or canola oil, crude')]['Domestic supply quantity'].values[0]
    
    oil_sugar.loc[oil_sugar['Item'].isin(['Oil of castor beans', 'Castor oil, hydrogenated']), 'crop_factor'] = sua[(sua['Area']=='World') & (sua['Item']=='Castor oil seeds')][
                                    'Processing'].values[0] / sua[(sua['Area']=='World') & (sua['Item'].isin([
        'Oil of castor beans', 'Castor oil, hydrogenated']))]['Domestic supply quantity'].sum()
    oil_sugar.loc[oil_sugar['Item']=='Oil of castor beans', 'supply'] = sua[(sua['Area']=='World') & (sua['Item']=='Oil of castor beans')]['Domestic supply quantity'].values[0]
    oil_sugar.loc[oil_sugar['Item']=='Castor oil, hydrogenated', 'supply'] = sua[(sua['Area']=='World') & (sua['Item']=='Castor oil, hydrogenated')]['Domestic supply quantity'].values[0]
    
    oil_sugar.loc[oil_sugar['Item']=='Olive oil', 'crop_factor'] = (sua[(sua['Area']=='World') & (sua['Item']=='Olives')][
        'Processing'].values[0] - sua[(sua['Area']=='World') & (sua['Item']=='Olives preserved')][
        'Domestic supply quantity'].values[0]) / sua[(sua['Area']=='World') & (sua['Item']=='Olive oil')]['Domestic supply quantity'].values[0]
    oil_sugar.loc[oil_sugar['Item']=='Olive oil', 'supply'] = sua[(sua['Area']=='World') & (sua['Item']=='Olive oil')]['Domestic supply quantity'].values[0]
    
    oil_sugar.loc[oil_sugar['Item']=='Coconut oil', 'crop_factor'] = 1
    oil_sugar.loc[oil_sugar['Item']=='Coconut oil', 'supply'] = sua[(sua['Area']=='World') & (sua['Item']=='Coconut oil')]['Domestic supply quantity'].values[0]
    
    oil_sugar.loc[oil_sugar['Item'].isin(['Palm oil', 'Oil of palm kernel']), 'crop_factor'] = sua[(sua['Area']=='World') & (sua['Item']=='Oil palm fruit')][
                                    'Processing'].values[0] / sua[(sua['Area']=='World') & (sua['Item'].isin([
        'Palm oil', 'Oil of palm kernel']))]['Domestic supply quantity'].sum()
    oil_sugar.loc[oil_sugar['Item']=='Palm oil', 'supply'] = sua[(sua['Area']=='World') & (sua['Item']=='Palm oil')]['Domestic supply quantity'].values[0]
    oil_sugar.loc[oil_sugar['Item']=='Oil of palm kernel', 'supply'] = sua[(sua['Area']=='World') & (sua['Item']=='Oil of palm kernel')]['Domestic supply quantity'].values[0]
    
    # merging with lancet regions
    sua['Area Code (M49)'] = sua.apply(lambda row: int(row['Area Code (M49)'][1:]), axis=1)
    
    sua = sua.merge(regions[['Abbreviation', 'M49 Code','Region or country', 
                           ]], left_on='Area Code (M49)', right_on='M49 Code').drop(['M49 Code',
                                                                                   'Area', 'Area Code (M49)', 
                                                                                   ], axis=1)
    sua = sua.groupby(['Abbreviation', 'Region or country', 'Item']).sum()[['Feed', 'Food', 'Losses', 
                                                                            'Other uses (non-food)', 'Residuals', 'Seed', 'Processing', 
                                                                            'Tourist consumption']].reset_index()



    # fbs
    fbs = pd.read_csv('../../data/FAOSTAT_A-S_E/FoodBalanceSheets_E_All_Data_(Normalized)/FoodBalanceSheets_E_All_Data_(Normalized).csv',
                      encoding='latin1')
    
    # match units (all in tonnes)
    fbs['Value'] = fbs['Value'] * 1000
    
    fbs = fbs[(fbs['Element'].isin(['Feed', 'Food', 'Domestic supply quantity', 'Losses', 'Other uses (non-food)',
                                    'Residuals', 'Seed', 'Processing', 'Tourist consumption']))
        & (fbs['Year'].isin([2018, 2019, 2020, 2021, 2022])) 
    ].groupby(['Area', 'Area Code (M49)', 'Element', 'Item'])[['Value']].mean().reset_index().pivot(
        index=['Area', 'Area Code (M49)', 'Item'], columns='Element', values='Value').reset_index()
    
    fbs = fbs[fbs['Item'].isin(['Wheat and products', 'Rice and products', 'Maize and products',
                                'Barley and products', 'Sorghum and products', 'Rye and products', 'Oats', 'Millet and products', 'Cereals, Other',
                                'Starchy Roots', 'Pulses', 'Vegetables', 'Fruits - Excluding Wine', 'Nuts and products',
                                'Sugar Crops', 'Sugar & Sweeteners'])]
    
    fbs = fbs.fillna(0)
    
    # processing factors
    oil_sugar.loc[oil_sugar['Item']=='Sugar & Sweeteners', 'crop_factor'] = fbs[(fbs['Area']=='World') & (fbs['Item']=='Sugar Crops')][
        'Processing'].values[0] / fbs[(fbs['Area']=='World') & (fbs['Item']=='Sugar & Sweeteners')]['Domestic supply quantity'].values[0]
    oil_sugar.loc[oil_sugar['Item']=='Sugar & Sweeteners', 'supply'] = fbs[(fbs['Area']=='World') & (fbs['Item']=='Sugar & Sweeteners')]['Domestic supply quantity'].values[0]
    
    # join with eat lancet regions
    fbs['Area Code (M49)'] = fbs.apply(lambda row: int(row['Area Code (M49)'][1:]), axis=1)
    
    fbs = fbs.merge(regions[['Abbreviation', 'M49 Code','Region or country', 
                           ]], left_on='Area Code (M49)', right_on='M49 Code').drop(['M49 Code',
                                                                                   'Area', 'Area Code (M49)', 
                                                                                   ], axis=1)
    fbs = fbs.groupby(['Abbreviation', 'Region or country', 'Item']).sum()[['Feed', 'Food', 
                                                                            'Losses', 'Other uses (non-food)', 'Processing', 'Residuals', 
                                                                            'Seed', 'Tourist consumption']].reset_index()

    oil_sugar.to_csv('../../OPSIS/Data/FAOSTAT/processing_factors.csv', index=False)


    