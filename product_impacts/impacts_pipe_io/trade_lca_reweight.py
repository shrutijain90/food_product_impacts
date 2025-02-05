# Usage: python -m product_impacts.impacts_pipe_io.trade_lca_reweight

import pandas as pd
import numpy as np
import logging

logging.basicConfig(filename="../../SFS/environmental_impacts/Data Inputs/LCA_data_by_country/trade_info.txt", level=logging.INFO, format="%(message)s")

def scale_weights(g):
    w = g['weight_true'].values[0]
    g['Weight'] = (g['Weight'] / g['Weight'].sum()) * w
    return g

def reweight_lca(lca, country_groups, category, fao_cats, iso3):
    
    lca = lca.merge(country_groups[['Country', 'iso3']])
    
    if category in ['Fish (farmed)', 'Crustaceans (farmed)']:
        df_D = pd.concat([pd.read_csv(f'../../SFS/FAOSTAT/FAO_re_export/supply_matrix_{cat}_2019_2022.csv') for cat in fao_cats],
                         axis=0, ignore_index=True)
    else:
        df_D = pd.concat([pd.read_csv(f'../../SFS/FAOSTAT/FAO_re_export/supply_matrix_{cat}_2017_2021.csv') for cat in fao_cats],
                         axis=0, ignore_index=True)
    df_D = df_D.groupby('iso3').sum().reset_index()
    df_D = df_D[['iso3', iso3]]
    df_D['weight_true'] = (df_D[iso3] / df_D[iso3].sum()) * 100
    df_D.loc[df_D['weight_true']<0.1, 'weight_true'] = 0
    df_D.loc[df_D['weight_true']==0, iso3] = 0
    df_D['weight_true'] = (df_D[iso3] / df_D[iso3].sum()) * 100
    df_D = df_D[df_D['weight_true']>0]
    df_D = df_D.merge(country_groups)
    
    df_lca = lca[lca['Data S2 Name']==category]
    df_lca = df_lca.merge(country_groups)
    
    # first use existing information for countries that supply
    df_lca_sub = df_lca.merge(df_D[['iso3', 'Intermediate Region Name', 'Sub-region Name', 'Region Name', 'weight_true']])
    if len(df_lca_sub)>0:
        df_lca_sub = df_lca_sub.groupby('iso3').apply(lambda g: scale_weights(g)).reset_index(drop=True)
    df_D = df_D[~df_D['iso3'].isin(df_lca_sub['iso3'].values)]
    df_lca_sub = df_lca_sub.drop(['iso3', 'M49 Code', 
                                  'Intermediate Region Name', 'Sub-region Name', 
                                  'Region Name', 'weight_true'], axis=1)
    logging.info(f'{df_lca_sub["Weight"].sum()}% from supplying countries')
    
    # for missing countries, assume supply from intermediate region (if not null), then sub-region, then region, then global
    if len(df_D)>0:
        add = df_lca.merge(df_D[df_D['Intermediate Region Name'].notnull()][['iso3', 'Intermediate Region Name', 'weight_true']],
                           left_on='Intermediate Region Name', right_on='Intermediate Region Name', how='left')
        if len(add)>0:
            add = add.groupby('iso3_y').apply(lambda g: scale_weights(g)).reset_index(drop=True)
        df_D = df_D[~df_D['iso3'].isin(add['iso3_y'].values)]
        add = add.drop(['iso3_x','M49 Code', 'Intermediate Region Name', 
                        'Sub-region Name','Region Name', 'iso3_y', 'weight_true'], axis=1)
        logging.info(f'{add["Weight"].sum()}% from supplying intermediate regions')
        df_lca_sub = pd.concat([df_lca_sub, add], axis=0, ignore_index=True)
        
        if len(df_D)>0:
            add = df_lca.merge(df_D[['iso3', 'Sub-region Name', 'weight_true']],
                               left_on='Sub-region Name', right_on='Sub-region Name', how='left')
            if len(add)>0:
                add = add.groupby('iso3_y').apply(lambda g: scale_weights(g)).reset_index(drop=True)
            df_D = df_D[~df_D['iso3'].isin(add['iso3_y'].values)]
            add = add.drop(['iso3_x','M49 Code', 'Intermediate Region Name', 
                            'Sub-region Name','Region Name', 'iso3_y', 'weight_true'], axis=1)
            logging.info(f'{add["Weight"].sum()}% from supplying sub-regions')
            df_lca_sub = pd.concat([df_lca_sub, add], axis=0, ignore_index=True)
            
            if len(df_D)>0:
                add = df_lca.merge(df_D[['iso3', 'Region Name', 'weight_true']],
                                   left_on='Region Name', right_on='Region Name', how='left')
                if len(add)>0:
                    add = add.groupby('iso3_y').apply(lambda g: scale_weights(g)).reset_index(drop=True)
                df_D = df_D[~df_D['iso3'].isin(add['iso3_y'].values)]
                add = add.drop(['iso3_x','M49 Code', 'Intermediate Region Name', 
                                'Sub-region Name','Region Name', 'iso3_y', 'weight_true'], axis=1)
                logging.info(f'{add["Weight"].sum()}% from supplying regions')
                df_lca_sub = pd.concat([df_lca_sub, add], axis=0, ignore_index=True)
                
                if len(df_D)>0:
                    add = df_lca
                    add['weight_true'] = df_D['weight_true'].sum()
                    add = scale_weights(add)
                    add = add.drop(['iso3', 'M49 Code', 'Intermediate Region Name', 
                                    'Sub-region Name','Region Name', 'weight_true'], axis=1)
                    logging.info(f'{add["Weight"].sum()}% from global')
                    df_lca_sub = pd.concat([df_lca_sub, add], axis=0, ignore_index=True)
                    
    df_lca_sub = df_lca_sub.groupby([c for c in df_lca_sub.columns if c!='Weight'])[['Weight']].sum().reset_index()
    
    # if all the above results in too few rows, assume global weights
    if len(df_lca_sub)<5:
        if len(df_lca) > len(df_lca_sub):
            logging.info(f'{category} has {len(df_lca_sub)} rows remaining, so considering all {len(df_lca)} rows with global weights')
            df_lca_sub = df_lca
            df_lca_sub['weight_true'] = 100
            df_lca_sub = scale_weights(df_lca_sub)
            df_lca_sub = df_lca_sub.drop(['iso3', 'M49 Code', 'Intermediate Region Name', 
                                          'Sub-region Name','Region Name', 'weight_true'], axis=1)
        
    return df_lca_sub

if __name__ == '__main__':
    
    lca = pd.read_csv('../../SFS/environmental_impacts/Data Inputs/jp_lca_dat.csv', 
                      encoding = "ISO-8859-1")
    country_groups = pd.read_csv('../../SFS/FAOSTAT/country_groups.csv')
    
    # removing 3 rows where weight is '-' and converting weights to float
    lca.loc[lca['Weight']=='-', 'Weight'] = np.nan
    lca['Weight'] = lca['Weight'].str.rstrip('%').astype('float')
    lca = lca[lca['Weight'].notnull()]
    lca['Weight'] = lca['Weight'] + 0.1 # to remove zeros
    
    # changing some country names to match with FAO data
    lca = lca.replace("C\x99te d'Ivoire",'Ivory Coast')
    lca = lca.replace("Iran (Islamic Republic of)",'Iran, Islamic Republic of')
    lca = lca.replace("Russian Federation",'Russia')
    lca = lca.replace("United States of America",'United States')
    lca = lca.replace("Viet Nam",'Vietnam')
    
    # dictionary to match LCA categories with FAO categories (trade matrices were exported with more disaggregation, so combining some here)
    # missing: 'tea' [not needed as it is added manually with 100% weight]
    categories_dict = {
        'Wheat & Rye (Bread)': ['wheat', 'rye', 'millet', 'sorghum', 'other_cereals', 'other_oilcrops'], 
        'Maize (Meal)': ['maize'], 
        'Barley (Beer)': ['barley'], 
        'Oatmeal': ['oats'],
        'Rice': ['rice'], 
        'Potatoes': ['potatoes', 'sweet_potatoes_yams', 'other_tubers'], 
        'Cassava': ['cassava'], 
        'Cane Sugar': ['cane_sugar'], 
        'Beet Sugar': ['beet_sugar'],
        'Other Pulses': ['beans', 'lentils', 'lupins'], 
        'Peas': ['peas'], 
        'Nuts': ['almonds', 'cashews', 'hazelnuts', 'chestnuts', 'walnuts', 'snfl_seed', 'othr_nuts'], 
        'Groundnuts': ['jgrnd'], 
        'Soymilk': ['jsoyb'],
        'Tofu': ['jsoyb'], 
        'Soybean Oil': ['soyb_oil'], 
        'Palm Oil': ['palm_oil'], 
        'Sunflower Oil': ['snfl_seed_oil'], 
        'Rapeseed Oil': ['jrpsd'],
        'Olive Oil': [ 'olive_oil'], 
        'Tomatoes': [ 'tomatoes'],
        'Onions & Leeks': ['onions', 'leeks'], 
        'Root Vegetables': ['carrots'],
        'Brassicas': ['cabbage', 'broc_cauli'], 
        'Other Vegetables': ['cucumber', 'squash', 'artichokes', 'lettuce_chicory', 'green_beans', 'other_veg'], 
        'Citrus Fruit': ['oranges', 'lemons', 'other_citrus'], 
        'Bananas': ['bananas'],
        'Apples': ['apples'], 
        'Berries & Grapes': ['strawberries', 'raspberries', 'grapes'], 
        'Wine': ['wine'], 
        'Other Fruit': ['pears', 'peach', 'avacado', 'melon', 'kiwi', 'other_fruit', 'olives'], 
        'Coffee': ['coffee'],
        'Dark Chocolate': ['chocolate'], 
        'Bovine Meat (beef herd)': ['beef'],
        'Bovine Meat (dairy herd)': ['beef'], 
        'Lamb & Mutton': ['lamb'], 
        'Pig Meat': ['pork'],
        'Poultry Meat': ['chicken', 'turkey', 'other_poultry'], 
        'Milk': ['milk'], 
        'Cheese': ['milk'],
        'Eggs': ['eggs'], 
        'Fish (farmed)': ['fish'],
        'Crustaceans (farmed)': ['crustaceans']
    }
    
    # countries with products 
    for iso3 in ['ARE', 'ARG', 'AUS', 'AUT', 'BEL', 'BGR', 'BOL', 'BRA', 'CAN',
                'CHE', 'CHL', 'COL', 'CRI', 'CUB', 'CYP', 'CZE', 'DEU', 'DNK',
                'DZA', 'ECU', 'EGY', 'ESP', 'EST', 'FIN', 'FRA', 'GBR', 'GLP',
                'GRC', 'HKG', 'HRV', 'HUN', 'IDN', 'IND', 'IRL', 'IRQ', 'ISR',
                'ITA', 'JPN', 'KOR', 'KWT', 'LBN', 'LTU', 'LUX', 'LVA', 'MAR',
                'MEX', 'MTQ', 'MYS', 'NCL', 'NLD', 'NOR', 'NZL', 'PAN', 'PER',
                'PHL', 'POL', 'PRI', 'PRT', 'PYF', 'QAT', 'REU', 'ROU', 'RUS',
                'SAU', 'SGP', 'SRB', 'SVK', 'SVN', 'SWE', 'THA', 'TUN', 'TUR',
                'UKR', 'USA', 'VEN', 'ZAF']:
        logging.info(iso3)
        lca_list = []
        for category in categories_dict.keys():
            logging.info(category)
            fao_cats = categories_dict[category]
            df_lca_sub = reweight_lca(lca, country_groups, category, fao_cats, iso3)
            lca_list.append(df_lca_sub)

        lca_country = pd.concat(lca_list, axis=0, ignore_index=True)
        lca_country.to_csv(f'../../SFS/environmental_impacts/Data Inputs/LCA_data_by_country/jp_lca_dat_{iso3}.csv', index=False)