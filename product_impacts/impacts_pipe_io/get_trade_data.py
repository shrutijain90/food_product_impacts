# Usage: python -m product_impacts.impacts_pipe_io.get_trade_data

import pandas as pd
import numpy as np

def get_prod_matrix(prod, item, year, FAO_area_codes, processing_factors):
    
    # preparing production matrix
    prod = prod[['Area Code (M49)', 'Area', 'Item', 'Element', 'Year', 'Unit', 'Value']]
    prod = prod[(prod['Year']==year) & (prod['Item']==item) & (prod['Element']=='Production')].reset_index(drop=True)
    prod = prod.rename(columns={'Value': 'Production'})
    prod = prod.drop('Element', axis=1)
    
    prod['Area Code (M49)'] = prod.apply(lambda row: int(row['Area Code (M49)'][1:]), axis=1)
    prod = prod.rename(columns={'Area Code (M49)': 'M49 Code'})
    prod = prod.merge(FAO_area_codes, how='right')
    prod = prod[['iso3', 'M49 Code', 'Production']]
    prod = prod.fillna(0)
    prod = prod.groupby('iso3').sum()[['Production']].reset_index().sort_values(by='iso3')
    
    # applying a processing factor for sugar and oil crops where needed
    prod['Item'] = item
    prod = prod.merge(processing_factors, how='left')
    prod.loc[prod['crop_factor'].isna(), 'crop_factor'] = 1
    prod['Production'] = prod['Production'] * prod['crop_factor']
    
    P = prod[['Production']].to_numpy()

    # adjustments to rice production
    if item=='Rice':
        P = P * 0.7 # scaling down to account for milling
    
    return P

def get_fish_prod_matrix(prod, item, year, FAO_area_codes):
    
    # preparing production matrix
    prod = prod[['Area Code (M49)', 'Area', 'Item', 'Element', 'Year', 'Unit', 'Value']]
    prod = prod[(prod['Year']==year) & (prod['Item'].isin(item)) & (prod['Element']=='Production')].reset_index(drop=True)
    prod = prod.rename(columns={'Value': 'Production'})
    prod = prod.drop('Element', axis=1)
    
    prod['Area Code (M49)'] = prod.apply(lambda row: int(row['Area Code (M49)'][1:]), axis=1)
    prod = prod.rename(columns={'Area Code (M49)': 'M49 Code'})
    prod = prod.groupby(['M49 Code', 'Area'])[['Production']].sum().reset_index()
    prod = prod.merge(FAO_area_codes, how='right')
    prod = prod[['iso3', 'M49 Code', 'Production']]
    prod = prod.fillna(0)
    prod = prod.groupby('iso3').sum()[['Production']].reset_index().sort_values(by='iso3')
    
    P = prod[['Production']].to_numpy()
    
    return P

def get_trade_matrix(mat, prod, item, year, FAO_area_codes, trade_items, trade_factors, processing_factors):
    
    # special function to split refined and row sugars into sugar cane and sugar beet categories 
    # the split flow functions cannot be used here as not all subcategories within the sugar cane and sugar beet categories need to be split
    def _split_sugars(m, prod, item):
        prod = prod[['Area Code (M49)', 'Area', 'Item', 'Element', 'Year', 'Unit', 'Value']]
        prod = prod[(prod['Year']==year) & (prod['Item'].isin(['Sugar beet', 'Sugar cane'])) & (prod['Element']=='Production')].reset_index(drop=True)
        prod = prod.rename(columns={'Value': 'Production'})
        prod = prod.drop('Element', axis=1)
        prod = prod.merge(prod.groupby('Area')[['Production']].sum().reset_index().rename(columns={'Production': 'total'}))
        prod = prod[prod['total']!=0].reset_index(drop=True)
        prod['prop'] = prod['Production'] / prod['total']
        prod = prod[prod['Item']==item]
        
        m = m.merge(prod[['Area', 'prop']], how='left', left_on='Reporter Countries', right_on='Area').drop('Area', axis=1).rename(
            columns={'prop': 'prop_reporter'})
        m = m.merge(prod[['Area', 'prop']], how='left', left_on='Partner Countries', right_on='Area').drop('Area', axis=1).rename(
            columns={'prop': 'prop_partner'})
        m = m.fillna(0)

        m.loc[(m['Element']=='Export Quantity') 
              & (m['Item'].isin(['Raw cane or beet sugar (centrifugal only)',
                                 'Refined sugar'])), 'Value'] = m[
            (m['Element']=='Export Quantity') 
            & (m['Item'].isin(['Raw cane or beet sugar (centrifugal only)',
                               'Refined sugar']))]['Value'] * m[(m['Element']=='Export Quantity')
                                                                & (m['Item'].isin(['Raw cane or beet sugar (centrifugal only)',
                                                                                   'Refined sugar']))]['prop_reporter']

        m.loc[(m['Element']=='Import Quantity') 
              & (m['Item'].isin(['Raw cane or beet sugar (centrifugal only)',
                                 'Refined sugar'])), 'Value'] = m[
            (m['Element']=='Import Quantity') 
            & (m['Item'].isin(['Raw cane or beet sugar (centrifugal only)',
                               'Refined sugar']))]['Value'] * m[(m['Element']=='Import Quantity')
                                                                & (m['Item'].isin(['Raw cane or beet sugar (centrifugal only)',
                                                                                   'Refined sugar']))]['prop_partner']
        
        return m.drop(['prop_reporter', 'prop_partner'], axis=1)

        
    # preparing trade matrix

    if len(trade_items)==0:
        E = np.zeros((FAO_area_codes['iso3'].nunique(), FAO_area_codes['iso3'].nunique()))
        return E
    else:
        mat = mat[(mat['Year']==year) & (mat['Item'].isin(trade_items)) 
                & (mat['Unit']=='t')][['Reporter Country Code (M49)', 'Partner Country Code (M49)',
                                            'Reporter Countries', 'Partner Countries', 'Item', 'Element', 
                                            'Year', 'Unit', 'Value']].reset_index(drop=True)

        if len(mat)==0:
            E = np.zeros((FAO_area_codes['iso3'].nunique(), FAO_area_codes['iso3'].nunique()))
            return E
        
        # applying a factor to account for trade of a differently processed product where needed 
        mat = mat.merge(trade_factors, how='left')
        mat.loc[mat['factor'].isna(), 'factor'] = 1
        mat['Value'] = mat['Value'] * mat['factor']

        # applying a processing factor for sugar and oil crops where needed
        mat = mat.merge(processing_factors, how='left')
        mat.loc[mat['crop_factor'].isna(), 'crop_factor'] = 1
        if item in ['Sugar beet', 'Sugar cane']:
            mat['crop_factor'] = processing_factors[processing_factors['Item']=='Sugar & Sweeteners']['crop_factor'].values[0]
            mat.loc[mat['Item'].isin(['Sugar beet', 'Sugar cane']), 'crop_factor'] = 1
        mat['Value'] = mat['Value'] * mat['crop_factor']
        
        # split refined and raw sugars into sugar cane and sugar beet
        if item in ['Sugar beet', 'Sugar cane']:
            mat = _split_sugars(mat, prod, item)
        
        mat = mat.groupby(['Reporter Country Code (M49)', 'Partner Country Code (M49)',
                           'Reporter Countries', 'Partner Countries', 'Element', 'Year', 'Unit'])[['Value']].sum().reset_index()
        mat['Item'] = item
        
    # reliability index from "Reconciling bilateral trade data for use in GTAP"
    # accuracy level for each import 
    imports = mat[mat['Element']=='Import Quantity']
    imports = imports.rename(columns={'Reporter Countries': 'Country A', 'Partner Countries': 'Country B',
                                      'Reporter Country Code (M49)': 'Country A Code', 'Partner Country Code (M49)': 'Country B Code'})
    exports = mat[mat['Element']=='Export Quantity']
    exports = exports.rename(columns={'Reporter Countries': 'Country B', 'Partner Countries': 'Country A',
                                      'Reporter Country Code (M49)': 'Country B Code', 'Partner Country Code (M49)': 'Country A Code'})

    df = pd.concat([imports, exports], axis=0, ignore_index=True)
    df = pd.pivot(df, index=['Country A', 'Country B', 'Country A Code', 'Country B Code', 'Item', 'Year', 'Unit'], 
                  columns = 'Element',values = 'Value').reset_index()
    df = df.fillna(0)

    if 'Import Quantity' not in df.columns.tolist():
        df['Import Quantity'] = 0
    if 'Export Quantity' not in df.columns.tolist():
        df['Export Quantity'] = 0
    
    df = df.rename(columns={'Import Quantity': 'Import rep A', 'Export Quantity': 'Export rep B',})
    df = df[(df['Import rep A']!=0) | (df['Export rep B']!=0)].reset_index(drop=True)
    df['AL'] = 2*(df['Import rep A'] - df['Export rep B']).abs()/(df['Import rep A'] + df['Export rep B'])
    row_cond = (df['AL']==2) & (df['Import rep A']<10) & (df['Export rep B']<10)
    df.loc[row_cond, 'AL'] = -1
    
    def _calc_rel_index(g, col, ind_col):
        d = g.copy()
        d = d[d['AL']!=-1]
        # d['WAL'] = d[col] / d[col].sum() * d['AL']
        # d = d[d['WAL']!=d['WAL'].max()]
        RI = d[d['AL']<=0.2][col].sum() / d[col].sum()
        g[ind_col] = RI 
        return g
    
    df = df.groupby('Country A').apply(lambda g: _calc_rel_index(g, 'Import rep A', 'RIM')).reset_index(drop=True)
    df = df.groupby('Country B').apply(lambda g: _calc_rel_index(g, 'Export rep B', 'RIX')).reset_index(drop=True)
    df = df.fillna(0)
    
    def _select_qty(row):
        if row['RIM'] >= row['RIX']:
            return row['Import rep A']
        else:
            return row['Export rep B']
        
    df['From B to A'] = df.apply(lambda row: _select_qty(row), axis=1)
    df['Country A Code'] = df.apply(lambda row: int(row['Country A Code'][1:]), axis=1)
    df['Country B Code'] = df.apply(lambda row: int(row['Country B Code'][1:]), axis=1)
    
    trade_mat = df[['Country A', 'Country B', 'Country A Code', 'Country B Code', 'Item', 'Year',
                    'Unit', 'From B to A']]
    
    trade_mat = trade_mat.merge(FAO_area_codes, left_on='Country A Code', right_on='M49 Code', how='right')
    trade_mat = trade_mat.drop(['Country A', 'Country A Code'], axis=1)
    trade_mat = trade_mat.rename(columns={'M49 Code': 'Country A M49', 'iso3': 'Country A iso3'})
    trade_mat = trade_mat.sort_values(by='Country A iso3')
    
    def _add_all_countries(m):
        m = m.merge(FAO_area_codes, left_on='Country B Code', right_on='M49 Code', how='right')
        m = m.drop(['Country B', 'Country B Code', 'Country A iso3', 'Country A M49'], axis=1)
        m = m.rename(columns={'M49 Code': 'Country B M49', 'iso3': 'Country B iso3'})
        m = m.sort_values(by='Country B iso3')
        return m

    trade_mat = trade_mat.groupby(['Country A iso3', 'Country A M49']).apply(lambda g: _add_all_countries(g)).reset_index()

    trade_mat = trade_mat.drop('level_2', axis=1)
    trade_mat = trade_mat.fillna(0)
    trade_mat.loc[trade_mat['Country A iso3']==trade_mat['Country B iso3'], 'From B to A'] = 0
    trade_mat = trade_mat[['Country A iso3', 'Country B iso3', 'From B to A']].groupby(['Country A iso3', 'Country B iso3']).sum().reset_index()
    trade_mat = pd.pivot(trade_mat, index=['Country B iso3'], columns = 'Country A iso3',values = 'From B to A').reset_index()
    E = trade_mat.drop(['Country B iso3'], axis=1).to_numpy()
    return E

def get_fish_trade_matrix(mat, year, FAO_area_codes, category):
    
    # to map this data on to categories of interest and on to our area codes
    item_codes = pd.read_csv('../../data/FI_Trade_Partners_2024.1.0/CL_FI_COMMODITY_ISSCFC.csv')
    country_codes = pd.read_csv('../../data/FI_Trade_Partners_2024.1.0/CL_FI_COUNTRY_GROUPS.csv')
    
    mat = mat.merge(item_codes, left_on='COMMODITY.FAO_CODE', right_on='Code')
    mat = mat[mat['SITCv4']<38] # removing all misc categrories like feed
    mat['Item'] = 'fish'
    mat.loc[(mat['SITCv4']<37) & (mat['SITCv4']>=36), 'Item'] = 'crustaceans'
    
    mat = mat[(mat['PERIOD']==year) & (mat['Item']==category) 
             & (mat['TRADE_FLOW.ALPHA_CODE']=='I')] # considering the imports as the data has both exports and re-exports separately
                                                    # re-export algorithm will take care of the allocation later
    mat = mat[['COUNTRY_REPORTER.UN_CODE', 'COUNTRY_PARTNER.UN_CODE', 'VALUE']]
    mat = mat.groupby(['COUNTRY_REPORTER.UN_CODE', 'COUNTRY_PARTNER.UN_CODE']).sum().reset_index()
    mat = mat.merge(country_codes[['UN_Code', 'ISO3_Code']], left_on='COUNTRY_REPORTER.UN_CODE', right_on='UN_Code').drop(
        ['COUNTRY_REPORTER.UN_CODE', 'UN_Code'], axis=1).rename(columns={'ISO3_Code': 'Country A iso3', 'VALUE': 'From B to A'})
    mat = mat.merge(country_codes[['UN_Code', 'ISO3_Code']], left_on='COUNTRY_PARTNER.UN_CODE', right_on='UN_Code').drop(
        ['COUNTRY_PARTNER.UN_CODE', 'UN_Code'], axis=1).rename(columns={'ISO3_Code': 'Country B iso3'})
    
    trade_mat = mat.merge(FAO_area_codes, left_on='Country A iso3', right_on='iso3', how='right')
    trade_mat = trade_mat.drop(['M49 Code', 'Country A iso3'], axis=1).rename(columns={'iso3': 'Country A iso3'})
    trade_mat = trade_mat.sort_values(by='Country A iso3')
    
    def _add_all_countries(m):
        m = m.merge(FAO_area_codes, left_on='Country B iso3', right_on='iso3', how='right')
        m = m.drop(['M49 Code', 'Country A iso3', 'Country B iso3'], axis=1).rename(columns={'iso3': 'Country B iso3'})
        m = m.sort_values(by='Country B iso3')
        return m
    
    trade_mat = trade_mat.groupby(['Country A iso3']).apply(lambda g: _add_all_countries(g)).reset_index()
    trade_mat = trade_mat.fillna(0)
    trade_mat.loc[trade_mat['Country A iso3']==trade_mat['Country B iso3'], 'From B to A'] = 0
    trade_mat = trade_mat[['Country A iso3', 'Country B iso3', 'From B to A']].groupby(['Country A iso3', 'Country B iso3']).sum().reset_index()
    trade_mat = pd.pivot(trade_mat, index=['Country B iso3'], columns = 'Country A iso3',values = 'From B to A').reset_index()
    E = trade_mat.drop(['Country B iso3'], axis=1).to_numpy()
    return E

def re_export_algo(P, E):
    # Implements the trade matrix re-export algorithm as given in Croft et al., 2018 (https://www.sciencedirect.com/science/article/pii/S0959652618326180#appsec2)
    
    # Number of iterations
    N = 100000 

    # Number of countries
    num_ctry = len(P)

    # Pre-calculate diagonal Production matrix
    Pd = np.diagflat(P)

    # Pre-allocate Domestic Supply matrix
    D = np.zeros((num_ctry, num_ctry))

    for n in range(1,N+1):
        # STEP 1: allocate production
        # Allocate production to domestic supply
        D += Pd / N

        # STEP 2: perform trade
        # Calculate proportions of domestic supply required for each component of export iteration
        temp1 = E / N / np.tile(np.sum(D, axis=0), (num_ctry, 1)).T

        # Sum to check if greater than 1 (if domestic supply is less than desired export total)
        temp2 = np.tile(np.nansum(temp1, axis=1)[:, np.newaxis], (1, num_ctry))

        # Constrain export greater than domestic supply to be equal to domestic supply
        mask = np.tile(np.nansum(temp1, axis=1) > 1, (num_ctry, 1)).T # or np.tile(np.nansum(temp1, axis=1)[:, np.newaxis]>1, (1, num_ctry))
        temp1[mask] = temp1[mask] / temp2[mask]
        
        # Proportional change in domestic supply
        e_n = np.ones((num_ctry, 1)) - np.nansum(temp1, axis=1)[:, np.newaxis]

        # Apply to domestic supply of domestic production (non-traded component)
        e_n = np.diagflat(e_n) + temp1

        # Take care of 0/0 cases
        e_n[np.isnan(e_n)] = 0

        # Take care of x/0 cases
        e_n[np.isinf(e_n)] = 0

        # Rescale domestic supply to redistribute according to trade
        D = D.dot(e_n)
        
    return D

def split_flows(sua, item, year, FAO_area_codes, proc):
    sua = sua[(sua['Element'].isin(['Feed', 'Food supply quantity (tonnes)', 'Loss', 'Other uses (non-food)',
                                    'Residuals', 'Seed', 'Processed', 'Tourist consumption']))
    & (sua['Year']==year) ].groupby(['Area', 'Area Code (M49)', 'Element', 'Item'])[['Value']].mean().reset_index().pivot(
        index=['Area', 'Area Code (M49)', 'Item'], columns='Element', values='Value').reset_index()

    sua = sua.rename(columns={'Food supply quantity (tonnes)': 'Food', 'Loss': 'Losses', 'Processed': 'Processing'})
    sua = sua.fillna(0)
    sua['Domestic supply quantity'] = sua['Feed'] + sua['Food'] + sua['Losses'] + sua['Seed'] + sua['Processing'] + sua['Other uses (non-food)'] \
    + sua['Tourist consumption'] + sua['Residuals']
    sua.loc[sua['Domestic supply quantity']<0, 'Domestic supply quantity'] = 0

    if item=='Olives':
        sua1 = sua[sua['Item']=='Olives']
        sua2 = sua[sua['Item']=='Olives preserved']
        sua = sua1[['Area Code (M49)', 'Processing']].merge(sua2[['Area Code (M49)', 'Domestic supply quantity']], how='left')
        sua.loc[sua['Domestic supply quantity'].isna(), 'Domestic supply quantity'] = 0
        sua['Processing'] = sua['Processing'] - sua['Domestic supply quantity']
        sua = sua[['Area Code (M49)', 'Processing']].merge(sua1[['Area Code (M49)', 'Domestic supply quantity']], how='left')
        
    elif item=='Coconuts, in shell':
        sua1 = sua[sua['Item']=='Coconuts, in shell']
        sua2 = sua[sua['Item']=='Coconuts, desiccated']
        sua = sua1[['Area Code (M49)', 'Processing']].merge(sua2[['Area Code (M49)', 'Domestic supply quantity']], how='left')
        sua.loc[sua['Domestic supply quantity'].isna(), 'Domestic supply quantity'] = 0
        sua['Processing'] = sua['Processing'] - sua['Domestic supply quantity']
        sua = sua[['Area Code (M49)', 'Processing']].merge(sua1[['Area Code (M49)', 'Domestic supply quantity']], how='left')
        
    elif item=='Groundnuts, excluding shelled':
        sua1 = sua[sua['Item']=='Groundnuts, excluding shelled']
        sua2 = sua[sua['Item'].isin(['Groundnuts, shelled', 'Prepared groundnuts'])]
        sua2 = sua2.groupby('Area Code (M49)')[['Domestic supply quantity']].sum().reset_index()
        sua = sua1[['Area Code (M49)', 'Processing']].merge(sua2[['Area Code (M49)', 'Domestic supply quantity']], how='left')
        sua.loc[sua['Domestic supply quantity'].isna(), 'Domestic supply quantity'] = 0
        sua['Processing'] = sua['Processing'] - sua['Domestic supply quantity']
        sua = sua[['Area Code (M49)', 'Processing']].merge(sua1[['Area Code (M49)', 'Domestic supply quantity']], how='left')
        
    elif item=='Soya beans':
        sua1 = sua[sua['Item']=='Soya beans']
        sua2 = sua[sua['Item'].isin(['Soya paste', 'Soya sauce', 'Soya curd'])]
        sua2 = sua2.groupby('Area Code (M49)')[['Domestic supply quantity']].sum().reset_index()
        sua = sua1[['Area Code (M49)', 'Processing']].merge(sua2[['Area Code (M49)', 'Domestic supply quantity']], how='left')
        sua.loc[sua['Domestic supply quantity'].isna(), 'Domestic supply quantity'] = 0
        sua['Processing'] = sua['Processing'] - sua['Domestic supply quantity']
        sua = sua[['Area Code (M49)', 'Processing']].merge(sua1[['Area Code (M49)', 'Domestic supply quantity']], how='left')
        
    elif item=='Milk, Total':
        sua1 = sua[sua['Item'].isin(['Raw milk of buffalo', 'Raw milk of camel', 'Raw milk of cattle',
                                     'Raw milk of goats', 'Raw milk of sheep'])]
        sua1 = sua1.groupby('Area Code (M49)')[['Processing', 'Domestic supply quantity']].sum().reset_index()
        sua2 = sua[sua['Item'].isin(['Butter and ghee of sheep milk', 'Butter of cow milk', 
                                     'Butter of goat milk', 'Buttermilk, curdled and acidified milk', 
                                     'Buttermilk, dry', 'Cream, fresh', 'Dairy products n.e.c.', 'Ghee from buffalo milk',
                                     'Ghee from cow milk', 'Ice cream and other edible ice', 
                                     'Skim milk of cows', 'Skim milk, condensed', 'Skim milk, evaporated', 
                                     'Whole milk powder', 'Whole milk, condensed', 'Whole milk, evaporated', 'Yoghurt', 
                                     'Yoghurt, with additives'])]
        sua2 = sua2.groupby('Area Code (M49)')[['Domestic supply quantity']].sum().reset_index()
        sua = sua1[['Area Code (M49)', 'Processing']].merge(sua2[['Area Code (M49)', 'Domestic supply quantity']], how='left')
        sua = sua.fillna(0)
        sua['Processing'] = sua['Processing'] - sua['Domestic supply quantity']
        sua = sua[['Area Code (M49)', 'Processing']].merge(sua1[['Area Code (M49)', 'Domestic supply quantity']], how='left')
        sua = sua.fillna(0)
        
    else:
        sua = sua[sua['Item']==item][['Area Code (M49)', 'Processing', 'Domestic supply quantity']]

    sua.loc[sua['Processing']<0, 'Processing'] = 0
    sua['Area Code (M49)'] = sua.apply(lambda row: int(row['Area Code (M49)'][1:]), axis=1)
    sua = sua.rename(columns={'Area Code (M49)': 'M49 Code'})
    sua = sua.merge(FAO_area_codes, how='right')
    sua = sua[['iso3', 'M49 Code', 'Domestic supply quantity', 'Processing']]
    sua = sua.fillna(0)
    sua = sua.groupby('iso3').sum()[['Domestic supply quantity', 'Processing']].reset_index().sort_values(by='iso3')
    sua['proc_prop'] = sua['Processing'] / sua['Domestic supply quantity']
    sua.loc[sua['Domestic supply quantity']==0, 'proc_prop'] = 0
    sua.loc[sua['proc_prop']>1, 'proc_prop'] = 1
    if proc==0:
        sua['proc_prop'] = 1 - sua['proc_prop']
    sua = sua.sort_values('iso3').reset_index(drop=True)

    return sua['proc_prop'].values

if __name__ == '__main__':
    
    # Enter crops and years
    
    # some categories in faostat were ignored - like bread, breakfast cereals, homogenized vegetable preparations etc
    trade_dict = {'Wheat': ['Wheat', 'Germ of wheat', 'Wheat and meslin flour'], 
                  'Rice': ['Rice, paddy (rice milled equivalent)', 'Flour of rice'], # not considering bran of rice as it is removed during milling
                  'Maize (corn)': ['Maize (corn)', 'Germ of maize', 'Flour of maize', 'Bran of maize', 'Cake of maize', 'Forage and silage, maize'], 
                  'Rye': ['Rye', 'Flour of rye', 'Bran of rye'], 
                  'Barley': ['Barley', 'Barley flour and grits', 'Barley, pearled', 'Bran of barley', 'Pot barley'], 
                  'Oats': ['Oats', 'Oats, rolled', 'Bran of oats'], 
                  'Sorghum': ['Sorghum', 'Flour of sorghum', 'Bran of sorghum'], 
                  'Buckwheat': ['Buckwheat', 'Flour of buckwheat'], 
                  'Millet': ['Millet', 'Flour of millet', 'Bran of millet'], 
                  'Quinoa': ['Quinoa'], 
                  'Canary seed': ['Canary seed'], 
                  'Fonio': ['Fonio'], 
                  'Mixed grain': ['Mixed grain', 'Flour of mixed grain', 'Bran of mixed grain'], 
                  'Triticale': ['Triticale'], 
                  'Cereals n.e.c.': ['Cereals n.e.c.', 'Flour of cereals n.e.c.', 'Bran of cereals n.e.c.'],
                  'Cassava, fresh': ['Cassava, dry', 'Cassava, fresh', 'Flour of cassava', 'Starch of cassava', 'Tapioca of cassava'], 
                  'Potatoes': ['Potatoes', 'Potatoes, frozen', 'Tapioca of potatoes'], 
                  'Sweet potatoes': ['Sweet potatoes'], 
                  'Taro': ['Taro'], 
                  'Yams': ['Yams'], 
                  'Edible roots and tubers with high starch or inulin content, n.e.c., fresh': ['Edible roots and tubers with high starch or inulin content, n.e.c., fresh'], 
                  'Artichokes': ['Artichokes'], 
                  'Asparagus': ['Asparagus'], 
                  'Broad beans and horse beans, green': ['Broad beans and horse beans, green'], 
                  'Cabbages': ['Cabbages'], 
                  'Carrots and turnips': ['Carrots and turnips'], 
                  'Cauliflowers and broccoli': ['Cauliflowers and broccoli'], 
                  'Chillies and peppers, green (Capsicum spp. and Pimenta spp.)': ['Chillies and peppers, green (Capsicum spp. and Pimenta spp.)'], 
                  'Cucumbers and gherkins': ['Cucumbers and gherkins'], 
                  'Eggplants (aubergines)': ['Eggplants (aubergines)'], 
                  'Green corn (maize)': ['Green corn (maize)'], 
                  'Green garlic': ['Green garlic'], 
                  'Leeks and other alliaceous vegetables': ['Leeks and other alliaceous vegetables'], 
                  'Lettuce and chicory': ['Lettuce and chicory'], 
                  'Mushrooms and truffles': ['Canned mushrooms', 'Dried mushrooms', 'Mushrooms and truffles'], 
                  'Okra': ['Okra'], 
                  'Onions and shallots, dry (excluding dehydrated)': ['Onions and shallots, dry (excluding dehydrated)'], 
                  'Onions and shallots, green': ['Onions and shallots, green'], 
                  'Other beans, green': ['Other beans, green'], 
                  'Other vegetables, fresh n.e.c.': ['Other vegetables, fresh n.e.c.', 'Other vegetables provisionally preserved', 'Other vegetable juices',
                                                     'Sweet corn, frozen', 'Sweet corn, prepared or preserved',
                                                     'Vegetables frozen', 'Vegetables preserved (frozen)', 'Vegetables preserved nes (o/t vinegar)', 
                                                     'Vegetables, dehydrated', 'Vegetable products, fresh or dry n.e.c.', 
                                                     'Vegetables, pulses and potatoes, preserved by vinegar or acetic acid'
                                                    ], 
                  'Peas, green': ['Peas, green'], 
                  'Pumpkins, squash and gourds': ['Pumpkins, squash and gourds'], 
                  'Spinach': ['Spinach'], 
                  'String beans': ['String beans'], 
                  'Tomatoes': ['Paste of tomatoes', 'Tomato juice', 'Tomatoes', 'Tomatoes, peeled (o/t vinegar)'], 
                  'Apples': ['Apple juice', 'Apple juice, concentrated', 'Apples'], 
                  'Apricots': ['Apricots', 'Apricots, dried'], 
                  'Avocados': ['Avocados'], 
                  'Bananas': ['Bananas'], 
                  'Blueberries': ['Blueberries'], 
                  'Cantaloupes and other melons': ['Cantaloupes and other melons'], 
                  'Cashewapple': ['Cashewapple'], 
                  'Cherries': ['Cherries'], 
                  'Cranberries': ['Cranberries'], 
                  'Currants': ['Currants'], 
                  'Dates': ['Dates'], 
                  'Figs': ['Figs', 'Figs, dried'], 
                  'Gooseberries': ['Gooseberries'], 
                  'Grapes': ['Grape juice', 'Grapes'], 
                  'Kiwi fruit': ['Kiwi fruit'], 
                  'Lemons and limes': ['Juice of lemon', 'Lemon juice, concentrated', 'Lemons and limes'], 
                  'Locust beans (carobs)': ['Locust beans (carobs)'], 
                  'Mangoes, guavas and mangosteens': ['Juice of mango', 'Mangoes, guavas and mangosteens', 'Mango pulp'], 
                  'Oranges': ['Orange juice', 'Orange juice, concentrated', 'Oranges'], 
                  'Other berries and fruits of the genus vaccinium n.e.c.': ['Other berries and fruits of the genus vaccinium n.e.c.'], 
                  'Other citrus fruit, n.e.c.': ['Citrus juice, concentrated n.e.c.', 'Juice of citrus fruit n.e.c.', 'Other citrus fruit, n.e.c.'], 
                  'Other fruits, n.e.c.': ['Fruit prepared n.e.c.', 'Juice of fruits n.e.c.', 'Other fruit n.e.c., dried', 'Other fruits, n.e.c.', 'Raisins'], 
                  'Other pome fruits': [],
                  'Other stone fruits': ['Other stone fruits'], 
                  'Other tropical fruits, n.e.c.': ['Other tropical fruit, dried', 'Other tropical fruits, n.e.c.'], 
                  'Papayas': ['Papayas'], 
                  'Peaches and nectarines': ['Peaches and nectarines'], 
                  'Pears': ['Pears'], 
                  'Persimmons': ['Persimmons'], 
                  'Pineapples': ['Juice of pineapples, concentrated', 'Pineapple juice', 'Pineapples', 'Pineapples, otherwise prepared or preserved'],  
                  'Plantains and cooking bananas': ['Plantains and cooking bananas'], 
                  'Plums and sloes': ['Plums and sloes', 'Plums, dried'], 
                  'Pomelos and grapefruits': ['Grapefruit juice', 'Grapefruit juice, concentrated', 'Pomelos and grapefruits'], 
                  'Quinces': ['Quinces'], 
                  'Raspberries': ['Raspberries'], 
                  'Sour cherries': ['Sour cherries'], 
                  'Strawberries': ['Strawberries'], 
                  'Tangerines, mandarins, clementines': ['Juice of tangerine', 'Tangerines, mandarins, clementines'], 
                  'Watermelons': ['Watermelons'], 
                  'Olives': ['Olives', 'Olives preserved', 'Olive oil'], # trade to be split, oil should be converted to oil crop
                  'Coconuts, in shell': ['Coconuts, in shell', 'Coconuts, desiccated', 'Coconut oil'], # trade to be split, oil should be converted to oil crop
                  'Bambara beans, dry': ['Bambara beans, dry'], 
                  'Beans, dry': ['Beans, dry'], 
                  'Broad beans and horse beans, dry': ['Broad beans and horse beans, dry'], 
                  'Chick peas, dry': ['Chick peas, dry'], 
                  'Cow peas, dry': ['Cow peas, dry'], 
                  'Lentils, dry': ['Lentils, dry'], 
                  'Lupins': [], 
                  'Other pulses n.e.c.': ['Flour of pulses', 'Other pulses n.e.c.'], 
                  'Peas, dry': ['Peas, dry'], 
                  'Pigeon peas, dry': ['Pigeon peas, dry'], 
                  'Vetches': [], 
                  'Soya beans': ['Cake of soya beans', 'Soya beans', 'Soya curd', 'Soya paste', 'Soya sauce', 'Soya bean oil'], 
                  'Almonds, in shell': ['Almonds, in shell', 'Almonds, shelled'], 
                  'Brazil nuts, in shell': ['Brazil nuts, in shell', 'Brazil nuts, shelled'], 
                  'Cashew nuts, in shell': ['Cashew nuts, in shell', 'Cashew nuts, shelled'], 
                  'Chestnuts, in shell': ['Chestnuts, in shell'], 
                  'Hazelnuts, in shell': ['Hazelnuts, in shell', 'Hazelnuts, shelled'], 
                  'Other nuts (excluding wild edible nuts and groundnuts), in shell, n.e.c.': ['Other nuts (excluding wild edible nuts and groundnuts), in shell, n.e.c.'], 
                  'Pistachios, in shell': ['Pistachios, in shell'], 
                  'Walnuts, in shell': ['Walnuts, in shell', 'Walnuts, shelled'], 
                  'Groundnuts, excluding shelled': ['Groundnuts, excluding shelled', 'Groundnuts, shelled', 
                                                    'Prepared groundnuts', 'Groundnut oil'], # trade to be split, oil should be converted to oil crop
                  'Linseed': ['Linseed', 'Oil of linseed'], # trade to be split, oil should be converted to oil crop
                  'Hempseed': ['Hempseed'], # trade to be split, oil should be converted to oil crop
                  'Sunflower seed': ['Sunflower seed', 'Sunflower-seed oil, crude'], # trade to be split, oil should be converted to oil crop
                  'Safflower seed': ['Safflower seed', 'Safflower-seed oil, crude'], # trade to be split, oil should be converted to oil crop
                  'Poppy seed': ['Poppy seed'], 
                  'Sesame seed': ['Sesame seed', 'Oil of sesame seed'], # trade to be split, oil should be converted to oil crop
                  'Castor oil seeds': ['Castor oil seeds', 'Oil of castor beans'], # oil to be converted to oil crop
                  'Cotton seed': ['Cottonseed oil'], # oil to be converted to oil crop
                  'Mustard seed': ['Mustard seed', 'Mustard seed oil, crude'], # oil to be converted to oil crop
                  'Rape or colza seed': ['Rape or colza seed', 'Rapeseed or canola oil, crude'], # oil to be converted to oil crop
                  'Oil palm fruit': ['Palm oil'], # oil to be converted to oil crop
                  'Sugar beet': ['Sugar beet', 'Refined sugar', 'Raw cane or beet sugar (centrifugal only)'], # trade of sugar to be split, sugar to be converted to sugar crop
                  'Sugar cane': ['Sugar cane', 'Refined sugar', 'Raw cane or beet sugar (centrifugal only)'], # trade of sugar to be split, sugar to be converted to sugar crop
                  'Milk, Total': ['Butter and ghee of sheep milk',
                                  'Butter of cow milk', 'Butter of goat milk', 'Buttermilk, curdled and acidified milk', 
                                  'Buttermilk, dry', 'Casein', 'Cheese from milk of goats, fresh or processed',
                                  'Cheese from milk of buffalo, fresh or processed', 
                                  'Cheese from milk of sheep, fresh or processed', 'Cheese from skimmed cow milk', 
                                  'Cheese from whole cow milk', 'Cream, fresh', 'Dairy products n.e.c.', 'Ghee from buffalo milk',
                                  'Ghee from cow milk', 
                                  'Ice cream and other edible ice', 'Processed cheese', 'Raw milk of cattle', 
                                  'Raw milk of sheep', 'Reconstituted milk', 'Skim milk and whey powder',
                                  'Skim milk of cows', 'Skim milk, condensed', 'Skim milk, evaporated', 'Standardized milk', 
                                  'Whey cheese', 'Whey, condensed', 'Whey, dry', 'Whey, fresh', 'Whole milk powder', 
                                  'Whole milk, condensed', 'Whole milk, evaporated', 'Yoghurt', 
                                  'Yoghurt, with additives'], ##### assuming that processing factors and by products take care of each other
                  'Beef and Buffalo Meat, primary': ['Bovine meat, salted, dried or smoked', 'Meat of buffalo, fresh or chilled',
                                                     'Meat of cattle boneless, fresh or chilled', 'Meat of cattle with the bone, fresh or chilled',
                                                     'Sausages and similar products of meat, offal or blood of beef and veal'],
                  'Eggs Primary': ['Hen eggs in shell, fresh', 'Eggs from other birds in shell, fresh, n.e.c.',
                                   'Eggs, dried', 'Eggs, liquid'],
                  'Meat of chickens, fresh or chilled': ['Meat of chickens, fresh or chilled'],
                  'Meat of turkeys, fresh or chilled': ['Meat of turkeys, fresh or chilled'],
                  'Meat of ducks, fresh or chilled': ['Meat of ducks, fresh or chilled'], 
                  'Meat of geese, fresh or chilled': ['Meat of geese, fresh or chilled'], 
                  'Meat of pigeons and other birds n.e.c., fresh, chilled or frozen': ['Meat of pigeons and other birds n.e.c., fresh, chilled or frozen'],
                  'Sheep and Goat Meat': ['Meat of goat, fresh or chilled', 'Meat of sheep, fresh or chilled'],
                  'Meat of pig with the bone, fresh or chilled': ['Meat of pig boneless, fresh or chilled',
                                                                  'Meat of pig with the bone, fresh or chilled',
                                                                  'Pig meat, cuts, salted, dried or smoked (bacon and ham)',
                                                                  'Sausages and similar products of meat, offal or blood of pig'],
                  'Coffee, green': ['Coffee, green', 'Coffee husks and skins', 'Coffee, decaffeinated or roasted'], ##### not including coffee extracts as it includes other things in the weight
                  'Tea leaves': ['Tea leaves'],
                  'Cocoa beans': ['Cocoa beans'], ##### not including processed items as they include weight of other ingredients
                  'Wine': ['Wine']
    }

    # based on lca data categories 
    # it is probably okay to consider the primary crop overall instead of processed products like cheese or bread, 
    # because the lca ratios for processed products are also applied to the primary crop lca estimates for a given region  
    items_dict = {
                  # wheat
                  # 'wheat': ['Wheat'],
                  
                  # rice
                  # 'rice': ['Rice'],
                  
                  # maize
                  # 'maize': ['Maize (corn)'],
                  
                  # othr_grains
                  # 'barley': ['Barley'],
                  # 'oats': ['Oats'],
                  # 'rye': ['Rye'],
                  # 'millet': ['Millet'], 
                  # 'sorghum': ['Sorghum'], 
                  # 'other_cereals': ['Buckwheat', 'Quinoa', 'Canary seed', 'Fonio', 'Mixed grain', 'Triticale', 'Cereals n.e.c.'], 
                  
                  # roots
                  # 'cassava': ['Cassava, fresh'], 
                  # 'potatoes': ['Potatoes'], 
                  # 'sweet_potatoes_yams': ['Sweet potatoes', 'Yams'],
                  # 'other_tubers': ['Taro', 'Edible roots and tubers with high starch or inulin content, n.e.c., fresh'],
                  
                  # vegetables
                  # 'peas': ['Peas, green'],
                  # 'onions': ['Onions and shallots, green', 'Onions and shallots, dry (excluding dehydrated)'],
                  # 'leeks': ['Leeks and other alliaceous vegetables'],
                  # 'tomatoes': ['Tomatoes'],
                  # 'carrots': ['Carrots and turnips'],
                  # 'cabbage': ['Cabbages'],
                  # 'broc_cauli': ['Cauliflowers and broccoli'],
                  # 'cucumber': ['Cucumbers and gherkins'],
                  # 'squash': ['Pumpkins, squash and gourds'],
                  # 'artichokes': ['Artichokes'],
                  # 'lettuce_chicory': ['Lettuce and chicory'],
                  # 'green_beans': ['Broad beans and horse beans, green', 'Other beans, green', 'String beans'],
                  # 'other_veg': ['Asparagus', 'Chillies and peppers, green (Capsicum spp. and Pimenta spp.)', 'Eggplants (aubergines)', 'Green corn (maize)',
                  #               'Green garlic', 'Mushrooms and truffles', 'Okra', 'Other vegetables, fresh n.e.c.', 'Spinach'],
                  
                  # fruits
                  # 'bananas': ['Bananas', 'Plantains and cooking bananas'], 
                  # 'oranges': ['Oranges'],
                  # 'lemons': ['Lemons and limes'],
                  # 'other_citrus': ['Tangerines, mandarins, clementines', 'Pomelos and grapefruits', 'Other citrus fruit, n.e.c.'],
                  # 'apples': ['Apples'],
                  # 'strawberries': ['Strawberries'],
                  # 'raspberries': ['Raspberries'],
                  # 'pears': ['Pears'],
                  # 'peach': ['Peaches and nectarines', 'Plums and sloes', 'Apricots'],
                  # 'avacado': ['Avocados'],
                  # 'melon': ['Watermelons', 'Cantaloupes and other melons'],
                  # 'kiwi': ['Kiwi fruit'],
                  # 'grapes': ['Grapes'],
                  # 'other_fruit': ['Cashewapple', 'Dates', 'Figs', 'Locust beans (carobs)', 'Mangoes, guavas and mangosteens',
                  #                 'Other fruits, n.e.c.', 'Other tropical fruits, n.e.c.', 'Papayas', 'Pineapples',   
                  #                 'Coconuts, in shell', 'Blueberries', 'Cherries', 'Cranberries', 'Currants', 'Gooseberries', 
                  #                 'Other berries and fruits of the genus vaccinium n.e.c.', 'Other pome fruits', 'Other stone fruits', 
                  #                 'Persimmons', 'Quinces', 'Sour cherries'], 
                  # 'olives': [ 'Olives'],
                  
                  # legumes
                  # 'beans': ['Bambara beans, dry', 'Beans, dry', 'Broad beans and horse beans, dry'], 
                  # 'lentils': ['Lentils, dry', 'Other pulses n.e.c.', 'Chick peas, dry', 'Cow peas, dry', 'Pigeon peas, dry', 'Peas, dry'], 
                  # 'lupins': ['Lupins'],  
                  
                  # soybeans
                  # 'jsoyb': ['Soya beans'],
                  # 'soyb_oil': ['Soya beans'],
                  
                  # nuts_seeds
                  # 'jgrnd': ['Groundnuts, excluding shelled'], 
                  # 'almonds': ['Almonds, in shell'],
                  # 'cashews': ['Cashew nuts, in shell'],
                  # 'hazelnuts': ['Hazelnuts, in shell'],
                  # 'chestnuts': ['Chestnuts, in shell'],
                  # 'walnuts': ['Walnuts, in shell'],
                  # 'snfl_seed': ['Sunflower seed'],
                  # 'othr_nuts': ['Brazil nuts, in shell',
                  #           'Other nuts (excluding wild edible nuts and groundnuts), in shell, n.e.c.', 'Pistachios, in shell', 
                  #           'Linseed', 'Hempseed', 'Safflower seed', 'Poppy seed', 'Sesame seed'],
                  
                  
                  # oil_veg
                  # 'jrpsd': ['Rape or colza seed'], 
                  # 'snfl_seed_oil': ['Sunflower seed'],
                  # 'olive_oil': ['Olives'], 
                  # 'other_oilcrops': ['Groundnuts, excluding shelled', 'Linseed', 'Hempseed', 'Safflower seed', 'Sesame seed',
                  #                    'Castor oil seeds', 'Cotton seed', 'Coconuts, in shell', 'Mustard seed'], 
                  
                  # oil_palm
                  # 'palm_oil': ['Oil palm fruit'], 
                  
                  # sugar
                  # 'beet_sugar': ['Sugar beet'], 
                  # 'cane_sugar': ['Sugar cane'],   

                  # meat, eggs, dairy
                  # 'milk': ['Milk, Total'],
                  # 'cheese': ['Milk, Total'],
                  # 'beef': ['Beef and Buffalo Meat, primary'],
                  # 'eggs': ['Eggs Primary'],
                  # 'chicken': ['Meat of chickens, fresh or chilled'],
                  # 'turkey': ['Meat of turkeys, fresh or chilled'],
                  # 'other_poultry': ['Meat of ducks, fresh or chilled', 'Meat of geese, fresh or chilled', 'Meat of pigeons and other birds n.e.c., fresh, chilled or frozen'],
                  # 'lamb': ['Sheep and Goat Meat'],
                  # 'pork': ['Meat of pig with the bone, fresh or chilled'],
                  
                  # other - chocolate, coffee, tea, wine 
                  # 'coffee': ['Coffee, green'], 
                  # 'tea': ['Tea leaves'],
                  # 'chocolate': ['Cocoa beans'],
                  # 'wine': ['Wine'],
                  'fish': [['Freshwater Fish', 'Demersal Fish', 'Pelagic Fish', 'Marine Fish, Other']],
                  'crustaceans': [['Crustaceans', 'Cephalopods', 'Molluscs, Other']]
    }

    # prod = pd.read_csv('../../data/FAOSTAT_A-S_E/Production_Crops_Livestock_E_All_Data_(Normalized)/Production_Crops_Livestock_E_All_Data_(Normalized).csv',
    #                    encoding='latin1')
    # mat = pd.read_csv('../../data/FAOSTAT_T-Z_E/Trade_DetailedTradeMatrix_E_All_Data_(Normalized)/Trade_DetailedTradeMatrix_E_All_Data_(Normalized).csv',
    #                   encoding='latin1')
    # sua = pd.read_csv('../../data/FAOSTAT_A-S_E/SUA_Crops_Livestock_E_All_Data_(Normalized)/SUA_Crops_Livestock_E_All_Data_(Normalized).csv',
    #                       encoding='latin1', low_memory=False)
    # trade_factors = pd.read_csv('../../OPSIS/Data/FAOSTAT/trade_factors.csv') # for things like cassava starch etc
    # processing_factors = pd.read_csv('../../OPSIS/Data/FAOSTAT/processing_factors.csv') # for oil and sugar crops
    FAO_area_codes = pd.read_csv('../../SFS/FAOSTAT/area_codes.csv')
    
    fish_prod = pd.read_csv('../../data/FAOSTAT_A-S_E/FoodBalanceSheets_E_All_Data_(Normalized)/FoodBalanceSheets_E_All_Data_(Normalized).csv',
                 encoding='latin1')
    fish_prod['Value'] = fish_prod['Value'] * 1000 # as FBS reports production in 1000t units
    fish_mat = pd.read_csv('../../data/FI_Trade_Partners_2024.1.0/TRADE_PARTNERS_QUANTITY.csv')
    
    for category in items_dict.keys():
        print(category)
        if category in ['fish', 'crustaceans']:
            years = [2019, 2020, 2021, 2022] # years available for fish data
        else:
            years = [2017, 2018, 2019, 2020, 2021] # 2022 has some information missing (e.g. production for coconut oil), so considering 2017-2021 
        items = items_dict[category]

        df_P_list = []
        df_E_list = []
        df_D_list = []
    
        for year in years:
            print(year)
            
            # create empty matrix
            category_P = np.zeros((FAO_area_codes['iso3'].nunique(),1))
            category_E = np.zeros((FAO_area_codes['iso3'].nunique(), FAO_area_codes['iso3'].nunique()))
            category_D = np.zeros((FAO_area_codes['iso3'].nunique(), FAO_area_codes['iso3'].nunique()))
            
            for item in items:
                print(item)
                
                if category in ['fish', 'crustaceans']:
                    P = get_fish_prod_matrix(fish_prod, item, year, FAO_area_codes)
                    E = get_fish_trade_matrix(fish_mat, year, FAO_area_codes, category)
                else:
                    P = get_prod_matrix(prod, item, year, FAO_area_codes, processing_factors)
                    E = get_trade_matrix(mat, prod, item, year, FAO_area_codes, trade_dict[item], trade_factors, processing_factors)
                D = re_export_algo(P, E)

                # adjustments due to duplication across categories
                if item in ['Olives', 'Coconuts, in shell', 'Groundnuts, excluding shelled', 'Linseed', 'Hempseed', 'Sunflower seed', 'Safflower seed', 'Sesame seed', 'Soya beans', 'Milk, Total']:
                    if category in ['snfl_seed_oil', 'olive_oil', 'other_oilcrops', 'soyb_oil', 'cheese']:
                        proc_prop = split_flows(sua, item, year, FAO_area_codes, proc=1)
                    else:
                        proc_prop = split_flows(sua, item, year, FAO_area_codes, proc=0)
                    P = P * proc_prop[:, np.newaxis]
                    E = E * proc_prop[:, np.newaxis]
                    D = D * proc_prop[:, np.newaxis]
    
                # add to existing matrix for category
                category_P = np.add(category_P, P)
                category_E = np.add(category_E, E)
                category_D = np.add(category_D, D)

            abbr = FAO_area_codes['iso3'].unique()
            abbr.sort()
    
            df_category_P = pd.DataFrame(category_P, columns=['prod'])
            df_category_P['iso3'] = pd.Series(abbr.tolist(), index=df_category_P.index)
            first_column = df_category_P.pop('iso3')
            df_category_P.insert(0, 'iso3', first_column)
            df_P_list.append(df_category_P)
            
            df_category_E = pd.DataFrame(category_E, columns = abbr.tolist())
            df_category_E['iso3'] = pd.Series(abbr.tolist(), index=df_category_E.index)
            first_column = df_category_E.pop('iso3')
            df_category_E.insert(0, 'iso3', first_column)
            df_E_list.append(df_category_E)
            
            df_category_D = pd.DataFrame(category_D, columns = abbr.tolist())
            df_category_D['iso3'] = pd.Series(abbr.tolist(), index=df_category_D.index)
            first_column = df_category_D.pop('iso3')
            df_category_D.insert(0, 'iso3', first_column)
            df_D_list.append(df_category_D)
        
        df_category_P = pd.concat(df_P_list).groupby('iso3').mean().reset_index()
        df_category_P.to_csv(f'../../SFS/FAOSTAT/FAO_prod_mat/prod_matrix_{category}_{years[0]}_{years[-1]}.csv', index=False)
        df_category_E = pd.concat(df_E_list).groupby('iso3').mean().reset_index()
        df_category_E.to_csv(f'../../SFS/FAOSTAT/FAO_bal_trade_mat/trade_matrix_{category}_{years[0]}_{years[-1]}.csv', index=False)
        df_category_D = pd.concat(df_D_list).groupby('iso3').mean().reset_index()
        df_category_D.to_csv(f'../../SFS/FAOSTAT/FAO_re_export/supply_matrix_{category}_{years[0]}_{years[-1]}.csv', index=False)
