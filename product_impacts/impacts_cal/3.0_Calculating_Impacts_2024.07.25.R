#######
# Original code from: Clark et al., 2022, Estimating the environmental impacts of 57,000 food products
#######


#!/usr/bin/env Rscript

# removing memory space
rm(list = ls())

# Libraries
# I almost certainly do not need all these
# But this works
library(ggplot2)
library(plotly)
library(cowplot)
library(tidyr)
library(plyr)
library(dplyr)
# library(extrafont)
# library(extrafontdb)
# library(showtext)
library(readr)
library(ggrepel)
library(stringr)
library(parallel)
library(matrixStats)

# Setting working directory
setwd('/Users/shruti/OneDrive - Nexus365/DPhil/SFS/environmental_impacts')


# Loading functions
source('/Users/shruti/OneDrive - Nexus365/DPhil/code/food_product_impacts/product_impacts/impacts_cal/0.0_Functions_Estimating_Impacts_2024.06.17.R')

# Number of cores
n_cores = 5


#####
###
# Importing and stacking data

#####
###
# Adding fruit, veg, and nut composition by product
# Getting data frame of products we could interpolate
file.list = list.files(path = paste0(getwd(),'/Outputs'),
                       pattern = 'percent composition by ingredient UK', full.names = TRUE) #%>%
  #.[grepl(Sys.Date(),.)]

# file.list <- list.files(path = '/Users/macuser/Desktop/foodDB_outputs',full.names=TRUE,pattern = 'composition')


# stacking these
stacked.dat = data.frame()

for(i in 1:length(file.list)) {
  tmp.dat = read.csv(file.list[i], stringsAsFactors = FALSE)
  if(i == 1) {
    tmp.dat <- tmp.dat[, c("id", "product_name", "Retailer", "country", "Department", "Aisle", "Shelf", 
                           "variable", "value", "percent", "Food_Category", "Food_Category_sub", "Food_Category_sub_sub",
                           "value_not_embedded")]
    stacked.dat = rbind(stacked.dat,tmp.dat) 
  } else if(length(names(tmp.dat)) != length(names(stacked.dat))) {
    tmp.dat <- tmp.dat[,names(stacked.dat)]
    stacked.dat = rbind(stacked.dat, tmp.dat)
  } else if(length(names(tmp.dat)) == length(names(stacked.dat))) {
    stacked.dat = rbind(stacked.dat, tmp.dat)
  }
  
}
# Taking unique
stacked.dat <- unique(stacked.dat)

###
# Identifying brewed coffee
# Used later for the impact calculators
brewed.coffee <- brewed.coffee.tea(stacked.dat)
brewed.coffee <- unique(stacked.dat$id[brewed.coffee])



# Identifying products that had xxxg per 100g
# E.g., those where percent composition of an ingredient is > 100 and value is NA
# For these products, dropping all other instances of that food category
# And only removing the row that corresponds with xxxg per 100g product
# List of food cats in these products
stacked.dat.100g.cats <-
  stacked.dat %>%
  filter(percent > 100 & is.na(value)) %>%
  unique(.) %>%
  mutate(percent = as.numeric(percent)) %>%
  group_by(id, product_name, Retailer, country, Department, Aisle, Shelf, variable, value, Food_Category, Food_Category_sub, Food_Category_sub_sub, value_not_embedded) %>%
  summarise(percent = mean(percent, na.rm = TRUE)) %>%
  mutate(total_composition = NA)

# Reorganising columns
stacked.dat.100g.cats <-
  stacked.dat.100g.cats[,names(stacked.dat)]

# Filtering out this product food category combinations from stacked dat
stacked.dat <-
  stacked.dat %>% # Filtering out this combo of food cats, products, etc to avoid replicates
  filter(!(paste0(id,product_name,Retailer,country,Department,Aisle,Shelf,Food_Category, Food_Category_sub, Food_Category_sub_sub) %in%
           paste0(stacked.dat.100g.cats$id, stacked.dat.100g.cats$product_name,stacked.dat.100g.cats$Retailer,stacked.dat.100g.cats$country,stacked.dat.100g.cats$Department,stacked.dat.100g.cats$Aisle,stacked.dat.100g.cats$Shelf,stacked.dat.100g.cats$Food_Category)))

# And rbinding two data frames
stacked.dat <-
  rbind(as.data.frame(stacked.dat),
        as.data.frame(stacked.dat.100g.cats)) %>%
  unique(.)



#######

# # Data from products with no listed ingredients
# # These normally correspond to i.e. potatoes, tomatoes (other produce) or things like "milk", "cheese", etc
# # Also managing to have same column names and orders as stacked dat above
# dat.no.ingredients = 
#   read.csv(list.files(paste0(getwd(),'/Outputs'),pattern='no.*list',full.names=TRUE),
#   # read.csv("/Users/macuser/Desktop/foodDB_outputs/FoodDB estimated products no ingredient list 18January2022.csv",
#            stringsAsFactors = FALSE) %>%
#   mutate(Food_Category_sub = NA, Food_Category_sub_sub = NA) %>%
#   dplyr::select(id = product_id, product_name, Retailer, Department = department, Aisle = aisle, Shelf = shelf, Food_Category,Food_Category_sub, Food_Category_sub_sub, url) %>%
#   mutate(percent = 100, value_not_embedded = NA, variable = NA, value = NA) %>%
#   unique(.)
# # Updating column names
# dat.no.ingredients <- 
#   dat.no.ingredients[,names(stacked.dat)]

#######

# Stacking data sets
stacked.dat <-
  rbind(stacked.dat %>% dplyr::select(id, product_name, value, Retailer, country, Department, Aisle, Shelf, Food_Category, Food_Category_sub, Food_Category_sub_sub, percent)) %>%
        #dat.no.ingredients %>% dplyr::select(id, product_name, value, Retailer, Department, Aisle, Shelf, Food_Category,  Food_Category_sub, Food_Category_sub_sub,percent)) %>%
  mutate(percent = as.numeric(percent)) %>%
  unique(.)


# Identifying salt and water ----
# Doing this to avoid skewing composition of product with NAs
# And to accurately identify prods where >= 75% of composition is known

# Identifying salt
stacked.dat <-
  stacked.dat %>%
  mutate(Food_Category = ifelse(is.na(Food_Category) & grepl('\\bsalt\\b', value, ignore.case = TRUE) & !is.na(value),'Salt',Food_Category)) %>%
  mutate(Salt = ifelse(Food_Category %in% 'Salt', percent / 2.5 * 1000,0))

# Identifying Water
stacked.dat <-
  stacked.dat %>%
  mutate(Food_Category = ifelse(is.na(Food_Category) & grepl('water', value, ignore.case = TRUE) & !is.na(value), 'Water',Food_Category))

# Saving dataset used for later classification of products into e.g. oils, fats, cheese, etc
stacked.dat.save <-
  stacked.dat %>%
  dplyr::select(id, product_name, Retailer, country, Department, Aisle, Shelf) %>%
  unique(.)

# Identifying info for nutriscore ----
# Identifying walnut oil
walnut.oil <-
  stacked.dat %>%
  mutate(Food_Category_Old = Food_Category) %>%
  mutate(Food_Category = NA) %>%
  mutate(Food_Category = ifelse(grepl("walnut.*oil", value, ignore.case = TRUE), 'Walnut Oil',NA)) %>%
  filter(!is.na(Food_Category)) 

# Updating walnut oil in stacked dat (assuming it is olive oil, because we don't really have a better match)
stacked.dat <-
  stacked.dat %>%
  mutate(Food_Category = ifelse(grepl('walnut.*oil',value,ignore.case=TRUE),'Olive Oil',Food_Category))

# Getting fvno (fruit, veg, nut, and oil) and sugar composition
fvno.sugar <-
  stacked.dat %>%
  mutate(Sugar = 0) %>%
  mutate(percent = as.numeric(percent)) %>%
  mutate(FVNO = ifelse(Food_Category %in% c('Apples','Bananas','Berries & Grapes',
                                            'Brassicas','Citrus Fruit','Groundnuts',
                                            'Nuts','Olives','Onions & Leeks',
                                            'Other Pulses','Other Vegetables','Peas',
                                            'Root Vegetables','Tofu','Tomatoes',
                                            'Olive Oil','Rapeseed Oil'),percent,0)) %>%
  mutate(Sugar = ifelse(Food_Category %in% c('Cane Sugar','Beet Sugar'), percent, 0)) %>%
  unique(.) %>%
  group_by(id, product_name, Retailer, country, Department, Aisle, Shelf) %>% 
  summarise(FVNO = sum(FVNO,na.rm=TRUE), 
            Sugar = sum(Sugar,na.rm=TRUE))

# tmp <-
#   fvno.sugar %>% 
#   filter(grepl('Morrisons Take Away Stuffed Crust Pepperoni Pizza|no added sugar diet cola|lemonade|dark chocolate fruit nut|beef flavour potato sticks|italian lasagne',product_name, ignore.case=TRUE)) %>%
#   filter(grepl('Morri',product_name)) %>%
#   filter(Retailer %in% 'Morissons') %>%
#   as.data.frame() %>%
#   group_by(product_name) %>%
#   dplyr::summarise(FVNO = mean(FVNO),
#                    min_fvno = min(FVNO),
#                    max_fvno = max(FVNO))

# write.csv(tmp,'/Users/macuser/Desktop/FVNO for Richie.csv',row.names=FALSE)

# Merging with info on walnut oil
fvno.sugar <-
  left_join(fvno.sugar,
            walnut.oil) %>%
  mutate(FVNO = ifelse(!is.na(percent), FVNO + percent, FVNO)) %>%
  dplyr::select(-percent)

# Updating for soy/almond/oat/rice milks ----
stacked.dat <-
  stacked.dat %>%
  mutate(Food_Category = ifelse(grepl("soy.*milk|soy.*drink",product_name, ignore.case = TRUE),'Soymilk',Food_Category)) %>%
  mutate(Food_Category = ifelse(grepl("almond.*milk|almond.*drink",product_name, ignore.case = TRUE) & !grepl('with|bar|milk chocolate|tubes|[0-9]{1,}g',product_name, ignore.case = TRUE),'Almond milk',Food_Category)) %>%
  mutate(Food_Category = ifelse(grepl("cashew.*milk|\\bnut.*milk|cashew.*drink|nut.*drink",product_name, ignore.case = TRUE) & !grepl('with|bar|milk chocolate|tubes|[0-9]{1,}g',product_name, ignore.case = TRUE),'Other nut milk',Food_Category)) %>%
  mutate(Food_Category = ifelse(grepl("\\brice.*milk|rice.*drink",product_name, ignore.case = TRUE),'Rice milk',Food_Category)) %>%
  mutate(Food_Category = ifelse(grepl("\\boat.*milk|\\boat.*drink",product_name, ignore.case = TRUE),'Oat milk',Food_Category)) %>%
  mutate(Food_Category = ifelse(Food_Category %in% 'Milk' & percent >= 50 & grepl('cheese',product_name, ignore.case = TRUE),'Cheese',Food_Category)) # And catching cheese/milk - these were flagged because of e.g cheese (milk) in the ingredients list, which were idnetified as cheese. But updating here

# Getting rid of sub categories for nut milks/rice milks/etc
stacked.dat <-
  stacked.dat %>%
  mutate(Food_Category_sub = ifelse(Food_Category %in% c('Almond milk','Rice milk','Soymilk','Oat milk','Other nut milk'), NA, Food_Category_sub)) %>%
  mutate(Food_Category_sub_sub = ifelse(Food_Category %in% c('Almond milk','Rice milk','Soymilk','Oat milk', 'Other nut milk'), NA, Food_Category_sub_sub))

# Identifying broths and stocks
stacked.dat <- broth.stock(stacked.dat)

# Identifying organic ingredients
stacked.dat <- organic.ingredients(stacked.dat)

# Aggregating data by food category ----
# This will be used throughout the script
# And used to identify environmental and nutrition impact
# Of each unique entry of each product in the database
stacked.dat <-
  stacked.dat %>%
  group_by(id, product_name, Retailer, country, Department, Aisle, Shelf, Food_Category, Food_Category_sub, Food_Category_sub_sub, Organic_ingredient) %>% 
  summarise(percent = sum(percent,na.rm=TRUE)) %>%
  as.data.frame(.)

# List of products with less than 75% of composition identified
filter.prods <-
  stacked.dat %>%
  filter(!is.na(Food_Category)) %>%
  group_by(id, product_name, Retailer, country, Department, Aisle, Shelf) %>%
  summarise(tot_percent = sum(percent, na.rm = TRUE)) %>%
  filter(tot_percent >= 75) %>%
  unique(.) %>% as.data.frame(.) %>%
  mutate(id.drop = paste0(id,product_name,Retailer,country,Department,Aisle,Shelf))


# Getting list of product IDs to drop because ingredients have <0 % composition
prods.negs <-
  stacked.dat %>%
  filter(percent < 0) %>%
  dplyr::select(id, product_name, Retailer, country, Department, Aisle, Shelf) %>%
  unique(.) %>%
  mutate(id.drop = paste0(id,product_name,Retailer,country,Department,Aisle,Shelf))


# Adding nutritional information ----
# This is in case info for one of the nutrients is not available from back-of-package information
# Or alternatively, if back-of-package information clearly isn't correct (i.e. >100g fat / 100g product)

# Importing nutritional data from GeNUS
nut.info =
  read.csv(paste0(getwd(),"/Data Inputs/Nutrient Info By LCA Category 24April2020.csv"),
           stringsAsFactors = FALSE)

# Calculating nutrient info by product
# This is average for that product across all retail outlets, departments, etc
# This takes a while. I'm not sure why.
dat <-
  left_join(stacked.dat,
            nut.info %>% dplyr::select(Food_Category = food.group,
                                       Calories, Protein, Fat, SaturatedFat = Saturated.FA,
                                       Fiber = Dietary.Fiber, Sodium, Carbohydrates)) %>%
  mutate(Calories = Calories * percent/100, # Calculating composition
         Protein = Protein * percent/100,
         Fat = Fat * percent/100,
         SaturatedFat = SaturatedFat * percent/100,
         Fiber = Fiber * percent/100,
         Carbohydrates = Carbohydrates * percent/100,
         Sodium = Sodium * percent/100)

# Adding salt
# This is much much faster than using ifelse in dplyr
dat$Salt[dat$Food_Category %in% 'Salt'] <- 
  dat$percent[dat$Food_Category %in% 'Salt'] / 2.5 * 1000

dat$Sodium[dat$Food_Category %in% 'Salt'] <- 
  dat$percent[dat$Food_Category %in% 'Salt'] / 2.5 * 1000

# and summarising by product name
dat <- 
  dat %>%
  group_by(id, product_name, Retailer, country, Department, Aisle, Shelf) %>% # Summing by product name and id
  summarise(Calories = sum(Calories, na.rm = TRUE),
            Protein = sum(Protein, na.rm = TRUE),
            Fat = sum(Fat, na.rm = TRUE),
            SaturatedFat = sum(SaturatedFat, na.rm = TRUE),
            Fiber = sum(Fiber, na.rm = TRUE),
            Sodium = sum(Sodium, na.rm = TRUE),
            Carbohydrates = sum(Carbohydrates, na.rm = TRUE),
            Salt = sum(Salt, na.rm = TRUE)) %>% 
  unique(.) %>% as.data.frame(.) %>%
  mutate(id.drop = paste0(id,product_name,Retailer,country,Department,Aisle,Shelf)) %>%
  filter(!(id.drop %in% prods.negs$id.drop)) %>% # Getting rid of products with negative compositional values
  filter(id.drop %in% filter.prods$id.drop) # Keeping products with > 75% composition identified

# Joining in fnvo and sugar data
dat <-
  left_join(dat,
            fvno.sugar %>% dplyr::select(id, product_name, Retailer, country, Department, Aisle, Shelf, FVNO, Sugar))

# Correcting back of package information ----
# Using listed back of package info if available
# Assuming none of the info for a product is incorrect
# If it is incorrect, then using the estimated information

# Importing back of package information
dat.nutrition <- 
  read.csv(paste0(getwd(),"/Products_dat/products_categories.csv")) %>% #Importing data
  dplyr::select(id = product_id, product_name, # Limiting to select columns
                Sugar_pack = sugar_per_100_value, # Needed to calculate NutriScore
                Fat_pack = fat_per_100_value,
                SatFat_pack = saturates_per_100_value,
                Salt_pack = salt_per_100_value,
                Protein_pack = protein_per_100_value,
                Fibre_pack = fibre_per_100_value,
                Carbs_pack = carbohydrate_per_100_value,
                Energy_pack = energy_per_100_value,
                serving = serving_size, serving_data = serving_size, serving_value = serving_size_value, serving_unit = serving_size_unit)

# already done this in Python
# # Getting col indices of nutrients needed for NutriScore
# nutrient.list <-
#   names(raw.dat)[which(names(raw.dat) %in% 'Sugar_pack') : which(names(raw.dat) %in% 'Energy_pack')]
# 
# # Identifying and adjusting units for each nutrient
# # This makes sure i.e. units are g/mg
# # And the numeric value for the nutrient is correct
# dat.nutrition <- 
#   nutrition.adjust.function(dat = raw.dat,
#                             nutrient.list = nutrient.list)

# Converting 'NaNs' to 'NA's
dat.nutrition[which(dat.nutrition[,'Sugar_pack'] %in% 'NaN'),'Sugar_pack'] <- NA
dat.nutrition[which(dat.nutrition[,'Fat_pack'] %in% 'NaN'),'Fat_pack'] <- NA
dat.nutrition[which(dat.nutrition[,'SatFat_pack'] %in% 'NaN'),'SatFat_pack'] <- NA
dat.nutrition[which(dat.nutrition[,'Salt_pack'] %in% 'NaN'),'Salt_pack'] <- NA
dat.nutrition[which(dat.nutrition[,'Protein_pack'] %in% 'NaN'),'Protein_pack'] <- NA
dat.nutrition[which(dat.nutrition[,'Fibre_pack'] %in% 'NaN'),'Fibre_pack'] <- NA
dat.nutrition[which(dat.nutrition[,'Carbs_pack'] %in% 'NaN'),'Carbs_pack'] <- NA

# Summarising nutrition by product
# And performing logic checks to make sure a product doesn't e.g. have >100g fat per 100g product
# It's impossible to tell what is correct
# But very easy to tell what is incorrect
dat.nutrition <-
  dat.nutrition %>%
  group_by(product_name) %>%
  summarise(Sugar_pack_value = mean(Sugar_pack, na.rm = TRUE),
            Fat_pack_value = mean(Fat_pack, na.rm = TRUE),
            SatFat_pack_value = mean(SatFat_pack, na.rm = TRUE),
            Salt_pack_value = mean(Salt_pack, na.rm = TRUE),
            Protein_pack_value = mean(Protein_pack, na.rm = TRUE),
            Fibre_pack_value = mean(Fibre_pack, na.rm = TRUE),
            Carbs_pack_value = mean(Carbs_pack, na.rm = TRUE),
            Energy_pack_value = mean(Energy_pack, na.rm = TRUE)) %>%
  mutate(check_pack = ifelse(Sugar_pack_value > 100 & !is.na(Sugar_pack_value), 1, # Logical checks
                             ifelse(Fat_pack_value > 100  & !is.na(Fat_pack_value), 1, 
                                    ifelse(SatFat_pack_value > 100  & !is.na(SatFat_pack_value), 1,
                                           ifelse(SatFat_pack_value > (Fat_pack_value + 5)  & !is.na(SatFat_pack_value) & !is.na(Fat_pack_value), 1,
                                                  ifelse(Salt_pack_value > 100  & !is.na(Salt_pack_value), 1,
                                                         ifelse(Protein_pack_value > 100  & !is.na(Protein_pack_value), 1,
                                                                ifelse(Salt_pack_value > 100  & !is.na(Salt_pack_value), 1,
                                                                       ifelse(Carbs_pack_value > 100  & !is.na(Carbs_pack_value), 1,
                                                                              ifelse(Fibre_pack_value > 100  & !is.na(Fibre_pack_value), 1, 0)))))))))) %>%
  mutate(Calories_pack_value = Fat_pack_value * 8.84 + Carbs_pack_value * 4 + Protein_pack_value * 4) %>%
  as.data.frame(.)

# Merging in estimated nutritional value
# ANd using estimated values in cases where back of package info is clearly incorrect
dat <- 
  left_join(dat %>% unique(.), # Merging
            dat.nutrition %>% unique(.)) %>%
  mutate(Sugar = ifelse(!is.na(Sugar_pack_value) & !(check_pack %in% 1), Sugar_pack_value, Sugar), # Logic checks
         Fat = ifelse(!is.na(Fat_pack_value) & !(check_pack %in% 1), Fat_pack_value, Fat), # Basically, if back of package info is crap
         SaturatedFat = ifelse(!is.na(SatFat_pack_value) & !(check_pack %in% 1), SatFat_pack_value, SaturatedFat), # Then estimating based on our estimates
         Salt = ifelse(!is.na(Salt_pack_value) & !(check_pack %in% 1), Salt_pack_value * 1000 / 2.5, Sodium),
         Protein = ifelse(!is.na(Protein_pack_value) & !(check_pack %in% 1), Protein_pack_value, Protein),
         Fiber = ifelse(!is.na(Fibre_pack_value) & !(check_pack %in% 1), Fibre_pack_value, Fiber),
         Carbs = ifelse(!is.na(Carbs_pack_value) & !(check_pack %in% 1), Carbs_pack_value, Carbohydrates),
         Calories = ifelse(!is.na(Energy_pack_value) & !(check_pack %in% 1), Energy_pack_value, Calories)) %>%
  mutate(Sodium = Salt) %>%
  unique(.) %>%
  mutate(Calories = Fat * 8.84 + Carbs * 4 + Protein * 4)

# Classifying products for nutriscore ----
# Drinks
# Cheese
# Oils

# Identifying drink products
drinks <- 
  stacked.dat.save %>%
  dplyr::select(product_name, Retailer, country, Department, Aisle, Shelf) %>%
  unique(.) %>%
  mutate(drink_department = ifelse(Department %in% c('Drinks','Soft Drinks, Tea & Coffee','Tea, Coffee & Soft Drinks'), 'Drink', 
         ifelse(Department %in% c('Beer, wine & spirits','Beer, Wine & Spirits','Beer, Wines & Spritis'),'Alcohol','No'))) %>%
  mutate(drink_aisle = ifelse(Aisle %in% c('All Drinks','Ambient Juice','Arla Shop','Bottled Water','Chilled fruit juice & smoothies','Chilled Fruit Juice & Smoothies','Chilled Juice',
                      'Chilled Juice & Drinks','Chilled Juice & Smoothies','Chilled Juice, Smoothies & Drinks','Christmas drinks','Christmas Drinks','Coca Cola Shop','Coffee','Cordials',
                      'Drinks','Drinks Bigger Packs','Energy & Health Drinks','Fizzy drinks','Fizzy Drinks','Fizzy Drinks & Cola','Fruit juice & drinks','Go to Category: \n£1 Value Drinks','Go to Category: \nHalloween Drinks',
                      'Hot chocolate & malted drinks','Hot Chocolate & Malted Drinks','Hot Chocolate & Malts','Hot chocolate & milky drinks','Hot Drinks','Juices','Juices & Smoothies','Kids Drinks','Longer life juice & juice drinks',
                      'Milk & Dairy Drinks','Milk & milk drinks','Milkshake','Mixers','Mixers & adult soft drinks','Mixers & Adult Soft Drinks','Premium Drinks & Mixers','Smoothies','Smoothies, Juice & Yogrhut Drinks','Soft Drinks',
                      'Soft Drinks & Juices','Sports & Energy Drinks','Still & Sparkling','Still & Sparkling Fruit Drinks','Squash','Squash & Cordial','Squash & cordials','Squash & Cordials',
                      'Tea','Tea & Hot Drinks','Tea, coffee & hot drinks','Tea, Coffee & Hot Drinks','Tea Coffee & Juices','Tonic & Mixers','Tonic Water & Mixers','Water'),'Drink',
                      ifelse(Aisle %in% c('Alcohol Free','Alcohol gifts','Alcoholic Drinks','Ales & Stouts','Beer','Beer & Cider','Beer, Wine & Spirits','Beers and Ciders','Champagne & sparkling wine',
                                          'Champagne & Sparkling Wine','Cider','Cider, Wine & Spirits','Cocktails','Craft Beer','Lager','Low & No Alcohol','Low Alcohol & Alcohol Free Drinks','Low alcohol & gluten free',
                                          'Sparkling Wine','Spirits','Spirits & liqueurs','Spirits & Liqueurs','Spirits and Liqueurs','Wine','Wine & Champagne','Wine, Fizz & Drinks'),'Alcohol','No'))) %>%
  mutate(drink_shelf = ifelse(Shelf %in% c('Chilled Drinks','Chilled Juice & Smoothies','Chilled Juice, Smoothies & Drinks','Daily Yoghurt Drinks','Dairy Alternative Drinks','Drink Coolers','Drinks','Energy Drinks','Energy drinks','Essences, Juices & Flavourings',
                                     'Extracts, Essences & Juices','Fizzy Drinks','Fresh Fruit juice','Fresh Juice and Herbal Tea','Frozen smoothie mixes','Half & half','Goats Milk','Italian Coffee','Juices','Kings & Tonic','Long Life Drinks','Long Life Juice','Longlife Milk','Long Life Milk',
                                     'Long Life UHT Milk','Milk','Milk & cream','Milkshake, Iced Coffee & Protein Drinks','Mixers','Original Lemonade','Schweppes 1783','Soft Drinks','Sporks drinks','Sports nutritional drinks','Sports Drinks','Sports cap water','Stock','Stocks','Stocks & Gravies',
                                     'Tea','Tea, Sake & Other Beverages','Tea, Coffee & Hot Drinks','Tea, Coffee & Soft Drinks','The Ultimate Light Mixer','The Ultimate Mixer','Yoghurt Drinks','Yogurt drinks','Yogurt Drinks'), 'Drinks',
                        ifelse(Shelf %in% c('Beer','Beer & Cider','Beer & Spirits','Beers, Wine & Spirits','Dessert Wine','Italian Wine','Spirits, Beer & Cider','Wine','Wine & Champagne','Wine & Fizz','Wines'), 'Alcohol','No'))) %>%
  mutate(drink = ifelse(drink_department %in% 'Drink' | drink_aisle %in% 'Drink' | drink_shelf %in% 'Drink', 'Drinks',
                        ifelse(drink_department %in% 'Alcohol' | drink_aisle %in% 'Alcohol' | drink_shelf %in% 'Alcohol','Alcohol','No'))) %>%
  dplyr::select(product_name, drink) %>%
  filter(drink %in% c('Drinks','Alcohol')) %>%
  unique(.)

# or if water > 90% of the product
drinks <- 
  rbind(drinks,
        stacked.dat %>% 
          filter(Food_Category %in% c('Water','Milk','Soymilk','Rice milk','Almond milk','Oat milk')) %>%
          group_by(id, product_name, Department, Aisle, Shelf) %>%
          dplyr::summarise(percent = sum(percent, na.rm = TRUE)) %>%
          mutate(drink = ifelse(percent >= 90,'Drink',NA)) %>%
          filter(drink %in% 'Drink') %>%
          as.data.frame(.) %>%
          dplyr::select(product_name, drink) %>%
          unique(.)) %>%
  unique(.)

# Identifying Cheese
cheese <- 
  stacked.dat.save %>%
  dplyr::select(product_name, Retailer, country, Department, Aisle, Shelf) %>%
  unique(.) %>%
  mutate(cheese_department = ifelse(grepl('cheese',Department, ignore.case = TRUE), 'Cheese','No')) %>%
  mutate(cheese_aisle = ifelse(grepl('cheese',Aisle, ignore.case = TRUE), 'Cheese','No')) %>%
  mutate(cheese_shelf = ifelse(Shelf %in% c('All Cheese','Blue Cheese','Build Your Cheeseboard','Brie & Camembert','Cheddar Cheese','Cheese','Cheese & Accompaniments','Cheese & Crackers','Cheese Counter','Cheese for Entertaining','Cheese Selections',
                                            'Cheese Slices, Spreads & Triangles','Cheese Snacking','Cheese Spreads & Snacks','Cheeseboards','Cheese Snacks & Spreads','Cheeseboard & Deli','Cheeseboards & Selections','Cheesemongers','Continental Cheese',
                                            'Continental & Specialty Cheese','Cottage & Soft Cheese','Counter Cheese','Cottage Cheese & Soft Cheese','Counter - Cheese','Cream, Soft & Cottage Cheese','Dairy Alternative Cheese','Dairy Free Cheese & Alternatives',
                                            'Deli, Cheese & Accompaniments','Deli Style Cheese','Feta & Goats Cheese','Feta & Halloumi','Feta, Halloumi & Paneer','Grated & Sliced Cheese','Grated & Sliced','Goats Cheese','Hard Cheese','Italian Cheese','Lighter & Low Fat Cheese',
                                            'Lunch Box Cheese','Mozzarella, Mascarpone & Ricotta','No.1 Cheese','Parmesan & Pecorino','Quark','Quark, Soft Cream & Cottage Cheese','Reduced Fat Cheese','Regional Cheese','Sliced & Grated Cheese','Snacking Cheese & Lunchboxes',
                                            'Stilton & Blue Cheese','Vegan & Dairy Free Cheese'),'Cheese','No')) %>%
  mutate(cheese = ifelse(cheese_department %in% c('Cheese'), 'Cheese',
                        ifelse(cheese_aisle %in% 'Cheese','Cheese',
                               ifelse(cheese_shelf %in% 'Cheese','Cheese','No')))) %>%
  dplyr::select(product_name, cheese) %>%
  filter(cheese %in% 'Cheese') %>%
  unique(.)

# or if cheese > 90% of the product
cheese <- 
  rbind(cheese,
        stacked.dat %>% 
          filter(grepl('Cheese',Food_Category)) %>%
          group_by(id, product_name, Department, Aisle, Shelf) %>%
          dplyr::summarise(percent = sum(percent, na.rm = TRUE)) %>%
          mutate(cheese = ifelse(percent >= 90,'Cheese',NA)) %>%
          as.data.frame(.) %>%
          unique(.) %>%
          dplyr::select(product_name, cheese) %>%
          filter(cheese %in% 'Cheese')) %>%
  unique(.)

# Identifying Oils and Fats
fats.oils <- 
  stacked.dat.save %>%
  dplyr::select(product_name, Retailer, country, Department, Aisle, Shelf) %>%
  unique(.) %>%
  mutate(fats.oils = ifelse(Shelf %in% c('Butter, Spreads & Margarine','Butter, Fats & Spreads','Butter, spreads & pastry','Butter, Spreads & Pastry','Butters, Fats & Spreads','Oils','Oil','Oils & Fats',
                                         'Oils & Vinegar','Oils & Vinegars'),'Fats.Oils','No')) %>%
  mutate(fats.oils = ifelse(grepl('Vinegar|pastry|jus ros|jus-ros|jr feuille|dough|balsamic', product_name, ignore.case = TRUE), 'No',fats.oils)) %>%
  dplyr::select(product_name, fats.oils) %>%
  filter(fats.oils %in% 'Fats.Oils') %>%
  unique(.)

# or if fats/oils > 90% of the product
fats.oils <- 
  rbind(fats.oils,
        stacked.dat %>% 
          filter(Food_Category %in% c('Butter, Cream & Ghee',"Oils Misc.",'Olive Oil','Palm Oil','Rapeseed Oil',"Soybean Oil","Sunflower Oil")) %>%
          group_by(id, product_name, Department, Aisle, Shelf) %>%
          dplyr::summarise(percent = sum(percent, na.rm = TRUE)) %>%
          mutate(fats.oils = ifelse(percent >= 90,'Fats.Oils',NA)) %>%
          filter(fats.oils %in% 'Fats.Oils') %>%
          as.data.frame(.) %>%
          dplyr::select(product_name, fats.oils) %>%
          unique(.)) %>%
  unique(.)

# Updating classifications for these products in the big data set
dat <-
  dat %>%
  mutate(cheese = ifelse(product_name %in% cheese$product_name,'Cheese','No'),
         fat.oil = ifelse(product_name %in% fats.oils$product_name, 'Fat.Oil','No'),
         alcohol = ifelse(product_name %in% drinks$product_name[drinks$drink %in% 'Alcohol'],'Alcohol','No'),
         drinks = ifelse(product_name %in% drinks$product_name[drinks$drink %in% 'Drinks'],'Drinks','No'))

# Calculating nutriscore
# And making sure we only have products we're keeping
nutriscore = 
  nutriscore.function(dat = dat) %>%
  filter(!(id.drop %in% prods.negs$id.drop)) %>%
  filter(id.drop %in% filter.prods$id.drop)

# Creating managed data folder if not already there
if('Managed_Data' %in% list.files(getwd())) {
  # Nothing
} else {
  dir.create(paste0(getwd(),"/Managed_Data"))
}

# Saving for radar plots
write.csv(nutriscore %>% dplyr::select(id, product_name, Retailer, country, Department, Aisle, Shelf,
                                       NutriCal, NutriSugar, NutriSatFats, NutriSodium, NutriFatRatioScore, NutriFVNO, NutriFiber, NutriProtein,
                                       NutriScoreNeg, NutriScorePos, NutriScorePoints),
          paste0(getwd(),"/Managed_Data/NutriScore for radar plots ",Sys.Date()," Log2.csv"),
          row.names = FALSE)


#####
###
# Doing this in a series of loops for each country 
# Saving stacked.dat before starting this
stacked.dat.whole = stacked.dat

# List of countries to loop through
countries = sort(unique(stacked.dat$country))

# Mapping of country names to iso codes (to read region specific LCA files)
country_groups <- read.csv(file.path(getwd(), '..', 'FAOSTAT', 'country_groups.csv'))

for(c in (countries)) {
  
  # if (!(c %in% c('Serbia', 'South Africa', 'France'))) {
  if (c %in% c('United Kingdom')) {
    # Getting data limited to the country
    stacked.dat = stacked.dat.whole %>% filter(country %in% c)
    iso3 = country_groups[country_groups$Country == c, "iso3"]
  
    # Calculating environmental impacts per 100g ----
    # Importing LCA data set
    # Managing lca dat
    
    # Global LCA data
    # lca.dat <-
      # read.csv(paste0(getwd(),"/Data Inputs/jp_lca_dat.csv"),
      #          stringsAsFactors = FALSE) %>%
      # mutate(Weight = as.numeric(gsub("%","",Weight)))
    
    # LCA data with sourcing incorporated 
    lca.dat <-
      read.csv(paste0(getwd(),"/Data Inputs/LCA_data_by_country/jp_lca_dat_",iso3,".csv"),
               stringsAsFactors = FALSE) %>%
      mutate(Weight = as.numeric(gsub("%","",Weight)))
    
    # Adding translation for subcategories
    lca.subcats <- read.csv(paste0(getwd(),'/Data Inputs/Search words, second round, 2024.07.22.csv'))
    
    # And updating lca categories
    lca.dat <-
      left_join(lca.dat, # Merging
                lca.subcats %>% dplyr::select(Data.S2.Name = LCA_Category, Product_details, LCA_Category_sub = LCA_sub_category, LCA_Category_sub_sub = LCA_sub_sub_category, 
                                              Average_of_original_category, Average_of_sub_category) %>% 
                  filter(LCA_Category_sub != '') %>% unique(.)) %>%
      unique(.) #%>%
      # mutate(LCA_Category_sub_sub = ifelse(Average_of_sub_category %in% 'Yes',NA,LCA_Category_sub_sub)) %>%
      # mutate(LCA_Category_sub = ifelse(Average_of_original_category %in% 'Yes',NA,LCA_Category_sub))
    
    # Adding conversion estimates
    # lca.dat <-
    #   rbind(lca.dat,
    #         conversion.function(indicators = c('^Land.Use','GHG','Eutrophication','Scarcity','Acidification','^Water','Biodiversity')) %>% dplyr::select(-food.group)) %>%
    #   .[,c('Data.S2.Name','LCA_Category_sub','LCA_Category_sub_sub','Weight','Land.Use..m2.year.','GHG.Emissions..kg.CO2eq..IPCC2013.incl.CC.feedbacks.',
    #        'Eutrophication..g.PO43.eq.','Scarcity.Weighted.Water.Use..L.eq.','Acidification..g.SO2eq.','Water.Use..L.','Biodiversity..sp.yr.10.14.',
    #        'Average_of_original_category','Average_of_sub_category')]
    
    
    lca.dat <-
      rbind(conversion.function(indicators = c('^Land.Use','GHG','Eutrophication','Scarcity','Acidification','^Water','Biodiversity')) %>% dplyr::select(-food.group),
            lca.dat %>% filter(!grepl('Cheese',Data.S2.Name))) %>% # Conversion function goes from cheese to other types of cheese
      .[,c('Data.S2.Name','LCA_Category_sub','LCA_Category_sub_sub','Weight','Land.Use..m2.year.','GHG.Emissions..kg.CO2eq..IPCC2013.incl.CC.feedbacks.',
           'Eutrophication..g.PO43.eq.','Scarcity.Weighted.Water.Use..L.eq.','Acidification..g.SO2eq.','Water.Use..L.','Biodiversity..sp.yr.10.14.',
           'Average_of_original_category','Average_of_sub_category','Sys')] 
    # And adding in other cheese category
    lca.dat <-
      rbind(lca.dat,
            lca.dat %>% filter(grepl('Medium Cheese',LCA_Category_sub)) %>% mutate(LCA_Category_sub = 'Other Cheese')) # And adding in the other cheese category
    
    # Updating categories for almond milk vs other milk
    lca.dat <-
      lca.dat %>%
      mutate(Data.S2.Name = ifelse(Data.S2.Name %in% 'Almond milk' & !(LCA_Category_sub %in% 'Almonds'),'Other nut milk', Data.S2.Name)) %>%
      mutate(LCA_Category_sub = ifelse(Data.S2.Name %in% c('Almond milk','Other nut milk','Oat milk','Soymilk','Rice milk'),NA, LCA_Category_sub)) %>%
      mutate(LCA_Category_sub_sub = ifelse(Data.S2.Name %in% c('Almond milk','Other nut milk','Oat milk','Soymilk','Rice milk'),NA, LCA_Category_sub_sub))
    
      
    
    # Adding butter, misc oils, and pig meat
    # These weightings recommended by Joseph Poore, folliwng methods in Poore and Nemecek 2018 Science
    lca.dat <-
      rbind(lca.dat,
            lca.dat %>% filter(Data.S2.Name %in% 'Milk') %>% mutate(Data.S2.Name = 'Butter, Cream & Ghee'),
            lca.dat %>% filter(Data.S2.Name %in% 'Rapeseed Oil') %>% mutate(Data.S2.Name = 'Oils Misc.'),
            lca.dat %>% filter(Data.S2.Name %in% 'Pig Meat') %>% mutate(Data.S2.Name = 'Animal Fats')) %>%
      rbind(., # Adding info for tea, coffee, chocolate
            read.csv(paste0(getwd(),"/Data Inputs/lcadat 17october2019.csv"),
                     stringsAsFactors = FALSE) %>%
              filter(Data.S2.Name %in% 'Tea') %>% mutate(Weight = 100) %>% 
              mutate(LCA_Category_sub_sub = '',LCA_Category_sub = '', Sys = 'C',
                     Average_of_original_category = NA, Average_of_sub_category = NA) %>% # Adding column names
              .[,names(lca.dat)]) # And ordering columns to rbind
    
    # Adding data on tea
    
    # and updating names to merge with rest of script
    names(lca.dat)[names(lca.dat) %in% 'Data.S2.Name'] <- 'Food_Category'
    names(lca.dat)[names(lca.dat) %in% 'Land.Use..m2.year.'] <- 'Land'
    names(lca.dat)[names(lca.dat) %in% 'GHG.Emissions..kg.CO2eq..IPCC2013.incl.CC.feedbacks.'] <- 'GHG'
    names(lca.dat)[names(lca.dat) %in% 'Eutrophication..g.PO43.eq.'] <- 'Eut'
    names(lca.dat)[names(lca.dat) %in% 'Scarcity.Weighted.Water.Use..L.eq.'] <- 'WatScar'
    names(lca.dat)[names(lca.dat) %in% 'Biodiversity..sp.yr.10.14.'] <- 'Biodiversity'
    names(lca.dat)[names(lca.dat) %in% 'Acidification..g.SO2eq.'] <- 'Acidification'
    names(lca.dat)[names(lca.dat) %in% 'Water.Use..L.'] <- 'WaterUse'
    
    
    # and limiting lca dat to only necessary columns
    lca.dat <- 
      lca.dat[,c('Food_Category','LCA_Category_sub','LCA_Category_sub_sub','Weight','Land','GHG','Eut','WatScar','Biodiversity','Acidification','WaterUse','Average_of_original_category','Average_of_sub_category','Sys')] %>%
      mutate(Food_Category = ifelse(Food_Category %in% c('Fish (farmed)','Fish (wild caught)','Crustaceans (farmed)','Crustaceans (wild caught)'),
                                    gsub(" \\(farmed\\)| \\(wild caught\\)","",Food_Category),
                                    Food_Category)) %>%
      rbind(., data.frame(Food_Category = 'Salt',LCA_Category_sub = NA, LCA_Category_sub_sub = NA, Weight = 1, Land = 0, GHG = 0, Eut = 0, WatScar = 0, Biodiversity = 0, Acidification = 0, WaterUse = 0,Average_of_original_category=NA,Average_of_sub_category=NA, Sys = 'C')) %>% # Adding data for salt
      rbind(., data.frame(Food_Category = 'Water',LCA_Category_sub = NA, LCA_Category_sub_sub = NA, Weight = 1, Land = 0, GHG = 0, Eut = 0, WatScar = 0, Biodiversity = 0, Acidification = 0, WaterUse = 0,Average_of_original_category=NA,Average_of_sub_category=NA, Sys = 'C')) # Adding data for water
    
    # Renaming column - doing this for merging with food data later
    lca.dat <-
      lca.dat %>%
      dplyr::rename(Food_Category_sub = LCA_Category_sub,
                    Food_Category_sub_sub = LCA_Category_sub_sub)
    
    # Adding in fisheries data
    lca.dat <-
      rbind(lca.dat,
            fish.env.function('yay'))
    
    # Updating food category
    stacked.dat <-
      stacked.dat %>%
      mutate(Food_Category = ifelse(Food_Category %in% c('Fish (farmed)','Fish (wild caught)','Crustaceans (farmed)','Crustaceans (wild caught)'),
                                    gsub(" \\(farmed\\)| \\(wild caught\\)","",Food_Category),
                                    Food_Category))
    
    # Formatting  data for monte carlo for the env estimates
    food.df <-
      stacked.dat %>%
      mutate(check.id = paste0(product_name, id, Retailer, country, Department, Aisle, Shelf)) %>%
      filter(!(check.id %in% .$check.id[.$percent <= 0])) %>%
      group_by(product_name, id, Retailer, country, Department, Aisle, Shelf, Food_Category, Food_Category_sub, Food_Category_sub_sub, Organic_ingredient) %>%
      dplyr::summarise(amount = sum(percent, na.rm = TRUE)) %>%
      as.data.frame(.) %>%
      mutate(product_name = paste0(product_name, id, Retailer, country, Department, Aisle, Shelf)) %>% 
      filter(!is.na(Food_Category))
    
    # Merging in the coffee data
    food.df <-
      food.df %>%
      mutate(brewed_coffee = 'NO') %>%
      mutate(brewed_coffee = ifelse(id %in% brewed.coffee & Food_Category %in% 'Coffee' & amount >= 10, 'Coffee','No')) %>%
      mutate(Food_Category = ifelse(Food_Category %in% c('Vitamin','Mineral','Other Food Additive'), NA, Food_Category))
    
    # getting list of products to run for
    # these are only products that have at least some ingredient information
    # that has been paired with an lca dataset
    food.df.count <-
      food.df %>%
      filter(!is.na(Food_Category)) %>%
      dplyr::group_by(product_name) %>%
      dplyr::summarise(count = n())
      
    
    # Getting list of product ids for parallelization
    chunk2 <- function(x,n) split(x, cut(seq_along(x), n, labels = FALSE))
    # splitting
    # split.product.list <- chunk2(unique(food.df.count$product_name), 100)
    # split.product.list <- chunk2(unique(food.df$product_name)[1:100], 2)
    split.product.list <-
      unique(food.df$product_name) %>%
      .[. %in% food.df.count$product_name]
    
    # splitting the product list into chunks because code is too slow for ~600K items
    chunk_vector <- function(x, chunk.size) {
      # This splits `x` into slices of length `chunk.size`.
      # E.g. chunk.size = 10,000, etc.
      split(x, ceiling(seq_along(x) / chunk.size))
    }
    
    chunk.size <- 1000
    product.chunks <- chunk_vector(split.product.list, chunk.size)
    
    # Save the list to an RDS file (to rerun a specific chunk/set of chunks)
    # saveRDS(product.chunks, file = paste0(getwd(),"/Managed_Data/product_chunks_",c,".rds"))
    product.chunks <- readRDS(paste0(getwd(),"/Managed_Data/product_chunks_",c,".rds"))
    
    # and running
    # this will take a while...
    
    # Running monte carlo for all products in chunks
    for (k in seq_along(product.chunks)) {
      
      if (k > 0) {
        
        # Chunk k
        current.chunk <- product.chunks[[k]]
        
        # Process this chunk
        chunk.df <- do.call(
          rbind,
          lapply(current.chunk, function(x) {
            # Call function
            monte.carlo.lca(x)
          })
        )
        
        # Save partial result to disk
        # Name the file to show the chunk index:
        chunk.file <- file.path(
          getwd(),
          'Managed_Data',
          paste0('Impact_Proportions_By_Product_sourcing_', c,'_chunk_', k, '_of_',
                 length(product.chunks), '_', Sys.Date(), '.csv')
        )
        write.csv(chunk.df, chunk.file, row.names = FALSE)
      }
    }
  }
  
}
