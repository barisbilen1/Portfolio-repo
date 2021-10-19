
                        ##-- BASIC EXPLATORY DATA ANALYSIS --##
# Importing libraries

import pandas as pd
import getpass
import math
import numpy as np
from tqdm import trange
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest


if getpass.getuser() == "asus":
    file_path = "C:/Users/asus/Desktop/trendyol_data_v2/"
    
if getpass.getuser() == "your computers name":
    file_path = "C:/Users/..desiredlocation..."  

# Reading the data

transactions = pd.read_csv(file_path +'transactions_v2.csv') #took 10 seconds, 11m rows
products = pd.read_csv(file_path +'products_v2.csv') #1.4m rows
qa = pd.read_csv(file_path +'qa_v2.csv') #1.4m rows
reviews = pd.read_csv(file_path + 'reviews_v2.csv') #6m rows
supplier_disputed_return = pd.read_csv(file_path +'supplier_disputed_return_v2.csv') #20k rows
supplier_return = pd.read_csv(file_path +'supplier_return_v2.csv') #22k rows
supplier_defective_return = pd.read_csv(file_path +'supplier_defective_return_v2.csv') #22k rows
test_data = pd.read_csv(file_path +'test_data_v2.csv') # 7k rows (prediction set)
user_demographics = pd.read_csv(file_path+'user_demographics_v2.csv') #706k rows.


# we should recall the format of the test data first.
test_data.head(3)
#id column consists of USER_ID, PRODUCT_CONTENT_ID, ORDER_PARENT_ID concatenated together. So those are the only columns we can use in order to predict the return probability.

#1) transactions data

transactions #10m rows.
transactions.head(3)
transactions_eda = transactions.copy()
transactions_eda.describe()
#some findings: mean sales price is 67 TL, mean shipping cost is 5 TL, mean promotion percent is 27%, mean shipping percent is 12%.
transactions_eda.nunique() #binning promotion&shipping percents might be a good idea. I later conclude that it is not :).

plt.hist(transactions_eda.promotion_percent, bins = 10) #right-skewed.
min_promo_perc = np.min(np.array(transactions_eda.promotion_percent)[np.nonzero(np.array(transactions_eda.promotion_percent))])
#I conclude that skewness in an independent variable is not a major problem. Regression models mostly do not assume anything about the distributions of predictor variables.
bins = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
transactions_eda["promotion_buckets"] = pd.cut(transactions_eda.promotion_percent, bins)
# I changed my mind and conclude that 'binning's really only a good idea when you'd expect a discontinuity in the response at the cut-points—say the temperature something boils at, or the legal age for driving.' I will continue with the original continuous variable.
transactions_eda = transactions_eda.drop(['promotion_buckets'],axis=1)

plt.hist(transactions_eda.shipping_percent, bins = 10) #extremely skewed. Only taking values between 0 and 10%. This is expected because no one will pay more than, say, 15-20% shipping cost for an item.

sns.set_style('darkgrid')
sns.displot(transactions_eda.shipping_percent) #again highly right skewed. I continue with outlier removal in these two variables.

for col in ["shipping_percent", "promotion_percent"]:
    upper_quantile = transactions_eda[col].quantile(0.95)
    transactions_eda[col] = np.where(transactions_eda[col] > upper_quantile, upper_quantile,transactions_eda[col])
    
transactions_eda.describe()

sns.set(rc={'axes.facecolor':"#F8F9F9",
            "figure.facecolor":"#CACFD2",
            "grid.color":"#E5E7E9",
            "axes.edgecolor":"#17202A",
            "axes.labelcolor":"#17202A",
            "text.color":"#17202A"
           }) 

sns.histplot(transactions_eda.discounted_price, binrange = (0,500),bins=20)

for col in ["discounted_price","original_price"]:
    upper_quantile = transactions_eda[col].quantile(0.95)
    transactions_eda[col] = np.where(transactions_eda[col] > upper_quantile, upper_quantile,transactions_eda[col])
    
transactions_eda.describe() #no negative values
#transactions_eda['discounted_price'] = np.ceil(transactions_eda.discounted_price).astype(int)
#transactions_eda['original_price'] = np.ceil(transactions_eda.original_price).astype(int)

transactions_eda.isna().sum() #just checking
len(transactions_eda) #10.7m observations
sns.histplot(transactions_eda.is_returned)  #approximately 9.7m non-returned items, and 1m returned items

#an attempt to isolate outliers that cannot possibly be identified by quantiles

isolation_forest = IsolationForest(n_estimators=100)
isolation_forest.fit(transactions_eda['discounted_price'].values.reshape(-1, 1))
xx = np.linspace(transactions_eda['discounted_price'].min(), transactions_eda['discounted_price'].max(), len(transactions_eda)).reshape(-1,1)
anomaly_score = isolation_forest.decision_function(xx)
outlier = isolation_forest.predict(xx)

plt.figure(figsize=(10,4))
plt.plot(xx, anomaly_score, label='anomaly score')
plt.fill_between(xx.T[0], np.min(anomaly_score), np.max(anomaly_score), 
                 where=outlier==-1, color='r', 
                 alpha=.4, label='outlier region')
plt.legend()
plt.ylabel('anomaly score')
plt.xlabel('discounted_price')
plt.show();

#above plot reminded me nothing but the fact I should also remove outliers from down. Why would I just think that outliers can only be at the top?

for col in ["discounted_price","original_price","shipping_percent","promotion_percent"]:
    lower_quantile = transactions_eda[col].quantile(0.02)
    transactions_eda[col] = np.where(transactions_eda[col] < lower_quantile, lower_quantile,transactions_eda[col])
    

transactions_eda.nunique()


#2) products data

products.head(3)
products.nunique()
products.attribute_value.unique()

#let's merge this dataframe with transactions dataframe in order to make use of these product attributes when training the model
merged_transactions = pd.merge(transactions_eda,products, how ='left', on ='product_variant_id')
merged_transactions.head(3)
merged_transactions.isna().sum() #all good.

#3) qa data

qa.head(3)
qa = qa.drop(['platform'],axis=1)

def word_checker4(word1,string_of_interest):
    if str(word1) in string_of_interest:
        return True
    else:
        return False
    
def word_checker5(word1,word2,word3,word4,string_of_interest):
    if str(word1) in string_of_interest or str(word2) in string_of_interest or str(word3) in string_of_interest or str(word4) in string_of_interest:
        return True
    else:
        return False

qa['is_return_mentioned'] = 0

for i in trange(len(qa)):
    string_of_interest = str(qa.iloc[i].question)
    currentrow = qa.iloc[i]
    if word_checker4('iade',string_of_interest)==True:
        qa.is_return_mentioned.iloc[i] = 1
        
    
qa.is_return_mentioned.sum() #585
qa[qa.is_return_mentioned==1]

for i in trange(len(qa)):
    string_of_interest = str(qa.iloc[i].answer).lower()
    currentrow = qa.iloc[i]
    if word_checker5('ambalaj','paket','hijyen','maalesef',string_of_interest) and currentrow.is_return_mentioned==1:
        qa.is_return_mentioned.iloc[i] = 0
        
qa.is_return_mentioned.sum() #now it is 519.


#ideas: Recall, we can only use 3 id's for prediction, and supplier_id is not one of them. So I guess we should somewhat 'separate' or 'decouple' the effects of supplier_id and the other variables. An item may have high return probability, but  if the supplier of that item also has high return probability, we should separate this 'supplier effect', because we only have product_content_id for the prediction. Think: how can we extract predictive power, from suppliers' information?

qa
qa_summary = qa.drop(['question','answer'],axis=1)
temp1 = qa_summary.groupby(['product_content_id']).agg(
    return_probability = pd.NamedAgg(column='is_return_mentioned',aggfunc='mean')
)
temp1.return_probability.sum()
high_return_products = temp1[temp1.return_probability > 0.1]
high_return_products = pd.DataFrame(high_return_products.reset_index())
list_highreturn_products = list(high_return_products.product_content_id)

temp2 = qa_summary.groupby(['supplier_id']).agg(
    return_probability = pd.NamedAgg(column='is_return_mentioned',aggfunc='mean')
)
temp2.return_probability.sum()
high_return_suppliers = temp2[temp2.return_probability > 0.1]
high_return_suppliers = pd.DataFrame(high_return_suppliers.reset_index())
list_highreturn_suppliers = list(high_return_suppliers.supplier_id)

temp3 = []
for i in trange(len(qa)):
    currentrow = qa.iloc[i]
    if currentrow.product_content_id in high_return_products and currentrow.supplier_id not in high_return_suppliers:
        temp3 = temp3.append(currentrow.product_content_id)

temp3
        
#looks like there is no supplier effect.       
        
#let's bring 'is_respondent' and 'is_return_mentioned' columns to the main dataframe.
temp4 = qa_summary.drop(['product_content_id','is_return_mentioned'],axis=1)

#merged_transactions = pd.merge(merged_transactions,temp4, how ='left', on = ['supplier_id']) MEMORY USAGE ERROR

merged_transactions.to_csv(file_path+'merged_transactions_10_10merged_transactions.csv',index=False)
high_return_products.to_csv(file_path+'high_return_products.csv',index=False)
temp4.to_csv(file_path+'respondent_suppliers.csv',index=False)

merged_transactions.info(memory_usage='deep')

del merged_transactions
                           
merged_transactions = pd.read_csv(file_path +'merged_transactions_10_10.csv')
                           
merged_transactions.info(memory_usage='deep')
#merged_transactions = merged_transactions.drop(['color_name'],axis=1)
merged_transactions = merged_transactions.drop(['brand_name'],axis=1)

merged_transactions.info(memory_usage='deep')                          
temp4.info(memory_usage='deep')                            
#merged_transactions = pd.merge(merged_transactions,temp4, how ='left', on = ['supplier_id']) 
len(merged_transactions)

#I should reduce memory usage.
                           
merged_transactions = merged_transactions.drop(['product_content_id_y'],axis=1)                        
merged_transactions = merged_transactions.rename({'product_content_id_x':'product_content_id'},axis=1)
merged_transactions = merged_transactions.drop(['original_price'],axis=1)
merged_transactions.category_name.unique()
len(merged_transactions.attribute_value.unique())  
merged_transactions.is_wallet_trx.unique()
merged_transactions['is_saved_card_trx'] = merged_transactions['is_saved_card_trx'].astype(int)
merged_transactions_v2 = merged_transactions.copy()

merged_transactions_v2['is_shipcost_incurred'] = merged_transactions_v2['is_shipcost_incurred'].astype(np.int32)
merged_transactions_v2.info(memory_usage='deep')
merged_transactions_v2['user_id'] = merged_transactions_v2['user_id'].astype(np.int32)             
merged_transactions_v2['is_elite_user'] = merged_transactions_v2['is_elite_user'].astype(np.int32)       
merged_transactions_v2['coupon_used'] = merged_transactions_v2['coupon_used'].astype(np.int32)
merged_transactions_v2['is_discounted'] = merged_transactions_v2['is_discounted'].astype(np.int32)                
merged_transactions_v2['is_different_sizes'] = merged_transactions_v2['is_different_sizes'].astype(np.int32)
merged_transactions_v2['is_wallet_trx'] = merged_transactions_v2['is_wallet_trx'].astype(np.int32)
#lets encode 'gender' column
                           
merged_transactions_v2.info(memory_usage='deep')      
                           
dummy_gender = pd.get_dummies(merged_transactions_v2['gender_name'])                          
merged_transactions_v3 = pd.merge(
    left=merged_transactions_v2,
    right=dummy_gender,
    left_index=True,
    right_index=True,
)                           
merged_transactions_v3 = merged_transactions_v3.drop(['gender_name'],axis=1)                          
merged_transactions_v3 = merged_transactions_v3.drop(['attribute_value'],axis=1)                              
merged_transactions_v3.info(memory_usage='deep') 
            
dummy_category = pd.get_dummies(merged_transactions_v3['category_name'])                          
merged_transactions_v4 = pd.merge(
    left=merged_transactions_v3,
    right=dummy_category,
    left_index=True,
    right_index=True,
)
merged_transactions_v4 = merged_transactions_v4.drop(['category_name'],axis=1)                           

merged_transactions_v4.info(memory_usage='deep') #memory usage increased significantly, will continue with v3.

dummy_colors = pd.get_dummies(merged_transactions_v3['color_name'])                          
merged_transactions_v5 = pd.merge(
    left=merged_transactions_v3,
    right=dummy_colors,
    left_index=True,
    right_index=True,
)
merged_transactions_v5 = merged_transactions_v5.drop(['color_name'],axis=1)                            
                           
merged_transactions_v5.info(memory_usage='deep')
                           
# from 5.6 gb to 3.4 gb memory usage, good job so far.

                           
high_return_products = pd.read_csv(file_path+'high_return_products.csv')
respondent_suppliers = pd.read_csv(file_path +'respondent_suppliers.csv')
respondent_suppliers.info(memory_usage='deep')

merged_transactions_v6 = pd.merge(merged_transactions_v5,respondent_suppliers,how='left',on='supplier_id')
#unfortunately it didn't work again. I really couldn't figure it out.
                           
merged_transactions_v6 = pd.merge(merged_transactions_v5,high_return_products,how='left',on='product_content_id')

#this is .._v6 is our main dataframe so far. Let's continue.
                           
#4) reviews data
                    
#We continue in the other notebook.
merged_transactions_v6.to_csv(file_path+'merged_transactions_v6.csv',index=False)
merged_transactions_v6.info(memory_usage='deep')
