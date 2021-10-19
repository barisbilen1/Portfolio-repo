
                        ##-- INITIAL LOOK ON THE DATASET --##
# Importing libraries

import pandas as pd
import getpass
import math
import numpy as np
from tqdm import trange
import gc

if getpass.getuser() == "asus":
    file_path = "C:/Users/asus/Desktop/trendyol_data/"
    
if getpass.getuser() == "your computers name":
    file_path = "C:/Users/..desiredlocation..."  

# Reading the data

transactions = pd.read_csv(file_path +'transactions.csv') #took 10 seconds, 11m rows
products = pd.read_csv(file_path +'products.csv') #1.4m rows
qa = pd.read_csv(file_path +'qa.csv') #1.4m rows
reviews = pd.read_csv(file_path + 'reviews.csv') #6m rows
supplier_disputed_return = pd.read_csv(file_path +'supplier_disputed_return.csv') #20k rows
supplier_return = pd.read_csv(file_path +'supplier_return.csv') #22k rows
supplier_defective_return = pd.read_csv(file_path +'supplier_defective_return.csv') #22k rows
test_data = pd.read_csv(file_path +'test_data.csv') # 7k rows (prediction set)
user_demographics = pd.read_csv(file_path+'user_demographics.csv') #706k rows.

#STEP ZERO: Before doing anything, we should handle missing values by doing some imputation/elimination. And then we can have a look at the data, distributions etc.

#1) cleaning 'transactions' data

transactions.isna().sum()
transactions['coupon_used'] = np.where(pd.isnull(transactions.coupon_id), 0, 1)
transactions = transactions.drop(["coupon_id"],axis=1)
transactions = transactions.drop(["order_date"],axis=1) # I don't think we need this column.
#we choose to ignore seasonality effect on whether an item will be returned, at least at this stage.
transactions.head(3)
#An idea: bir sipariş içinde aynı ürünün birden fazla bedeni (size) varsa, bu durum iade ihtimalini artırabilir, çünkü müşteri iki tane alayım bana uymayanını iade edeyim diye düşünmüş olabilir.
# Bunu bir binary kolon olarak tabloya ekliyorum. Feature olarak kullanılabilir belki.

transactions2 = transactions
transactions2_orders_grouped = transactions2.groupby(['order_parent_id']).agg({'product_content_id':'nunique','product_variant_id':'nunique'})
#there are 4.3m distinct orders in total.

transactions2_orders_grouped.head(3)
transactions2_orders_grouped[transactions2_orders_grouped.product_content_id != transactions2_orders_grouped.product_variant_id]
#FOUND THEM! 97k rows in total.

#Let's investigate some of themm.
transactions[transactions.order_parent_id==595759043]
transactions[transactions.order_parent_id==729960907]
transactions[transactions.order_parent_id==729960049]
#their is_returned values are 0 and NaN. So I can't make any inference right now. But I'll label them anyway.

list_different_sizes = transactions2_orders_grouped[transactions2_orders_grouped.product_content_id != transactions2_orders_grouped.product_variant_id]

list_different_sizes2 = pd.DataFrame(list_different_sizes.reset_index())
different_sizes_orders = list(list_different_sizes2.order_parent_id)
len(different_sizes_orders) #all good.

boolean_series1 = transactions2.order_parent_id.isin(different_sizes_orders)

transactions2['is_different_sizes'] = np.where(boolean_series1,1,0)
    
transactions2[transactions2.order_parent_id==729960049]
transactions2[transactions2.order_parent_id==729960907]
transactions2[transactions2.order_parent_id==729966335]
#perfect.

transactions = transactions2

transactions = transactions.drop(["order_line_item_id"],axis=1) #no need.
transactions['is_discounted'] = np.where(transactions.discounted_price < transactions.original_price, 1, 0) #whether the item is bought at a discounted price, binary column
transactions = transactions.drop(["coupon_discount"],axis=1) # I don't think this column will bring predictive power to the model

def word_checker(word,string_of_interest):#making the entire column percent. 
    if str(word) not in string_of_interest:
        return True
    else:
        return False

transactions['promotion_percent'] = np.where(word_checker('%',transactions.promotion_name), ((transactions.original_price-transactions.discounted_price)/transactions.original_price) , transactions.promotion_award_value)

transactions[transactions.promotion_name=='Influencer Hediye Çeki -zehraipek '] #checked, all good.

#Remark: There may also be cases where there is no promotion but the user buys the item with a discount.

#I don't think we need 'promotion_name' anymore.
transactions = transactions.drop(["promotion_name"],axis=1)
transactions = transactions.drop(["promotion_award_value"],axis=1) #neither this one.

transactions.isna().sum()
#An idea: If the shipping cost is incurred by the customer, the possibility of a return is lower I guess. We need this information.
#ship cost incurred by whom? trendyol, supplier, customer?
#Also, what does NaN mean in ship cost column? I assume NaN means that no shipping cost is incurred by the part who is responsible to pay shipping cost. I define a binary column.
transactions['is_shipcost_incurred'] = np.where(pd.isnull(transactions.ship_cost),0,1)
transactions['ship_cost'] = np.where(pd.isnull(transactions.ship_cost),0,transactions.ship_cost)

#max ship cost is 13 TL, where the average is 5 TL.
#let's define a column: shipping cost/discounted_price (actual sales price)

transactions['shipping_percent'] = transactions.ship_cost / transactions.discounted_price


#A BIG PROBLEM !!

#There are over 420k rows with NaN is_returned value. What does NaN mean in terms of is_returned? (maybe due to the loss of track originating from the miscommunication/lack of communication between the cargo firm and Trendyol.)
#It is very hard to guess what those NaN values mean in this column.
#We simply decided to have a copy of the data frame, and omit those rows in one version and replace them with 0 in the other version. However, we'll continue our analysis with 'NaNs dropped version'. The other one will just be stored in this code, because we might want to use it later.

transactions_is_ret_nandropped = transactions.dropna(subset=['is_returned'])

transactions_is_ret_madezero = transactions

transactions_is_ret_madezero['is_returned'] = np.where(pd.isnull(transactions_is_ret_madezero.is_returned),0,transactions_is_ret_madezero.is_returned)

transactions_is_ret_nandropped.isna().sum() #there are still 2 missing values in promotion percent column, let's have a look.
is_NaN = transactions_is_ret_nandropped.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = transactions_is_ret_nandropped[row_has_NaN]
rows_with_NaN #prices are zero, that's why promo. perc. column got Na.
transactions_is_ret_nandropped = transactions_is_ret_nandropped.dropna() #dropping these two rows for good.
transactions_is_ret_nandropped.isna().sum() #done


#2) cleaning 'products' data

products.isna().sum() #1.4m rows in total.
is_NaN = products.isnull()
products[is_NaN.any(axis=1)]
# I realized that NaN gender_id values are for the items which really do not have gender, like carpets, tissues, baby. Anyway, let's just omit this column (it is ID), and we will account for genderless items with 'gender_name' column.

products = products.drop(["gender_id"],axis=1)
products['gender_name'] = np.where(products.gender_name=='Unknown','GenderlessItem',products.gender_name)
products['gender_name'] = np.where(products.gender_name == 'Kadın / Kız','Kadın_Kız',products.gender_name) #better way to denote.

products.isna().sum() #let's look at NaN color_ids.(nan color id and nan color names are paired)
products[products['color_id'].isna()]
#let's denote those NaN color_names as multicolor and drop color_id column.
products['color_name'] = np.where(pd.isnull(products.color_id),'Multicolor',products.color_name)
products = products.drop(["color_id"],axis=1)
products.isna().sum() #there is one Na in supplier_color_name
products[products['supplier_color_name'].isna()] #ok it is 'siyah' but why do we need another color column? dropping this column...
products = products.drop(["supplier_color_name"],axis=1)

#Na values are handle, let's investigate further.
products = products.drop(["attributet_name"],axis=1) #same value for all rows.
#dropping some id columns
products = products.drop(["category_id"],axis=1)
products = products.drop(["brand_id"],axis=1)
products.head(2)
products = products.drop(["product_name"],axis=1) #this columns is not suitable for ML purposes.

#3) cleaning 'qa' data (question&answer)
qa.isna().sum() # no NAs.
qa.head(2)
len(pd.unique(qa.supplier_id)) #only 16588 unique suppliers.

#Logic: if the supplier has responded to a question, this is an indicator that the supplier is taking care of the questions raised by the customers. If so, customers might feel a little bit more comfortable returning the product, believing that the supplier is easy to communicate and there will be no problem. Let's add this information as binary valued column and we will decide whether we use it as a predictor later.

qa['is_respondent'] = 1
respondent_suppliers = qa[['supplier_id','is_respondent']].drop_duplicates()
#later we can join this little table with another table.
#Is there anything that we can do with this 'qa' data?

#4) cleaning 'reviews' data
reviews.head(4)
reviews.isna().sum()
#like count for a review is totally irrelevant with the return possibility (at least IMHO)

reviews = reviews.drop(["review_like_count"],axis=1)

#This is the part where we may use NLP to classify reviews.
#what I (baris) think is; 'rate' is sufficient for this purpose, BUT, in e-commerce platforms I have witnessed some examples of high rate&negative comment pairs. The reason is that some customers think their comment will be deleted by the store if they give them 1* star, so they give 5* star and write their negative comment. Should we account for this? (am I being over-detail-oriented?)

reviews5 = reviews[reviews.rate==5]

def word_checker2(word,string_of_interest):
    if str(word) in string_of_interest:
        return True
    else:
        return False
    
badcomments = []
badcomment_count = 0

for i in trange(10000):
    string_of_interest = str(reviews5.iloc[i].comment)
    if word_checker2('kötü',string_of_interest) == True: #kötü(TR)=bad(EN)
        badcomments.append(string_of_interest)
        badcomment_count += 1

badcomment_count  #bingo, 52 rows found.
badcomments

#you know what, most of them are 'kötü değil' (EN:not bad). But I observed one thing, if a comment includes 'kötü' and 'yorum' (EN:comment) at the same time, then that comment is a bad comment (people explicitly state that 'this item is BAD and gave 5 STARS just to have my COMMENT published') Let's check my assertion.

def word_checker3(word1,word2,word2_2,word3,word4,string_of_interest):
    if str(word1) in string_of_interest and str(word2) in string_of_interest and word2_2 in string_of_interest and (str(word3) in string_of_interest or str(word4) in string_of_interest):
        return True
    else:
        return False

#baby NLP model.
    
badcomments2 = []
badcomment_count2 = 0
    
for i in trange(10000):
    string_of_interest = str(reviews5.iloc[i].comment)
    if word_checker3('kötü','yorum','göz','gör',string_of_interest) == True: 
        badcomments2.append(string_of_interest)
        badcomment_count2 += 1
        
    
badcomment_count2 #5 comments found in first 10k rows.Let's generalize.
badcomments2

badcomments2 = []
badcomment_count2 = 0

for i in trange(len(reviews5)):
    string_of_interest = str(reviews5.iloc[i].comment)
    if word_checker3('kötü','yorum','yıldız','göz','gör',string_of_interest) == True: 
        badcomments2.append(string_of_interest)
        badcomment_count2 += 1
  
#'yıldız' means 'star' in English. 
#this for loop took minutes to 9 minutes complete.

badcomment_count2  #there 724 such comments.
badcomments2

# I am going to rely on my assertion, comments in badcomments2 managed to persuade me. Probably it will make no difference at all but I am going to change their 'rate's to '1' anyway.

# in original reviews.csv, sum of rates is 25,863,779 (just for control)

reviews['new_rate'] = np.where(reviews.rate == 5 & word_checker3('kötü','yorum','yıldız','göz','gör',str(reviews.comment)),1,reviews.rate) #didn't work

#we expect sum of rates are now lower than the initial sum.

reviews.new_rate.sum() #it is the same, something is wrong.

#Let's do it with a for loop instead (slow but safer)
#reviews = reviews.drop(["new_rate"],axis=1)
reviewscopy3 = reviews
reviewscopy3 = reviewscopy3.iloc[0:10000, :]
pd.options.mode.chained_assignment = None

def word_checker_try(word1,string_of_interest):
    if str(word1) in string_of_interest:
        return True
    else:
        return False
    
reviewscopy3['new_rate'] = 0

#CHECKING IF THE FOLLOWING BLOCK OF CODE WORKS

for i in trange(len(reviewscopy3)):
    string_of_interest = str(reviewscopy3.iloc[i].comment)
    currentrow = reviewscopy3.iloc[i]
    if word_checker_try('beden',string_of_interest)==True and currentrow.rate == 5:
        reviewscopy3.new_rate.iloc[i] = 1
    else:
        reviewscopy3.new_rate.iloc[i] = currentrow.rate
        
#I checked the first row, it works. But I guess we should eliminate else: block, it slows down the execution.

#real one:

reviews['new_rate'] = reviews.rate

for i in trange(len(reviews)):
    string_of_interest = str(reviews.iloc[i].comment)
    currentrow = reviews.iloc[i]
    if word_checker3('kötü','yorum','yıldız','göz','gör',string_of_interest)==True and currentrow.rate == 5:
        reviews.new_rate.iloc[i] = 1
        
reviews.new_rate.sum() #it is 25,860,883, it was 25,863,779 before (just checking). Let's move on to the next data file. More NLP can be made on this reviews data later.

gc.collect()

#continuing to analyze 'reviews.csv'
#important: we should label reviews that include 'iade' (EN:return). If a comment includes that word, the item is most probably going to be returned.

def word_checker4(word1,string_of_interest):
    if str(word1) in string_of_interest:
        return True
    else:
        return False


#trials
reviews2 = reviews.copy()
reviews2['is_return_mentioned'] = 0

boolean_series2 = word_checker4('iade',reviews2.comment)
reviews2.is_return_mentioned.sum() #0.

reviews2['is_return_mentioned'] = np.where(boolean_series2,1,reviews2.is_return_mentioned)

reviews2.is_return_mentioned.sum() 

reviews2 = reviews2.drop(['is_return_mentioned'],axis=1)

#above code didn't work.

for i in trange(len(reviews2)):
    string_of_interest = str(reviews2.iloc[i].comment)
    currentrow = reviews2.iloc[i]
    if word_checker4('iade',string_of_interest)==True:
        reviews2.is_return_mentioned.iloc[i] = 1
        

#Above for loop took 2 hours in my 8 gb ram machine.
    
reviews2
reviews2.is_return_mentioned.sum() #looks like 480k people mentioned 'return' in their comments.

        
#5) cleaning 'supplier_disputed_return' data

supplier_disputed_return.head(5) # 20k rows
supplier_disputed_return.isna().sum() #no Na





#6) cleaning 'supplier_disputed_return' data

supplier_return.head(5) #22,526 rows
supplier_return.isna().sum() #no Na

#this data is so far the most useful one in terms of predictive power I guess.





#7) cleaning 'supplier_defective_return' data

supplier_defective_return.head(5)  #22,536 rows.
supplier_defective_return.isna().sum() #no Na

#8) cleaning 'test_data' data (this is our test data)

test_data.isna().sum()
test_data['expected'] = 'will_be_predicted'

#9) cleaning 'user_demographics' data

user_demographics.head(3) #706k rows
user_demographics.birth_date.isna().sum() #268k birth_date Na
#There really is no way to retrieve 'birth_date' from other columns (neither from this table nor from other tables)
#Nearly one-third of birth_date info is missing.
# It seems we cannot use this column (btw, I believe age is totally irrelevant with return possibility).

user_demographics = user_demographics.drop(['birth_date'],axis=1)

user_demographics.gender.describe()
user_demographics[user_demographics.gender=='kvkktalepsilindi'] # there are only 225 rows with deleted gender info.

user_demographics = user_demographics.drop(user_demographics[user_demographics.gender=='kvkktalepsilindi'].index)

#19k rows with UNKNOWN gender info, we can delete those rows.
user_demographics = user_demographics.drop(user_demographics[user_demographics.gender=='UNKNOWN'].index)
user_demographics.isna().sum()

#all data files are cleaned and ready for analysis. Let's write those data files to our computers, in order to save time.

file_path2 = "C:/Users/asus/Desktop/trendyol_data_v2/"

transactions_is_ret_nandropped.to_csv(file_path2+'transactions_v2.csv',index=False) # took 4 mins.
user_demographics.to_csv(file_path2+'user_demographics_v2.csv',index=False)
reviews2.to_csv(file_path2+'reviews_v2.csv',index=False)
supplier_return.to_csv(file_path2+'supplier_return_v2.csv',index=False)
supplier_disputed_return.to_csv(file_path2+'supplier_disputed_return_v2.csv',index=False)
supplier_defective_return.to_csv(file_path2+'supplier_defective_return_v2.csv',index=False)
products.to_csv(file_path2+'products_v2.csv',index=False)
test_data.to_csv(file_path2+'test_data_v2.csv',index=False)
qa.to_csv(file_path2+'qa_v2.csv',index=False)

justcheckingthisfile = pd.read_csv(file_path2 +'reviews_v2.csv')

# Data cleaning phase is over


