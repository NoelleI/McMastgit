##Note that comments in this file are for pedagogical purposes
##the serious data cleaning appears around line 81
##the first portion up to that point is a simple data load and 

import os
os.chdir("C:\\Users\\Admin\\AppData\\Local\\rodeo\\app-2.5.2\\resources\\conda\\Scripts")
#this was for importing libraries using pip on my computer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
from sklearn.model_selection import train_test_split # to split the data into two parts
import json 

#os.chdir("C:\\Users\\Admin\\Documents\\McMaster\\Week 3")
os.chdir("C:\\Users\\Admin\\Documents\\McMaster\\McMastgit")
#this is where the files are stored on my computer, change to equivalent drive on your computer


file = 'tmdb_5000_movies.csv'
file2 = 'tmdb_5000_credits.csv'

#read in raw data

df = pd.read_csv(file)
df2 = pd.read_csv(file2)
df_sub = df.iloc[:,:20]   #iloc allows slicing of dataframe by index, get rid of empty columns
df_sub2 = df2.iloc[:,:4]
df_sub = pd.merge(df_sub, df_sub2)
df_sub.head()
#df_sub.fillna(value=0,inplace = True)#value can be changed as needed or data line can be removed - exclude line if na here as 0 will skew data
df_sub.isnull().any()
#neither budget nor revenue have any missing values, so we will leave the data as is for this initial regression analysis, no exlcusions are necessary

df_final = df_sub[['budget','revenue']]
df_final.head()
#df_final.dropna()     #choose to drop na rather than replace b/c of bias - but there are no na values

train, test = train_test_split(df_final,test_size=0.30)  #make sure this is randomized (see reference link)

#read about train_test_split function here: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

train.head() #there is no reason to believe a priori that there is a reason to remove "0" values for revenue, but will investigate later 
#perhaps information in some of the other columns will reveal why these points are equal to 0.

X_train = pd.DataFrame(np.array(train['budget'])) 
y_train = train['revenue']
X_test = pd.DataFrame(np.array(test['budget']))
y_test = test['revenue']

#the linear regression method fit works with DataFrame format

from sklearn import linear_model
from sklearn import metrics
lin = linear_model.LinearRegression()  
#http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html




lin.fit(X_train, y_train)  #fit the regression model

lin_score_test = lin.score(X_test, y_test)  #returns the coefficient of variation R-squared
lin_score_train = lin.score(X_train, y_train)

#The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares ((y_true - y_pred) ** 2).sum() and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum(). 
#The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
#A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.
#You will sometimes find an adjusted R-squared value is used. This is adjusted so that models with different numbers of predictors may be compared


print("Training score: ",lin_score_train)
print("Testing score: ",lin_score_test)
#If the R-squared value for the test score is much lower than for the training score, overfitting has occurred. 
#The R-squared score for the test set is what is expected to be seen with new data


#Continue the analysis with additional features extracted from the json data: 
#this utility function extracts the data from the json file
def get_data(column, df, key):  #credits, c and role are not used
    df_temp = pd.DataFrame([])
    for i in column:
        json_data = json.loads(i)
        df1 = pd.DataFrame.from_dict(json_data)
        if len(i) > 2:
            df2 = pd.DataFrame([df1[key][0]])
        else:
            df2 = df1
        df_temp = pd.concat([df_temp,df2],ignore_index=True)
    df = pd.concat([df,df_temp],axis=1)
    return df
    
df_final = get_data(df_sub["production_countries"], df_final, 'name')
df_final.columns = ["budget","revenue", "First Listed Production Country"]

df_final.head()

#there are 71 different production companies noted. Many of them appear only once.
#eliminate all countries with less than 100 appearances and create a category called "other"
#some of the values are NaN, create a new category called "None Listed" for none listed

country_count = 100


def consolidate_infrequent_categories(column, count):
    column = column.fillna("None Listed")
    lst= list(column.value_counts(dropna = False) < count)
    replst=list((column.value_counts(dropna = False)[lst]).index)
    withlst = ["other"]*len(replst)
    column = column.replace(replst,withlst)
    print(column.value_counts(dropna = False)) #note that all of the NaN and countries with <100 occurances are gone
    return(column)

df_final["First Listed Production Country"] = consolidate_infrequent_categories(df_final["First Listed Production Country"], country_count)




df_final = pd.get_dummies(df_final, columns=["First Listed Production Country"])

#train a new regression with the numerical dummies
#fisrt split the data and arrange so that target and predictors are separated
predictors = list(df_final.columns[0:1]) + list(df_final.columns[2:]) ###the target column is in column 2
 
train, test = train_test_split(df_final,test_size=0.30)
X_train = train[predictors].dropna()
y_train = train['revenue'].dropna()
X_test = test[predictors].dropna()
y_test = test['revenue'].dropna()


lin = linear_model.LinearRegression()
lin.fit(X_train, y_train) 

lin_score_test = lin.score(X_test, y_test)
lin_score_train = lin.score(X_train, y_train)


print("Training score: ",lin_score_train)
print("Testing score: ",lin_score_test)
print("Coefficients",lin.coef_)  #coef are around the size of the ratio of revenue to 1 (magnitude of dummy vars), but signs should be investigated 
print("Intercept",lin.intercept_)
print("Average Revenue", df_final["revenue"].mean())


#When you get the output, you will see that the training score and test scores have not improved. The coefficients also have different signs and are large. 
#They seem to change when more or less countries are added by changing the minimum frequency (100). Could be they are not independent (multicollinearity)

#Note::due to potential multi-collinearity, noted above, the explanatory power of the model is limited
#This means that the coefficients of the model that accompany each predictor are not necessarily meaningful
#Note that we have not quoted an error on these coefficients
#For small enough data sets, this error can be found using the statsmodels package
#Determining the size of the error on the coefficients when there are many predictors compared to samplesize is a subject of current research in high dimensional statistics
# see https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
#It is best to think of these models as being useful for prediction but not necessarily explanation


#Adding production country does not improve our fit.
#You can try this instead with Genre or another predictor you believe can do a better job

###Assignment 4 solution - Add Genre and Production Country#####

###Add scaling above so that coefficients can be compared for RFECV
from sklearn.preprocessing import MinMaxScaler

#first add new predictors, then scale, then use RFECV

#additional predictors 
df_final = get_data(df_sub["genres"], df_final, 'name')
new_columns = df_final.columns.values; new_columns[-1] = 'First Listed Genre'; df_final.columns = new_columns


df_final = get_data(df_sub["production_companies"], df_final, 'name')
new_columns = df_final.columns.values; new_columns[-1] = 'First Listed Production Company'; df_final.columns = new_columns

#reduce number of columns by consolidating infequent categories

genre_count = 3000
company_count = 3000
df_final["First Listed Genre"] = consolidate_infrequent_categories(df_final["First Listed Genre"], genre_count)
df_final["First Listed Production Company"] = consolidate_infrequent_categories(df_final["First Listed Production Company"], company_count)

drop_categories = []

if len(df_final["First Listed Production Company"].value_counts()) >=2:
    drop_categories.append("First Listed Production Company_other")

if len(df_final["First Listed Genre"].value_counts()) >=2:
    drop_categories.append("First Listed Genre_other")

df_final = pd.get_dummies(df_final, columns=["First Listed Production Company", "First Listed Genre" ])

#Scale the numbers to get more meaningful coefficients
scaler = MinMaxScaler(copy=False)
scaler.fit(df_final)
columns_temp = df_final.columns
df_final = pd.DataFrame(scaler.transform(df_final))
df_final.columns = columns_temp

#perform regression
predictors = list(df_final.columns[0:1]) + list(df_final.columns[2:])
 
train, test = train_test_split(df_final,test_size=0.30)
X_train = train[predictors].dropna()
y_train = train['revenue'].dropna()
X_test = test[predictors].dropna()
y_test = test['revenue'].dropna()


lin.fit(X_train, y_train) 

lin_score_test = lin.score(X_test, y_test)
lin_score_train = lin.score(X_train, y_train)


print("Training score: ",lin_score_train)
print("Testing score: ",lin_score_test)
print("Coefficients",lin.coef_)  #coef are around the size of the ratio of revenue to 1 (magnitude of dummy vars), but signs should be investigated 
print("Intercept",lin.intercept_)
print("Average Revenue", df_final["revenue"].mean())

#From this output, note that there are major impacts from multi-colinearity. This is unimportant for prediction, but for 
#recursive feature elimination to be as effective as possible, we wish the coefficients to be as meaningful as possible
#For this reason, we will eliminate one of the categories for problem set 4, where we will use the coefficients to rank factors
#Note that, because this is a high dimensional dataset, we should use caution when interpreting the meaning of the coefficients
#the "other" categories are actually several merged categories, for example
#statistical significance of coefficients for high dimensional datasets is not clear

#Now drop the "other" categories because, for the purpose of this model, they are collinear with the rest of the features from 
#the same original categorical model
#However, note that in "reality" they are not complimentary as are the categories "male" vs. "female"
#Also note that depending on how we select to group the "other" categories, we may eliminate all of the consolidate_infrequent_categories
#so, control for this using drop_categories



X_train = train[predictors].dropna().drop(drop_categories, axis = 1)
y_train = train['revenue'].dropna()
X_test = test[predictors].dropna().drop(drop_categories, axis = 1)
y_test = test['revenue'].dropna()



lin.fit(X_train, y_train) 

lin_score_test_drop = lin.score(X_test, y_test)
lin_score_train_drop = lin.score(X_train, y_train)


print("Training score adjusted " + str(lin_score_train_drop) + "  Original training score " + str(lin_score_train))
print("Testing score: " + str(lin_score_test_drop) + "  Original testing score " + str(lin_score_test))
print("We can see that the r-squared goodness of fit is exactly the same for both training and test sets with and without multi-collinearity \n")
print("Coefficients",lin.coef_)  #coef are around the size of the ratio of revenue to 1 (magnitude of dummy vars), but signs should be investigated 
print("For predictors " + str(X_test.columns))
print("Intercept",lin.intercept_)
print("However, the coefficients look much more reasonable. This is important for recursive feature elimination in the model")
print("However, we do not know what the p-values or error on these coefficents are, they seem close to 0, but interpret with caution")
print("Average Revenue", df_final["revenue"].mean())
print("The Average Revenue for the Entire Data Set is Much Smaller than the Maximum. Consider investigating outliers.")

print("\n now compare to the case with budget alone with the same train_test_split\n ")

predictors2= ['budget']

X_train = train[predictors2].dropna()
y_train = train['revenue'].dropna()
X_test = test[predictors2].dropna()
y_test = test['revenue'].dropna()



lin.fit(X_train, y_train) 

lin_score_test_budget = lin.score(X_test, y_test)
lin_score_train_budget = lin.score(X_train, y_train)

print("Training score with categoricals " + str(lin_score_train_drop) + "  Training score with only budget " + str(lin_score_train_budget))
print("Testing score: " + str(lin_score_test_drop) + " Testing score with budget only " + str(lin_score_test_budget))
#print("We can see that the r-squared goodness of fit is exactly the same for both training and test sets with and without multi-collinearity \n")
print("Coefficients",lin.coef_)  #coef are around the size of the ratio of revenue to 1 (magnitude of dummy vars), but signs should be investigated 
print("For predictors " + str(X_test.columns))
print("Intercept",lin.intercept_)
print("Note that for the categorical variables First Listed Production Company, First Listed Production Country and First Listed Genre chosen above the impact to the r-squared is negligeable")
#print("However, we do not know what the p-values or error on these coefficents are, they seem close to 0, but interpret with caution")
#print("Average Revenue", df_final["revenue"].mean())
#print("The Average Revenue for the Entire Data Set is Much Smaller than the Maximum. Consider investigating outliers.")







