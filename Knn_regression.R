library(readr)
library(dplyr)

#Reading training data
tr = read_csv("train.csv")
summary(tr)

#Analyzing the number of missing values
sum(is.na(tr))

## Data Exploration/Data Cleaning ##
#Finding columns which have missing values
na_cnt = as.data.frame(apply(tr,2,function(x) sum(is.na(x))))
names(na_cnt) = "Value"
na_cnt$Name = rownames(na_cnt)
na_cnt1 = arrange(na_cnt,na_cnt$Value)
na_cnt1 = na_cnt1[na_cnt1$Value!=0,]

## Linear Modeling ##
#Keeping relevant columns in dataframe
tr_mod = as.data.frame(tr[,!names(tr) %in% na_cnt1$Name])

#Performing logarithmic transformation on saleprice
tr_mod$SalePrice = log(tr_mod$SalePrice,2)

#Dividing dataset into test and training
smp = sample(1:nrow(tr_mod),nrow(tr_mod)*.7)
tr_tr = tr_mod[smp,]
tr_tst = tr_mod[-smp,]

#Performing linear regression using all variables, to get a baseline score
reg = lm(SalePrice ~. -Id, data = tr_mod)
summary(reg)

#Checking the accuracy of the dataset on test data
reg_pr = predict(reg, newdata = tr_tst)
mse = mean((tr_tst$SalePrice-reg_pr)^2)

#Reading test data
test = read_csv("test.csv")
summary(test)

#Analyzing the number of missing values
sum(is.na(test))
test_mod = test

#Keeping relevant columns in dataframe
test_mod = as.data.frame(test_mod[,!names(test_mod) %in% na_cnt1$Name])

#Using regression to predict values for training dataset
sale_p = predict(reg_all, newdata = test_mod)
prediction = as.data.frame(cbind(test_mod$Id,sale_p))

#As we performed log transformation earlier, performing 2^value to get actual sale price
prediction$sale_p = 2^prediction$sale_p

#Exporting the predictions as csv
write.table(prediction, file = "C1-2-house-prices.csv", row.names=F, col.names=c("Id","SalePrice"), sep=",")

###-----------------------------------------------------------------------------------###

#KNN Implementation

#Updating column names
colnames(tr_mod)[colnames(tr_mod)=="1stFlrSF"] = "firstFlrSF"
colnames(tr_mod)[colnames(tr_mod)=="2ndFlrSF"] = "secFlrSF"

colnames(test_mod)[colnames(test_mod)=="1stFlrSF"] = "firstFlrSF"
colnames(test_mod)[colnames(test_mod)=="2ndFlrSF"] = "secFlrSF"

# Data cleaning/exploration #
#Combining values for different categorical variables
tr_mod$Condition1 = ifelse(tr_mod$Condition1!="Norm","Oth","Norm")
tr_mod$Exterior1st = ifelse(tr_mod$Exterior1st=="BrkComm"|tr_mod$Exterior1st=="BrkFace","Brk","Oth")
tr_mod$Heating = ifelse(tr_mod$Heating=="GasA","GasA","Oth")
tr_mod$RoofMatl = ifelse(tr_mod$RoofMatl=="CompShg","CompShg","Oth")
tr_mod$SaleCondition = ifelse((tr_mod$SaleCondition=="Partial"),"Partial",ifelse((tr_mod$SaleCondition=="Normal"),"Normal","Oth"))

test_mod$Condition1 = ifelse(test_mod$Condition1!="Norm","Oth","Norm")
test_mod$Exterior1st = ifelse(test_mod$Exterior1st=="BrkComm"|test_mod$Exterior1st=="BrkFace","Brk","Oth")
test_mod$Heating = ifelse(test_mod$Heating=="GasA","GasA","Oth")
test_mod$RoofMatl = ifelse(test_mod$RoofMatl=="CompShg","CompShg","Oth")
test_mod$SaleCondition = ifelse((test_mod$SaleCondition=="Partial"),"Partial",ifelse((test_mod$SaleCondition=="Normal"),"Normal","Oth"))

#Creating sample and training sets
smp = sample(1:nrow(tr),nrow(tr)*.7)
tr_tr = tr_mod[smp,]
tr_tst = tr_mod[-smp,]

#Creating array of relevant quantitative variables
rel_fact1 = c("firstFlrSF", "secFlrSF", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "Fireplaces", "GarageArea", "KitchenAbvGr", "LotArea", "MasVnrArea", "OverallCond", "OverallQual", "ScreenPorch", "YearBuilt","WoodDeckSF")

#Creating array of relevant qualitative variables 
rel_fact2 = c("Id", "Functional", "MSZoning", "Condition1","Exterior1st", "Neighborhood", "CentralAir","Heating","KitchenQual","LandSlope","RoofMatl","SaleCondition")

#Creating a subset of data for modelling, with only the relevant quantitative variables
tr_knn = tr_tr[,names(tr_tr) %in% rel_fact1]
test_knn = tr_tst[,names(tr_tst) %in% rel_fact1]

#Creating a subset of data for modelling, with ID and Sale Prices 
tr_knn_n = tr_tr[,names(tr_tr) %in% c("Id","SalePrice")] 

#Creating a subset of data for modelling, with only the relevant quantitative variables
tr_knn_c = tr_tr[,names(tr_tr) %in% rel_fact2]
test_knn_c = tr_tst[,names(tr_tst) %in% rel_fact2]

#Creating dummy values for each of the categorical variables
tr_knnc = as.data.frame(model.matrix(Id ~ . -1, tr_knn_c))
test_knnc = as.data.frame(model.matrix(Id ~ . -1, test_knn_c))

#Combining quantitative and categorical for both test and train datasets
tr_knn = cbind(tr_knn,tr_knnc)
test_knn = cbind(test_knn,test_knnc)

#Creating a function to standardize all the variables on scale of maximum and minimum values for the column
strd <- function(x) # creates a function to reduce code complexity and redundancy
{
    return ((x - min(x)) / (max(x) - min(x))) 
}

#Calling the function to standardize all variables
tr_knn = as.data.frame(apply(tr_knn,2,strd))
test_knn = as.data.frame(apply(test_knn,2,strd))

#Substituting all NA values as 0
test_knn[is.na(test_knn)]=0
tr_knn[is.na(tr_knn)]=0

#Checking whether there are any records with NA values
sum(is.na(test_knn))

#Declaring k and prediction variable
k=3
pred_sp = NA

#Iterating through all records and performing KNN clustering
for(i in 1:nrow(test_knn))
{
    tot_dis = NA
    dis = sweep(as.matrix(tr_knn),2,as.matrix(test_knn[i,]),"-")
    dis = dis^2
    tot_dis = apply(dis,1,sum)
    tot_dis = sqrt(tot_dis)
    tot_dis = cbind(tot_dis,tr_knn_n)
    tot_dis = tot_dis[order(tot_dis$tot_dis),]
    pred_sp[i] = mean(tot_dis[1:k,3])
}

#Calculating the sale price from prediction and train data, as we performed log transformation of the dependent variable earlier
tr_err = 2^tr_tst$SalePrice
pred_sp = 2^pred_sp

#Calculating the R-squared for the model
SSt = sum((tr_err - mean(tr_err))^2)
SSr = sum((pred_sp - mean(tr_err))^2)
SSr/SSt

#Iterating through different values of k, we also found that setting k = 3 gave us the most accurate results, hence thats the value of k in our final model.

###-----------------------------------------------------------------------------------###

#Performing clustering for the entire dataset using both quantitative and categorical variables
tr_tr = tr_mod
tr_tst = test_mod

#Creating a subset of data for modelling, with only the relevant quantitative variables
tr_knn = tr_tr[,names(tr_tr) %in% rel_fact1]
test_knn = tr_tst[,names(tr_tst) %in% rel_fact1]

#Creating a subset of data for modelling, with ID and Sale Prices 
tr_knn_n = tr_tr[,names(tr_tr) %in% c("Id","SalePrice")] 

#Creating a subset of data for modelling, with only the relevant quantitative variables
tr_knn_c = tr_tr[,names(tr_tr) %in% rel_fact2]
test_knn_c = tr_tst[,names(tr_tst) %in% rel_fact2]

#Creating dummy values for each of the categorical variables
tr_knnc = as.data.frame(model.matrix(Id ~ . -1, tr_knn_c))
test_knnc = as.data.frame(model.matrix(Id ~ . -1, test_knn_c))

#Combining quantitative and categorical for both test and train datasets
tr_knn = cbind(tr_knn,tr_knnc)
test_knn = cbind(test_knn,test_knnc)

#Calling the function to standardize all variables
tr_knn = as.data.frame(apply(tr_knn,2,strd))
test_knn = as.data.frame(apply(test_knn,2,strd))

#Substituting all NA values as 0
test_knn[is.na(test_knn)]=0
tr_knn[is.na(tr_knn)]=0

#Checking whether there are any records with NA values
sum(is.na(test_knn))

#Declaring k and prediction variable
k=3
pred_sp = NA

#Iterating through all records and performing KNN clustering
for(i in 1:nrow(test_knn))
{
    tot_dis = NA
    dis = sweep(as.matrix(tr_knn),2,as.matrix(test_knn[i,]),"-") # using sweep simplifies distance calculation
    dis = dis^2
    tot_dis = apply(dis,1,sum)
    tot_dis = sqrt(tot_dis)
    tot_dis = cbind(tot_dis,tr_knn_n)
    tot_dis = tot_dis[order(tot_dis$tot_dis),]
    pred_sp[i] = mean(tot_dis[1:k,3])
}

#Calculating the sale price from prediction and train data, as we performed log transformation of the dependent variable earlier
tr_err = 2^tr_tst$SalePrice
pred_sp = 2^pred_sp

#Creating a data frame with prediction and the unique ID for each record
pred_val = as.data.frame(cbind(test_mod$Id,pred_sp))

#Exporting the predictions as csv
write.table(pred_val, file = "house-prices-knn-quan+qual.csv", row.names=F, col.names=c("Id","SalePrice"), sep=",")