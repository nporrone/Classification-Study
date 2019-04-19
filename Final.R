library(pROC)
library(ISLR)
library(MASS)
library(caret)
library(e1071)
library(FNN)
library(faraway)
library(gam)
library(tree)
library(randomForest)
library(gbm)
library(boot)
library(glmnet)
library(knitr)
library(tidyverse)
library(GGally)
library(stargazer)
library(xtable)
library(boot)
library(rpart)
set.seed(13)


# Data Summary

data=read.csv("C:/Users/nporr/OneDrive/Desktop/Bank_Marketing/Bank.csv",sep=";")
data = data[,-12]  # get rid of duration
data
table = as.data.frame(apply(FUN=summary,X=data[,c(1,6,10,12,13,14)],MARGIN = 2)) # Create a table summary
print(xtable(table, caption= "The Summaries of the Continous variables."))

# plot histogram for the continous variables
par(mfrow= c(2,3))
hist(data$age,breaks = 25)
hist(data$balance,breaks = 25)
hist(data$day,breaks = 25)
hist(data$campaign,breaks = 25)
hist(data$pdays,breaks = 25)
hist(data$previous,breaks = 25)

pairs(data[,c(1,6,10,12,13,14)]) # check for correlation between continous 

# Data Split

split = sample(length(data$y), round(length(data$y)*0.8), rep=FALSE) # 80% training, 20% test
data_trn = data[split, ]
data_tst = data[-split, ]

# Methods

## Logistic Regression

### BIC Aprroach

logmodelf = glm(y ~ ., data = data_trn, family = binomial(link = "logit"))
logbicfit = step(logmodelf, direction = "both",k=log(nrow(data_trn)),scope = list(upper = ~.,lower= ~1), trace = F )

summary(logmodelf)
summary(logbicfit)

# full model

cutoff = 0.5
y_hatf = rep(0,nrow(data_tst))
idxf = which(predict(logmodelf,data_tst,type = "response") > cutoff )
y_hatf[idxf] = 1

confful = table(y_hatf, data_tst$y)
confful

ErrorRtf = 1 - sum(diag(confful)/sum(confful))
ErrorRtf

## CV on full

control = trainControl(method="cv", number=5)
logmodelfcv= train(y ~ .,data = data,trControl = control, method = "glm", family=binomial())

# reduced model

y_hat = rep(0,nrow(data_tst))
idx = which(predict(logbicfit,data_tst,type = "response") > cutoff )
y_hat[idx] = 1

conf = table(y_hat, data_tst$y)
conf

ErrorRt = 1 - sum(diag(conf)/sum(conf))
ErrorRt

## CV on reduced

controlr = trainControl(method="cv", number=5)
logmodelcvr= train(y ~ housing + loan + contact + campaign + poutcome,data = data,trControl = controlr, method = "glm", family=binomial())

# RoC Curve

logpred = predict(logbicfit,data_tst,type="response")
RoC= roc(data_tst$y~logpred,plot=T,col="red")
pROC::auc(RoC)

## Decsion Trees

#unpruned tree

dtree = tree(y ~ ., data = data_trn)
plot(dtree)
text(dtree, pretty = 1)
title(main = "Unpruned Classification Tree")

## cv to see the best tree size
dtree.cv=cv.tree(dtree)
plot(dtree.cv$size,sqrt(dtree.cv$dev/nrow(data_trn)),type="b",xlab="Tree Size",ylab="Classification Error")

dtree_pred = predict(dtree,newdata = data_tst,type="class")

conf = table(dtree_pred, data_tst$y)
conf

ErrorRt = 1 - sum(diag(conf)/sum(conf))
ErrorRt


## Pruned tree did not help

## Bagging

bag = randomForest(y ~ ., data = data_trn, mtry = 15, importance = TRUE, ntree = 500)
bag

bag_pred = predict(bag, newdata = data_tst)

# test error (bagging)

conf = table(bag_pred, data_tst$y)
conf

ErrorRt = 1 - sum(diag(conf)/sum(conf))
ErrorRt


## Random Forest

rf = randomForest(y ~ ., data = data_trn, mtry = 4, importance = TRUE, ntree = 500)
rf

rf_pred = predict(rf, newdata = data_tst)
# test error (bagging)

conf = table(rf_pred, data_tst$y)
conf

ErrorRt = 1 - sum(diag(conf)/sum(conf))
ErrorRt

# Boosting

boost = gbm((unclass(y) - 1) ~ ., data = data_trn, distribution = "bernoulli", n.trees = 500, interaction.depth = 4, shrinkage = 0.01)
boost_pred = predict(boost, newdata = data_tst, n.trees = 500, type="response")
boost_pred[boost_pred >= 0.5] = "yes"
boost_pred[boost_pred < 0.5] = "no"

# test error(boosting)
conf = table(boost_pred, data_tst$y)
conf

ErrorRt = 1 - sum(diag(conf)/sum(conf))
ErrorRt

