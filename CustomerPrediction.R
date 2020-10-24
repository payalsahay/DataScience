#--------------------------------------Begin--------------------------------------#
#Objective: Develop a Classifier to predict whether 
#a customer is going to open a deposit account.
# load libraries
library(ggplot2)
library(caret)
library(plyr)
library(dplyr)
library(C50)
library(kernlab)
library(reshape)
library(MLmetrics)
library(caretEnsemble)
library(naivebayes)
library(neuralnet)

# load traning and test dataset
banktraining <- read.csv("~/Desktop/620 - Data Mining/Assignment 3/bank-training.csv")
banktest <- read.csv("~/Desktop/620 - Data Mining/Assignment 3/bank-test.csv")

# summary of datasets
summary(banktraining)
summary(banktest)

# view top rows of training and test datasets
head(banktraining)
head(banktest)

# check class of dataset
class(banktraining)

# variables in the dataset
names(banktraining)

#variable types of dataset
str(banktraining)

# class summary for datasets
summary(banktraining$y)
summary(banktest$y)

# plot the class distribution
qplot(y, data=banktraining, geom = "bar") + theme(axis.text.x = element_text(angle = 90, hjust = 1))

# relevel training data to get "yes" as positive class
trainingdata <- banktraining
trainingdata$y <- relevel(trainingdata$y, "yes")
summary(trainingdata$y)

# relevel test data to get "yes" as positive class
testdata <- banktest
testdata$y <- relevel(testdata$y, "yes")
summary(testdata$y)

#--------------------------------------Ques 1-------------------------------------#

# we will train and evaluate the model using 10 fold cross validation
trainingparameter <- trainControl(method = "repeatedcv", number = 10, repeats = 1)

#-----C5.0 algorithm for Decision Tree Classifier------#
DecTree <- train(trainingdata[,-21], trainingdata$y, 
             method = "C5.0", 
             preProcess = c("nzv", "corr"),
             trControl= trainingparameter,
             na.action = na.omit)
DecTree

# make predictions on test set & see results
DecTreePred <-predict(DecTree, testdata, na.action = na.pass)
DecTreePred
# create confusion matrix & see results
ConfMatDT <-confusionMatrix(DecTreePred, testdata$y, mode = "everything")
ConfMatDT
t(ConfMatDT$table)

#--------Naïve Bayes Classifer--------#
NaiveBayes <- train(trainingdata[,-21], trainingdata$y, 
             method = "nb", 
             preProcess = c("nzv", "corr"),
             trControl= trainingparameter,
             na.action = na.omit)
NaiveBayes

# make predictions on test set & see results
NaiveBayesPred <-predict(NaiveBayes, testdata, na.action = na.pass)
NaiveBayesPred
# create confusion matrix & see results
ConfMatNB <-confusionMatrix(NaiveBayesPred, testdata$y, mode = "everything")
ConfMatNB
t(ConfMatNB$table)

#---------SVM Classifier----------#
SVM <- train(y ~ ., data = trainingdata,
             method = "svmPoly",
             trControl= trainingparameter,
             tuneGrid = data.frame(degree = 1,scale = 1,C = 1),
             na.action = na.omit)
SVM

# make predictions on test set & see results
SVMPred <-predict(SVM, testdata)
SVMPred
# create confusion matrix & see results
ConfMatSVM <-confusionMatrix(SVMPred, testdata$y, mode = "everything")
ConfMatSVM
t(ConfMatSVM$table)

#---------Neural Network Classifier------------#
NeuNet <- train(trainingdata[,-21], trainingdata$y,
             method = "nnet",
             preProcess = c("nzv", "corr"),
             trControl= trainingparameter,
             tuneGrid = data.frame(size = 5, decay = 0.1))
NeuNet

# make predictions on test set & see results
NeuNetPred <-predict(NeuNet, testdata)
NeuNetPred
# create confusion matrix & see results
ConfMatNN <-confusionMatrix(NeuNetPred, testdata$y, mode = "everything")
ConfMatNN
t(ConfMatNN$table)

#--------------------------------------Ques 2-------------------------------------#

#----Weighted F Measure----#

# Decision Tree model F measure
F1_Score(testdata$y, DecTreePred)
FBeta_Score(testdata$y, DecTreePred, beta = 0.5)
FBeta_Score(testdata$y, DecTreePred, beta = 0.1)
FBeta_Score(testdata$y, DecTreePred, beta = 1.1)
FBeta_Score(testdata$y, DecTreePred, beta = 1.5)

# Naive Bayes model F measure
F1_Score(testdata$y, NaiveBayesPred)
FBeta_Score(testdata$y, NaiveBayesPred, beta = 0.5)
FBeta_Score(testdata$y, NaiveBayesPred, beta = 0.1)
FBeta_Score(testdata$y, NaiveBayesPred, beta = 1.1)
FBeta_Score(testdata$y, NaiveBayesPred, beta = 1.5)

# SVM model F measure
F1_Score(testdata$y, SVMPred)
FBeta_Score(testdata$y, SVMPred, beta = 0.5)
FBeta_Score(testdata$y, SVMPred, beta = 0.1)
FBeta_Score(testdata$y, SVMPred, beta = 1.1)
FBeta_Score(testdata$y, SVMPred, beta = 1.5)

# Neural Network model F measure
F1_Score(testdata$y, NeuNetPred)
FBeta_Score(testdata$y, NeuNetPred, beta = 0.5)
FBeta_Score(testdata$y, NeuNetPred, beta = 0.1)
FBeta_Score(testdata$y, NeuNetPred, beta = 1.1)
FBeta_Score(testdata$y, NeuNetPred, beta = 1.5)

# header for classifier list
classifier <- c("C5.0","nb","svm","nnet")

# since we want to give more weightage to recall the Beta should be greater than 1
# we consider the Beta = 1.1 since it gives higher value of FMeasure between 1.1 and 1.5
FMeasure <- c(FBeta_Score(testdata$y, DecTreePred, beta = 1.1),
            FBeta_Score(testdata$y, NaiveBayesPred, beta = 1.1),
            FBeta_Score(testdata$y, SVMPred, beta = 1.1),
            FBeta_Score(testdata$y, NeuNetPred, beta = 1.1))

# create dataframe to view classifiers with respective FMeasures for Beta = 1.1
FMeasureDF <- data.frame(classifier,FMeasure) 
FMeasureDF

#--------------------------------------Ques 3-------------------------------------#

#----Cost Analysis----#

# create cost matrix
CostMatrix <- cbind(c(0,10), c(1,0))
t(CostMatrix)

# cost analysis for C5.0 algorithm
CostDecTree <- C5.0(y ~., trials = 5, data=trainingdata, cost=CostMatrix)
CostDecTree
summary(CostDecTree)

CostPred <- predict(CostDecTree, testdata)
CostPred

CostConfMat <-confusionMatrix(CostPred, testdata$y, mode = "everything")
CostConfMat
t(CostConfMat$table)

# multiply cost and consfusion matrix
TotCostDT <- CostMatrix*CostConfMat$table
# find total cost
sum(TotCostDT)

# cost analysis for Naive Bayes classifier
CostNaiveBayes <- naive_bayes(y ~., trials = 5, data=trainingdata, cost=CostMatrix)
CostNaiveBayes
summary(CostNaiveBayes)

CostPredNB <- predict(CostNaiveBayes, testdata)
CostPredNB

CostConfMatNB <-confusionMatrix(CostPredNB, testdata$y, mode = "everything")
CostConfMatNB
t(CostConfMatNB$table)

#multiply cost and consfusion matrix
TotCostNB <- CostMatrix*CostConfMatNB$table
# find total cost
sum(TotCostNB)
