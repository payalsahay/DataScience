# Load the caret and related libraries
library(plyr)
library(caret)
library(mlbench)
library(foreign)
library(ggplot2)
library(dplyr)
library(scales)
library(reshape)
library(e1071)
library(klaR)
library(caret)
library(e1071)
Churn <- read.csv("~/Documents/data Mining/assignment4/WA_Fn-UseC_-Telco-Customer-Churn.csv", header = TRUE)
str(Churn)
head(Churn)
##3
# Remove missing data rows
sum(is.na(Churn))
Churn <- na.omit(Churn)
Churn <- droplevels(Churn)
summary(Churn$Churn)
group_tenure <- function(tenure){
  if (tenure >= 0 & tenure <= 12){
    return('0-12 Month')
  }else if(tenure > 12 & tenure <= 24){
    return('12-24 Month')
  }else if (tenure > 24 & tenure <= 48){
    return('24-48 Month')
  }else if (tenure > 48 & tenure <=60){
    return('48-60 Month')
  }else if (tenure > 60){
    return('> 60 Month')
  }
}
Churn$tenure_group <- sapply(Churn$tenure,group_tenure)
Churn$tenure_group <- as.factor(Churn$tenure_group)
head(Churn$SeniorCitizen )
#plyr::mapvalues
Churn$SeniorCitizen <- as.factor(plyr::mapvalues(Churn$SeniorCitizen,
                                           from=c("0","1"),
                                           to=c("No", "Yes")))
Churn$customerID <- NULL
Churn$tenure <- NULL
library(corrplot)
numeric.var <- sapply(Churn, is.numeric)
corr.matrix <- cor(Churn[,numeric.var])
corrplot(corr.matrix, main="\n\nCorrelation Plot for Numerical Variables", method="number")

Churn$TotalCharges <- NULL
# Plot the class distribution

plot(CleanDataset$Churn, cex.names = 0.4)

# Plot the distribution in a nicer way using ggplot, see chaining in ggplot with + theme
qplot(Churn, data=CleanDataset, geom = "bar") + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Split data into train/test Random sampling with replacement
trainingData <- sample_frac(Churn, 0.75, replace = TRUE)
# See percentages across classes
prop.table(table(trainingData$Churn))

# Compare percentages across classes between training and orginal data
DistributionCompare <- cbind(prop.table(table(trainingData$Churn)), 
                             prop.table(table(CleanDataset$Churn)))
colnames(DistributionCompare) <- c("Training", "Orig")
DistributionCompare
# Melt Data - Convert from columns to rows
meltedDComp <- melt(DistributionCompare)
meltedDComp

# Plot to see distribution of training vs original - is it representative or is there over/under sampling?
ggplot(meltedDComp, aes(x= X1, y = value)) + geom_bar( aes(fill = X2), stat = "identity", position = "dodge") + theme(axis.text.x = element_text(angle = 90, hjust = 1))
rm(meltedDComp)
rm(DistributionCompare)

# Lets do stratified sampling. Select rows to based on Class variable as strata
TrainingDataIndex <- createDataPartition(Churn$Churn, p=0.75, list = FALSE)

# Create Training Data as subset of soyabean dataset with row index numbers as identified above and all columns
trainingData <- Churn[TrainingDataIndex,]
# See percentages across classes
prop.table(table(trainingData$Churn))

# Compare percentages across classes between training and orginal data
DistributionCompare <- cbind(prop.table(table(Churn$Churn)), 
                             prop.table(table(Churn$Churn)))
colnames(DistributionCompare) <- c("Training", "Orig")
DistributionCompare

# Run lines 39 - 47 to visualize Orig and training proportions by class

# Everything else not in training is test data. Note the - (minus)sign

testData <- Churn[-TrainingDataIndex,]
dim(trainingData)
dim(testData)
head(trainingData)
trainingData$customerID <- NULL
testData$customerID <- NULL

#logModel
ChurnLogModel <- glm(Churn ~ .,family=binomial(link="logit"),data=trainingData)
print(summary(ChurnLogModel))
anova(ChurnLogModel, test="Chisq")

#testing$Churn <- as.character(testing$Churn)
#testing$Churn[testing$Churn=="No"] <- "0"
#testing$Churn[testing$Churn=="Yes"] <- "1"
?predict
fitted.results <- predict(ChurnLogModel,newdata=testData,type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results != testData$Churn)
print(paste('Logistic Regression Accuracy',1-misClasificError))

testData$customerID <- NULL
head(testData)
testData$Churn <- as.character(testData$Churn)
testData$Churn[testData$Churn=="No"] <- "0"
testData$Churn[testData$Churn=="Yes"] <- "1"
fitted.results <- predict(LogModel,newdata=testData,type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results != testData$Churn)
print(paste('Logistic Regression Accuracy',1-misClasificError))
print("Confusion Matrix for Logistic Regression")
table(testData$Churn, fitted.results > 0.5)
#For illustration purpose, we are going to use only three variables for
#plotting Decision Trees, 
#they are “Contract”, “tenure_group” and “PaperlessBilling”.
head(trainingData)
# We will use 10 fold cross validation to train and evaluate model
TrainingParameters <- trainControl(method = "repeatedcv", number = 5, repeats = 1)

# Train a model with above parameters. We will use C5.0 algorithm
DecTreeModel <- train(Churn ~., data =trainingData, 
                      method = "C5.0",
                      trControl= TrainingParameters,
                      preProcess = c("center", "scale"),
                      na.action = na.omit
)
DecTreeModel
plot(DecTreeModel)
# Now make predictions on test set
DTPredictions <-predict(DecTreeModel, testData, na.action = na.pass)
DTPredictions
DTPredictions
testData$Churn
# Print confusion matrix and results
cm <-confusionMatrix(DTPredictions, testData$Churn,mode = "prec_recall",positive = "Yes")
cm
head(trainingData)
library(partykit)
Dtree <- ctree(Churn~Contract+tenure+PaperlessBilling, trainingData)
plot(Dtree)

#NB bayes
library(bnclassify)
library(NB)
TrainPara <- trainControl(method = "repeatedcv", number = 10, repeats = 1)
head(trainingData)
NB_Model <- train(Churn ~., data =trainingData,
                  method = "nb", 
                  trControl= TP,
                  na.action = na.omit)
NB_Model
# make predictions on test set & see results
NB_Predict <-predict(NB_Model, testData, na.action = na.pass)
NB_Predict
# create confusion matrix & see results
cmnb <-confusionMatrix(NB_Predict, testData$Churn ,mode = "prec_recall",positive = "Yes")
cmnb
### SVM MODEL
SVModel <- train(Churn ~ ., data = trainingData,
                        method = "svmPoly",
                        trControl= TrainPara,
                        tuneGrid = data.frame(degree = 1,
                                              scale = 1,
                                              C = 1
                        ))


summary(SVModel)
SVModel
SVMPredictions<-predict(SVModel, testData)
# See predictions
SVMPredictions
 # Create confusion matrix
cmSVM <-confusionMatrix(SVMPredictions,testData$Churn,mode = "prec_recall",positive = "Yes")
cmSVM$overall
cmSVM
  
#Random Forest
library(randomForest)
randomModel <- randomForest(Churn ~., data = trainingData,ntree = 500, mtry = 2)
print(randomModel)
pred_random <- predict(randomModel, testData)
cm_rf <- confusionMatrix(pred_random, testData$Churn,mode = "prec_recall",positive = "Yes")
cm_rf
pred_random
library(MLmetrics)
table(testData$Churn,pred_random)
Accuracy(pred_random, testData$Churn)
F1_Score(testData$Churn, pred_random, positive = "Yes")
FBeta_Score(testData$Churn, pred_random, positive = "Yes", beta = 0.5)
FBeta_Score(testData$Churn, pred_random, positive = "Yes", beta = 0.1)

plot(randomModel)

###
# create training control
library()
#Traincontrol_Bank_new <- trainControl(method="cv", number=5, summaryFunction = twoClassSummary, savePredictions=TRUE, classProbs=TRUE)
# Stacking Algorithms
control <- trainControl(method="repeatedcv", number=5, repeats=2, savePredictions=TRUE, classProbs=TRUE)
#algorithmList <- c( 'knn','glm','rpart','rf')
algorithmList <- c( "knn", "rf", "rpart")
stack_models <- caretList(Churn~., data=testData, trControl=control, methodList=algorithmList)
stacking_results <- resamples(stack_models)
summary(stacking_results)
dotplot(stacking_results)
modelCor(stacking_results)
splom(stacking_results)
# stack using Logistics Regression
stackControl <- trainControl(method="repeatedcv", number=5, repeats=2, savePredictions=TRUE, classProbs=TRUE)
set.seed(1024)
stackControl$method
##knn 
stack.glm <- caretStack(stack_models, method="knn",  trControl=stackControl)
print(stack.glm)
# evaluate results on test set
stack.glm
stack.glm$models
# Create confusion matrix
Ensemble_pred1 <- predict(stack.glm, newdata=testData)
Ensemble_pred1
con_matrix_ensemble <- confusionMatrix(Ensemble_pred1,testData$Churn,mode = "prec_recall",positive = "Yes")
con_matrix_ensemble
library(MLmetrics)
table(testData$Churn,Ensemble_pred1)
Accuracy(Ensemble_pred1, testData$Churn)
F1_Score(testData$Churn, Ensemble_pred1, positive = "Yes")
FBeta_Score(testData$Churn, Ensemble_pred1, positive = "Yes", beta = 0.5)
FBeta_Score(testData$Churn, Ensemble_pred1, positive = "Yes", beta = 0.1)


