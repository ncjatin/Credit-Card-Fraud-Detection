
#Import all packages

if (!require(ROSE)) install.packages('ROSE')
library(ROSE)
if (!require(caTools)) install.packages('caTools')
library(caTools)
if (!require(dplyr)) install.packages('dplyr')
library(dplyr)
if (!require(e1071)) install.packages('e1071')
library(e1071)
if (!require(corrplot)) install.packages('corrplot')
library(corrplot)
if (!require(gridExtra)) install.packages('gridExtra')
library(gridExtra)
if (!require(class)) install.packages('class')
library(class)
if (!require(caret)) install.packages('caret')
library(caret)

set.seed(123)

#Import Data from csv
datasetfull <- read.csv('creditcard1.csv')
datasetfull$Class <- factor(datasetfull$Class,levels = c(0,1))
datasetfull[-31] <- scale(datasetfull[-31])



#Splitting of datasets 
smpl.split <- sample.split(datasetfull$Class,SplitRatio = 0.80)
traindata <- subset(datasetfull,smpl.split==TRUE)
testdata <- subset(datasetfull,smpl.split==FALSE)

Orig_N <- nrow(traindata)
Orig_Ratio <-table(traindata$Class)/Orig_N



#Balancing the train dataset using Rose p=0.2 N= 1.2 * Original N
traindata.balance <- ROSE(Class ~ ., data = traindata, seed = 1,p=0.15,N=round(Orig_N*1.15))$data

plot_before <- ggplot(traindata,aes(x=Class)) +geom_bar(fill='skyblue' ,color='red')+
  theme_grey() +ggtitle('Before Balance')

plot_after <-ggplot(traindata.balance,aes(x=Class)) +geom_bar(fill='skyblue' ,color='red')+
  theme_grey() +ggtitle('After Balance')

grid.arrange(plot_before, plot_after, ncol=2)


############ Balancing Done####################################




############## 1. LOGISTIC MODEL #####################################
logistic_classifier <- readRDS('Logistic')
logistic_predictions <- predict(logistic_classifier,newdata=testdata[,-31],type='response')
logistic.end.time <- Sys.time()

logistic_classifier_Result <- ifelse(logistic_predictions>0.5,1,0)
conf_matrix_logistic <- table(Actual=testdata[,31],Predicted=logistic_classifier_Result)


#Evaluating model
#Parameters for evaluation
logistic.TN <-conf_matrix_logistic[1,1]
logistic.TP <-conf_matrix_logistic[2,2]
logistic.FP <-conf_matrix_logistic[1,2]
logistic.FN <-conf_matrix_logistic[2,1]


logistic.Precision <- logistic.TP/(logistic.TP+logistic.FP)
logistic.specificity <- logistic.TN/(logistic.TN+logistic.FP)
logistic.sensitivity <- logistic.TP/(logistic.TP+logistic.FN)
logistic.fallout <- logistic.FP/(logistic.FP+logistic.TN)

logistic.evaluation.parameters <- c('Precision','Specificity','Sensitivity','Fallout')
logistic.evaluation.values <- c(round(logistic.Precision,3),round(logistic.specificity,3),round(logistic.sensitivity,3),round(logistic.fallout,3))
logistic.pf <-as.data.frame(cbind(logistic.evaluation.parameters,logistic.evaluation.values))

ggplot(logistic.pf,aes(x=logistic.evaluation.parameters ,y=logistic.evaluation.values)) +geom_col(color='Red',fill='Pink')+
  ggtitle('Logistic Model Performance')+
  xlab('Evaluation Parameters')+ ylab('Scores')+
  theme_minimal()

roc.curve(testdata[,31],logistic_predictions)

################## LOGISTIC DONE ##################################################




################# 2.NAIVE BAYES CLASSIFIER #########################################

Naive_Bayes_classifier <- readRDS('NaiveBayes')

Naive_Bayes_pred = predict(Naive_Bayes_classifier,newdata = testdata[,-c(31)])
conf_matrix_naivebayes<- table(Actual=testdata[,31],Prediction=Naive_Bayes_pred)

naivebayes.TP <- conf_matrix_naivebayes[2,2]
naivebayes.TN <- conf_matrix_naivebayes[1,1]
naivebayes.FN <- conf_matrix_naivebayes[2,1]
naivebayes.FP <- conf_matrix_naivebayes[1,2]

naivebayes.Precision <- naivebayes.TP/(naivebayes.TP+naivebayes.FP)
naivebayes.specificity <- naivebayes.TN/(naivebayes.TN+naivebayes.FP)
naivebayes.sensitivity <- naivebayes.TP/(naivebayes.TP+naivebayes.FN)
naivebayes.fallout <- naivebayes.FP/(naivebayes.FP+naivebayes.TN)

naivebayes.evaluation.parameters <- c('Precision','Specificity','Sensitivity','Fallout')
naivebayes.evaluation.values <- c(round(naivebayes.Precision,3),round(naivebayes.specificity,3),round(naivebayes.sensitivity,3),round(naivebayes.fallout,3))
naivebayes.pf <-as.data.frame(cbind(naivebayes.evaluation.parameters,naivebayes.evaluation.values))

ggplot(naivebayes.pf,aes(x=naivebayes.evaluation.parameters ,y=naivebayes.evaluation.values)) +geom_col(color='Red',fill='Pink')+
  ggtitle('Naive Bayes Model Performance')+
  xlab('Evaluation Parameters')+ ylab('Scores')+
  theme_minimal()

roc.curve(testdata[,31],Naive_Bayes_pred)

####################### NAIVE BAYES DONE ##################################################




###########################3.KNN Classification begins here #########################################

min.max.normalize <- function(dataset){
  for (i in 1:ncol(dataset)){
    col_min = min(dataset[,i])
    col_max = max(dataset[,i])
    normal.column = sapply(dataset[,i],function(x){ (x-col_min)/(col_max-col_min)})
    dataset[,i] = normal.column
  }
  
  return (dataset)
}


traindata.normalize <- min.max.normalize(traindata.balance[-31])
traindata.normalize$Class <- traindata.balance$Class
testdata.normalize <- min.max.normalize(testdata[-31])
testdata.normalize$Class <- testdata$Class

Knn_classifier_pred <- readRDS('KNN')
conf_matrix_knn = table(Actual=testdata$Class, Predicted =Knn_classifier_pred)

#Parameters for evaluation

Knn.TP <- conf_matrix_knn[2,2]
Knn.TN <- conf_matrix_knn[1,1]
Knn.FN <- conf_matrix_knn[2,1]
Knn.FP <- conf_matrix_knn[1,2]

Knn.Precision <- Knn.TP/(Knn.TP+Knn.FP)
Knn.specificity <- Knn.TN/(Knn.TN+Knn.FP)
Knn.sensitivity <- Knn.TP/(Knn.TP+Knn.FN)
Knn.fallout <- Knn.FP/(Knn.FP+Knn.TN)

Knn.evaluation.parameters <- c('Precision','Specificity','Sensitivity','Fallout')
Knn.evaluation.values <- c(round(Knn.Precision,3),round(Knn.specificity,3),round(Knn.sensitivity,3),round(Knn.fallout,3))
Knn.pf <-as.data.frame(cbind(Knn.evaluation.parameters,Knn.evaluation.values))

ggplot(Knn.pf,aes(x=Knn.evaluation.parameters ,y=Knn.evaluation.values)) +geom_col(color='Red',fill='Pink')+
  ggtitle('kNN Model Performance')+
  xlab('Evaluation Parameters')+ ylab('Scores')+
  theme_minimal()


roc.curve(testdata[,31],Knn_classifier_pred)

############################KNN Classification Done#####################################