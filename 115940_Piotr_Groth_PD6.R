library(tidyverse)
library(skimr)
library(corrplot)
library(rpart.plot)
library(caret)
library(dplyr)
library(caTools)
library(ROCR)
set.seed(4)

#ładowanie zbioru danych
wine <- read_csv("winequality-red.csv")

#wyświetlenie pierwszych wierszy zbioru
head(wine)

#przeglądanie zbioru danych
skim(wine) 
#zbiór danych jest kompletny i zawiera tylko kolumny liczbowe

#sprawdzaenie korelacji między danymi
corrplot(cor(wine), method="number")

#dobre wino to takie, które ma quality powyżej 6.5
wine$quality = ifelse(wine$quality > 6.5, 1,0 ) 

#podział danych na zestaw treningowy i testowy
split = sample.split(wine$quality, SplitRatio = 0.70)

data.train = subset(wine, split == TRUE)
data.test = subset(wine, split == FALSE)

#model regresji logistycznej
log.model <- glm(formula = quality ~ . , family = binomial(link='logit'), data = data.train)
summary(log.model)

fitted.probabilities <- predict(log.model, newdata = data.test, type = 'response')
fitted.results <- ifelse(fitted.probabilities > 0.5, 1, 0)

log.misClasificError <- mean(fitted.results != data.test$quality)

#Dokładnośc modelu
print(paste('Accuracy for log', 1 - log.misClasificError))

table(data.test$quality, fitted.probabilities > 0.5)

# Krzywa  ROC 
roc_obj <- prediction(fitted.probabilities, data.test$quality)
roc_perf <- performance(roc_obj, "tpr", "fpr")
plot(roc_perf, main = "ROC Curve for Logistic Regression Model")

# Decision Tree Model

tree_model <- rpart(quality ~ ., method = 'class', data = data.train)

predictions <- predict(tree_model, newdata = data.test, type = 'class')

#Wyświetlanie drzewa decyzyjnego
rpart.plot(tree_model, box.palette = "auto", shadow.col = "gray", nn = TRUE)

accuracy <- sum(predictions == data.test$quality) / nrow(data.test)
#Dokładnośc modelu
print(paste('Accuracy for Tree', accuracy))







