#import the packages
install.packages('tidyverse')
install.packages('caret')
install.packages('MASS')
install.packages('randomForest')
install.packages('lime')
library(tidyverse)
library(dplyr)
library(caret)
library(MASS)
library(pROC)
library(randomForest)
library(lime)
diabetes_data <- read.csv('diabetes_data.csv', sep = ';')

#Check the dataset
str(diabetes_data)
sum(is.na(diabetes_data))

#Create a new subset for modeling
selected_vars <- c("gender", "polyuria", "polydipsia", "sudden_weight_loss", "polyphagia", "irritability", "partial_paresis", "class")
df_diabetes <- diabetes_data %>% dplyr::select(all_of(selected_vars))
df_diabetes$gender <- factor(df_diabetes$gender, levels = c(0, 1), labels = c("Male", "Female")) #split into male and female

#Check the dataset again
head(df_diabetes)
sum(is.na(df_diabetes))

#setseed for re-examinate
set.seed(1)

#Split the dataset into training and testing
splitdata_dia <- createDataPartition(df_diabetes$class, p = 0.80, list = FALSE)
train_data_full <- df_diabetes[splitdata_dia, selected_vars]
test_data_full <- df_diabetes[-splitdata_dia, selected_vars]

#The first model included all the variables
model_1 <- glm(class ~ ., family = binomial(link = "logit"), data = train_data_full)
summary(model_1)

#The first model with testing set
predictions_1 <- predict(model_1, test_data_full, type = "response")
predicted_class_1 <- ifelse(predictions_1 > 0.5, 1, 0)

#Confusion matrix and AUC
confusionMatrix_1 <- confusionMatrix(factor(predicted_class_1), factor(test_data_full$class))
auc_1 <- auc(roc(factor(test_data_full$class), predictions_1))


#remove the polyphagia and partial_paresis due to the low correlation
selected_vars_2 <- c("gender", "polyuria", "polydipsia", "sudden_weight_loss", "irritability", "class")

#Second model without polyphagia and partial_paresis
train_data_2 <- train_data_full[selected_vars_2]
test_data_2 <- test_data_full[selected_vars_2]

model_2 <- glm(class ~ ., family = binomial(link = "logit"), data = train_data_2)

summary(model_2)

# Second model with testing set
predictions_2 <- predict(model_2, test_data_2, type = "response")
predicted_class_2 <- ifelse(predictions_2 > 0.5, 1, 0)

#Confusion matrix and AUC
confusionMatrix_2 <- confusionMatrix(factor(predicted_class_2), factor(test_data_2$class))
auc_2 <- auc(roc(factor(test_data_2$class), predictions_2))

#Result for both 
print(confusionMatrix_1)
print(auc_1)
print(confusionMatrix_2)
print(auc_2)

#K-fold
set.seed(2) 
folds <- createFolds(df_diabetes$class, k = 10)
cv_results_1 <- lapply(folds, function(x) {
  train_fold <- df_diabetes[-x, selected_vars]
  test_fold <- df_diabetes[x, selected_vars]
  model_fold <- glm(class ~ ., family = binomial(link = "logit"), data = train_fold)
  predictions_fold <- predict(model_fold, test_fold, type = "response")
  roc_auc <- auc(roc(test_fold$class, predictions_fold))
  return(roc_auc)
})

mean_auc_1 <- mean(unlist(cv_results_1))
print(mean_auc_1)


set.seed(2) 
cv_results_2 <- lapply(folds, function(x) {
  train_fold <- df_diabetes[-x, selected_vars_2]
  test_fold <- df_diabetes[x, selected_vars_2]
  model_fold <- glm(class ~ ., family = binomial(link = "logit"), data = train_fold)
  predictions_fold <- predict(model_fold, test_fold, type = "response")
  roc_auc <- auc(roc(test_fold$class, predictions_fold))
  return(roc_auc)
})

mean_auc_2 <- mean(unlist(cv_results_2))
print(mean_auc_2)

#Try other model: Random Forest
train_data_full$class <- as.factor(train_data_full$class)
test_data_full$class <- as.factor(test_data_full$class)
train_data_2$class <- as.factor(train_data_2$class)
test_data_2$class <- as.factor(test_data_2$class)

#Random Forest
model_rf_full <- randomForest(class ~ ., data = train_data_full, ntree = 500)
print(model_rf_full)

predictions_rf_full <- predict(model_rf_full, test_data_full)
confusionMatrix_rf_full <- table(test_data_full$class, predictions_rf_full)
print(confusionMatrix_rf_full)

#Second model without polyphagia and partial_paresis - Random Forest
model_rf_simplified <- randomForest(class ~ ., data = train_data_2, ntree = 500)
print(model_rf_simplified)
predictions_rf_simplified <- predict(model_rf_simplified, test_data_2)
confusionMatrix_rf_simplified <- table(test_data_2$class, predictions_rf_simplified)
print(confusionMatrix_rf_simplified)


#Re-ajust the model to 600 for 7 variables
model_rf_full <- randomForest(class ~ ., data = train_data_full, ntree = 600)
print(model_rf_full)

predictions_rf_full <- predict(model_rf_full, test_data_full)
confusionMatrix_rf_full <- table(test_data_full$class, predictions_rf_full)
print(confusionMatrix_rf_full)

#Second model without polyphagia and partial_paresis
model_rf_simplified <- randomForest(class ~ ., data = train_data_2, ntree = 600)
print(model_rf_simplified)

predictions_rf_simplified <- predict(model_rf_simplified, test_data_2)
confusionMatrix_rf_simplified <- table(test_data_2$class, predictions_rf_simplified)
print(confusionMatrix_rf_simplified)

#Random forest with mtry modified 
mtry_values_full <- c(1, 3, 5, 7)
results_full <- list()

for (mtry_val in mtry_values_full) {
  set.seed(2)  
  model_rf_full <- randomForest(class ~ ., data = train_data_full, ntree = 500, mtry = mtry_val)
  predictions_rf_full <- predict(model_rf_full, test_data_full)
  confusionMatrix_rf_full <- table(test_data_full$class, predictions_rf_full)
  
  results_full[[paste("mtry =", mtry_val)]] <- list(Model = model_rf_full, ConfusionMatrix = confusionMatrix_rf_full)
}

for (result_name in names(results_full)) {
  cat(result_name, "\n")
  print(results_full[[result_name]]$Model)
  print(results_full[[result_name]]$ConfusionMatrix)
  cat("\n")
}

#Random forest with mtry modified - second model
mtry_values_simplified <- c(1, 3, 5)
results_simplified <- list()

for (mtry_val in mtry_values_simplified) {
  set.seed(2)  
  model_rf_simplified <- randomForest(class ~ ., data = train_data_2, ntree = 500, mtry = mtry_val)
  predictions_rf_simplified <- predict(model_rf_simplified, test_data_2)
  confusionMatrix_rf_simplified <- table(test_data_2$class, predictions_rf_simplified)
  
  results_simplified[[paste("mtry =", mtry_val)]] <- list(Model = model_rf_simplified, ConfusionMatrix = confusionMatrix_rf_simplified)
}

for (result_name in names(results_simplified)) {
  cat(result_name, "\n")
  print(results_simplified[[result_name]]$Model)
  print(results_simplified[[result_name]]$ConfusionMatrix)
  cat("\n")
}

#Random forest with mtry modified and N600
mtry_values_full <- c(1, 3, 5, 7)
results_full <- list()

for (mtry_val in mtry_values_full) {
  set.seed(2) 
  model_rf_full <- randomForest(class ~ ., data = train_data_full, ntree = 600, mtry = mtry_val)
  predictions_rf_full <- predict(model_rf_full, test_data_full)
  confusionMatrix_rf_full <- table(test_data_full$class, predictions_rf_full)
  
  results_full[[paste("mtry =", mtry_val)]] <- list(Model = model_rf_full, ConfusionMatrix = confusionMatrix_rf_full)
}

for (result_name in names(results_full)) {
  cat(result_name, "\n")
  print(results_full[[result_name]]$Model)
  print(results_full[[result_name]]$ConfusionMatrix)
  cat("\n")
}

#Random forest with mtry modified - second model N600
mtry_values_simplified <- c(1, 3, 5)
results_simplified <- list()

for (mtry_val in mtry_values_simplified) {
  set.seed(123) 
  model_rf_simplified <- randomForest(class ~ ., data = train_data_2, ntree = 600, mtry = mtry_val)
  predictions_rf_simplified <- predict(model_rf_simplified, test_data_2)
  confusionMatrix_rf_simplified <- table(test_data_2$class, predictions_rf_simplified)
  
  results_simplified[[paste("mtry =", mtry_val)]] <- list(Model = model_rf_simplified, ConfusionMatrix = confusionMatrix_rf_simplified)
}

for (result_name in names(results_simplified)) {
  cat(result_name, "\n")
  print(results_simplified[[result_name]]$Model)
  print(results_simplified[[result_name]]$ConfusionMatrix)
  cat("\n")
}

#Accuracy for rf models
calculate_accuracy <- function(conf_matrix) {
  true_positive <- conf_matrix[2, 2]
  true_negative <- conf_matrix[1, 1]
  total <- sum(conf_matrix)
  accuracy <- (true_positive + true_negative) / total
  return(accuracy)
}


#Rf_full_N500
conf_matrix_rf_full_500 <- matrix(c(38, 1, 7, 58), nrow = 2, byrow = TRUE)
accuracy_rf_full_500 <- calculate_accuracy(conf_matrix_rf_full_500)
print(paste("Accuracy for RF Full Features, 500 trees:", accuracy_rf_full_500))


#Rf_simplified_N500
conf_matrix_rf_simplified_500 <- matrix(c(34, 5, 4, 61), nrow = 2, byrow = TRUE)
accuracy_rf_simplified_500 <- calculate_accuracy(conf_matrix_rf_simplified_500)
print(paste("Accuracy for RF Simplified Features, 500 trees:", accuracy_rf_simplified_500))


#Rf_full_N600
conf_matrix_rf_full_600 <- matrix(c(38, 1, 7, 58), nrow = 2, byrow = TRUE)
accuracy_rf_full_600 <- calculate_accuracy(conf_matrix_rf_full_600)
print(paste("Accuracy for RF Full Features, 600 trees:", accuracy_rf_full_600))


#Rf_simplified_N600
conf_matrix_rf_simplified_600 <- matrix(c(34, 5, 4, 61), nrow = 2, byrow = TRUE)
accuracy_rf_simplified_600 <- calculate_accuracy(conf_matrix_rf_simplified_600)
print(paste("Accuracy for RF Simplified Features, 600 trees:", accuracy_rf_simplified_600))


#Rf_full_dif_mtry_N500
conf_matrix_rf_full_mtry1 <- matrix(c(37, 2, 8, 57), nrow = 2, byrow = TRUE)
accuracy_rf_full_mtry1 <- calculate_accuracy(conf_matrix_rf_full_mtry1)
print(paste("Accuracy for RF Full Features, mtry=1:", accuracy_rf_full_mtry1))

conf_matrix_rf_full_mtry3 <- matrix(c(38, 1, 7, 58), nrow = 2, byrow = TRUE)
accuracy_rf_full_mtry3 <- calculate_accuracy(conf_matrix_rf_full_mtry3)
print(paste("Accuracy for RF Full Features, mtry=3:", accuracy_rf_full_mtry3))

conf_matrix_rf_full_mtry5 <- matrix(c(39, 0, 6, 59), nrow = 2, byrow = TRUE)
accuracy_rf_full_mtry5 <- calculate_accuracy(conf_matrix_rf_full_mtry5)
print(paste("Accuracy for RF Full Features, mtry=5:", accuracy_rf_full_mtry5))

conf_matrix_rf_full_mtry7 <- matrix(c(39, 0, 6, 59), nrow = 2, byrow = TRUE)
accuracy_rf_full_mtry7 <- calculate_accuracy(conf_matrix_rf_full_mtry7)
print(paste("Accuracy for RF Full Features, mtry=7:", accuracy_rf_full_mtry7))


#Rf_simplified_dif_mtry_N500
conf_matrix_rf_simplified_mtry1 <- matrix(c(34, 5, 4, 61), nrow = 2, byrow = TRUE)
accuracy_rf_simplified_mtry1 <- calculate_accuracy(conf_matrix_rf_simplified_mtry1)
print(paste("Accuracy for RF Simplified Features, mtry=1:", accuracy_rf_simplified_mtry1))

conf_matrix_rf_simplified_mtry3 <- matrix(c(34, 5, 4, 61), nrow = 2, byrow = TRUE)
accuracy_rf_simplified_mtry3 <- calculate_accuracy(conf_matrix_rf_simplified_mtry3)
print(paste("Accuracy for RF Simplified Features, mtry=3:", accuracy_rf_simplified_mtry3))

conf_matrix_rf_simplified_mtry5 <- matrix(c(34, 5, 4, 61), nrow = 2, byrow = TRUE)
accuracy_rf_simplified_mtry5 <- calculate_accuracy(conf_matrix_rf_simplified_mtry5)
print(paste("Accuracy for RF Simplified Features, mtry=5:", accuracy_rf_simplified_mtry5))


#Rf_full_dif_mtry_N600
conf_matrix_rf_full_mtry1_600 <- matrix(c(36, 3, 8, 57), nrow = 2, byrow = TRUE)
accuracy_rf_full_mtry1_600 <- calculate_accuracy(conf_matrix_rf_full_mtry1_600)
print(paste("Accuracy for RF Full Features, mtry=1, 600 trees:", accuracy_rf_full_mtry1_600))

conf_matrix_rf_full_mtry3_600 <- matrix(c(38, 1, 7, 58), nrow = 2, byrow = TRUE)
accuracy_rf_full_mtry3_600 <- calculate_accuracy(conf_matrix_rf_full_mtry3_600)
print(paste("Accuracy for RF Full Features, mtry=3, 600 trees:", accuracy_rf_full_mtry3_600))

conf_matrix_rf_full_mtry5_600 <- matrix(c(39, 0, 6, 59), nrow = 2, byrow = TRUE)
accuracy_rf_full_mtry5_600 <- calculate_accuracy(conf_matrix_rf_full_mtry5_600)
print(paste("Accuracy for RF Full Features, mtry=5, 600 trees:", accuracy_rf_full_mtry5_600))

conf_matrix_rf_full_mtry7_600 <- matrix(c(39, 0, 6, 59), nrow = 2, byrow = TRUE)
accuracy_rf_full_mtry7_600 <- calculate_accuracy(conf_matrix_rf_full_mtry7_600)
print(paste("Accuracy for RF Full Features, mtry=7, 600 trees:", accuracy_rf_full_mtry7_600))


#Rf_simplified_dif_mtry_N500
conf_matrix_rf_simplified_mtry1_600 <- matrix(c(36, 3, 8, 57), nrow = 2, byrow = TRUE)
accuracy_rf_simplified_mtry1_600 <- calculate_accuracy(conf_matrix_rf_simplified_mtry1_600)
print(paste("Accuracy for RF Simplified Features, mtry=1, 600 trees:", accuracy_rf_simplified_mtry1_600))

conf_matrix_rf_simplified_mtry3_600 <- matrix(c(34, 5, 4, 61), nrow = 2, byrow = TRUE)
accuracy_rf_simplified_mtry3_600 <- calculate_accuracy(conf_matrix_rf_simplified_mtry3_600)
print(paste("Accuracy for RF Simplified Features, mtry=3, 600 trees:", accuracy_rf_simplified_mtry3_600))

conf_matrix_rf_simplified_mtry5_600 <- matrix(c(34, 5, 4, 61), nrow = 2, byrow = TRUE)
accuracy_rf_simplified_mtry5_600 <- calculate_accuracy(conf_matrix_rf_simplified_mtry5_600)
print(paste("Accuracy for RF Simplified Features, mtry=5, 600 trees:", accuracy_rf_simplified_mtry5_600))


#Model Interpretability
#Train a random forest model
model_rf_full <- randomForest(class ~ ., data = train_data_full, ntree = 500)

#Calculate Feature Importance
importance_rf <- importance(model_rf_full)

#Draw a graph
varImpPlot(model_rf_full, main = "Feature Importance in Random Forest Model")
