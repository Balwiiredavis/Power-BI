



Thyroid <-read.csv("/Users/balwiiredavis/Desktop/project/Thyroid_Diff.csv", header=TRUE)

 
head(Thyroid)


str(Thyroid)


summary(Thyroid)


# Check for missing values

missing_values <- colSums(is.na(Thyroid))
print(missing_values)


# Convert categorical variables into factors

Thyroid$Gender <- factor(Thyroid$Gender)
Thyroid$Smoking <- factor(Thyroid$Smoking)
Thyroid$Hx.Smoking <- factor(Thyroid$Hx.Smoking)
Thyroid$Hx.Radiothreapy <- factor(Thyroid$Hx.Radiothreapy)
Thyroid$Thyroid.Function <- factor(Thyroid$Thyroid.Function)
Thyroid$Physical.Examination <- factor(Thyroid$Physical.Examination)
Thyroid$Adenopathy <- factor(Thyroid$Adenopathy)
Thyroid$Pathology <- factor(Thyroid$Pathology)
Thyroid$Focality <- factor(Thyroid$Focality)
Thyroid$Risk <- factor(Thyroid$Risk)
Thyroid$T <- factor(Thyroid$T)
Thyroid$N <- factor(Thyroid$N)
Thyroid$M <- factor(Thyroid$M)
Thyroid$Stage <- factor(Thyroid$Stage)
Thyroid$Response <- factor(Thyroid$Response)
Thyroid$Recurred <- factor(Thyroid$Recurred)


# library for decision trees

library(rpart)

# Set seed for reproducibility

set.seed(123)

# Split the data set into training and testing sets 

train_indices <- sample(1:nrow(Thyroid), 0.8 * nrow(Thyroid))
train_data <- Thyroid[train_indices, ]
test_data <- Thyroid[-train_indices, ]

# Train a decision tree classifier

decision_tree_model <- rpart(Risk ~ ., data = train_data, method = "class")

# Print the decision tree

print(decision_tree_model)

# Evaluate the classifier's performance on the testing data

predictions <- predict(decision_tree_model, test_data, type = "class")
confusion_matrix <- table(predictions, test_data$Risk)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)

# Print accuracy

print(paste("Accuracy:", round(accuracy, 4)))

# deeper evaluation of the model's performance

# Calculate confusion matrix

conf_matrix <- table(predictions, test_data$Risk)

# Define function to calculate precision, recall, and F1-score

calculate_metrics <- function(conf_matrix) {
  precision <- diag(conf_matrix) / rowSums(conf_matrix)
  recall <- diag(conf_matrix) / colSums(conf_matrix)
  f1_score <- 2 * (precision * recall) / (precision + recall)
  return(data.frame(Precision = precision, Recall = recall, F1_Score = f1_score))
}

# Calculate precision, recall, and F1-score for each class

metrics <- calculate_metrics(conf_matrix)

# Print the calculated metrics

print("Precision, Recall, and F1-Score for Each Class:")
print(metrics)



# Hyperparameter Tuning

library(caret)

# Define the training control

ctrl <- trainControl(method = "cv",    
                     number = 5,       
                     verboseIter = TRUE)  

# Define the grid of hyper parameters to search over

hyper_grid <- expand.grid(cp = seq(0.01, 0.5, by = 0.01))  

# Perform grid search using caret's train function

hyper_tuned_model <- train(Risk ~ .,               
                           data = train_data,     
                           method = "rpart",      
                           trControl = ctrl,     
                           tuneGrid = hyper_grid)  


# Print the best hyperparameters found

print("Best Hyperparameters:")
print(hyper_tuned_model$bestTune)

# Print the tuned decision tree model

print(hyper_tuned_model$finalModel)


# Evaluate the tuned model's performance on the testing data

predictions_hyper_tuned <- predict(hyper_tuned_model, test_data)
confusion_matrix_hyper_tuned <- table(predictions_hyper_tuned, test_data$Risk)
accuracy_hyper_tuned <- sum(diag(confusion_matrix_hyper_tuned)) / sum(confusion_matrix_hyper_tuned)

# Print accuracy of the tuned model

print(paste("Accuracy of Tuned Model:", round(accuracy_hyper_tuned, 4)))

# random forest model

# Load required library for Random Forest

library(randomForest)

# Train Random Forest classifier

random_forest_model <- train(Risk ~ .,              
                             data = train_data,     
                             method = "rf",         
                             trControl = ctrl,     
                             tuneGrid = data.frame(mtry = c(2, 3, 4)))  

# Print the trained Random Forest model

print(random_forest_model)

# Evaluate the Random Forest model's performance on the testing data

predictions_random_forest <- predict(random_forest_model, test_data)
confusion_matrix_random_forest <- table(predictions_random_forest, test_data$Risk)
accuracy_random_forest <- sum(diag(confusion_matrix_random_forest)) / sum(confusion_matrix_random_forest)

# Print accuracy of the Random Forest model

print(paste("Accuracy of Random Forest Model:", round(accuracy_random_forest, 4)))


# Gradient Boosting Machines

# Train Gradient Boosting Machine classifier

gbm_model <- train(Risk ~ .,                   
                   data = train_data,         
                   method = "gbm",           
                   trControl = ctrl,        
                   tuneGrid = expand.grid(n.trees = c(100, 200, 300),    
                                          interaction.depth = c(1, 2, 3),  
                                          shrinkage = c(0.1, 0.01),       
                                          n.minobsinnode = c(10, 20)))     



# Print the trained GBM model

print(gbm_model)

# Evaluate the GBM model's performance on the testing data

predictions_gbm <- predict(gbm_model, test_data)
confusion_matrix_gbm <- table(predictions_gbm, test_data$Risk)
accuracy_gbm <- sum(diag(confusion_matrix_gbm)) / sum(confusion_matrix_gbm)

# Print accuracy of the GBM model

print(paste("Accuracy of GBM Model:", round(accuracy_gbm, 4)))

# Interpreting the decision tree model

# Extract important features and their relationships from the decision tree model

important_features <- varImp(decision_tree_model, scale = FALSE)
print("Important Features and their Relationships:")
print(important_features)

# features like Gender, Hx.Smoking, Hx.Radiotherapy, and Thyroid Function have no importance according to the decision tree model.

# cross-validation 

# Define cross-validation control

ctrl <- trainControl(method = "cv",    
                     number = 5,       
                     verboseIter = TRUE)  

# Train decision tree model with cross-validation

decision_tree_model_cv <- train(Risk ~ .,            
                                data = Thyroid,     
                                method = "rpart",   
                                trControl = ctrl)  

# Print cross-validated results

print(decision_tree_model_cv)






