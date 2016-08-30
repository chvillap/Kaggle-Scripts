# =============================================================================
# training_test.R
#
# Script to train a predictive model and apply it to the test set.
# =============================================================================

for (k in 1:4) {
    df.train <- read.csv(sprintf("train%d.csv", k))
    df.test <- read.csv(sprintf("test%d.csv", k))

    # -------------------------------------------------------------------------
    # Random forest model
    
    # require(randomForest)

    # # Get the target variable as a factor.
    # df.train$Survived <- as.factor(df.train$Survived)

    # # Use the Random Forest algorithm to train an ensemble predictive model.
    # rf <- randomForest(Survived ~ .,
    #                    data = subset(df.train, select = -PassengerId),
    #                    ntree = 200, importance = FALSE)

    # # Predict outcomes in the test set.
    # y.pred <- predict(rf, subset(df.test, select = -PassengerId))
    # df.pred <- data.frame(PassengerId = df.test$PassengerId,
    #                       Survived = y.pred)

    # -------------------------------------------------------------------------
    # XGBoost model
    
    require(Matrix)
    require(xgboost)
    
    # Use one-hot encoding to convert each categorical variable to a set of
    # binary variables.
    X.train <- sparse.model.matrix(
        Survived ~ . - 1,
        data = subset(df.train, select = -PassengerId))
    X.test <- sparse.model.matrix(
        ~ . - 1,
        data = subset(df.test, select = -PassengerId))
    
    # Get the target variable as a logic vector.
    y.train <- df.train$Survived == 1
    
    print(colnames(X.train))
    print(dim(X.train))
    print(colnames(X.test))
    print(dim(X.test))
    
    # Use the XGBoost algorithm to train an ensemble predictive model.
    xgb <- xgboost(data = X.train, label = y.train,
                   eta = 0.1, nround = 10, max.depth = 6,
                   objective = "binary:logistic", verbose = 1)
    
    # Predict outcomes in the test set.
    y.pred <- predict(xgb, X.test)
    df.pred <- data.frame(PassengerId = df.test$PassengerId,
                          Survived = ifelse(y.pred < 0.5, 0, 1))

    # -------------------------------------------------------------------------

    # Write the submission file.
    fname <- sprintf("submission%d.csv", k)
    write.csv(df.pred, fname, quote = FALSE, row.names = FALSE)
}
