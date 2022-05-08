
##### EE418 Intro to Machine Learning - Project - Paris Housing Prices #####


# install.packages("ggcorrplot")
# install.packages("skimr")
# install.packages("randomForest")
# install.packages("caTools")
# install.packages("caret")


# Importing data set
housing <- read.csv("ParisHousing.csv")


###------------------Exploratory data analysis and pre-processing------------------###

# Check for missing data
sum(is.na(housing))


# First and last ten observations
head(housing, 10)
tail(housing, 10)


# Examining data
str(housing)


# Summary of data
summary(housing)


# Library for additional stats about dataset
library(skimr)
skim(housing)


# Summary, histogram, and box-plot of housing price
summary(housing$price)


boxplot(housing$price, main = "Housing prices")

hist(housing$price, xlab = "Price", col = "lightblue", main = "Housing prices")



# Square meters histogram
hist(housing$squareMeters, xlab = "Square meters", col = "lightblue",
     main = "Square meters")



# Number of rooms histogram
hist(housing$numberOfRooms, 
     xlab = "Rooms", 
     col = "lightblue",
     main = "Number of rooms")



# Graphs for other features


# 1 row, 2 column layout for graphs
par(mfrow = c(1, 2))


hist(housing$made, 
     xlab = "Year", 
     col = "lightblue",
     main = "Year built")

hist(housing$floors, 
     xlab = "Floors", 
     col = "lightblue",
     main = "Number of floors")




# 2x2 layout for upcoming graphs
par(mfrow = c(2, 2))

# Mapping some numeric variables to categorical ones for some graphs
vars_ctg <- within(housing, {
  isNewBuilt_ctg <- NA
  isNewBuilt_ctg[isNewBuilt == 1] <- "Yes"
  isNewBuilt_ctg[isNewBuilt == 0] <- "No"
  hasYard_ctg <- NA
  hasYard_ctg[hasYard == 1] <- "Yes"
  hasYard_ctg[hasYard == 0] <- "No"
  hasPool_ctg <- NA
  hasPool_ctg[hasPool == 1] <- "Yes"
  hasPool_ctg[hasPool == 0] <- "No"
  hasStorageRoom_ctg <- NA
  hasStorageRoom_ctg[hasStorageRoom == 1] <- "Yes"
  hasStorageRoom_ctg[hasStorageRoom == 0] <- "No"
})

# Bar-plots regarding some features
barplot(table(vars_ctg$isNewBuilt_ctg), col = "lightblue", 
        main = "Is the house newly built?")
barplot(table(vars_ctg$hasYard_ctg), col = "lightblue", 
        main = "Does the house have a yard?")
barplot(table(vars_ctg$hasPool_ctg), col = "lightblue", 
        main = "Does the house have a pool?")
barplot(table(vars_ctg$hasStorageRoom_ctg), col = "lightblue", 
        main = "Does the house have a storage room?")




# Correlation coefficient between features

cor(housing$price, housing)

library(ggcorrplot)

# Reset to default plot layout
par(mfrow = c(1, 1))


ggcorrplot(cor(housing), lab = TRUE, title = "Correlation matrix", 
           legend.title = "Correlation") + theme(plot.title = element_text(hjust = 0.5))

plot(housing$price, housing$squareMeters, 
     xlab = "price", ylab = "square meters", 
     main="Price per square meters")



# Normalizing dataset
min_max_norm <- function(x) {(x - min(x)) / (max(x) - min(x))}
housing <- as.data.frame(lapply(housing, min_max_norm))



library(caTools)

# Setting seed so that the sample can be reproduced
set.seed(123)

#80% Train and 20% Test
split = sample.split(housing$price, SplitRatio = .8)
housing_train = subset(housing, split == TRUE)
housing_test = subset(housing, split == FALSE)




###---------------------Random Forest--------------------###
###------------------Training the model------------------###


library(randomForest)

set.seed(123)
rf_model <- randomForest(price ~ ., data = housing_train)

rf_model

# Variable importance plot
importance(rf_model)
varImpPlot(rf_model, main = "Variable importance")


plot(rf_model, col = "blue", main = "Error and number of trees")

hist(treesize(rf_model),
     main = "Number of Nodes for the Trees",
     col = "lightblue")



# Finding optimal mtry
set.seed(123)
tuned_mtry <- tuneRF(housing_train[,-17], housing_train[,17], stepFactor = 0.5,
            trace = TRUE,
            plot = TRUE)

tuned_mtry


# Final model
set.seed(123)
rf_model <- randomForest(price ~ ., data = housing_train, 
                         ntree = 100, 
                         mtry = 10)

rf_model


###------------------Evaluating model------------------###


rf_pred <- predict(rf_model, housing_test)


plot(rf_pred, housing_test$price, main = "Actual and predicted values")


cor(rf_pred, housing_test$price)


# Mean Absolute Error
MAE(housing_test$price, rf_pred)


# Mean Squared Error
MSE(housing_test$price, rf_pred)


#  R Squared
RSQ(housing_test$price, rf_pred)


# Root Mean Squared Error
RMSE(housing_test$price, rf_pred)




###------------------------LASSO-------------------------###
###------------------Training the model------------------###


library(caret)

# k-fold cross validation framework
crossval <- trainControl(method = 'cv', number = 10, savePredictions = 'all')


#  Generating all potential lambda values
lambdas <- 10^seq(5, -5, length = 500)


set.seed(123)
# Creating model
lasso_model <- train(price ~ .,
                     data = housing_train,
                     preProcess = c("center"),
                     method = "glmnet",
                     tuneGrid = expand.grid(alpha = 1, lambda = lambdas),
                     trControl = crossval)


# Finding best lambda value
lasso_model$bestTune


# Finding regression model coefficients
coef(lasso_model$finalModel, lasso_model$bestTune$lambda)
plot(log(lasso_model$results$lambda), lasso_model$results$RMSE)


# Importance of each variable
varImp(lasso_model)


###------------------Evaluating model------------------###


# Model prediction
lasso_pred <- predict(lasso_model, newdata = housing_test)


plot(lasso_pred, housing_test$price, main = "Actual and predicted values")


# Mean Absolute Error
MAE(housing_test$price, lasso_pred)


# Mean Squared Error
MSE(housing_test$price, lasso_pred)


#  R Squared
RSQ(housing_test$price, lasso_pred)


# Root Mean Squared Error
RMSE(housing_test$price, lasso_pred)




###------------------------SVM------------------------###
###------------------Training the model------------------###
library(e1071)
library()

set.seed(123)
model = svm(price ~ ., data = housing)

##--------------------Linear training model------------------##
print(model)
model <- train(
  price ~ .,
  data = housing_train,
  method = 'svmRadial',
  preProcess = c("center", "scale"))
model

plot(model)


##CV
set.seed(123)
tr_ctr <- trainControl(
  method = "cv",
  number = 10,
)
##Tuning
set.seed(123)
tuneGrid <- expand.grid(
  C = c(0.25, .5, 1),
  sigma = 0.1
)
model_1 <- train(
  price ~ .,
  data = housing_test,
  method = 'svmRadial',
  preProcess = c("center", "scale"),
  trControl = tr_ctr,
  tuneGrid = tuneGrid
)

model_1
plot(model_1)
plot(predictions, housing_test$price, main = "Actual and predicted values")

###------------------Evaluating model------------------###
test_features = subset(housing_test, select=-c(price))
test_target = subset(housing_test, select=price)[,1]

predictions = predict(model_1, newdata = test_features)

# Mean Absolute Error
MAE(housing_test$price, predictions)


# Mean Squared Error
MSE(housing_test$price, predictions)


#  R Squared
RSQ(housing_test$price, predictions)


# Root Mean Squared Error
RMSE(housing_test$price, predictions)




###------------------Evaluating model------------------###

# Mean Absolute Error
MAE(housing_test$price, y_pred)


# Mean Squared Error
MSE(housing_test$price, y_pred)


#  R Squared
RSQ(housing_test$price, y_pred)


# Root Mean Squared Error
RMSE(housing_test$price, y_pred)




###-------------------Linear Regression-------------------###
head(housing)
nrow(housing)
ncol(housing)
colnames(housing)
summary(housing)
summary(housing$price)

#mean
meanprice <- mean(housing$price)
print( meanprice )

#median
medianP <- median(housing$price)
print( medianP )

#mode
getmode <- function(PriceMode){
  uniqd <- unique(PriceMode)
  uniqd[which.max(tabulate(match(PriceMode, uniqd)))]
}
hprice <- housing$price

result1 <- getmode(hprice)
print(result1)

#subsetting
squareMeters <- housing$squareMeters
Price <- housing$price

#NA value
is.na(squareMeters)
is.na(Price)

#linear regression model 
housing <- lm(Price ~ squareMeters )
summary(housing)
attributes(housing)
coef(housing)

#visualisation
plot(squareMeters, Price, col ="green", main="House price vs SquareMeters",
     abline(housing), xlab="SquareMeters", ylab="House Price")

#check model
x <- data.frame(squareMeters=50)
resulthp <- predict (housing, x)
print(resulthp)

#multiple
## 75% of the sample size
smp_size <- floor(0.75*nrow(housing))

##set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(housing)), size = smp_size)

train <-housing[train_ind, ]
test <- housing[-train_ind, ]
train
test

#MR
inputDataM <- housing_train[ , c("price", "squareMeters", "numberOfRooms", "hasPool")]
inputDataM

modelMR <- lm(price ~ squareMeters + numberOfRooms + hasPool, data=inputDataM )
modelMR

cat("### The COEFFICIENT VALUE ###", "/n")
a <- coef(modelMR)[1]
print(a)

xsquareMeters <- coef(modelMR)[2]
xnumberOfRooms <- coef(modelMR)[3]
xhasPool <- coef(modelMR)[4]

print(xsquareMeters)
print(xnumberOfRooms)
print(xhasPool)

# y=a + xsquareMeters.x1 + xnumberofRooms.x2 + xhasPool.x3
# price = 4963.720 + (100.000)*(60) + (1.516)*(1) + (2941.703)*(1)

# Check #

View(housing)
squareMeters=60
numberOfRooms=1
hasPool = 1
price = 4963.720 + (100.000)*(60) + (1.516)*(1) + (2941.703)*(1)
price







###------------------Evaluation functions------------------###

# Mean Absolute Error
MAE <- function(actual, predicted) {
  mean(abs(actual - predicted))
}


# Mean Squared Error
MSE <- function(actual, predicted) {
  mean((actual - predicted)^2)
}


#  R Squared
RSQ <- function(actual, predicted) {
  (cor(actual, predicted))^ 2
}


# Root Mean Squared Error
RMSE <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}
