library(ggplot2)
library(ggthemes)
library(tidyverse)
library(caret)
library(Metrics)
library(mice)

set.seed(101)
data <- read.csv('data.csv', stringsAsFactors = F, strip.white = T, na.strings = "")

# DATA FORMATTING
data$Total.Raised <- data$Total.Raised - data$Last.Financing.Size # to get rid of cheque size in this variable
data <- as.data.frame(lapply(data, factor))
data$Employees <- as.integer(levels(data$Employees))[data$Employees]
data$Last.Financing.Size <- as.numeric(levels(data$Last.Financing.Size))[data$Last.Financing.Size]
data$Last.Financing.Valuation <- as.numeric(levels(data$Last.Financing.Valuation))[data$Last.Financing.Valuation]
data$Total.Raised <- as.numeric(levels(data$Total.Raised))[data$Total.Raised]

data$HQ.Location <- str_sub(data$HQ.Location, start = -2, end = -1) # extract states from locations
data$HQ.Location[data$State == "DC"] <- "MD" # combine DC and MD
data$HQ.Location <- factor(data$HQ.Location)

# IMPUTATION WITH MICE
sub.data <- data %>%
  select(c("Employees", "Total.Raised", "Last.Financing.Size", "Last.Financing.Valuation", "Primary.Industry.Sector", 
           "Primary.Industry.Group", "Last.Financing.Deal.Type",
           "Last.Financing.Deal.Type.2", "X..Active.Investors"))
mice.output <- mice(data = sub.data, nnet.MaxNWts = 2000, m = 3)
sub.data <- complete(mice.output)

data <- dplyr::select(data, -c(1, 4, 5, 6, 9, 15, 16))
data$Last.Financing.Deal.Type.2 <- sub.data$Last.Financing.Deal.Type.2
data$Last.Financing.Valuation <- sub.data$Last.Financing.Valuation
data$Total.Raised <- sub.data$Total.Raised
data$Last.Financing.Size <- sub.data$Last.Financing.Size

# PREPROCESSING FOR CLASSIFICATION
classes <- c()
for(i in 1 : length(sub.data$Last.Financing.Size)) {
  dummy <- sub.data$Last.Financing.Size[i]
  if(dummy <= 5)       { classes[i] <- 1 }
  else if(dummy <= 25) { classes[i] <- 2 }
  else if(dummy <= 45) { classes[i] <- 3 }
  else                 { classes[i] <- 4 }
}

data$Classes <- classes

# MODEL BUILDING
train.sample <- sample(nrow(data), 700)
data <- data[, !(colnames(data) %in% c("ID", "Last.Financing.Valuation", "Total.Raised", "Last.Financing.Size"))]
data$Classes <- factor(data$Classes)
train <- data[train.sample, ]
total.test <- data[-train.sample, ]
test <- total.test[, !(colnames(data) %in% c("ID", "Classes"))]
target.test <- total.test[, "Classes"]

# RANDOM FOREST MODEL
rf.model <- caret::train(Classes ~ ., train, "rf")
rf.data <- predict(cart.model, test)

# LINEAR XGBoost
linxgb.model <- caret::train(Classes ~ ., train, "xgbLinear")
linxgb.data <- predict(linxgb.model, test)

# TREE XGBOOST
treexgb.model <- caret::train(Classes ~ ., train, "xgbTree")
treexgb.data <- predict(treexgb.model, test)

# EXTREME LEARNING MACHINE
elm.model <- caret::train(Classes ~ ., train, "elm")
elm.data <- predict(elm.model, test)

# CONDITIONAL INFERENCE TREE
ctree.model <- caret::train(Classes ~ ., train, "ctree")
ctree.data <- predict(deep.model, test)

# BOOSTED CART
treebag.model <- caret::train(Classes ~ ., train, "treebag")
treebag.data <- predict(mlp.model, test)

# NEURAL NET
nnet.model <- caret::train(Classes ~ ., train, "nnet")
nnet.data <- predict(nnet.model, test)

# K-NEAREST NEIGHBORS
knn.model <- caret::train(Classes ~ ., train, "knn")
knn.data <- predict(knn.model, test)

# MODEL VALIDATION
q <- list(rf.data, linxgb.data, treexgb.data, elm.data, treebag.data, ctree.data, nnet.data, knn.data)
accuracies <- unlist(lapply(q, Metrics::accuracy, actual = target.test))
scaled.acc <- accuracies / sum(accuracies)
predictions.df <- data.frame(rf   = q[[1]], lxgb  = q[[2]],    txgb = q[[3]], 
                             elm  = q[[4]], ctree = q[[5]], treebag = q[[6]], 
                             nnet = q[[7]],   knn = q[[8]], 
                             target = target.test)
class.prob <- with(predictions.df, 
                  scaled.acc[1] * as.numeric(rf)    + scaled.acc[2] * as.numeric(lxgb)    +
                  scaled.acc[3] * as.numeric(txgb)  + scaled.acc[4] * as.numeric(elm)     +
                  scaled.acc[5] * as.numeric(ctree) + scaled.acc[6] * as.numeric(treebag) +
                  scaled.acc[7] * as.numeric(nnet)  + scaled.acc[8] * as.numeric(knn))
final.predictions <- round(class.prob)
write.csv(final.predictions, file = "results.csv", row.names = F)