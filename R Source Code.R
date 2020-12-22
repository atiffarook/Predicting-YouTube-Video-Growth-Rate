## Load the libraries
library(tidyverse)
library(tidyr)
library(ggforce)
library(ggcorrplot)
library(class)
library(MASS)
library(reshape2)
library(grid)
library(gridExtra)
library(caret)
library(mclust)
library(boot)
library(MLeval)
library(ISLR) 
library(glmnet)
library(gbm)
library(pls)
library(randomForest)
library(tree)
library(e1071)
library(Metrics)
library(VSURF)
library(Boruta)
library(ggraph)
library(igraph)
library(rpart.plot)
library(knitr)

## Loading and cleaning the data 
## Load the dataset 
data_main <- read.csv("training.csv")

## Check if there are any NA's 
any(is.na(data_main)  == TRUE)
data_nop <- data_main[,-c(1,2)]

## Check the percentage of 0's/NULL Values in each column 
res <- colSums(data_nop==0)/nrow(data_nop)*100

## Names of every column 
names <- names(data_nop)

## Variables to remove 
var_remove <- c(c(names[which(res > 99)]), c("max_blue", "max_red", "max_green"))

## Variables to keep
vars_to_keep = names[!(names %in% var_remove)]
vars_to_keep <- c(vars_to_keep[-226], "growth_2_6")
data <- data_nop[,vars_to_keep]


# Removing varaiables that are highly correlated 

df2 = cor(data_sd)
hc = findCorrelation(df2, cutoff=0.85) # putt any value as a "cutoff"
hc = sort(hc)
names(data_sd[,hc])

## Creating categorical variables for Num_Subscribers, Num_Views, avg_growth & count_vids

categories <- c("Num_Subscribers_Base_low", "Num_Subscribers_Base_low_mid", 
                "Num_Views_Base_mid_high", "Num_Views_Base_low", 
                "Num_Views_Base_low_mid", "Num_Views_Base_mid_high", 
                "avg_growth_low", "avg_growth_low_mid", "avg_growth_mid_high", 
                "count_vids_low", "count_vids_low_mid", "count_vids_mid_high") 

Num_Subscribers <-  rep(0, nrow(data))
Num_Views <- rep(0, nrow(data))
avg_growth <- rep(0, nrow(data))
count_vids <-  rep(0, nrow(data))

##Num_Subscribers
i <- 1 
index1 <- which(data[,categories[i]] == 1)
index2 <- which(data[,categories[i+1]] == 1)
index3 <- which(data[,categories[i+2]] == 1)

Num_Subscribers[index1] <- 1
Num_Subscribers[index2] <- 2
Num_Subscribers[index3] <- 3
#table(Num_Subscribers)

## Num_Views
i <- 4
index1 <- which(data[,categories[i]] == 1)
index2 <- which(data[,categories[i+1]] == 1)
index3 <- which(data[,categories[i+2]] == 1)

Num_Views[index1] <- 1
Num_Views[index2] <- 2
Num_Views[index3] <- 3
#table(Num_Views)

## avg_growth
i <- 7
index1 <- which(data[,categories[i]] == 1)
index2 <- which(data[,categories[i+1]] == 1)
index3 <- which(data[,categories[i+2]] == 1)

avg_growth[index1] <- 1
avg_growth[index2] <- 2
avg_growth[index3] <- 3
#table(avg_growth)

##count_vids
i <- 10
index1 <- which(data[,categories[i]] == 1)
index2 <- which(data[,categories[i+1]] == 1)
index3 <- which(data[,categories[i+2]] == 1)

count_vids[index1] <- 1
count_vids[index2] <- 2
count_vids[index3] <- 3
#table(count_vids)

add_categories <- cbind(Num_Subscribers, Num_Views, avg_growth, count_vids)
n <- ncol(data)

## Data set after preprocessing
data <- cbind(data[,1:(n-1)],add_categories, "growth_2_6" = data$growth_2_6)

## Boruta Algorithm 

# Running the algorithm 

# boruta <- Boruta(growth_2_6~., data = data, doTrace = 2)
# plot(boruta)
# save(boruta, file = "boruta_og.R")

# Load the file 
load("boruta_og.R")

## Predictors that matter
getSelectedAttributes(boruta, withTentative = F)

## Selecting the predictors and ordering them 
importance_df <- attStats(boruta)
ord <- order(importance_df$meanImp, decreasing = TRUE)
importance_df[ord,]

pred <- rownames(importance_df[ord,])

# choosing the imp predictors after the second "drop" in graph: n = 47
n_pred <- 47
pred1 <- pred[1:n_pred]
pred1 <- c(pred1, "growth_2_6")

# Final data set after feature selection 
data<- data[, pred1]

## Plot the graph 
plot(boruta, xlab = "", xaxt = "n", outline=FALSE,  
     whisklty = 0, staplelty = 0, ylim = c(0,28.5), xlim = c(70,229))
lz<-lapply(1:ncol(boruta$ImpHistory),function(i)
  boruta$ImpHistory[is.finite(boruta$ImpHistory[,i]),i])
names(lz) <- colnames(boruta$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
     at = 1:ncol(boruta$ImpHistory), cex.axis = 0.7)
abline(h = 22, col = "red", lty = 2, lwd = 0.5)
abline(h = 17.5, col = "red", lty = 2, lwd = 0.5 )
abline(h = 11.5, col = "red", lty = 2, lwd = 0.5 )
abline(h = 6.6, col = "red", lty = 2, lwd = 0.5 )
abline(h = 3.02, col = "red", lty = 2, lwd = 0.5 )

text(72, 4.02, "n = 95", cex = 0.8 )
text(72, 8, "n = 47", cex = 0.8 )
text(72, 12.9, "n = 33", cex = 0.8 )
text(72, 18.9, "n = 08", cex = 0.8 )
text(72, 23.4, "n = 04", cex = 0.8 )

#70% of data for train and 30% of data for test
train_size = floor(0.7 * nrow(data))

#set the seed
set.seed(1234)

#get training indices
train_ind = sample(seq_len(nrow(data)), size = train_size)

data_train = data[train_ind, ]
data_test = data[-train_ind, ]

X_train = model.matrix(growth_2_6~., data_train)[,-1]
y_train = data_train$growth_2_6

X_test = model.matrix(growth_2_6~., data_test)[,-1]
y_test = data_test$growth_2_6

mtry <- floor((ncol(data_train) - 1)/3)

## Random Forest - Cross Validation + best values for mtry and ntree 

## Might take a while to run ~ 5 minutes 
# Fit classification tree using the 'randomForest' library.
set.seed(123)

# Use the out-of-bag estimator to select the optimal parameter values.
oob_train_control <- trainControl(method="oob", 
                                  classProbs = TRUE, 
                                  savePredictions = TRUE)

# We find the best value for m using cross validation
rf_default <- train(growth_2_6~. , 
                    data = data_train, method = 'rf',
                    trControl = oob_train_control) 

rf_default

# Adjust Settings of Random Forest (If necessary)

## Might take a while to run  ~ 10 minutes+ 
# Grid search many different RF settings
mtry_vals = c(10, 16, 20, 25, 30, 47)
nodesize_vals = c(2, 5, 10, 20)

results = matrix(rep(0, length(mtry_vals)*length(nodesize_vals)),
                 length(mtry_vals),length(nodesize_vals))
rownames(results) = apply(as.matrix(mtry_vals), 2, 
                          function(t){return(paste("m =",t))})
colnames(results) = apply(as.matrix(nodesize_vals), 2, 
                          function(t){return(paste("n =",t))})

for (i1 in 1:length(mtry_vals))
{
  m = mtry_vals[i1]
  for (i2 in 1:length(nodesize_vals))
  {
    n = nodesize_vals[i2]
    rf_model = randomForest(growth_2_6~., data = data_train, 
                            ntree=400, mtry=m, nodesize=n, importance=TRUE)
    rf_preds = predict(rf_model, data_test)
    
    D = as.data.frame(cbind(data_test$growth_2_6, rf_preds))
    colnames(D) = c("true_growth_2_6", "rf_preds")
    
    test_set_r_sq = rmse(y_test, rf_preds)
    
    results[i1,i2] = test_set_r_sq
    
  }
}

recommended.mtry <- mtry ## uses (predictors/3)

## can explore mtry = 24 as that was the best m from CV 
tunegrid <- expand.grid(mtry=recommended.mtry)
set.seed(123)
oob_train_control <- trainControl(method="oob", 
                                  classProbs = TRUE, 
                                  savePredictions = TRUE)

forestfit.m <- train(growth_2_6~. , 
                     data = data_train, method = 'rf',
                     trControl = oob_train_control, tuneGrid = tunegrid,
                     ntree = 500) 
print(forestfit.m, digits = 2)

#predict with RF after CV 
rf_preds_cv = predict(forestfit.m, data_test)

test_set_r_sq = rmse(y_test, rf_preds_cv)

tree.diagram <- tree::tree(growth_2_6~. , data_train)
plot(tree.diagram) 
text(tree.diagram)


test <- read.csv("test.csv")
sample <- read.csv("sample.csv")
data <- test

categories <- c("Num_Subscribers_Base_low", "Num_Subscribers_Base_low_mid", 
                "Num_Views_Base_mid_high", "Num_Views_Base_low", 
                "Num_Views_Base_low_mid", "Num_Views_Base_mid_high", 
                "avg_growth_low", "avg_growth_low_mid", "avg_growth_mid_high", 
                "count_vids_low", "count_vids_low_mid", "count_vids_mid_high") 

Num_Subscribers <-  rep(0, nrow(data))
Num_Views <- rep(0, nrow(data))
avg_growth <- rep(0, nrow(data))
count_vids <-  rep(0, nrow(data))

##Num_Subscribers
i <- 1 
index1 <- which(data[,categories[i]] == 1)
index2 <- which(data[,categories[i+1]] == 1)
index3 <- which(data[,categories[i+2]] == 1)

Num_Subscribers[index1] <- 1
Num_Subscribers[index2] <- 2
Num_Subscribers[index3] <- 3
table(Num_Subscribers)

## Num_Views
i <- 4
index1 <- which(data[,categories[i]] == 1)
index2 <- which(data[,categories[i+1]] == 1)
index3 <- which(data[,categories[i+2]] == 1)

Num_Views[index1] <- 1
Num_Views[index2] <- 2
Num_Views[index3] <- 3
table(Num_Views)

## avg_growth
i <- 7
index1 <- which(data[,categories[i]] == 1)
index2 <- which(data[,categories[i+1]] == 1)
index3 <- which(data[,categories[i+2]] == 1)

avg_growth[index1] <- 1
avg_growth[index2] <- 2
avg_growth[index3] <- 3
table(avg_growth)

##count_vids
i <- 10
index1 <- which(data[,categories[i]] == 1)
index2 <- which(data[,categories[i+1]] == 1)
index3 <- which(data[,categories[i+2]] == 1)

count_vids[index1] <- 1
count_vids[index2] <- 2
count_vids[index3] <- 3
table(count_vids)
add_categories <- cbind(Num_Subscribers, Num_Views, avg_growth, count_vids)

test <- cbind(test, add_categories)
test <- test[,pred1[1:n_pred]]

## Used the CV Rf model 
sample.prob <- predict(forestfit.m, test)

sample$growth_2_6 <- sample.prob

## FINAL DOCUMENT 
write.csv(sample,"Final draft", row.names = FALSE)
head(sample)
