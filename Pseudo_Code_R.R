

#### Pseudo Code for R


############## Library Loading

cat("\n### Packages Import...")
suppressPackageStartupMessages({
	# EDA & Visualization
	library(DataExplorer)
	library(ggplot2)

    # Data Manipulation
    library(forcats)
    library(tidyverse) 
    library(tidyr)
    library(data.table) 
    library(splitstackshape) 
    library(lubridate) 
    library(magrittr)
    library(purrr)
    library(reshape2)

    # ML 
    library(caret)
    library(randomforest)
    library(pROC)

    library(xgboost)
    library(lightgbm)
    library(Metrics)
})

set.seed(1234)

## Pre Set conditions
options(stringsAsFactors= FALSE)
options(digits= 4)

n.time <- Sys.time()
cat("\n### Data Loading...")

untibble <- function(tibble) {
  if("tbl" %in%  class(tibble)) {
    return(as.data.frame(unclass(tibble)))
  } else return(tibble)
}

#################### Data Loading
tr <- suppressMessages(untibble(read_csv("../input/train.csv")))
te <- suppressMessages(untibble(read_csv("../input/test.csv")))


##################### EDA Using DataExplorer

ncol(tr)
nrow(tr)

ncol(te)
nrow(te)


## Missing Value Plots
plot_missing(tr)

## Or
na_count_train <-data.frame(sapply(tr, function(y) sum(length(which(is.na(y))))))
colnames(na_count_train) = c("na_count")
na_count_train$na_percentage = as.integer(100*na_count_train$na_count/nrow(tr))


## Histogram for Continuous Variables
plot_histogram(tr)


### Target Variable Exploration

## If Continuous (Try Normal and Log Scale)
tr %>%
  ggplot(aes(target)) +
  geom_histogram(bins = 100, fill = "red") +
  #scale_x_log10() +
  labs(x = "Target") +
  ggtitle("Target feature distribution")

## If Categorical
prop.table(table(tr$target))

tr %>% 
  count(target) %>% 
  plot_ly(labels = ~target, values = ~n) %>%
  add_pie(hole = 0.3) %>%
  layout(title = "Target Distribution",  showlegend = T,
         xaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE),
         yaxis = list(showgrid = FALSE, zeroline = FALSE, showticklabels = FALSE))


######################## Data Manipulation

### Treat Nulls and Categorical Variables
  tr <- tr %>%
  			mutate_if(is.character, funs(factor(.))) %>% 
  			mutate_if(is.factor, fct_lump, prop = 0.025) %>%  # Lump Factors with Least values into Other
  			mutate_if(is.factor, fct_explicit_na) %>% # Add 
  			#mutate_if(is.character, funs(factor(.) %>% as.integer)) %>%
  			mutate_if(is.numeric, funs(ifelse(is.na(.), 0L, .))) 


### Group By/ Summarise/ Joins

### If Column1 is a categorical variable
fn <- funs(mean, sd, min, max, sum, n_distinct, .args = list(na.rm = TRUE))

df_stats <- . %>%
  #mutate_if(is.factor, funs(as.integer)) %>% 
  group_by(column1) %>% 
  summarise_all(fn) 

tr_stats <- df_stats(tr)
te_stats <- df_stats(te)

## Filter, Select and Arrange
tr_x = tr  %>% 
			filter(flag == 1) %>% 
			select(c("column2","column3"),grep("xyz",colnames(tr))) %>% arrange(-column4)

### Row Index 
tri <- 1:nrow(tr)
### Target Variable
y <- tr$target


### Combining Columns
tr <- tr %>% bind_cols(tr_stats)
te <- te %>% bind_cols(te_stats)


## Joins and Combining train and test sets
tr_te <- tr %>% 
  select(-target) %>% 
  bind_rows(te) %>%
  left_join(df, by = "id") 


###################### Modeling Section

## GLMNET

# Create CV fold indexes
# Create custom indices: myFolds
myFolds <- createFolds(y, k = 5)


# Create reusable trainControl object: myControl
myControl <- trainControl(method = "cv",
  summaryFunction = twoClassSummary,
  classProbs = TRUE, # IMPORTANT!
  verboseIter = TRUE,
  savePredictions = TRUE,
  index = myFolds
)


model_Tune_glmnet <- train(
 target ~ ., tr,
 metric = "ROC", # Accuracy  RMSE logLoss 
 method = "glmnet",
 tuneGrid = expand.grid(
 alpha = 0:1,
 lambda = 0:10/10
 ),
 trControl = myControl,
 preProcess = c("nzv","center", "scale","pca") 
)

fit_T_GLMNET = predict(model_Tune_glmnet,tr %>% select(-target))

#confusion matrix is a very useful tool for calibrating the output of a model and examining all possible outcomes of your predictions (true positive, true negative, false positive, false negative).
cf = confusionMatrix(fit_T_GLMNET, y,positive = 1)
print(cf)

test_GLMNET = predict(model_Tune_glmnet,te)



## Random Forest
model_rf <- train(target ~ ., tr,
                    method = "rf",
                    #metric = "ROC",
                    preProcess = c("scale", "center"),
                    tuneGrid = expand.grid(mtry = c(1,4,8,9)),tunelength=15,
                    trControl = trainControl(method="cv",number = 5, index = myFolds ,search="grid",verboseIter = TRUE, summaryFunction = twoClassSummary,
                                             classProbs = TRUE)
)


fit_T_rf = predict(model_rf,tr %>% select(-target))
cf = confusionMatrix(fit_T_rf, y,positive = 1)
print(cf)

test_rf = predict(model_rf,te,type = "prob")

##*************************************************** 
# Accuracy - It determines the overall predicted accuracy of the model. Accuracy  = (True Positives + True Negatives)/(True Positives + True Negatives + False Positives + False Negatives)

#True Positive Rate (TPR) - It indicates how many positive values, out of all the positive values, have been correctly predicted. TPR = TP/(TP + FN). Also Called Sensitivity Or Recall

#True Negative Rate (TNR) - It indicates how many negative values, out of all the negative values, have been correctly predicted. TNR =  TN/(TN + FP). It is also known as Specificity.

#False Negative Rate (FNR) - It indicates how many positive values, out of all the positive values, have been incorrectly predicted. FNR =  FN/(FN + TP).

#Precision: It indicates how many values, out of all the predicted positive values, are actually positive. formula:TP /(TP + FP).

#F Score: F score is the harmonic mean of precision and recall. It lies between 0 and 1. Higher the value, better the model. It is formulated as 2((precision*recall) / (precision+recall)). 
##*************************************************** 


### Other Metrics

# AIC stands for (Akaike’s Information Criteria). The basic idea of AIC is to penalize the inclusion of additional variables to a model. 
# It adds a penalty that increases the error when including additional terms. The lower the AIC, the better the model.

# BIC (or Bayesian information criteria) is a variant of AIC with a stronger penalty for including additional variables to the model. Lower BIC is better.

library(broom)
glance(model_rf) %>%
  dplyr::select(adj.r.squared, sigma, AIC, BIC, p.value)

## OR
summary(model_rf)
AIC(model_rf)
BIC(model_rf)


######################## Class Imbalance Treatment 

## Undersampling
ctrl <- trainControl(method="cv",number = 5, index = myFolds,
                     verboseIter = TRUE,
                     sampling = "down")

model_rf_under <- caret::train(target ~ .,
                         data = tr,
                         method = "rf",
                         preProcess = c("scale", "center"),
                         trControl = ctrl)


test_rf_under <- predict(model_rf_under, newdata = te, type = "prob")

## Oversampling
ctrl <- trainControl(method="cv",number = 5, index = myFolds,
                     verboseIter = TRUE,
                     sampling = "up")

model_rf_over <- caret::train(target ~ .,
                         data = tr,
                         method = "rf",
                         preProcess = c("scale", "center"),
                         trControl = ctrl)

## SMOTE
ctrl <- trainControl(method="cv",number = 5, index = myFolds,
                     verboseIter = TRUE,
                     sampling = "smote")

model_rf_smote <- caret::train(target ~ .,
                         data = tr,
                         method = "rf",
                         preProcess = c("scale", "center"),
                         trControl = ctrl)



##### Different Approach for Imbalance

thresh_code <- getModelInfo("rf", regex = FALSE)[[1]]
# rf model modified with following parameters
#         thresh_code$type
#         thresh_code$parameters
#         thresh_code$grid
#         thresh_code$loop
#         thresh_code$fit
#         thresh_code$predict
#         thresh_code$prob

                thresh_code$type <- c("Classification") #...1
                ## Add the threshold as another tuning parameter
                thresh_code$parameters <- data.frame(parameter = c("mtry", "threshold"),  #...2
                                                     class = c("numeric", "numeric"),
                                                     label = c("#Randomly Selected Predictors",
                                                               "Probability Cutoff"))
                ## The default tuning grid code:
                thresh_code$grid <- function(x, y, len = NULL, search = "grid") {  #...3
                    p <- ncol(x)
                    if(search == "grid") {
                        grid <- expand.grid(mtry = floor(sqrt(p)),
                                            threshold = seq(.01, .99, length = len))
                    } else {
                        grid <- expand.grid(mtry = sample(1:p, size = len),
                                            threshold = runif(1, 0, size = len))
                    }
                    grid
                }
                
                ## Here we fit a single random forest model (with a fixed mtry)
                ## and loop over the threshold values to get predictions from the same
                ## randomForest model.
                thresh_code$loop = function(grid) {    #...4
                    library(plyr)
                    loop <- ddply(grid, c("mtry"),
                                  function(x) c(threshold = max(x$threshold)))
                    submodels <- vector(mode = "list", length = nrow(loop))
                    for(i in seq(along = loop$threshold)) {
                        index <- which(grid$mtry == loop$mtry[i])
                        cuts <- grid[index, "threshold"]
                        submodels[[i]] <- data.frame(threshold = cuts[cuts != loop$threshold[i]])
                    }
                    list(loop = loop, submodels = submodels)
                }
                
                ## Fit the model independent of the threshold parameter
                thresh_code$fit = function(x, y, wts, param, lev, last, classProbs, ...) {  #...5
                    if(length(levels(y)) != 2)
                        stop("This works only for 2-class problems")
                    randomForest(x, y, mtry = param$mtry, ...)
                }
                
                ## Now get a probability prediction and use different thresholds to
                ## get the predicted class
                thresh_code$predict = function(modelFit, newdata, submodels = NULL) {    #...6
                    class1Prob <- predict(modelFit,
                                          newdata,
                                          type = "prob")[, modelFit$obsLevels[1]]
                    ## Raise the threshold for class #1 and a higher level of
                    ## evidence is needed to call it class 1 so it should 
                    ## decrease sensitivity and increase specificity
                    out <- ifelse(class1Prob >= modelFit$tuneValue$threshold,
                                  modelFit$obsLevels[1],
                                  modelFit$obsLevels[2])
                    if(!is.null(submodels)) {
                        tmp2 <- out
                        out <- vector(mode = "list", length = length(submodels$threshold))
                        out[[1]] <- tmp2
                        for(i in seq(along = submodels$threshold)) {
                            out[[i+1]] <- ifelse(class1Prob >= submodels$threshold[[i]],
                                                 modelFit$obsLevels[1],
                                                 modelFit$obsLevels[2])
                        }
                    }
                    out
                }                  
                ## The probabilities are always the same but we have to create
                ## mulitple versions of the probs to evaluate the data across
                ## thresholds
                thresh_code$prob = function(modelFit, newdata, submodels = NULL) {   #...7
                    out <- as.data.frame(predict(modelFit, newdata, type = "prob"))
                    if(!is.null(submodels)) {
                        probs <- out
                        out <- vector(mode = "list", length = length(submodels$threshold)+1)
                        out <- lapply(out, function(x) probs)
                    }
                    out
                }
                
                ### for summaryFunction in trControl 
                fourStats <- function (data, lev = levels(data$obs), model = NULL) {
                    ## This code will get use the area under the ROC curve and the
                    ## sensitivity and specificity values using the current candidate
                    ## value of the probability threshold.
                    out <- c(twoClassSummary(data, lev = levels(data$obs), model = NULL))
                    
                    ## The best possible model has sensitivity of 1 and specificity of 1. 
                    ## How far are we from that value?
                    coords <- matrix(c(1, 1, out["Spec"], out["Sens"]),
                                     ncol = 2,
                                     byrow = TRUE)
                    colnames(coords) <- c("Spec", "Sens")
                    rownames(coords) <- c("Best", "Current")
                    c(out, Dist = dist(coords)[1])
                }
################## Modeling with customized RF model ###################

mod1 <- train(Class ~ ., data = trainingSet,
              method = thresh_code,  # modified model -- in lieu of "rf"
              ## Minimize the distance to the perfect model
              metric = "Dist",
              maximize = FALSE,
              tuneLength = 10,   #  20,
              ntree = 200,          # 1000,
              trControl = trainControl(method = "cv",
                                       classProbs = TRUE,
                                       summaryFunction = fourStats))

metrics <- mod1$results[, c(2, 4:6)]
metrics <- melt(metrics, id.vars = "threshold",
                variable.name = "Resampled",
                value.name = "Data")

ggplot(metrics, aes(x = threshold, y = Data, color = Resampled)) +
    geom_line() +
    ylab("") + xlab("Probability Cutoff") +
    theme(legend.position = "top")

## Compare models

models <- list(original = model_rf,
                       under = model_rf_under,
                       over = model_rf_over,
                       smote = model_rf_smote,
                       glmnet = model_Tune_glmnet)


comparison <- data.frame(model = names(models),
                         Sensitivity = rep(NA, length(models)),
                         Specificity = rep(NA, length(models)),
                         Precision = rep(NA, length(models)),
                         Recall = rep(NA, length(models)),
                         F1 = rep(NA, length(models)))

for (name in names(models)) {
  model <- get(paste0("cm_", name))
  
  comparison[comparison$model == name, ] <- filter(comparison, model == name) %>%
    mutate(Sensitivity = model$byClass["Sensitivity"],
           Specificity = model$byClass["Specificity"],
           Precision = model$byClass["Precision"],
           Recall = model$byClass["Recall"],
           F1 = model$byClass["F1"])
}



comparison %>%
  gather(x, y, Sensitivity:F1) %>%
  ggplot(aes(x = x, y = y, color = model)) +
    geom_jitter(width = 0.2, alpha = 0.5, size = 3)

