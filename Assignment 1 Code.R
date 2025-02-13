SDG_2019 <- read.csv2("~/Desktop/Master/Introduction to Data Science/Assignment 1/SDG_2019.csv")
GEI_2019 <- read.csv2("~/Desktop/Master/Introduction to Data Science/Assignment 1/gei_2019.csv", sep = ",", dec = ".", header = TRUE)
DB_2019 <- read.csv2("~/Desktop/Master/Introduction to Data Science/Assignment 1/Team28_DB_2019.csv", sep = ",", dec = ".", header = TRUE)
final_dataset <- merge(SDG_2019, GEI_2019, by.x = "Country.Name",by.y  = "COUNTRY")
final_dataset <- merge(final_dataset, DB_2019, by.x = "Country.Name", by.y = "COUNTRY")
final_dataset <- subset(final_dataset, subset = !(is.na(DB_score)))

##1.1

final_dataset <- transform(final_dataset, Electricity =  EG.ELC.ACCS.RU.ZS
                           + EG.ELC.ACCS.UR.ZS)
## Does not run due to Electricity
##regression1 <- lm(DB_score ~ Product.Innovation + Process.Innovation + High.Growth + Risk.Capital + Human.capital + Competition
                 ## +Internationalization + Start.up.skills + Networking + Cultural.Support + Oppurtunity.startup
                 ## +Technology.Absorption + Opputunity.perception + Risk.Acceptance + GEI_score + EG.ELC.ACCS.RU.ZS + 
                  ##  EG.ELC.ACCS.UR.ZS + Electricity, data = final_dataset, singular.ok = FALSE)
##Model without electricity
regression2 <- lm(DB_score ~ Product.Innovation + Process.Innovation + High.Growth + Risk.Capital + Human.capital + Competition
                  +Internationalization + Start.up.skills + Networking + Cultural.Support + Oppurtunity.startup
                  +Technology.Absorption + Opputunity.perception + Risk.Acceptance + GEI_score + EG.ELC.ACCS.RU.ZS + 
                    EG.ELC.ACCS.UR.ZS, data = final_dataset, singular.ok = FALSE)
summary(regression2)
## Model with just Electricity
regression3 <- lm(DB_score ~ Product.Innovation + Process.Innovation + High.Growth + Risk.Capital + Human.capital + Competition
                  +Internationalization + Start.up.skills + Networking + Cultural.Support + Oppurtunity.startup
                  +Technology.Absorption + Opputunity.perception + Risk.Acceptance + GEI_score + Electricity
                   , data = final_dataset, singular.ok = FALSE)
summary(regression3)
## Electricity is perfect multicollinearity so either model with both variables or just electricity
comparison <- (cbind(summary(regression2), summary(regression3)))
comparison

##1.2
library(glmnet)
library(tidyverse)
final_dataset_imputed <- final_dataset %>%
  mutate(across(everything(), ~ifelse(is.na(.), mean(., na.rm = TRUE), .)))

x <- model.matrix(~ Product.Innovation + Process.Innovation + High.Growth + Risk.Capital + Human.capital + Competition
                  +Internationalization + Start.up.skills + Networking + Cultural.Support + Oppurtunity.startup
                  +Technology.Absorption + Opputunity.perception + Risk.Acceptance + GEI_score + Electricity - 1
                  , data = final_dataset_imputed)
y <- final_dataset$DB_score
lregression2 <- glmnet(x,y, alpha = 1, lambda = 10^seq(-2,6,length.out = 50))
lregression2cv <- cv.glmnet(x,y,alpha = 1, nfolds = 10)
lregression2lambda <- lregression2cv$lambda.1se
lregression2 <- glmnet(x,y, alpha = 1, lambda = lregression2lambda)
coef(lregression2)


##2.1
sample_size <- floor(0.8*nrow(final_dataset_imputed))
set.seed(777)
picked = sample(seq_len(nrow(final_dataset_imputed)),size=sample_size)
train.final_dataset_imputed = final_dataset_imputed[picked, ]
test.final_dataset_imputed = final_dataset_imputed[-picked,]

## LASSO on training set
x1 <- model.matrix(~ Product.Innovation + Process.Innovation + High.Growth + Risk.Capital + Human.capital + Competition
                   +Internationalization + Start.up.skills + Networking + Cultural.Support + Oppurtunity.startup
                   +Technology.Absorption + Opputunity.perception + Risk.Acceptance + GEI_score + Electricity - 1
                   , data = train.final_dataset_imputed)
y1 <- train.final_dataset_imputed$DB_score
lassoregressioncv <- cv.glmnet(x1,y1,alpha = 1, nfolds = 10)
lassoregressionlambda1se <- lassoregressioncv$lambda.1se
lassoregressionlambdamin <- lassoregressioncv$lambda.min
lassoregressioncvfinal_1 <- glmnet(x1,y1,alpha = 1, lambda = lassoregressionlambda1se)
lassoregressioncvfinal_2 <- glmnet(x1,y1,alpha = 1, lambda = lassoregressionlambdamin)
round(lassoregressioncvfinal_1$beta,2)
round(lassoregressioncvfinal_2$beta,2)

##Elastic Net on training set
elasticregressioncv <- cv.glmnet(x1,y1, alpha = 0.5, nfolds = 10)
elasticregressionlambda1se <- elasticregressioncv$lambda.1se
elasticregressionlambdamin <- elasticregressioncv$lambda.min
elasticregressioncvfinal_1 <- glmnet(x1,y1,alpha = 0.5, lambda = elasticregressionlambda1se)
elasticregressioncvfinal_2 <- glmnet(x1,y1,alpha = 0.5, lambda = elasticregressionlambdamin)
coef(elasticregressioncvfinal_1)
coef(elasticregressioncvfinal_2)

##2.3
##Plotting
##plot(lassoregression1)
##plot(elasticregression1)

##2.4
x_test <- model.matrix(DB_score ~ Product.Innovation + Process.Innovation + High.Growth + Risk.Capital + Human.capital + Competition
                   +Internationalization + Start.up.skills + Networking + Cultural.Support + Oppurtunity.startup
                   +Technology.Absorption + Opputunity.perception + Risk.Acceptance + GEI_score + Electricity - 1
                   , data = test.final_dataset_imputed)
##Best model
predict1_lasso_1se <- predict(lassoregressioncvfinal_1, x_test)
sqrt(mean((test.final_dataset_imputed$DB_score-predict1_lasso_1se)^2))

predict1_lasso_1min <- predict(lassoregressioncvfinal_2, x_test)
sqrt(mean((test.final_dataset_imputed$DB_score-predict1_lasso_1min)^2))

predict1_elastic_1se <- predict(elasticregressioncvfinal_1, x_test)
sqrt(mean((test.final_dataset_imputed$DB_score-predict1_elastic_1se)^2))


predict1_elastic_1min <- predict(elasticregressioncvfinal_2, x_test)
sqrt(mean((test.final_dataset_imputed$DB_score-predict1_elastic_1min)^2))

##2.5
round(coef(lassoregressioncvfinal_1), 2)

##3
prediction_sample <- transform(prediction_sample, Electricity = EG.ELC.ACCS.RU.ZS
                               + EG.ELC.ACCS.UR.ZS)
x_train <- model.matrix(~ Product.Innovation + Process.Innovation + High.Growth + Risk.Capital + Human.capital + Competition
                                  +Internationalization + Start.up.skills + Networking + Cultural.Support + Oppurtunity.startup
                                  +Technology.Absorption + Opputunity.perception + Risk.Acceptance + GEI_score + Electricity - 1
                                  , data = prediction_sample)

prediction <- predict(lassoregressioncvfinal_1, s = lassoregressionlambda1se, newx = x_train)
prediction

##4.1
final_dataset_imputed <- transform(final_dataset_imputed, HighDB = ifelse(DB_score >= 70,1,0))

##4.2
##Ridge on new training dataset
sample_size2 <- floor(0.8*nrow(final_dataset_imputed))
set.seed(777)
picked2 = sample(seq_len(nrow(final_dataset_imputed)),size=sample_size2)
train.final_dataset_imputed2 = final_dataset_imputed[picked2, ]
test.final_dataset_imputed2 = final_dataset_imputed[-picked2,]

x_ridge <- model.matrix(~ Product.Innovation + Process.Innovation + High.Growth + Risk.Capital + Human.capital + Competition
                        +Internationalization + Start.up.skills + Networking + Cultural.Support + Oppurtunity.startup
                        +Technology.Absorption + Opputunity.perception + Risk.Acceptance + GEI_score + Electricity - 1
                        , data = train.final_dataset_imputed2)
y_ridge <- train.final_dataset_imputed2$HighDB

ridgeregressioncv <- cv.glmnet(x_ridge,y_ridge, alpha = 0, nfolds = 10)
ridgeregressionlambda1se <- ridgeregressioncv$lambda.1se
ridgeregressionlambdamin <- ridgeregressioncv$lambda.min
ridgeregressionfinal1<- glmnet(x_ridge,y_ridge, alpha = 0, lambda = ridgeregressionlambda1se)
ridgeregressionfinal2 <- glmnet(x_ridge,y_ridge, alpha = 0, lambda = ridgeregressionlambdamin)
coef(ridgeregressionfinal1)
coef(ridgeregressionfinal2)

##4.3
x_train_2 <- model.matrix(~ Product.Innovation + Process.Innovation + High.Growth + Risk.Capital + Human.capital + Competition
                        +Internationalization + Start.up.skills + Networking + Cultural.Support + Oppurtunity.startup
                        +Technology.Absorption + Opputunity.perception + Risk.Acceptance + GEI_score + Electricity - 1
                        , data = prediction_sample)

prediction2 <- predict(ridgeregressionfinal1, s = ridgeregressionlambda1se, newx = x_train_2)
prediction3 <- predict(ridgeregressionfinal2, s = ridgeregressionlambdamin, newx = x_train_2)

prediction2
prediction3

##best model
predict_ridge_1se <- predict(ridgeregressionfinal1, x_ridge)
sqrt(mean((test.final_dataset_imputed2$HighDB-predict_ridge_1se)^2))

predict_ridge_min <- predict(ridgeregressionfinal2, x_ridge)
sqrt(mean((test.final_dataset_imputed2$HighDB-predict_ridge_min)^2))