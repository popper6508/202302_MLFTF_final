# ==================================================================
# Load Library
library(devtools)  
library(HDeconometrics)
library(randomForest)
library(xgboost)
library(writexl)
library(readxl)
library(dplyr)
library(stringr)
library(Boruta)

# ==================================================================
# Load Data
#setwd("C:/User/Minho/Dropbox/R/ML/Homework/final")
dir()
data <- read_xlsx("data.xlsx", sheet = 3)
#===================================================================
# Data Transformation
tcode = data[1,-1] # first element: Transformation code
tcode
#as.numeric(tcode)
data <- data[-1,]
tdata <- data[-(1:2),] # 차분을 위해 앞에 2개 제거거
head(tdata)
ncol(data)

for (i in 2:ncol(data)){
  
  if(tcode[i-1] == 1){
    tdata[,i] <- data[-(1:2),i]
  } # no transformation  
  
  if(tcode[i-1] == 2){
    tdata[,i] <- diff(as.matrix(data[-1,i]))
  } # 1st diff
  
  #  if(tcode[i-1] == 3){
  #    tdata[,i] <- diff(diff(data[-1,i]))
  #  } # 2nd diff
  
  if(tcode[i-1] == 4){
    tdata[,i] <- log(data[-(1:2),i])
  } # log
  
  if(tcode[i-1] == 5){
    tdata[,i] <- diff(log(as.matrix(data[-1,i])))
  } # log differencing
  
  #  if(tcode[i-1] == 6){
  #    tdata[,i] <- diff(diff(log(data[,i])))
  #  } # 2 times log diff
  
  #  if(tcode[i-1] == 7){
  #    tdata[,i] <- diff(data[-1,i]/data[1:(nrow(data)-1),i])
  #  }  # 
}

tdata <- as.data.frame(tdata)
#write_xlsx(tdata, "Transformed_20231127.xlsx")

Y <- tdata[,-1] %>% 
  as.matrix()
mode(Y)
npred = nrow(tdata) - 416 # 8년 Window
lag = c(1,2,4,8)

# Stack용 변수들의 NULL 구성
rw_pred <- NULL
ar_pred <- NULL
lasso_pred <- NULL
adalasso_pred <- NULL
elasticnet_pred <- NULL
adaelasticnet_pred <- NULL
tfact_pred <- NULL
rf_pred <- NULL
xgb_pred <- NULL
har_pred <- NULL ## HAR 추가필요
harx_pred <- NULL
RMSE <- matrix(,11,1)
RMAE <- matrix(,11,1)


# For Har 
Yharx <- read_xlsx("VKOSPI_HARX.xlsx")
Yharx <- Yharx[,-1]
#npred = nrow(Yharx) - 416 # 8년 Window
Yharx <- na.omit(Yharx) %>% 
  as.matrix()
head(Yharx)
Yhar <- Yharx[,1:4]


for (i in lag){
  
  # ==================================================================
  # Heterogenous Auto Regressive
  
  source("functions/func-har.R")
 
  harx = har.rolling.window(Yharx, npred, 1, i, type = "fixed")
  harx_pred = cbind(harx_pred, harx$pred)
  
  har = har.rolling.window(Yhar, npred, 1, i, type = "fixed")
  har_pred = cbind(har_pred, har$pred)
  
  
  # ==================================================================
  # Random Walk
  
  source("functions/func-rw.R")
  
  rw1 = rw.rolling.window(Y, npred, 1, i)
  rw_pred = cbind(rw_pred, rw1$pred)
  
  # ==================================================================
  # AR Model
  source("functions/func-ar.R")
  
  ar1 = ar.rolling.window(Y,npred,1,i,type="fixed")
  ar_pred = cbind(ar_pred, ar1$pred)
  
  # ==================================================================
  # LASSO
  source("functions/func-lasso.R")
  alpha = 1
  
  lasso1 = lasso.rolling.window(Y,npred,1,i,alpha,type="lasso")
  
  lasso_pred = cbind(lasso_pred, lasso1$pred)
  
  # ==================================================================
  # Adaptive LASSO
  
  adalasso1=lasso.rolling.window(Y,npred,1,i,alpha,type="adalasso")
  adalasso_pred = cbind(adalasso_pred, adalasso1$pred)
  
  # ==================================================================
  # Elasticnet
  alpha=0.5
  
  elasticnet1=lasso.rolling.window(Y,npred,1,i,alpha,type="lasso")
  elasticnet_pred = cbind(elasticnet_pred, elasticnet1$pred)
  
  # ==================================================================
  # Adaptive Elasticnet
  
  adaelasticnet1 = lasso.rolling.window(Y,npred,1,i,alpha,type="adalasso")
  adaelasticnet_pred = cbind(adaelasticnet_pred, adaelasticnet1$pred)
  
  # ==================================================================
  # Target Factor
  
  source("functions/func-fact.R")
  source("functions/func-tfact.R")
  source("functions/func-baggit.R")
  
  tfact1 = tfact.rolling.window(Y,npred,1,i)
  
  tfact_pred = cbind(tfact_pred, tfact1$pred)
  
  # ==================================================================
  # Random Forest                 
  source("functions/func-rf.R")
  
  rf1 = rf.rolling.window(Y,npred,1,i)
  
  rf_pred = cbind(rf_pred, rf1$pred)
  
  # ==================================================================
  # XGBoost
  source('functions/func-xgb.R')
  
  xgb1 = xgb.rolling.window(Y,npred,1,i)
  
  xgb_pred = cbind(xgb_pred, xgb1$pred)
  
  
  # ==================================================================
  # MSE For each model
  harmse = cbind(har$errors[1])
  harxmse = cbind(harx$errors[1])
  rwmse = cbind(rw1$errors[1])
  armse = cbind(ar1$errors[1])
  lassomse = cbind(lasso1$errors[1])
  adalassomse = cbind(adalasso1$errors[1])
  elasticnetmse = cbind(elasticnet1$errors[1])
  adaelasticnetmse = cbind(adaelasticnet1$errors[1])
  tfactmse = cbind(tfact1$errors[1])
  rfmse = cbind(rf1$errors[1])
  xgbmse = cbind(xgb1$errors[1])
  
  # ==================================================================
  # MAE For each model
  harmae = cbind(har$errors[2])
  harxmae = cbind(harx$errors[2])
  rwmae = cbind(rw1$errors[2])
  armae = cbind(ar1$errors[2])
  lassomae = cbind(lasso1$errors[2])
  adalassomae = cbind(adalasso1$errors[2])
  elasticnetmae = cbind(elasticnet1$errors[2])
  adaelasticnetmae = cbind(adaelasticnet1$errors[2])
  tfactmae = cbind(tfact1$errors[2])
  rfmae = cbind(rf1$errors[2])
  xgbmae = cbind(xgb1$errors[2])
  
  # ==================================================================
  # Error Combine
  MSE = rbind(harmse,
              harxmse,
               rwmse, 
               armse, 
               lassomse, 
               adalassomse, 
               elasticnetmse, 
               adaelasticnetmse, 
               tfactmse, 
               rfmse,
               xgbmse)
  
  MAE = rbind(harmae,
              harxmae,
              rwmae,  
              armae, 
              lassomae, 
              adalassomae, 
              elasticnetmae, 
              adaelasticnetmae, 
              tfactmae, 
              rfmae,
              xgbmae)
  
  MSE <- as.data.frame(MSE)
  RMSE <- cbind(RMSE, MSE)
  MAE <- as.data.frame(MAE)
  RMAE <- cbind(RMAE, MAE)
  
  save.image("results_forecasts_weekly231127.RData")

}

VAR <- c("HAR", "HARX", "Random Walk", "AR", "LASSO", "ADALASSO", "E.NET", "ADA.E.NET", "TFACT", "RF", "XGB")
ErrorMatrix <- cbind(VAR, RMSE[,-1], RMAE[,-1])
names(ErrorMatrix) <- c("MODEL", "MSE1W", "MSE2W", "MSE4W", "MSE8W", "MAE1W", "MAE2W", "MAE4W", "MAE8W")

#output <- str_c("ERRORS_", Sys.Date(), ".xlsx")
#write_xlsx(ErrorMatrix, output)


#============================================================
# Using Boruta Algorithm

Y2 = Y                  ## Using the Whole Sample 
Boruta_XGB_MSE <- NULL
Boruta_XGB_MAE <- NULL
xgb_selected_pred <- NULL

Boruta_RF_MSE <- NULL
Boruta_RF_MAE <- NULL
rf_selected_pred <- NULL
VO <- matrix(,208,1) # Varialbe Order Memory
ST <- matrix(,1,1)  # Selected Var Memory

for (lag in c(1,2,4,8)){
  aux = embed(Y2,4+lag)
  y=aux[,1]
  X=aux[,-c(1:(ncol(Y2)*lag))]
  
  boruta <- Boruta(X, y, maxRuns = 100)
  
  plot = plot(boruta)
  plot
  
  attstats = attStats(boruta)
  attstats
  
  order = order(attstats$meanImp, decreasing = T) # Mean Importance, Decreasing Ordering
  
  order
  ## Cross Validation for Optimal Number of Variables # (Up to 70 Variables)
  
  Errors = rep(NA,68)
  
  for (i in 2:68){
    
    selected = order[1:i]
    
    model=randomForest(X[,selected], y, importance=TRUE)
    
    pred = model$predicted     
    error = mean((pred-y)^2)
    
    Errors[i] <- error
  }
  
  plot(c(1:68), Errors, xlab="# of Variables", ylab="Fitted Squared Error")
  
  Errors1 = Errors
  
  # Rolling Window with Selected Variables
  
  varOrder = order(attstats$meanImp, decreasing = T)    # Ordering of Variables
  which.min(Errors1)                                    # Optimal Number of Variables 
  selected = varOrder[1:which.min(Errors1)]             # The Set of Optimal Number of Variables
  VO <- cbind(VO, as.data.frame(varOrder))
  ST <- cbind(ST, as.data.frame(which.min(Errors1)))
  
  source("functions/func-xgb_selected.R")
  
  xgb_selected = xgb.rolling.window(Y2,npred,1,lag,selected)
  Boruta_XGB_MSE <- cbind(Boruta_XGB_MSE, xgb_selected$error[1])
  Boruta_XGB_MAE <- cbind(Boruta_XGB_MAE, xgb_selected$error[2])
  xgb_selected_pred <- cbind(xgb_selected_pred, xgb_selected$pred)
  
  source("functions/func-rf_selected2022.R")
  
  rf_selected = rf.rolling.window(Y2,npred,1,lag,selected)
  Boruta_RF_MSE <- cbind(Boruta_RF_MSE, rf_selected$error[1])
  Boruta_RF_MAE <- cbind(Boruta_RF_MAE, rf_selected$error[2])
  rf_selected_pred <- cbind(rf_selected_pred, rf_selected$pred)
}

Boruta_XGB_MSE <- as.data.frame(Boruta_XGB_MSE)
Boruta_XGB_MAE <- as.data.frame(Boruta_XGB_MAE)
Boruta_XGB_ERROR <- cbind(c("B-XGB"), Boruta_XGB_MSE, Boruta_XGB_MAE)
names(Boruta_XGB_ERROR) <- c("MODEL", "MSE1W", "MSE2W", "MSE4W", "MSE8W", "MAE1W", "MAE2W", "MAE4W", "MAE8W")
Boruta_RF_MSE <- as.data.frame(Boruta_RF_MSE)
Boruta_RF_MAE <- as.data.frame(Boruta_RF_MAE)
Boruta_RF_ERROR <- cbind(c("B-RF"), Boruta_RF_MSE, Boruta_RF_MAE)
names(Boruta_RF_ERROR) <- c("MODEL", "MSE1W", "MSE2W", "MSE4W", "MSE8W", "MAE1W", "MAE2W", "MAE4W", "MAE8W")
ErrorMatrix <- rbind(ErrorMatrix, Boruta_XGB_ERROR, Boruta_RF_ERROR)

output <- str_c("ERRORS_", Sys.Date(), ".xlsx")
write_xlsx(ErrorMatrix, output)


save.image("Forecasting_result_Boruta.RData")

write_xlsx(VO, "Boruta_VariableOrder.xlsx")
write_xlsx(ST, "Boruta_Selected.xlsx")
