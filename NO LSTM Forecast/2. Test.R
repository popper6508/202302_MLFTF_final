## For Test

# Library Load
library(MCS)
library(sandwich)
library(dplyr)
library(stringr)

#setwd("C:/R/ML/Homework/final") 
dir()

load("Forecasting_result_Boruta.RData")
     
# 겨레 LSTM 결과값 데려오기기
lstm_pred <- read_xlsx("lstmpred.xlsx") %>% 
  as.matrix()
hier_pred <- read_xlsx("hierpred.xlsx") %>% 
  as.matrix()


#### GW Test : Best Model for each forecast horizon, Based on MAE  ###
source("gwtest.R")
real=tail(Y[,1],112) # Real data for comparison


# Horizon1  : Hier and Adaptive Lasso
gwtest_hier_alasso = matrix(NA,1,1)
gwpvalue_hier_alasso = matrix(NA,1,1)

gw1 = gw.test(adalasso_pred[,1], hier_pred[,1], real, tau=1, T=112, method="NeweyWest")

gwtest_hier_alasso[1] <- gw1$statistic
gwpvalue_hier_alasso[1] <- gw1$p.value

# Horizon2  : Hier and HAR
gwtest_hier_har = matrix(NA,1,1)
gwpvalue_hier_har = matrix(NA,1,1)

gw2 = gw.test(har_pred[,2], hier_pred[,2], real, tau=2, T=112, method="NeweyWest")

gwtest_hier_har[1] <- gw2$statistic
gwpvalue_hier_har[1] <- gw2$p.value

# Horizon4  : Hier and XGB with Boruta
gwtest_hier_BXGB = matrix(NA,1,1)
gwpvalue_hier_BXGB = matrix(NA,1,1)

gw4 = gw.test(xgb_selected_pred[,3], hier_pred[,3], real, tau=4, T=112, method="NeweyWest")

gwtest_hier_BXGB[1] <- gw4$statistic
gwpvalue_hier_BXGB[1] <- gw4$p.value

# Horizon8  : Hier and RF with Boruta Selection
gwtest_hier_BRF = matrix(NA,1,1)
gwpvalue_hier_BRF = matrix(NA,1,1)

gw8 = gw.test(hier_pred[,4], rf_selected_pred[,4], real, tau=8, T=112, method="NeweyWest")

gwtest_hier_BRF[1] <- gw8$statistic
gwpvalue_hier_BRF[1] <- gw8$p.value

# P values
horz <- c("H1", "H2", "H4", "H8")
comp <- c("AdaLasso", "HAR", "XGB-Boruta", "RF-Boruta")
pv <- c(gw1$p.value, gw2$p.value, gw4$p.value, gw8$p.value)
pv
pvalues <- cbind(horz, comp, pv)
pvalues
pvalues <- as.data.frame(pvalues)

# write_xlsx(pvalues, "pvaluesGW.xlsx")

### MCS Test ###
# MCS Test
i = 1
Pred = cbind(har_pred[,i],
             harx_pred[,i],
             rw_pred[,i], 
             ar_pred[,i], 
             lasso_pred[,i], 
             adalasso_pred[,i], 
             elasticnet_pred[,i],  
             adaelasticnet_pred[,i], 
             tfact_pred[,i], 
             rf_pred[,i], 
             xgb_pred[, i],
             rf_selected_pred[, i],
             xgb_selected_pred[, i],
             lstm_pred[,i],
             hier_pred[,i])

LOSS  = Pred-real
LOSS1 = LOSS^2      # squared error
LOSS2 = abs(LOSS)   # absolute error

SSM_1 <- MCSprocedure(LOSS1, alpha = 0.5, B = 5000, statistic = "Tmax")

i = 2
Pred = cbind(har_pred[,i],
             harx_pred[,i],
             rw_pred[,i], 
             ar_pred[,i], 
             lasso_pred[,i], 
             adalasso_pred[,i], 
             elasticnet_pred[,i],  
             adaelasticnet_pred[,i], 
             tfact_pred[,i], 
             rf_pred[,i], 
             xgb_pred[, i],
             rf_selected_pred[, i],
             xgb_selected_pred[, i],
             lstm_pred[,i],
             hier_pred[,i])

LOSS  = Pred-real
LOSS1 = LOSS^2      # squared error
LOSS2 = abs(LOSS)   # absolute error

SSM_2 <- MCSprocedure(LOSS1, alpha = 0.5, B = 5000, statistic = "Tmax")  

i = 3
Pred = cbind(har_pred[,i],
             harx_pred[,i],
             rw_pred[,i], 
             ar_pred[,i], 
             lasso_pred[,i], 
             adalasso_pred[,i], 
             elasticnet_pred[,i],  
             adaelasticnet_pred[,i], 
             tfact_pred[,i], 
             rf_pred[,i], 
             xgb_pred[, i],
             rf_selected_pred[, i],
             xgb_selected_pred[, i],
             lstm_pred[,i],
             hier_pred[,i])

LOSS  = Pred-real
LOSS1 = LOSS^2      # squared error
LOSS2 = abs(LOSS)   # absolute error

SSM_4 <- MCSprocedure(LOSS1, alpha = 0.5, B = 5000, statistic = "Tmax")

i = 4
Pred = cbind(har_pred[,i],
             harx_pred[,i],
             rw_pred[,i], 
             ar_pred[,i], 
             lasso_pred[,i], 
             adalasso_pred[,i], 
             elasticnet_pred[,i],  
             adaelasticnet_pred[,i], 
             tfact_pred[,i], 
             rf_pred[,i], 
             xgb_pred[, i],
             rf_selected_pred[, i],
             xgb_selected_pred[, i],
             lstm_pred[,i],
             hier_pred[,i])

LOSS  = Pred-real
LOSS1 = LOSS^2      # squared error
LOSS2 = abs(LOSS)   # absolute error

SSM_8 <- MCSprocedure(LOSS1, alpha = 0.5, B = 5000, statistic = "Tmax")

save.image("results_Test.Rdata")
