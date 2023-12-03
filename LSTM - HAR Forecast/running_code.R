library(tensorflow)
library(keras)
library(readxl)
library(readr)
library(recipes)
library(timetk)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))
# data <- as.data.frame(read.csv("./Data/data_raw_weekly.csv"))
data <- as.data.frame(read_xlsx("./Data/data_raw_weekly.xlsx")) %>% data.matrix()

tcode = data[1,]  # first element: Transformation code
tcode

data = data[-1,]
tdata = data[-(1:2),]
ncol(data)

for (i in 2:ncol(data)){

  if(tcode[i] == 1){
    tdata[,i] <- data[-(1:2),i]
  }

  if(tcode[i] == 2){
    tdata[,i] <- diff(data[-1,i])
  }

  if(tcode[i] == 4){
    tdata[,i] <- log(data[-(1:2),i])
  } # log

  if(tcode[i] == 5){
    tdata[,i] <- diff(log(data[-1,i]))
  }

  if(tcode[i] == 6){
    tdata[,i] <- diff(diff(log(data[,i])))
  }

  if(tcode[i] == 7){
    tdata[,i] <- diff(data[-1,i]/data[1:(nrow(data)-1),i])
  }
}

head(tdata[,1])
tail(tdata[,1])

complete.cases(tdata)

row.names(tdata) <- tdata[,1]
data_use <- tdata[,2:ncol(tdata)]

colnames(data_use)

# y_data <- tdata[,'vkospi']
# X_data <- tdata[, c(-1, -2)]

library(reticulate)
library(tidyverse)

###### virtual environment connect #####
conda_list()

conda_list()[[1]][2] %>% 
  use_condaenv(required = TRUE)

###### Data Transformation function for LSTM ######
# prepare_data_for_model <- function(Y, lag=1) {
#   aux <- embed(Y, lag + 4) 
#   
#   X <- aux[, -(1:lag), drop = FALSE] # Not to drop dimensions if lag == 1
#   
#   if(lag == 1) {
#     X.out <- tail(aux, 1)[, -(1:lag), drop = FALSE]
#   } else {
#     # Correcting the indexing for lags greater than 1
#     X.out <- aux[(lag+1):nrow(aux), -(1:lag), drop = FALSE]
#     X.out <- tail(X.out, 1)
#   }
#   
#   X2 <- X
#   X2[!X2 == 0] <- 0 # Replacing non-zero values with zero
#   
#   # Adjusted the reorganization of the matrix
#   for(i in 0:(ncol(Y) - 1)) {
#     start_col <- i * lag + 1
#     end_col <- start_col + lag - 1
#     if (end_col <= ncol(X)) {
#       X2[, (lag * i + 1):(lag * i + lag)] <- X[, start_col:end_col]
#     }
#   }
#   
#   X.out2 <- X.out
#   
#   # Creating arrays from matrices
#   X.arr <- array(data = as.numeric(X2), dim = c(nrow(X), lag, ncol(Y)))
#   X.out.arr <- array(data = as.numeric(X.out2), dim = c(1, lag, ncol(Y)))
#   
#   return(list(X.arr = X.arr, X.out.arr = X.out.arr))
# }

normalize <- function(x) {
  return((x-min(x))/(max(x)-min(x)))
}

denormalize <- function(x, minval, maxval) {
  x*(maxval-minval) + minval
}

###### LSTM ######
run_multi_lstm=function(Y,indice,lag, batch_size = 30, unit_n = 32){
  comp=princomp(scale(Y,scale=FALSE))
  Y2 = cbind(Y, comp$scores[,1:4]) %>% as.data.frame()
  Y3 = lapply(Y2, normalize) %>% as.data.frame() %>% as.matrix()
  aux=embed(Y3,4+lag)
  y=aux[,indice]
  X=aux[,-c(1:(ncol(Y3)*lag))]  
  
  if(lag==1){
    X.out=tail(aux,1)[1:ncol(X)]  
  }else{
    X.out=aux[,-c(1:(ncol(Y3)*(lag-1)))]
    X.out=tail(X.out,1)[1:ncol(X)]
  }
  
  ###
  X2 <- X %>% replace(!0, 0) 
  
  for(i in 0:(ncol(Y3)-1)){
    X2[,(4*i+1)] <- X[,(i+1)]
    X2[,(4*i+2)] <- X[,(i+ncol(Y3)+1)]
    X2[,(4*i+3)] <- X[,(i+2*ncol(Y3)+1)]
    X2[,(4*i+4)] <- X[,(i+3*ncol(Y3)+1)]
  }
  
  X.out2 <- X.out %>% replace(!0, 0)
  
  for(i in 0:(ncol(Y3)-1)){
    X.out2[(4*i+1)] <- X.out[(i+1)]
    X.out2[(4*i+2)] <- X.out[(i+ncol(Y3)+1)]
    X.out2[(4*i+3)] <- X.out[(i+2*ncol(Y3)+1)]
    X.out2[(4*i+4)] <- X.out[(i+3*ncol(Y3)+1)]
  }
  
  ###  
  X.arr = array(
    data = as.numeric(unlist(X2)),
    dim = c(nrow(X), 4, ncol(Y3)))
  
  X.out.arr = array(
    data = as.numeric(unlist(X.out2)),
    dim = c(1, 4, ncol(Y3)))
  
  batch_size = batch_size
  unit_n = unit_n
  feature = ncol(Y3)
  epochs = 100
  
  model = keras_model_sequential()
  
  model %>% layer_lstm(units = unit_n, 
                       input_shape = c(4, feature),
                       stateful = FALSE) %>%
    layer_dense(units = 1) 

  model %>% compile(loss = 'mse', optimizer = 'adam')
  
  model %>% summary()
  
  history = model %>% fit(X.arr, y, epochs = epochs,
                          batch_size = batch_size, shuffle = FALSE, verbose = 2)

  pred = model(X.out.arr) %>% denormalize(min(Y2[,indice]),max(Y2[,indice])) # De-normalization
  
  return(list("model"=model,"pred"=pred))
}

###### Hierarchical LSTM ######
run_hierarchical_lstm=function(Y,indice,lag, batch_size = 30, unit_n = 32){
  comp=princomp(scale(Y,scale=FALSE))
  Y2 = cbind(Y, comp$scores[,1:4]) %>% as.data.frame()
  Y3 = lapply(Y2, normalize) %>% as.data.frame() %>% as.matrix()
  aux=embed(Y3,4+lag)
  y=aux[,indice]
  X=aux[,-c(1:(ncol(Y3)*lag))]

  if(lag==1){
    X.out=tail(aux,1)[1:ncol(X)]  
  }else{
    X.out=aux[,-c(1:(ncol(Y3)*(lag-1)))]
    X.out=tail(X.out,1)[1:ncol(X)]
  }

  ###
  X2_1 <- X[,1:48] %>% replace(!0, 0) 
  
  for(i in 0:11){
    X2_1[,(4*i+1)] <- X[,(i+1)]
    X2_1[,(4*i+2)] <- X[,(i+ncol(Y3)+1)]
    X2_1[,(4*i+3)] <- X[,(i+2*ncol(Y3)+1)]
    X2_1[,(4*i+4)] <- X[,(i+3*ncol(Y3)+1)]
  }
  
  X2_2 <- X[,1:36] %>% replace(!0, 0) 
  
  for(i in 12:20){
    X2_2[,(4*i-47)] <- X[,(i+1)]
    X2_2[,(4*i-46)] <- X[,(i+ncol(Y3)+1)]
    X2_2[,(4*i-45)] <- X[,(i+2*ncol(Y3)+1)]
    X2_2[,(4*i-44)] <- X[,(i+3*ncol(Y3)+1)]
  }
  
  X2_3 <- X[,1:124] %>% replace(!0, 0) 
  
  for(i in 21:51){
    X2_3[,(4*i-83)] <- X[,(i+1)]
    X2_3[,(4*i-82)] <- X[,(i+ncol(Y3)+1)]
    X2_3[,(4*i-81)] <- X[,(i+2*ncol(Y3)+1)]
    X2_3[,(4*i-80)] <- X[,(i+3*ncol(Y3)+1)]
  }
  
  X2_4 <- X[,1:16] %>% replace(!0, 0) 
  
  for(i in 52:55){
    X2_4[,(4*i-207)] <- X[,(i+1)]
    X2_4[,(4*i-206)] <- X[,(i+ncol(Y3)+1)]
    X2_4[,(4*i-205)] <- X[,(i+2*ncol(Y3)+1)]
    X2_4[,(4*i-204)] <- X[,(i+3*ncol(Y3)+1)]
  }
  
  ###
  X.out2_1 <- X.out[1:48] %>% replace(!0, 0) 
  
  for(i in 0:11){
    X.out2_1[(4*i+1)] <- X.out[(i+1)]
    X.out2_1[(4*i+2)] <- X.out[(i+ncol(Y3)+1)]
    X.out2_1[(4*i+3)] <- X.out[(i+2*ncol(Y3)+1)]
    X.out2_1[(4*i+4)] <- X.out[(i+3*ncol(Y3)+1)]
  }
  
  X.out2_2 <- X.out[1:36] %>% replace(!0, 0) 
  
  for(i in 12:20){
    X.out2_2[(4*i-47)] <- X.out[(i+1)]
    X.out2_2[(4*i-46)] <- X.out[(i+ncol(Y3)+1)]
    X.out2_2[(4*i-45)] <- X.out[(i+2*ncol(Y3)+1)]
    X.out2_2[(4*i-44)] <- X.out[(i+3*ncol(Y3)+1)]
  }
  
  X.out2_3 <- X.out[1:124] %>% replace(!0, 0) 
  
  for(i in 21:51){
    X.out2_3[(4*i-83)] <- X.out[(i+1)]
    X.out2_3[(4*i-82)] <- X.out[(i+ncol(Y3)+1)]
    X.out2_3[(4*i-81)] <- X.out[(i+2*ncol(Y3)+1)]
    X.out2_3[(4*i-80)] <- X.out[(i+3*ncol(Y3)+1)]
  }
  
  X.out2_4 <- X.out[1:16] %>% replace(!0, 0) 
  
  for(i in 52:55){
    X.out2_4[(4*i-207)] <- X.out[(i+1)]
    X.out2_4[(4*i-206)] <- X.out[(i+ncol(Y3)+1)]
    X.out2_4[(4*i-205)] <- X.out[(i+2*ncol(Y3)+1)]
    X.out2_4[(4*i-204)] <- X.out[(i+3*ncol(Y3)+1)]
  }
  
  ###  
  X.arr1 = array(
    data = as.numeric(unlist(X2_1)),
    dim = c(nrow(X), 4, 12))
  
  X.arr2 = array(
    data = as.numeric(unlist(X2_2)),
    dim = c(nrow(X), 4, 9))
  
  X.arr3 = array(
    data = as.numeric(unlist(X2_3)),
    dim = c(nrow(X), 4, 31))
  
  X.arr4 = array(
    data = as.numeric(unlist(X2_4)),
    dim = c(nrow(X), 4, 4))
  
  ###
  X.out.arr1 = array(
    data = as.numeric(unlist(X.out2_1)),
    dim = c(1, 4, 12))
  
  X.out.arr2 = array(
    data = as.numeric(unlist(X.out2_2)),
    dim = c(1, 4, 9))
  
  X.out.arr3 = array(
    data = as.numeric(unlist(X.out2_3)),
    dim = c(1, 4, 31))
  
  X.out.arr4 = array(
    data = as.numeric(unlist(X.out2_4)),
    dim = c(1, 4, 4))
  
  batch_size = batch_size
  epochs = 100
  unit_n = unit_n
  
  input_layer_1 <- layer_input(shape = c(4, 31))
  input_layer_2 <- layer_input(shape = c(4, 9))
  input_layer_3 <- layer_input(shape = c(4, 12))
  input_layer_4 <- layer_input(shape = c(4, 4))
  
  # Level-1 LSTM layers
  lstm_layer_1_1 <- layer_lstm(units = unit_n, return_sequences = TRUE)(input_layer_1)
  lstm_layer_1_2 <- layer_lstm(units = unit_n, return_sequences = TRUE)(input_layer_2)
  
  # Concatenation after Level-1
  concatenated_layer_1 <- layer_concatenate(c(lstm_layer_1_1, lstm_layer_1_2))
  
  # Level-2 LSTM layer
  lstm_layer_3_1 <- layer_lstm(units = unit_n, return_sequences = TRUE)(input_layer_3)
  lstm_layer_2 <- layer_lstm(units = unit_n, return_sequences = TRUE)(concatenated_layer_1)
  
  # Concatenation for Level-2 and Level-3
  concatenated_layer_2 <- layer_concatenate(c(lstm_layer_2, lstm_layer_3_1))
  
  # Level-3 LSTM layer
  lstm_layer_3_2 <- layer_lstm(units = unit_n, return_sequences = TRUE)(input_layer_4)
  
  # Concatenation for Level-3
  concatenated_layer_3 <- layer_concatenate(c(concatenated_layer_2, lstm_layer_3_2))
  
  # Level-4 LSTM layer
  lstm_layer_4 <- layer_lstm(units = unit_n)(concatenated_layer_3)
  
  # Output layer
  output_layer <- layer_dense(units = 1)(lstm_layer_4)
  
  # Model
  model <- keras_model(inputs = list(input_layer_1, input_layer_2, input_layer_3, input_layer_4), 
                       outputs = output_layer)
  
  # Compile the model
  model %>% compile(loss = 'mse', optimizer = 'adam')
  
  # Fit the model to the training data
  model %>% summary()
  
  history = model %>% fit(x = list(X.arr3, X.arr2, X.arr1, X.arr4), y, epochs = epochs,
                          batch_size = batch_size, shuffle = FALSE, verbose = 2)
  
  pred = model(list(X.out.arr3, X.out.arr2, X.out.arr1, X.out.arr4)) %>%
    denormalize(min(Y2[,indice]),max(Y2[,indice]))
  
  return(list("model"=model,"pred"=pred))
}

# run_hierarchical_lstm_2=function(Y,indice,lag){
#   comp=princomp(scale(Y,scale=FALSE))
#   Y2 = cbind(Y, comp$scores[,1:4]) %>% as.data.frame()
#   Y3 = lapply(Y2, normalize) %>% as.data.frame() %>% as.matrix()
#   aux=embed(Y3,4+lag)
#   y=aux[,indice]
#   X=aux[,-c(1:(ncol(Y3)*lag))]
#   
#   if(lag==1){
#     X.out=tail(aux,1)[1:ncol(X)]  
#   }else{
#     X.out=aux[,-c(1:(ncol(Y3)*(lag-1)))]
#     X.out=tail(X.out,1)[1:ncol(X)]
#   }
#   
#   ###
#   X2_1 <- X[,1:56]
#   
#   X2_2 <- X[,57:112]
#   
#   X2_3 <- X[,113:168]
#   
#   X2_4 <- X[,169:224]
#   
#   ###
#   X.out2_1 <- X.out[1:56]
#   
#   X.out2_2 <- X.out[57:112]
# 
#   X.out2_3 <- X.out[113:168]
#   
#   X.out2_4 <- X.out[169:224]
# 
#   ###  
#   X.arr1 = array(
#     data = as.numeric(unlist(X2_1)),
#     dim = c(nrow(X), 1, 56))
#   
#   X.arr2 = array(
#     data = as.numeric(unlist(X2_2)),
#     dim = c(nrow(X), 1, 56))
#   
#   X.arr3 = array(
#     data = as.numeric(unlist(X2_3)),
#     dim = c(nrow(X), 1, 56))
#   
#   X.arr4 = array(
#     data = as.numeric(unlist(X2_4)),
#     dim = c(nrow(X), 1, 56))
#   
#   ###
#   X.out.arr1 = array(
#     data = as.numeric(unlist(X.out2_1)),
#     dim = c(1, 1, 56))
#   
#   X.out.arr2 = array(
#     data = as.numeric(unlist(X.out2_2)),
#     dim = c(1, 1, 56))
#   
#   X.out.arr3 = array(
#     data = as.numeric(unlist(X.out2_3)),
#     dim = c(1, 1, 56))
#   
#   X.out.arr4 = array(
#     data = as.numeric(unlist(X.out2_4)),
#     dim = c(1, 1, 56))
#   
#   batch_size = 30
#   epochs = 100
#   
#   input_layer_1 <- layer_input(shape = c(1, 56))
#   input_layer_2 <- layer_input(shape = c(1, 56))
#   input_layer_3 <- layer_input(shape = c(1, 56))
#   input_layer_4 <- layer_input(shape = c(1, 56))
#   
#   # Level-1 LSTM layers
#   lstm_layer_1_1 <- layer_lstm(units = 16, return_sequences = TRUE)(input_layer_1)
#   lstm_layer_1_2 <- layer_lstm(units = 16, return_sequences = TRUE)(input_layer_2)
#   
#   # Concatenation after Level-1
#   concatenated_layer_1 <- layer_concatenate(c(lstm_layer_1_1, lstm_layer_1_2))
#   
#   # Level-2 LSTM layer
#   lstm_layer_3_1 <- layer_lstm(units = 16, return_sequences = TRUE)(input_layer_3)
#   lstm_layer_2 <- layer_lstm(units = 16, return_sequences = TRUE)(concatenated_layer_1)
#   
#   # Concatenation for Level-2 and Level-3
#   concatenated_layer_2 <- layer_concatenate(c(lstm_layer_2, lstm_layer_3_1))
#   
#   # Level-3 LSTM layer
#   lstm_layer_3_2 <- layer_lstm(units = 16, return_sequences = TRUE)(input_layer_4)
#   
#   # Concatenation for Level-3
#   concatenated_layer_3 <- layer_concatenate(c(concatenated_layer_2, lstm_layer_3_2))
#   
#   # Level-4 LSTM layer
#   lstm_layer_4 <- layer_lstm(units = 16)(concatenated_layer_3)
#   
#   # Output layer
#   output_layer <- layer_dense(units = 1)(lstm_layer_4)
#   
#   # Model
#   model <- keras_model(inputs = list(input_layer_1, input_layer_2, input_layer_3, input_layer_4), 
#                        outputs = output_layer)
#   
#   # Compile the model
#   model %>% compile(loss = 'mse', optimizer = 'adam')
#   
#   # Fit the model to the training data
#   model %>% summary()
#   
#   history = model %>% fit(x = list(X.arr4, X.arr3, X.arr2, X.arr1), y, epochs = epochs,
#                           batch_size = batch_size, shuffle = FALSE, verbose = 2)
#   
#   pred = model(list(X.out.arr4, X.out.arr3, X.out.arr2, X.out.arr1)) %>%
#     denormalize(min(Y2[,indice]),max(Y2[,indice]))
#   
#   return(list("model"=model,"pred"=pred))
# }

####### Rolling Window #######
set.seed(40)
set_random_seed(40)

rolling.window.lstm=function(Y,nprev,indice=1,lag=1, batch = 30, unit = 32){
  
  save.pred=matrix(NA,nprev,1)
  
  for(i in nprev:1){
    Y.window=Y[(1+nprev-i):(nrow(Y)-i),] %>% as.data.frame()
    lstm=run_multi_lstm(Y.window,indice,lag,batch,unit)
    save.pred[(1+nprev-i),]=as.numeric(lstm$pred) # Note as.numeric()
    cat("iteration",(1+nprev-i),"\n")
  }
  
  real=Y[,indice]
  plot(real,type="l")
  lines(c(rep(NA,length(real)-nprev),save.pred),col="red")
  
  rmse=sqrt(mean((tail(real,nprev)-save.pred)^2))
  mae=mean(abs(tail(real,nprev)-save.pred))
  errors=c("rmse"=rmse,"mae"=mae)
  
  return(list("pred"=save.pred,"errors"=errors))
}

## batch 30 - unit 32
lstm_result_1 <- rolling.window.lstm(data_use, 112, 1, 1)
lstm_result_2 <- rolling.window.lstm(data_use, 112, 1, 2)
lstm_result_3 <- rolling.window.lstm(data_use, 112, 1, 4)
lstm_result_4 <- rolling.window.lstm(data_use, 112, 1, 8)

## batch 30 - unit 16
lstm_result2_1 <- rolling.window.lstm(data_use, 112, 1, 1, 30, 16)
lstm_result2_2 <- rolling.window.lstm(data_use, 112, 1, 2, 30, 16)
lstm_result2_3 <- rolling.window.lstm(data_use, 112, 1, 4, 30, 16)
lstm_result2_4 <- rolling.window.lstm(data_use, 112, 1, 8, 30, 16)

## batch 25 - unit 32
lstm_result3_1 <- rolling.window.lstm(data_use, 112, 1, 1, 25, 32)
lstm_result3_2 <- rolling.window.lstm(data_use, 112, 1, 2, 25, 32)
lstm_result3_3 <- rolling.window.lstm(data_use, 112, 1, 4, 25, 32)
lstm_result3_4 <- rolling.window.lstm(data_use, 112, 1, 8, 25, 32)

## batch 35 - unit 32
lstm_result4_1 <- rolling.window.lstm(data_use, 112, 1, 1, 35, 32)
lstm_result4_2 <- rolling.window.lstm(data_use, 112, 1, 2, 35, 32)
lstm_result4_3 <- rolling.window.lstm(data_use, 112, 1, 4, 35, 32)
lstm_result4_4 <- rolling.window.lstm(data_use, 112, 1, 8, 35, 32)

## batch 25 - unit 40
lstm_result5_1 <- rolling.window.lstm(data_use, 112, 1, 1, 30, 40)
lstm_result5_2 <- rolling.window.lstm(data_use, 112, 1, 2, 30, 40)
lstm_result5_3 <- rolling.window.lstm(data_use, 112, 1, 4, 30, 40)
lstm_result5_4 <- rolling.window.lstm(data_use, 112, 1, 8, 30, 40)

## batch 20 - unit 40
lstm_result6_1 <- rolling.window.lstm(data_use, 112, 1, 1, 20, 40)
lstm_result6_2 <- rolling.window.lstm(data_use, 112, 1, 2, 20, 40)
lstm_result6_3 <- rolling.window.lstm(data_use, 112, 1, 4, 20, 40)
lstm_result6_4 <- rolling.window.lstm(data_use, 112, 1, 8, 20, 40)

rolling.window.lstm.hier=function(Y, nprev, indice=1, lag=1, batch = 30, unit = 32){
  
  save.pred=matrix(NA,nprev,1)
  
  for(i in nprev:1){
    Y.window=Y[(1+nprev-i):(nrow(Y)-i),] %>% as.data.frame()
    lstm=run_hierarchical_lstm(Y.window,indice,lag,batch,unit)
    save.pred[(1+nprev-i),]=as.numeric(lstm$pred) # Note as.numeric()
    cat("iteration",(1+nprev-i),"\n")
  }
  
  real=Y[,indice]
  plot(real,type="l")
  lines(c(rep(NA,length(real)-nprev),save.pred),col="red")
  
  rmse=sqrt(mean((tail(real,nprev)-save.pred)^2))
  mae=mean(abs(tail(real,nprev)-save.pred))
  errors=c("rmse"=rmse,"mae"=mae)
  
  return(list("pred"=save.pred,"errors"=errors))
}

## batch 30 - unit 32
hier_lstm_1 <- rolling.window.lstm.hier(data_use, 112, 1, 1)
hier_lstm_2 <- rolling.window.lstm.hier(data_use, 112, 1, 2)
hier_lstm_3 <- rolling.window.lstm.hier(data_use, 112, 1, 4)
hier_lstm_4 <- rolling.window.lstm.hier(data_use, 112, 1, 8)

## batch 30 - unit 16
hier_lstm2_1 <- rolling.window.lstm.hier(data_use, 112, 1, 1, 30, 16)
hier_lstm2_2 <- rolling.window.lstm.hier(data_use, 112, 1, 2, 30, 16)
hier_lstm2_3 <- rolling.window.lstm.hier(data_use, 112, 1, 4, 30, 16)
hier_lstm2_4 <- rolling.window.lstm.hier(data_use, 112, 1, 8, 30, 16)

## batch 25 - unit 32
hier_lstm3_1 <- rolling.window.lstm.hier(data_use, 112, 1, 1, 25, 32)
hier_lstm3_2 <- rolling.window.lstm.hier(data_use, 112, 1, 2, 25, 32)
hier_lstm3_3 <- rolling.window.lstm.hier(data_use, 112, 1, 4, 25, 32)
hier_lstm3_4 <- rolling.window.lstm.hier(data_use, 112, 1, 8, 25, 32)

## batch 35 - unit 32
hier_lstm4_1 <- rolling.window.lstm.hier(data_use, 112, 1, 1, 35, 32)
hier_lstm4_2 <- rolling.window.lstm.hier(data_use, 112, 1, 2, 35, 32)
hier_lstm4_3 <- rolling.window.lstm.hier(data_use, 112, 1, 4, 35, 32)
hier_lstm4_4 <- rolling.window.lstm.hier(data_use, 112, 1, 8, 35, 32)

## batch 30 - unit 40
hier_lstm5_1 <- rolling.window.lstm.hier(data_use, 112, 1, 1, 30, 40)
hier_lstm5_2 <- rolling.window.lstm.hier(data_use, 112, 1, 2, 30, 40)
hier_lstm5_3 <- rolling.window.lstm.hier(data_use, 112, 1, 4, 30, 40)
hier_lstm5_4 <- rolling.window.lstm.hier(data_use, 112, 1, 8, 30, 40)

## batch 20 - unit 40
hier_lstm6_1 <- rolling.window.lstm.hier(data_use, 112, 1, 1, 20, 40)
hier_lstm6_2 <- rolling.window.lstm.hier(data_use, 112, 1, 2, 20, 40)
hier_lstm6_3 <- rolling.window.lstm.hier(data_use, 112, 1, 4, 20, 40)
hier_lstm6_4 <- rolling.window.lstm.hier(data_use, 112, 1, 8, 20, 40)


# rolling.window.lstm.hier_2=function(Y, nprev, indice=1, lag=1){
#   
#   save.pred=matrix(NA,nprev,1)
#   
#   for(i in nprev:1){
#     Y.window=Y[(1+nprev-i):(nrow(Y)-i),] %>% as.data.frame()
#     lstm=run_hierarchical_lstm_2(Y.window,indice,lag)
#     save.pred[(1+nprev-i),]=as.numeric(lstm$pred) # Note as.numeric()
#     cat("iteration",(1+nprev-i),"\n")
#   }
#   
#   real=Y[,indice]
#   plot(real,type="l")
#   lines(c(rep(NA,length(real)-nprev),save.pred),col="red")
#   
#   rmse=sqrt(mean((tail(real,nprev)-save.pred)^2))
#   mae=mean(abs(tail(real,nprev)-save.pred))
#   errors=c("rmse"=rmse,"mae"=mae)
#   
#   return(list("pred"=save.pred,"errors"=errors))
# }
#
# test_1 <- rolling.window.lstm.hier_2(data_use, 104, 1, 1)
# test_2 <- rolling.window.lstm.hier_2(data_use, 104, 1, 2)
# test_3 <- rolling.window.lstm.hier_2(data_use, 104, 1, 4)
# test_4 <- rolling.window.lstm.hier_2(data_use, 104, 1, 8)


# save.image("lstm_result.RData")

load('lstm_result.RData')

###### HAR ######
library(readxl)
library(readr)
library(recipes)
library(timetk)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))
har_data <- as.data.frame(read_excel("./Data/data_raw_weekly_harx.xlsx")) %>% data.matrix()
colnames(har_data)

har_data_t <- har_data[,2:ncol(har_data)] %>% na.omit()

head(har_data_t[,1])
tail(har_data_t[,1])

complete.cases(har_data_t)

source('func-ar_111.R')

## harx
harx1 <- ar.rolling.window(har_data_t, 112, 1, 1)
harx2 <- ar.rolling.window(har_data_t, 112, 1, 2)
harx3 <- ar.rolling.window(har_data_t, 112, 1, 4)
harx4 <- ar.rolling.window(har_data_t, 112, 1, 8)

## har
har_Data <- har_data_t[,1:3]
<<<<<<< HEAD

har1 <- ar.rolling.window(har_Data, 112, 1, 1)
har2 <- ar.rolling.window(har_Data, 112, 1, 2)
har3 <- ar.rolling.window(har_Data, 112, 1, 4)
har4 <- ar.rolling.window(har_Data, 112, 1, 8)

load('har_result.RData')

# save.image("har_result.RData")
=======

har1 <- ar.rolling.window(har_Data, 112, 1, 1)
har2 <- ar.rolling.window(har_Data, 112, 1, 2)
har3 <- ar.rolling.window(har_Data, 112, 1, 4)
har4 <- ar.rolling.window(har_Data, 112, 1, 8)

load('har_result.RData')

# save.image("har_result.RData")
