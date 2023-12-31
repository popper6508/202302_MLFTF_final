<<<<<<< HEAD
runxgb=function(Y,indice,lag, selected){
  
  #comp=princomp(scale(Y,scale=FALSE))
  #Y2=cbind(Y,comp$scores[,1:4])
  Y2 = Y
  aux=embed(Y2,5+lag)
  y=aux[,1]
  X=aux[,-c(1:(ncol(Y2)*lag))]  
  
  if(lag==1){
    X.out=tail(aux,1)[1:ncol(X)]  
  }else{
    X.out=aux[,-c(1:(ncol(Y2)*(lag-1)))]
    X.out=tail(X.out,1)[1:ncol(X)]
  }
  
  model = xgboost(X[, selected],label = y,nrounds = 1000, verbose = FALSE,
                  params=list(eta=0.05,nthread=1,colsample_bylevel=2/3,subsample=1,max_depth=4,min_child_weigth=nrow(X)/200))
  
  pred=predict(model,t(X.out[selected]))
  
  return(list("model"=model,"pred"=pred))
}


xgb.rolling.window=function(Y,nprev,indice=1,lag=1, selected){
  
  save.pred=matrix(NA,nprev,1)
  for(i in nprev:1){
    Y.window=Y[(1+nprev-i):(nrow(Y)-i),]
    lasso=runxgb(Y.window,indice,lag,selected)
    save.pred[(1+nprev-i),]=lasso$pred
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

=======
runxgb=function(Y,indice,lag, selected){
  
  #comp=princomp(scale(Y,scale=FALSE))
  #Y2=cbind(Y,comp$scores[,1:4])
  Y2 = Y
  aux=embed(Y2,5+lag)
  y=aux[,1]
  X=aux[,-c(1:(ncol(Y2)*lag))]  
  
  if(lag==1){
    X.out=tail(aux,1)[1:ncol(X)]  
  }else{
    X.out=aux[,-c(1:(ncol(Y2)*(lag-1)))]
    X.out=tail(X.out,1)[1:ncol(X)]
  }
  
  model = xgboost(X[, selected],label = y,nrounds = 1000, verbose = FALSE,
                  params=list(eta=0.05,nthread=1,colsample_bylevel=2/3,subsample=1,max_depth=4,min_child_weigth=nrow(X)/200))
  
  pred=predict(model,t(X.out[selected]))
  
  return(list("model"=model,"pred"=pred))
}


xgb.rolling.window=function(Y,nprev,indice=1,lag=1, selected){
  
  save.pred=matrix(NA,nprev,1)
  for(i in nprev:1){
    Y.window=Y[(1+nprev-i):(nrow(Y)-i),]
    lasso=runxgb(Y.window,indice,lag,selected)
    save.pred[(1+nprev-i),]=lasso$pred
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

>>>>>>> 3c6f8a671c98dbedc98801ab2460f54e27383541
