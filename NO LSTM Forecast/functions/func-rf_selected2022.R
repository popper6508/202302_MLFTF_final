<<<<<<< HEAD
runrf=function(Y,indice,lag,selected){
#  comp=princomp(scale(Y,scale=FALSE))
#  Y2=cbind(Y,comp$scores[,1:4])
  Y2=Y
  
  aux=embed(Y2,5+lag)
  y=aux[,1]
  X=aux[,-c(1:(ncol(Y2)*lag))]  
  
  if(lag==1){
    X.out=tail(aux,1)[1:ncol(X)]  
  }else{
    X.out=aux[,-c(1:(ncol(Y2)*(lag-1)))]
    X.out=tail(X.out,1)[1:ncol(X)]
  }
  
  model=randomForest(X[,selected],y,importance=TRUE)
  pred=predict(model,X.out[selected])
    
  return(list("model"=model,"pred"=pred))
}


rf.rolling.window=function(Y,nprev,indice=1,lag=1,selected){
  
  save.importance=list()
  save.pred=matrix(NA,nprev,1)
  for(i in nprev:1){
    Y.window=Y[(1+nprev-i):(nrow(Y)-i),]
    lasso=runrf(Y.window,indice,lag,selected)
    save.pred[(1+nprev-i),]=lasso$pred
    save.importance[[i]]=importance(lasso$model)
    cat("iteration",(1+nprev-i),"\n")
  }
  
  real=Y[,indice]
  plot(real,type="l")
  lines(c(rep(NA,length(real)-nprev),save.pred),col="red")
  
  rmse=sqrt(mean((tail(real,nprev)-save.pred)^2))
  mae=mean(abs(tail(real,nprev)-save.pred))
  errors=c("rmse"=rmse,"mae"=mae)
  
  return(list("pred"=save.pred,"errors"=errors,"save.importance"=save.importance))
}
=======
runrf=function(Y,indice,lag,selected){
#  comp=princomp(scale(Y,scale=FALSE))
#  Y2=cbind(Y,comp$scores[,1:4])
  Y2=Y
  
  aux=embed(Y2,5+lag)
  y=aux[,1]
  X=aux[,-c(1:(ncol(Y2)*lag))]  
  
  if(lag==1){
    X.out=tail(aux,1)[1:ncol(X)]  
  }else{
    X.out=aux[,-c(1:(ncol(Y2)*(lag-1)))]
    X.out=tail(X.out,1)[1:ncol(X)]
  }
  
  model=randomForest(X[,selected],y,importance=TRUE)
  pred=predict(model,X.out[selected])
    
  return(list("model"=model,"pred"=pred))
}


rf.rolling.window=function(Y,nprev,indice=1,lag=1,selected){
  
  save.importance=list()
  save.pred=matrix(NA,nprev,1)
  for(i in nprev:1){
    Y.window=Y[(1+nprev-i):(nrow(Y)-i),]
    lasso=runrf(Y.window,indice,lag,selected)
    save.pred[(1+nprev-i),]=lasso$pred
    save.importance[[i]]=importance(lasso$model)
    cat("iteration",(1+nprev-i),"\n")
  }
  
  real=Y[,indice]
  plot(real,type="l")
  lines(c(rep(NA,length(real)-nprev),save.pred),col="red")
  
  rmse=sqrt(mean((tail(real,nprev)-save.pred)^2))
  mae=mean(abs(tail(real,nprev)-save.pred))
  errors=c("rmse"=rmse,"mae"=mae)
  
  return(list("pred"=save.pred,"errors"=errors,"save.importance"=save.importance))
}
>>>>>>> 3c6f8a671c98dbedc98801ab2460f54e27383541
