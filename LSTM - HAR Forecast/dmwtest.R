dmwtest<-function(real,f1,f2){

se1=(real-f1)^2  #MSE loss function from the first model
se2=(real-f2)^2  #MSE loss function from the second model

# may try abs(real-f1) for MAE

# may try QLIKE loss function 

#qlike1=RK/VF_garch-log(RK/VF_garch)-1;    RK: proxy for actual volatility, VF_garch: volatility forecast from GARCH
#qlike2=RK/VF_garchX-log(RK/VF_garchX)-1;



dt = se1 - se2   # forecast loss differential, 

# DMW test #

nf = length(real)

e=dt-mean(dt);
s0=t(e)%*%e/nf;
q=round(nf^(1/3)); #bandwidth
s2=0;
L=1;

while (L<=q){
  s1=0;
  t=L+1;
  while (t<=nf){
    sloop=(1-L/(q+1))*e[t]*e[t-L];
    s1=s1+sloop;
    t=t+1;
  } 
  s2=s2+s1/nf;
  L=L+1;
}

varcov=s0+2*s2;
se=sqrt(varcov);

DMW=sqrt(nf)*mean(dt)/se;   # DMW test statistic
mse1 = mean(se1);
mse2 = mean(se2);


PVAL <- 2 * pnorm(-abs(DMW))

return(c(DMW,PVAL,mse1,mse2))
}
