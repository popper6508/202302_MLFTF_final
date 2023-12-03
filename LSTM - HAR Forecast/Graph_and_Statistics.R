##### Graph and Basic Statistics #####
library(readxl)
library(readr)
library(recipes)
library(timetk)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))
load('lstm_result.RData')
load('har_result.RData')

plot(log(data[2:nrow(data),2]), type = 'l', xlab = "time", ylab = "Index")
title("lnVKOSPI")

library(moments)
summary(log(data[2:nrow(data),2]))
sd(log(data[2:nrow(data),2]))
skewness(log(data[2:nrow(data),2]))
kurtosis(log(data[2:nrow(data),2]))

library(urca)
adf.t=ur.df(log(data[2:nrow(data),2]), )
adf.t@testreg

kpss.t=ur.kpss(log(data[2:nrow(data),2]))
kpss.t@teststat

pp.t=ur.pp(log(data[2:nrow(data),2]))
pp.t@testreg

plot(tail(log(data[2:nrow(data),2]), 112), type = 'l', ylab = "index", xlab = 'time')
lines(hier_lstm5_1$pred, type = 'l', col = 'red')
lines(lstm_result5_1$pred, type = 'l', col = 'blue')
title("Horizon 1")
legend(4, 2, legend=c("lnVKOSPI","LSTM", "Multi LSTM"), fill = c("black","blue","red"))

plot(tail(log(data[2:nrow(data),2]), 112), type = 'l', ylab = "index", xlab = 'time')
lines(hier_lstm5_2$pred, type = 'l', col = 'red')
lines(lstm_result5_2$pred, type = 'l', col = 'blue')
title("Horizon 2")

plot(tail(log(data[2:nrow(data),2]), 112), type = 'l', ylab = "index", xlab = 'time')
lines(hier_lstm5_3$pred, type = 'l', col = 'red')
lines(lstm_result5_3$pred, type = 'l', col = 'blue')
title("Horizon 3")

plot(tail(log(data[2:nrow(data),2]), 112), type = 'l', ylab = "index", xlab = 'time')
lines(hier_lstm5_4$pred, type = 'l', col = 'red')
lines(lstm_result5_4$pred, type = 'l', col = 'blue')
title("Horizon 4")
