##### Graph and Basic Statistics #####
library(readxl)
library(readr)
library(recipes)
library(timetk)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))
load('lstm_result.RData')
load('har_result.RData')

data <- as.data.frame(read_xlsx("./Data/data_raw_weekly.xlsx")) %>% data.matrix()

data <- data[2:nrow(data), 2:ncol(data)]

plot(log(data[,2]), type = 'l', xlab = "time", ylab = "Index")
title("lnVKOSPI")

library(moments)
summary(log(data[,1]))
sd(log(data[,1]))
skewness(log(data[,1]))
kurtosis(log(data[,1]))

library(urca)
adf.t=ur.df(log(data[,1]), )
adf.t@testreg

kpss.t=ur.kpss(log(data[,1]))
kpss.t@teststat

pp.t=ur.pp(log(data[,1]))
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
