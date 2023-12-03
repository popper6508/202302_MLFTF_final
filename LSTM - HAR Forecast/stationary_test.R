library(urca)
library(readxl)
library(dplyr)
library(stringr)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))

### Data Read ###
data <- as.data.frame(read_xlsx("./Data/data_raw_weekly.xlsx")) %>% data.matrix()

data <- data[2:nrow(data), 2:ncol(data)]

######## ADF Test ########
## 1st ADF Test ##
rep <- ncol(data)
t.stat <- NULL

for (i in 1:rep){
  test <- data[,i] %>% 
    as.matrix()
  var <- colnames(test)
  
  adf.t=ur.df(test)
  result <- cbind(var, adf.t@teststat) %>% 
    as.data.frame()
  
  t.stat <- rbind(t.stat, result)
}

### 1st ADF Test Result ###
cval <- adf.t@cval # Critical Value

adf.stat <- t.stat[2] %>% 
  as.matrix() %>% 
  as.numeric()

t.r <- adf.stat > cval[1,1] # adf 통계량과 C.value 비교하는 Logical 변수
t.r <- as.character(t.r)

test.result <- cbind(adf.stat, t.r)
rownames(test.result) <- colnames(data[-1])
test.result # FALSE if Stationary

######## KPSS Test ########
## KPSS Test ##
rep <- ncol(data)
t.stat <- NULL

for (i in 2:rep){
  test <- data[i] %>% 
    as.matrix()
  var <- colnames(test)
  
  kpss.t=ur.kpss(test)
  result <- cbind(var, kpss.t@teststat) %>% 
    as.data.frame()
  
  t.stat <- rbind(t.stat, result)
}

### KPSS Test Result ###
cval <- kpss.t@cval # Critical Value

kpss.stat <- t.stat[2] %>% 
  as.matrix() %>% 
  as.numeric()

t.r <- kpss.stat > cval[1,1] # kpss 통계량과 C.value 비교하는 Logical 변수
t.r <- as.character(t.r)

test.result <- cbind(kpss.stat, t.r)
rownames(test.result) <- colnames(data[-1])
test.result

######## PP Test ########
rep <- ncol(data)
t.stat <- NULL

for (i in 2:rep){
  test <- data[i] %>% 
    as.matrix()
  var <- colnames(test)
  
  pp.t=ur.pp(test)
  result <- cbind(var, pp.t@teststat) %>%
    as.data.frame()
  
  t.stat <- rbind(t.stat, result)
}

### pp Test Result ###
cval <- adf.t@cval # Critical Value

pp.stat <- t.stat[2] %>% 
  as.matrix() %>% 
  as.numeric()

t.r <- pp.stat > cval[1,1] # pp 통계량과 C.value 비교하는 Logical 변수
t.r <- as.character(t.r)

test.result <- cbind(pp.stat, t.r)
rownames(test.result) <- colnames(data[-1])
test.result
