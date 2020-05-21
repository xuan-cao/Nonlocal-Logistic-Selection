rm(list = ls())
source("ind_pmom.R")
source("SSS.R")
library(MASS)
library(truncnorm)
set.seed(0)

n = 100
p = 150
t = 4

#True regression coefficients
b = runif(t,0.5,1.5) #setting 1
b = rep(1.5,t) #setting 2
b = runif(t,1.5,3) #setting 3
b = rep(3,t) #setting 4

#print(b)

Bc = c(b, array(0, p-t)) #Coeffecients






Sigma = matrix(0,p,p)
for(j in 1:(p-1)){
  for(k in (j+1):p)
    Sigma[j,k] = Sigma[k,j] = 0.3^abs(j-k)
}
diag(Sigma) = 1

Sigma <- diag(p)


#Generating data
X <- mvrnorm(n,rep(0, p),Sigma=Sigma)
X <- scale(X)
y = rlogis(n, location = X %*% Bc)
y = ifelse(y>0,1,0)           # Logistic response E in the paper

#Testing set
n_test = 50
X_test = mvrnorm(n_test,rep(0, p),Sigma=Sigma)
Y_test = rlogis(n_test, location = X_test %*% Bc)
Y_test = ifelse(Y_test>0,1,0) 

#Run SSS
fit_de_SSS = SSS(X,y,ind_fun=pmom_laplace,N=150,C0=1,verbose=TRUE)
source("result.R")
res_de_SSS = result(fit_de_SSS)
print(res_de_SSS$hppm) # the MAP model


Evaluation <- function(beta1, beta2){
  true.index <- which(beta1==1)
  false.index <- which(beta1==0)
  positive.index <- which(beta2==1)
  negative.index <- which(beta2==0)
  
  TP <- length(intersect(true.index,positive.index))
  FP <- length(intersect(false.index,positive.index))
  FN <- length(intersect(true.index,negative.index))
  TN <- length(intersect(false.index,negative.index))
  
  
  Precision <- TP/(TP+FP)
  if((TP+FP)==0) Precision <- 1
  Recall <- TP/(TP+FN)
  if((TP+FN)==0) Recall <- 1
  Sensitivity <- Recall
  Specific <- TN/(TN+FP)
  if((TN+FP)==0) Specific <- 1
  MCC.denom <- sqrt(TP+FP)*sqrt(TP+FN)*sqrt(TN+FP)*sqrt(TN+FN)
  if(MCC.denom==0) MCC.denom <- 1
  MCC <- (TP*TN-FP*FN)/MCC.denom
  if((TN+FP)==0) MCC <- 1
  
  return(list(Precision=Precision,Recall=Recall,Sensitivity=Sensitivity,Specific=Specific,MCC=MCC,TP=TP,FP=FP,TN=TN,FN=FN))
}


X_train = X
Y_train = y
nonzerobetaid = (res_de_SSS$hppm)
traindata = cbind(Y_train, X_train[,nonzerobetaid])
library(glmnet)
fittedmodel <- glm(Y_train ~., data = as.data.frame(traindata), family = binomial)
predfitted <- predict(fittedmodel, newdata = as.data.frame(cbind(Y_test, X_test[,nonzerobetaid])), type="response")


MSPE_nonlocal = round(mean((Y_test - predfitted)^2), digits = 4)

nonlocalgamma = rep(0, p)
nonlocalgamma[nonzerobetaid] = 1
true_model = c(rep(1, t), rep(0, p - t))

#Evaluation variable selection performance
Evaluation(true_model, nonlocalgamma)

#Calculating MSPE
print(MSPE_nonlocal)
