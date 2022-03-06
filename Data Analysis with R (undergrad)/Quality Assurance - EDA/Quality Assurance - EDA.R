---
title: "Quality Assurance EDA"
author: "Ali Baris"
date: "11/22/2021"
output: html_document
---

# The quality of a product depends on;
# A: density of a solution 
# B: acidity of the solution 
# C: treatment time
# D: temperature
# 
# On each row of the dataset, those four factors were set to certain levels, 1000
# batches at those levels were produced, and the percentage of unacceptable batches out of 1000 were
# reported in column P. We would like to understand the relation between production factors and the
# quality of product.


library(magrittr)
library(tidyverse)
library(modelr)
library(pander)
library(leaps)
library(ISLR)

library(readxl)
d <- read_excel("experiment.xlsx")
d %>% head() %>% pander(caption="Here is a glimpse of data that we will 
analyze.")

summary(d)

quant <- d %>% select_if(is.numeric)
names(quant)

cor(d)

# The table above suggests that only predictor variables 
# are not correlated with each other. B (acidity of the solution) is the only 
# one that is correlated with P (percentage of unacceptable batches) to a certain
# extent. Correlation coefficient is -0.6.

library(corrplot)

d %>% 
  select_if(is.numeric) %>% 
  cor(use = "pairwise.complete.obs") %>% 
  corrplot(type = "lower", diag = FALSE, order = "hclust")

# Above graph verifies our statement. Except B and P, no other 
# variables are correlated with each other. And correlation 
# between B and P is also not very strong (60% correlation). 
# One can only say that there is very weak correlation between 
# the pairs (A,P), (C,P) and (D,P). (You see above that color 
# is very transparent.)
# Let us go into more detail and look at histograms and distributions.

panel.hist <- function(x, ...)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(usr[1:2], 0, 1.5) )
  h <- hist(x, plot = FALSE)
  breaks <- h$breaks; nB <- length(breaks)
  y <- h$counts; y <- y/max(y)
  rect(breaks[-nB], 0, breaks[-1], y, col = "cyan", ...)
}
panel.cor <- function(x, y, digits = 2, prefix = "", cex.cor, ...)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  rr <- cor(x, y)
  r <- abs(rr)
  txt <- format(c(rr, 0.123456789), digits = digits)[1]
  txt <- paste0(prefix, txt)
  if(missing(cex.cor)) cex.cor <- 0.8/strwidth(txt)
  text(0.5, 0.5, txt, cex = cex.cor * r)
}
d %>% 
  pairs(diag.panel = panel.hist,
        lower.panel = panel.cor,
        upper.panel = function (...) panel.smooth(..., lwd = 2, col = "
darkgray"))

##Finding 1

# By looking at the above set of plots and the data itself, 
# we see that, although A, B, C and D are continuous 
# variables, their ranges are very limited. In fact they 
# can be treated as factors, but in this case
# I will continue with assuming they are discrete-continuous 
# random variables.

##Finding 2

# When we look at the bottom-right plot, we seen that our 
# response variable P is not normally distributed
# and it is highly right-skewed. We should fix this 
# problem since many of the statistical methods we
# might perform assume that response variable, as well 
# as the residual errors, are normally distributed.
# Let us try log transformation to fix this issue. Also, 
# I"m taking P to first plot so that I can look at it more
# easily.

panel.hist <- function(x, ...)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(usr[1:2], 0, 1.5) )
  h <- hist(x, plot = FALSE)
  breaks <- h$breaks; nB <- length(breaks)
  y <- h$counts; y <- y/max(y)
  rect(breaks[-nB], 0, breaks[-1], y, col = "cyan", ...)
}
panel.cor <- function(x, y, digits = 2, prefix = "", cex.cor, ...)
{
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  rr <- cor(x, y)
  r <- abs(rr)
  txt <- format(c(rr, 0.123456789), digits = digits)[1]
  txt <- paste0(prefix, txt)
  if(missing(cex.cor)) cex.cor <- 0.8/strwidth(txt)
  text(0.5, 0.5, txt, cex = cex.cor * r)
}
d %>% 
  mutate(logP = log(P)) %>%
  rev() %>% 
  pairs(diag.panel = panel.hist,
        lower.panel = panel.cor,
        upper.panel = function (...) panel.smooth(..., lwd = 2, col = "
darkgray"))

##Finding 3

# Log transformation really did well and put P's distribution 
# into a symmetric form. That's why I will not
# try Box-Cox or any other transformation technique further.
# Let us look at another plot type, which is density plot:

d %>% 
  ggplot(aes(P,B)) +
  geom_point(col="gray") +
  geom_density_2d()

# I couldn't extract any information by looking at this graph. 
# I also tried with aes(P,A),aes(P,C),aes(P,D),
# but they didn't help either.
# I will continue with the current setting.

##Linear Regression

d2 <- d %>% mutate(P=log(P)) #applying log transformation
linmod <- lm(P~., data=d2) #regressing all variable against log(P)
summary(linmod)

## 
## Call:
## lm(formula = P ~ ., data = d2)
## 
## Residuals:
##      Min       1Q   Median       3Q      Max 
## -0.36036 -0.14399  0.01731  0.09264  0.27243 
## 
## Coefficients:
##               Estimate Std. Error t value Pr(>|t|)   
## (Intercept) 10.3152793 24.2485217   0.425  0.67468   
## A           -5.6857670 21.5876078  -0.263  0.79471   
## B           -0.3162772  0.0863504  -3.663  0.00137 **
## C           -0.0030675  0.0043175  -0.710  0.48488   
## D           -0.0006089  0.0028783  -0.212  0.83442   
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.1832 on 22 degrees of freedom
## Multiple R-squared:  0.3895, Adjusted R-squared:  0.2785 
## F-statistic: 3.509 on 4 and 22 DF,  p-value: 0.02321


# Before looking at diagnostic plots, we can say that only significant variable 
# (at a significance level of 99%) is B. Other variables' 
# p-values are high indicating that they are not significant. 
# The model's R^2 is not very good, it is 39%. We should 
# improve the model. For this purpose, I will analyze effects 
# plots and diagnostic plots of the regression model.

library(effects)
allEffects(linmod, partial.residuals = TRUE) %>% plot()

##Finding 4

# It seems like the predictor D has nothing to do with our 
# response variable P. Its explanatory power is
# very low. We should remove it. (after checking interactions!)

##Finding 5

# Top-left plot suggests that we may use a quadratic term 
# for the predictor A, as one can detect the
# U-shape in its effect plot. This is also valid for C.

##Finding 6

# It is clear that as B increases P decreases. This is also 
# verified by its negative coefficient.

linmod2 <- lm(log(P)~. + poly(A,2)+ poly(C,2), data=d2) 
#adding quadratic terms for A and C
summary(linmod2)

## 
## Call:
## lm(formula = log(P) ~ . + poly(A, 2) + poly(C, 2), data = d2)
## 
## Residuals:
##       Min        1Q    Median        3Q       Max 
## -0.110081 -0.021915 -0.006611  0.034123  0.083778 
## 
## Coefficients: (2 not defined because of singularities)
##               Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  2.7608278  7.7205243   0.358 0.724392    
## A           -1.1661782  6.8733118  -0.170 0.866976    
## B           -0.1191192  0.0274932  -4.333 0.000323 ***
## C           -0.0011943  0.0013747  -0.869 0.395268    
## D           -0.0002014  0.0009164  -0.220 0.828258    
## poly(A, 2)1         NA         NA      NA       NA    
## poly(A, 2)2  0.1783472  0.0583220   3.058 0.006208 ** 
## poly(C, 2)1         NA         NA      NA       NA    
## poly(C, 2)2 -0.0654453  0.0583220  -1.122 0.275096    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.05832 on 20 degrees of freedom
## Multiple R-squared:  0.6017, Adjusted R-squared:  0.4822 
## F-statistic: 5.036 on 6 and 20 DF,  p-value: 0.002706

##Finding 6
# 
# Summary result showed us that quadratic term for the
# predictor A worked well, and our model's R^2 is
# increased from 39% to 60%. (Adjusted R^2 is still low!(48%)).
# On the other hand, quadratic term of C did not provide us
# any additional explanatory power. We don't need it.
# We should improve model further. I will look at interactions.
# Additionally, I am adding quadratic termfor the predictor B.
# (its effect plot gives us a hint)

linmod3 <- lm(P~. + poly(A,2)*D+D*poly(B,2), data=d2) 
#adding interaction between D and others.
summary(linmod3)

# Our model significantly got worse. We shouldn't try this.
# Continuing with the existing model, i.e. linmod2. 
# I am applying step() function to perform stepwise
# selection.

linmod2 %>% step()

## Start:  AIC=-147.56
## log(P) ~ A + B + C + D + poly(A, 2) + poly(C, 2)
## 
## 
## Step:  AIC=-147.56
## log(P) ~ A + B + D + poly(A, 2) + poly(C, 2)
## 
## 
## Step:  AIC=-147.56
## log(P) ~ B + D + poly(A, 2) + poly(C, 2)
## 
##              Df Sum of Sq      RSS     AIC
## - D           1  0.000164 0.068193 -149.49
## - poly(C, 2)  2  0.006850 0.074880 -148.97
## <none>                    0.068029 -147.56
## - poly(A, 2)  2  0.031906 0.099935 -141.18
## - B           1  0.063852 0.131881 -131.69
## 
## Step:  AIC=-149.49
## log(P) ~ B + poly(A, 2) + poly(C, 2)
## 
##              Df Sum of Sq      RSS     AIC
## - poly(C, 2)  2  0.006850 0.075044 -150.91
## <none>                    0.068193 -149.49
## - poly(A, 2)  2  0.031906 0.100099 -143.13
## - B           1  0.063852 0.132046 -133.65
## 
## Step:  AIC=-150.91
## log(P) ~ B + poly(A, 2)
## 
##              Df Sum of Sq      RSS     AIC
## <none>                    0.075044 -150.91
## - poly(A, 2)  2  0.031906 0.106950 -145.34
## - B           1  0.063852 0.138896 -136.29

## 
## Call:
## lm(formula = log(P) ~ B + poly(A, 2), data = d2)
## 
## Coefficients:
## (Intercept)            B  poly(A, 2)1  poly(A, 2)2  
##    1.376245    -0.119119    -0.009895     0.178347

##Finding 7

# Step() function performed stepwise selection with respect to AIC values and suggests the model with B
# and  as predictors.

# I will continue with this model (linmod2). I couldn't improve it further.
# Let's check its diagnostic plots:

plot(linmod2)

# There is pattern in the Residual vs. fitted plot, this 
# implies that there is correlation between error terms.
# Unfortunately log transformation didnt help to fix this 
# issue. On the other hand, Normal Q-Q plot tells us that 
# errors more or less follow normal distribution, as they
# lie on the straight line. Scale location does not show us 
# any pattern, which means the model is OK. I will revisit 
# Leverage plot in later questions.

## Evaluation of the model

# We may look at R^2 and/or 
# RSE/mean(P).

names(summary(linmod2))
summary(linmod2)$r.squared

# R^2 is 60%. This means that 60% of the variation in the 
# response variable is explained by the predictor
# variables. 60% is not a perfect R^2 value but it is not 
# the worst either. Let us also look at RSE/mean(P).

(summary(linmod2)$sigma)/mean(d2$P)

# We want this error percentage to be as low as possible. 
# It is 5% which means the model gives somewhat accurate results.

# Effect of temperature (D) on P and other variables: 

plot(d2$P,d2$D)

# Plot tells us nothing. This is actually what we expected 
# because previously I concluded that D is not a
# significant variable. Nonetheless, I want to check its 
# marginal relation (in isolation of other variables)
# anyway.

linmod.onlyD <- lm(P~D,data=d2)
summary(linmod.onlyD)

## 
## Call:
## lm(formula = P ~ D, data = d2)
## 
## Residuals:
##      Min       1Q   Median       3Q      Max 
## -0.29529 -0.14605 -0.02474  0.14578  0.42623 
## 
## Coefficients:
##               Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  2.7004947  0.4681398   5.769 5.18e-06 ***
## D           -0.0006089  0.0034535  -0.176    0.861    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.2198 on 25 degrees of freedom
## Multiple R-squared:  0.001242,   Adjusted R-squared:  -0.03871 
## F-statistic: 0.03108 on 1 and 25 DF,  p-value: 0.8615

# The p-value is more than 5% or 10%. This means that we are 
# unable to reject the null hypothesis of
# 'H_o:D's beta=0, i.e. it is not significant for the model'. 
# We conclude that D cannot be used to explain the variation 
# in P. To answer the latter question, one can look at the 
# correlation graph below.

d2 %>% 
  select_if(is.numeric) %>% 
  cor(use = "pairwise.complete.obs") %>% 
  corrplot(type = "lower", diag = FALSE, order = "hclust")

## Finding 8

# D is not correlated with any of the variables. I also 
# checked the interaction between them in linmod3
# previously. One can confidently say that D is not useful 
# for our analysis.

## Investigating outliers

# By looking at the leverage plot we can see 22 and 14 have 
# Cook's Distance of 0.3. They are influential.
# Let us check what happens if we remove them.

linmod4 <- linmod2 %>% update(subset = -c(22, 14))
linmod4 %>% summary()

# Our R^2 is increased to 69%. We improved the model. 
# Also quadratic term of C became significant.
# That means those observations (22 and 14) apply strong 
# force to the regression line. 

# Self-criticism: I should have checked the diagnostic plots
# again, because the data is changed.

#--------------------------------------------------------

## Binomial Regression
#Interpretation
#Firstly we need to add a binary outcome data to data to perform logistic regression. 
#I do it as follows;

mean(d$P)
d3<- d%>%mutate(Perc.above.average=as.factor(ifelse(P>=mean(P),"Yes","N
o"))) %>% select(-P)
d3

## # A tibble: 27 x 5
##        A     B     C     D Perc.above.average
##    <dbl> <dbl> <dbl> <dbl> <fct>             
##  1  1.12   3.5    40   150 No                
##  2  1.12   3.5    50   120 No                
##  3  1.12   3.5    30   135 Yes               
##  4  1.12   3      40   120 Yes               
##  5  1.12   3      50   135 Yes               
##  6  1.12   3      30   150 No                
##  7  1.12   4      40   135 Yes               
##  8  1.12   4      50   150 No                
##  9  1.12   4      30   120 No                
## 10  1.12   3.5    40   120 Yes               
## # ... with 17 more rows


res_logit <- glm(Perc.above.average ~ ., data = d3, family = binomial(link = logit))
res_null <- glm(Perc.above.average ~1,data=d3, family=binomial)
summary(res_logit)

## 
## Call:
## glm(formula = Perc.above.average ~ ., family = binomial(link = logit), 
##     data = d3)
## 
## Deviance Residuals: 
##     Min       1Q   Median       3Q      Max  
## -1.5006  -0.7791  -0.5718   1.0578   2.0726  
## 
## Coefficients:
##               Estimate Std. Error z value Pr(>|z|)  
## (Intercept)  176.94275  304.05177   0.582   0.5606  
## A           -147.95770  270.01225  -0.548   0.5837  
## B             -2.16577    1.14987  -1.883   0.0596 .
## C             -0.02959    0.05400  -0.548   0.5837  
## D             -0.01976    0.03606  -0.548   0.5838  
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## (Dispersion parameter for binomial family taken to be 1)
## 
##     Null deviance: 35.594  on 26  degrees of freedom
## Residual deviance: 30.747  on 22  degrees of freedom
## AIC: 40.747
## 
## Number of Fisher Scoring iterations: 4

## Finding 9

# Logistic regression also suggests that B is the only
# significant variable.

res_logit2 <- glm(Perc.above.average ~ B, data = d3, 
                  family = binomial
                  (link = logit))
summary(res_logit2)

# Let us perform analysis of "deviance" to check whether 
# the model is different than the null model.

anova(res_logit2,res_null,test='Chi')

# We can reject the "H_o:Proposed model and null model 
# and equivalent" and conclude that our model is
# better than the null model.

res_logit2 %>% step()

## Start:  AIC=35.61
## Perc.above.average ~ B
## 
##        Df Deviance    AIC
## <none>      31.612 35.612
## - B     1   35.594 37.594
## 
## Call:  glm(formula = Perc.above.average ~ B, family = binomial(link 
## = logit), 
##     data = d3)
## 
## Coefficients:
## (Intercept)            B  
##       6.677       -2.086  
## 
## Degrees of Freedom: 26 Total (i.e. Null);  25 Residual
## Null Deviance:       35.59 
## Residual Deviance: 31.61     AIC: 35.61

## Finding 10

# I again applied step() function and it suggests us to 
# use model with B as the only predictor.

## Model Evaluation

# To check goodness of fit in logistic regression models, 
# we use Hosmer-Lemeshow (H-L) test.

number_of_intervals <- 3
library(modelr)
d3 %>%  
  add_predictions(res_logit2, var = "pred", type = "response") %>% 
  mutate(pred_cut = cut(pred, breaks = c(0, quantile(pred, prob = 
       seq(number_of_intervals-1)/number_of_intervals), 1))) %>% 
  group_by(pred_cut) %>% 
  summarize(N = n(),
            S = sum(Perc.above.average == "Yes"),
            pbar = median(pred), .groups = "drop") %>% 
  mutate(mean = N*pbar, sd = sqrt(mean*(1-pbar)),
         Z = (S - mean)/sd) %>% arrange(pbar) %>% 
  summarize(HL = sum(Z^2), .groups = "drop") %>% 
  extract2("HL") -> HL
cat("HL pvalue", pchisq(HL, df = number_of_intervals - 1, lower.tail = 
                          FALSE), "\n")

## HL pvalue 0.740556

## Finding 11

# P-value of the H-L test is really high. We can not reject 
# the significance of the proposed model. So we
# can say that model that we propose (res_logit2) works well.
# Let us just perform H-L test with another number of intervals 
# to make sure our model is correct.

number_of_intervals <- 4
d3 %>%  
  add_predictions(res_logit2, var = "pred", type = "response") %>% 
  mutate(pred_cut = cut(pred, breaks = c(0, quantile(pred, prob = 
                      seq(number_of_intervals-1)/number_of_intervals), 1))) %>% 
  group_by(pred_cut) %>% 
  summarize(N = n(),
            S = sum(Perc.above.average == "Yes"),
            pbar = median(pred), .groups = "drop") %>% 
  mutate(mean = N*pbar, sd = sqrt(mean*(1-pbar)),
         Z = (S - mean)/sd) %>% arrange(pbar) %>% 
  summarize(HL = sum(Z^2), .groups = "drop") %>% 
  extract2("HL") -> HL
cat("HL pvalue", pchisq(HL, df = number_of_intervals - 1, lower.tail = 
                          FALSE), "\n")

## HL pvalue 0.8962703

# Again, we can not reject the significance of the proposed 
# model. We can say that model that we propose
# (res_logit2) works well.

## Effect of temperature on P and other variables:

logistic.with.onlyD <- glm(Perc.above.average ~ D, data = d3, 
                           family = 
                               binomial(link = logit))
summary(linmod.onlyD)

## 
## Call:
## lm(formula = P ~ D, data = d2)
## 
## Residuals:
##      Min       1Q   Median       3Q      Max 
## -0.29529 -0.14605 -0.02474  0.14578  0.42623 
## 
## Coefficients:
##               Estimate Std. Error t value Pr(>|t|)    
## (Intercept)  2.7004947  0.4681398   5.769 5.18e-06 ***
## D           -0.0006089  0.0034535  -0.176    0.861    
## ---
## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
## 
## Residual standard error: 0.2198 on 25 degrees of freedom
## Multiple R-squared:  0.001242,   Adjusted R-squared:  -0.03871 
## F-statistic: 0.03108 on 1 and 25 DF,  p-value: 0.8615

## Finding 12

# Percentage of unacceptable products does not change with D 
# as the variable is not significant again(p-value>0.05).

d2 %>% 
  select_if(is.numeric) %>% 
  cor(use = "pairwise.complete.obs") %>% 
  corrplot(type = "lower", diag = FALSE, order = "hclust")

# Again we conclude that D has nothing to do with A,B,C and P.

## Influential Points

plot(res_logit2)

#7 and 28 are influential observations that have high leverage and Cook"s distance.

#---------------------------------------------------

## Final Comments (which model is better?)

# I want to look at Test MSE of the linear model and create 
# a confusion matrix for the logistic regression
# to decide for this. First let us split data set into 
# training data set and test data set. This will make MSE
# calculation for the linear model more accurate.

trindexes <- (1:nrow(d) %>% sample())[1:round(3/4*nrow(d))]
d.training <- d %>% slice(trindexes)

d.test <- d %>% slice(setdiff(1:nrow(d),trindexes))
mse.linear <- mean(d.test$P-(exp(predict(linmod2,newdata=d.test))^2))
#exponentiating back to find predict real P value

## Warning in predict.lm(linmod2, newdata = d.test): prediction from a 
#rank-deficient fit may be misleading
mse.linear
## [1] 8.098517

# MSE for the linear model is 8.09.
# For logistic regression;

glm.probs=predict(res_logit2,type="response")
glm.pred=rep("No",nrow(d))
glm.pred[glm.probs >.5]="Yes" # I am using 0.5 as threshold,
#let us not bother ourselves with ROC/AUC here.
table(glm.pred,d3$Perc.above.average)

mean(glm.pred==d3$Perc.above.average)
## [1] 0.6666667

# Logistic model gives us 66% accuracy. One should 
# study its ROC and AUC to improve its threshold and to
# achieve a better model.

