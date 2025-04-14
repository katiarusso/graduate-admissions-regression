#DOWNLOAD LIBRARIES
install.packages("tidyverse") #for data cleaning
library(tidyverse)
install.packages("caret") #for cross validation
library(caret)

#SECTION 1

#READ THE DATASET
data<-read.csv("Admission_Predict_Ver1.1.csv")
View(data)
names(data)

#SECTION 2

#DATA CLEANING

data<-data[ ,-1] #drops the first column, as it is just a counter keeping track of the number of entries
sum(is.na(data)) #checks how many missing values --> returns 0 in our case
str(data) #returns the structure of our dataset to view variables' datatypes

#boxplots for handling outliers in continuos variables

boxplot(data$GRE.Score, main="Boxplot for GRE Scores ",ylab="GRE Score", col="lightblue")

boxplot(data$TOEFL.Score, main="Boxplot for TOEFL Scores ",ylab="TOEFL Score", col="lightgreen")

boxplot(data$CGPA, main="Boxplot for CGPA Scores ",ylab="CGPA Score", col="lightcoral")

#check whether Research is a Binary Variable or we need to use factor() 
unique(data$Research) #returns the unique values of the variable, since it returns 0,1 we confirm it's Bin

#histograms to verify scale of continuos variables

hist(data$GRE.Score, main="GRE Scores Distributions", xlab="GRE Score", col="lightblue")

hist(data$TOEFL.Score, main="TOEFL Scores Distributions", xlab="TOEFL Score", col="lightgreen")

hist(data$CGPA, main="CGPA Scores Distributions", xlab="CGPA Score", col="lightcoral")

#DATA VISUALIZATION

summary(data) #returns a summary of our dataset

#scatterplots of Chance of Admission against the predictors

plot(data$GRE.Score,xlab="GRE Score",data$Chance.of.Admit,ylab="Chance of Admission",main="Scatterplot")
reg1<-lm(data$Chance.of.Admit~data$GRE.Score)
summary(reg1)
abline(reg1,col="red")

plot(data$TOEFL.Score ,xlab="TOEFL Score",data$Chance.of.Admit,ylab="Chance of Admission",main="Scatterplot")
reg2<-lm(data$Chance.of.Admit~data$TOEFL.Score)
summary(reg2)
abline(reg2,col="red")

plot(data$University.Rating ,xlab="University Rating",data$Chance.of.Admit,ylab="Chance of Admission",main="Scatterplot")
reg3<-lm(data$Chance.of.Admit~data$University.Rating)
summary(reg3)

plot(data$SOP,xlab="SOP",data$Chance.of.Admit,ylab="Chance of Admission",main="Scatterplot")
reg4<-lm(data$Chance.of.Admit~data$SOP)
summary(reg4)

plot(data$LOR,xlab="LOR",data$Chance.of.Admit,ylab="Chance of Admission",main="Scatterplot")
reg5<-lm(data$Chance.of.Admit~data$LOR)
summary(reg5)

plot(data$CGPA,xlab="CGPA",data$Chance.of.Admit,ylab="Chance of Admission",main="Scatterplot")
reg6<-lm(data$Chance.of.Admit~data$CGPA)
summary(reg6)
abline(reg6,col="red")

plot(data$Research,xlab="Research",data$Chance.of.Admit,ylab="Chance of Admission",main="Scatterplot")
reg7<-lm(data$Chance.of.Admit~data$Research)
summary(reg7)

#SECTION 3

#REGRESSION MODEL 

#multiple linear regression including every predictor
mreg<-lm(data$Chance.of.Admit~data$GRE.Score+data$TOEFL.Score+data$University.Rating+data$SOP+data$LOR+data$CGPA+data$Research,data=data)
summary(mreg)

#VALIDITY ASSUMPTIONS

res<-residuals(mreg) #creates a variable for the residuals of the multiple linear regression

plot(mreg$fitted.values,xlab="Fitted Values" ,res,ylab="Residuals",main="Residuals vs Fitted Values")
abline(h="0",col="red")

qqnorm(res,main="Q-Q Plot of Residuals")
qqline(res,col="red")
shapiro.test(res) #Shapiro-Wilk test for normality assumption

hist(res,probability=TRUE,xlab="Residuals",main="Histogram of Residuals",col="darkgrey")
curve(dnorm(x, mean = mean(res), sd = sd(res)), col = "red", add = TRUE)

cor(data) #returns the correlation between predictors

#INTERACTION

#interaction plot between University Rating and Chance of Admission

interaction.plot(data$University.Rating,xlab="University Rating", data$Research,trace.label = "Research", data$Chance.of.Admit,ylab="Chance of Admission",type="b",col=c("purple","darkgreen"),lwd=2,pch=c(18,24),main="Interaction Plot",legend=FALSE)
legend("topright", legend = c("1", "0"), title = "Research",col = c("darkgreen", "purple"), pch = c(24, 18), lty = 1, lwd = 2, bty = "o")

#regression model with interaction effect
interaction_model <- lm(data$Chance.of.Admit ~ data$Research*data$University.Rating + data$CGPA + data$TOEFL.Score + data$SOP+data$LOR+data$GRE.Score)
summary(interaction_model)
anova(mreg,interaction_model) #anova to determine whether we have significant evidence for the interaction effect

#SECTION 4

#MODEL SELECTION

step_model<-step(interaction_model, direction="backward") #performs STEP DOWN
summary(step_model)

AIC(interaction_model,step_model) #performs AIC
BIC(interaction_model,step_model) #performs BIC

#VALIDITY ASSUMPTION

res1<-residuals(step_model) #creates a variable for the residuals of the multiple linear regression

plot(step_model$fitted.values,xlab="Fitted Values" ,res1,ylab="Residuals",main="Residuals vs Fitted Values")
abline(h="0",col="red")

qqnorm(res1,main="Q-Q Plot of Residuals")
qqline(res1,col="red")
shapiro.test(res1) #Shapiro-Wilk test for normality assumption

hist(res1,probability=TRUE,xlab="Residuals",main="Histogram of Residuals",col="khaki")
curve(dnorm(x, mean = mean(res1), sd = sd(res1)), col = "red", add = TRUE)

#CROSS VALIDATION

set.seed(123) #ensures reproducibility of the results

train_control<-trainControl(method= "cv",number=10) #10-fold cross validation

#train the model using the 10-fold cross-validation
kfold_model<-train(Chance.of.Admit ~  CGPA + TOEFL.Score + LOR + GRE.Score + Research*University.Rating,data=data,method="lm",trControl=train_control)
print(kfold_model$results) #returns the performance of the model
