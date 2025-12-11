setwd("C:/Users/unit0/Desktop/Python/projects/UF/STA3032/P5_2")
data <- read.csv("5_2.csv", header = TRUE)
data$Gender <- as.factor(data$Gender)

reg <- lm(Profit~Income+Gender+Family.Members, data=data)
summary(reg)

predictM <- data.frame(Income=50000, Gender='M', Family.Members=2)
predictF <- data.frame(Income=50000, Gender='F', Family.Members=2)

predict(reg, newdata=predictM)
predict(reg, newdata=predictF)

reg2 <- lm(Profit~Income+Gender+Family.Members+Income*Gender, data=data)
summary(reg2)

reg3 <- lm(Profit~Income:Gender, data=data)
summary(reg3)
