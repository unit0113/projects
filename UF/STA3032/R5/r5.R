setwd("C:/Users/unit0/Desktop/Python/projects/UF/STA3032/R5")
data <- read.csv("mydata.csv", header = TRUE)

plot(data$x, data$y, main="Scatter Plot", xlab="X", ylab="Y")

reg <- lm(y~x, data=data)
plot(data$x, data$y, main="Scatter Plot", xlab="X", ylab="Y", abline(reg))
summary(reg)

confint(reg, 'x', level=0.95)

min(data$x)
max(data$x)
predict.lm(reg, se.fit=TRUE, newdata = data.frame(x=20), interval="confidence", level=0.95)


data$xsq <- data$x * data$x
plot(data$xsq, data$y, main="Scatter Plot", xlab="X", ylab="Y")
reg2 <- lm(y~xsq, data=data)
plot(data$xsq, data$y, main="Scatter Plot", xlab="X", ylab="Y", abline(reg2))
summary(reg2)
confint(reg2, 'xsq', level=0.95)

predict.lm(reg2, se.fit=TRUE, newdata = data.frame(xsq=20*20), interval="confidence", level=0.95)


reg3 <- lm(y~x+xsq, data=data)
summary(reg3)
