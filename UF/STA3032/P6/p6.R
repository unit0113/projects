library(reshape2)
library(car)

#13.6
response <- c(1.06,0.79,0.82,0.89,1.05,0.95,0.65,1.15,1.12, 1.58,1.45,0.57,1.16,1.12,0.91,0.83,0.43, 0.29,0.06,0.44,0.55,0.61,0.43,0.51,0.1,0.53,0.34,0.06,0.09,0.17,0.17,0.6)
solvents <- factor(c(rep(1,9), rep(2,8), rep(3,15)))
data <- data.frame(Response=response, Solvents=solvents)

leveneTest(Response~Solvents, data=data)

reg <- aov(Response~Solvents, data=data)
summary(reg)

#13.1
response <- c(17.5,16.9,15.8,18.6, 16.4,19.2,17.7,15.4, 20.3,15.7,17.8,18.9, 14.6,16.7,20.8,18.9, 17.5,19.2,16.5,20.5, 18.3,16.2,17.5,20.1)
machines <-factor(rep(c(1:6), each=4))
data <- data.frame(Response=response, Machines=machines)

model <- aov(Response~Machines, data=data)
TukeyHSD(model, conf.level = 0.95)

summary(model)
