library(car)

response <- c(93,120,65,105,115,82,99,87,100,90,78,95,93,88,110, 85,45,80,28,75,70,65,55,50,40, 100,75,65,40,73,65,50,30,45,50,45,55, 96,58,95,90,65,80,85,95,82)
sites <- factor(c(rep(1,15), rep(2, 10), rep(3, 12), rep(4, 9)))
data <- data.frame(Response=response, Site=sites)

#a
boxplot(Response~Site, data=data)
datamean <-tapply(data$Response, data$Site, mean)
points(1:4, datamean)

#b
qqdata <- data$Response-datamean[as.numeric(data$Site)]
qqnorm(qqdata)+qqline(qqdata)

leveneTest(Response~Site, data=data)

#c
summary(aov(Response~Site, data=data))
