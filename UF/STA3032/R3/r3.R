data <- c(20.2, 22.9, 23.3, 20.0, 19.4, 22.0, 22.1, 22.0, 21.9, 21.5, 20.9)
hist(data)
t.test(data, alternative = "two.sided", mu=23)
t.test(data, alternative = "two.sided", mu=23, conf.level=0.98)

qqnorm(data, pch = 1, frame = FALSE)
qqline(data, col = "steelblue", lwd = 2)
