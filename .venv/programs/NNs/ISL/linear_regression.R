LoadLibraries <- function() {
    library(MASS)
    library(ISLR2)
    print("Libraries successfully loaded")
}

LoadLibraries()

# Allows use of column headers without referencing Boston everytime
#attach(Boston)

# Single regression
# Linear Model
#lm.fit <- lm(medv ~ lstat)
#plot(lstat, medv)

# draws line
#abline(lm.fit)

#Plot in 2x2 matrix
#par(mfrow = c(2, 2))
#plot(lm.fit)

# plot residuals
#plot(predict(lm.fit), residuals(lm.fit))
#plot(predict(lm.fit), rstudent(lm.fit))

# Leverage statistics
#plot(hatvalues(lm.fit))
#print(which.max(hatvalues(lm.fit)))


# Multiple regression
#lm.fit <- lm(medv ~ . -age, data = Boston)
#par(mfrow = c(2, 2))
#plot(lm.fit)
#print(summary(lm.fit))


# Interaction terms
#lm.fit <- lm(medv ~ lstat * age)
#par(mfrow = c(2, 2))
#plot(lm.fit)
#print(summary(lm.fit))


# Non-linear transformation
#lm.fit2 <- lm(medv ~ lstat + I(lstat^2))
#lm.fit5 <- lm(medv ~ poly(lstat, 5))
#par(mfrow = c(2, 2))
#plot(lm.fit5)
#print(summary(lm.fit5))
#print(anova(lm.fit, lm.fit2))


# Qualititve predictors
#attach(Carseats)
#lm.fit <- lm(Sales ~ . + Income:Advertising + Price:Age, data = Carseats)
#par(mfrow = c(2, 2))
#plot(lm.fit)
#print(summary(lm.fit))

#contrasts(ShelveLoc)