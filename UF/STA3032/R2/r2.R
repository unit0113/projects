# 4.1
# A
my_data <- c(6,9,7,3,6,7,9,6,3,6,6,7,1,9,1)
my_data == 6
my_data >= 6
my_data < 6 + 2
my_data != 6

# B
new_data = my_data[4:15]
new_data
dim(new_data) <- c(2,2,3)
new_data
new_data <= 6/2 + 4
new_data <- new_data + 2
new_data
new_data <= 6/2 + 4

# C
i10 <- diag(10)
i10
i10 == 0

# D
new_data = my_data[4:15]
new_data
dim(new_data) <- c(2,2,3)
new_data
new_data_i <- new_data <= 6/2 + 4
new_data_i
any(new_data_i)
all(new_data_i)
new_data <- new_data + 2
new_data
new_data_ii <- new_data <= 6/2 + 4
new_data_ii
any(new_data_ii)
all(new_data_ii)

# E
i10
any(i10[lower.tri(i10)])
any(i10[upper.tri(i10)])

# 16.2
# A
1-ppois(100,107)
dpois(0,107)
ppois(150,107) - ppois(60,107)
