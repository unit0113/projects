data <- read.table(header=TRUE, text='
Vital	Group
4.70	Brass
4.60	Brass
4.30	Brass
4.50	Brass
5.50	Brass
4.90	Brass
5.30	Brass
4.20	Control
4.70	Control
5.10	Control
4.70	Control
5.00	Control
                   ')
data
t.test(Vital~Group, data, conf.level=0.95, alternative="greater")

t.test(Vital~Group, data, conf.level=0.95, alternative="greater", var.equal=TRUE)
