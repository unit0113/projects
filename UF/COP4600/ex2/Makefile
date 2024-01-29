default: math

mean.o: mean.c
	cc -c mean.c

math: mean.o
	ar cr libmath.a mean.o

clean:
	-rm -f libmath.a
	-rm -f mean.o
