def karatsuba_mult(num1: int, num2: int) -> int:
    n = min(len(str(num1)), len(str(num2)))
    if n == 1:
        return num1 * num2
    else:
        a = num1 // (10 ** (n // 2))
        b = num1 % (10 ** (n // 2))
        c = num2 // (10 ** (n // 2))
        d = num2 % (10 ** (n // 2))
        p = a + b
        q = c + d

        ac = karatsuba_mult(a, c)
        bd = karatsuba_mult(b, d)
        pq = karatsuba_mult(p, q)
        adbc = pq - ac - bd

        return ac * (10 ** n) + (adbc) * 10 ** (n // 2) + bd


print(karatsuba_mult(56, 12))
print(karatsuba_mult(5678, 1234))