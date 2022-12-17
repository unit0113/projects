def rect_int_mult(num1: int, num2: int) -> int:
    return _mult_helper(str(num1), str(num2))


def _mult_helper(s1: str, s2: str) -> int:
    if len(s1) == 1:
        return int(s1) * int(s2)
    else:
        a = s1[:len(s1)//2]
        b = s1[len(s1)//2:]
        c = s2[:len(s2)//2]
        d = s2[len(s2)//2:]

        ac = _mult_helper(a, c)
        ad = _mult_helper(a, d)
        bc = _mult_helper(b, c)
        bd = _mult_helper(b, d)

        return ac * (10 ** len(s1)) + (ad + bc) * 10 ** (len(s1) // 2) + bd


print(rect_int_mult(56, 12))
print(rect_int_mult(5678, 1234))