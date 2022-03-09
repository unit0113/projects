def merge_bookings(B1, B2):
    n1, n2, i1, i2 = len(B1), len(B2), 0, 0
    x = 0
    B = []
    
    while i1 + i2 < n1 + n2:
        if i1 < n1:
            k1, s1, t1 = B1[i1]
        if i2 < n2:
            k2, s2, t2 = B2[i2]
        
        if i2 == n2:
            k, s, x = k1, max(x, s1), t1
            i1 += 1
        elif i1 == n1:
            k, s, x = k2, max(x, s2), t2
            i2 += 1
        
        else:
            if x < min(s1, s2):
                x = min(s1, s2)
            
            if t1 <= s2:
                k, s, x = k1, x, t1
                i1 += 1
            elif t2 <= s1:
                k, s, x = k2, x, t2
                i2 += 1
            elif x < s2:
                k, s, x = k1, x, s2
            elif x < s1:
                k, s, x = k2, x, s1
            else:
                k, s, x = k1 + k2, x, min(t1, t2)
                if t1 == x: i1 += 1
                if t2 == x: i2 += 1
        
        B.append((k, s, x))

    B_ = [B[0]]
    for k, s, t in B[1:]:
        k_, s_, t_ = B_[-1]
        if k == k_ and t_ == s:
            B_.pop()
            s = s_
        B_.append((k, s, t))

    return B_


def satisfying_booking(R):
    '''
    Input:  R | Tuple of |R| talk request tuples (s, t)
    Output: B | Tuple of room booking triples (k, s, t)
              | that is the booking schedule that satisfies R
    '''
    if len(R) == 1:
        s, t = R[0]
        return ((1, s, t),)

    m = len(R) // 2
    R1, R2 = R[:m], R[m:]
    B1 = satisfying_booking(R1)
    B2 = satisfying_booking(R2)
    B = merge_bookings(B1, B2)

    return tuple(B)
