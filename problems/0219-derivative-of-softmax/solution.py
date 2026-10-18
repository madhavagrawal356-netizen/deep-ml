def softmax_derivative(x: list[float]) -> list[list[float]]:
    if not x:
        return []

    m = max(x)                                   # shift for numerical stability
    e2 = 2.718281828459045                       # Euler's number, in place of math.exp
    e = [e2 ** (v - m) for v in x]
    total = sum(e)
    s = [v / total for v in e]                   # softmax output

    n = len(x)
    J = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(s[i] * (1 - s[i]))
            else:
                row.append(-s[i] * s[j])
        J.append(row)
    return J
