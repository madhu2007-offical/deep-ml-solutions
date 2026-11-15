import math

def pooled_stats(m1, s1, n1, m2, s2, n2):
    N = n1 + n2

    mean = (n1 * m1 + n2 * m2) / N

    variance = (
        (n1 - 1) * s1**2 +
        (n2 - 1) * s2**2 +
        n1 * (m1 - mean)**2 +
        n2 * (m2 - mean)**2
    ) / (N - 1)

    sd = math.sqrt(variance)

    return (mean, sd)