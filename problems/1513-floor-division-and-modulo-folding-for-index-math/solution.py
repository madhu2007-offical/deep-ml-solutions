def term_bounds(terms, ranges):
    vmin = terms.get("", 0)
    vmax = terms.get("", 0)

    for v, coef in terms.items():
        if v == "" or coef == 0:
            continue

        lo, hi = ranges[v]
        a = coef * lo
        b = coef * (hi - 1)

        vmin += min(a, b)
        vmax += max(a, b)

    return vmin, vmax


def fold_div(terms, c, ranges):
    quotient = {"": 0}
    remainder = {}

    for v, coef in terms.items():
        if coef % c == 0:
            quotient[v] = coef // c
        else:
            remainder[v] = coef

    quotient = {"": quotient.get("", 0)}
    for v, coef in terms.items():
        if v != "" and coef % c == 0:
            quotient[v] = coef // c

    if remainder:
        rem_min, rem_max = term_bounds(remainder, ranges)
    else:
        rem_min, rem_max = 0, 0

    if 0 <= rem_min and rem_max < c:
        return quotient, None

    result = {"": remainder.get("", 0)}
    for v, coef in remainder.items():
        if v != "":
            result[v] = coef

    return quotient, result


def fold_mod(terms, c, ranges):
    remainder = {}

    for v, coef in terms.items():
        if coef % c != 0:
            remainder[v] = coef

    exact = True

    if remainder:
        vmin, vmax = term_bounds(remainder, ranges)
        exact = 0 <= vmin and vmax < c

    result = {}
    for v, coef in remainder.items():
        if v != "":
            result[v] = coef

    result[""] = remainder.get("", 0)

    return result, exact


def unflatten(shape, flat_terms, ranges):
    result = []
    stride = 1

    strides = [1] * len(shape)

    for j in range(len(shape) - 1, -1, -1):
        strides[j] = stride
        stride *= shape[j]

    for j, s in enumerate(strides):
        quotient, remainder = fold_div(flat_terms, s, ranges)

        if remainder is None:
            modulo, exact = fold_mod(quotient, shape[j], ranges)
        else:
            modulo, exact = fold_mod(
                {**quotient, **remainder}, shape[j], ranges
            )

        if remainder is None and exact:
            result.append(modulo)
        else:
            result.append("div/mod")

    return result