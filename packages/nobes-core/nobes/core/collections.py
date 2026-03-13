def update(left: set, right: set) -> set:
    left.update(right)
    return left


def difference(left: set, right: set) -> set:
    return set(left).difference(set(right))
