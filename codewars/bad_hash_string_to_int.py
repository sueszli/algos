# https://www.codewars.com/kata/596d93bd9b6a5df4de000049


def hash_str(string: str) -> int:
    a: int = sum(ord(c) for c in string)
    print(f"a = {a}")

    get_prev_diff = lambda string, i: ord(string[i]) - ord(string[i - 1]) if i > 0 else 0
    b: int = sum(get_prev_diff(string, i) for i, _ in enumerate(string))  # sum of diff with previous char
    b_simplified: int = sum(ord(b) - ord(a) for a, b in zip(string, string[1:]))
    print(f"b = {b}")

    c: int = (a | b) & (~a << 2)
    print(f"c = {c}")

    d: int = c ^ (32 * (string.count(" ") + 1))
    print(f"d = {d}")

    return d


def hash_str_simplified(s):
    a = sum(ord(c) for c in s)
    b = sum(ord(b) - ord(a) for a, b in zip(s, s[1:]))
    c = (a | b) & (~a << 2)
    return c ^ (32 * (s.count(" ") + 1))


print(hash_str("ca"))
print(hash_str_simplified("ca"))
