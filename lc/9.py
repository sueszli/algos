import math


def is_palindrome_simple(x: int) -> bool:
    s = str(x)
    for i in range(int(len(s) / 2)):
        if s[i] != s[len(s) - (i + 1)]:
            return False
    return True


def is_palindrome_simple_v2(x: int) -> bool:
    return str(x) == str(x)[::-1]


def is_palindrome(x: int) -> bool:
    reversed = 0
    copy = x

    while copy != 0:
        digit = copy % 10
        reversed = reversed * 10 + digit
        copy //= 10

    return reversed == x


if __name__ == "__main__":
    num = 1234321
    result = is_palindrome(num)
    print(f"{num} --> {result}")
