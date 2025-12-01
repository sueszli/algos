from typing import List


def isValid(s: str) -> bool:
    stack: List[str] = []
    l = ["{", "[", "("]
    r = ["}", "]", ")"]

    for elem in s:
        assert (elem in l) or (elem in r)
        if elem in l:
            stack.append(elem)
        elif elem in r:
            if len(stack) == 0:
                return False
            ir = r.index(elem)
            il = l.index(stack.pop())
            if ir != il:
                return False

    return len(stack) == 0


if __name__ == "__main__":
    arg = "()[]"
    result = isValid(arg)
    print(f"{arg} --> {result}")
