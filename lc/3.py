from collections import deque


# deque: simple, good perf
def window_slider_deque(s: str) -> int:
    res = 0
    q = deque()

    for c in s:
        while c in q:
            q.popleft()

        q.append(c)
        res = max(res, len(q))

    return res


# set: simple, bad perf
def window_slider_set(s: str) -> int:
    char_set = set()
    max_length = 0
    j = 0

    for i in range(len(s)):
        while s[i] in char_set:
            char_set.remove(s[j])
            j += 1

        char_set.add(s[i])
        max_length = max(max_length, i - j + 1)

    return max_length


# dict: complex, good perf
def window_slider_dict(s: str) -> int:
    char_index = dict()  # { char: index + 1 }
    max_length = 0
    j = 0

    for i in range(len(s)):
        if s[i] in char_index:
            j = max(char_index[s[i]], j)

        char_index[s[i]] = i + 1  # index for j in case of repeat
        max_length = max(max_length, i - j + 1)

    return max_length


input = "abccd"
window_slider_deque(input)
window_slider_set(input)
window_slider_dict(input)
