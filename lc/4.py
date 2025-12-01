from collections import deque


def get_median_merge1(nums1, nums2):
    merged = []
    deque1 = deque(nums1)
    deque2 = deque(nums2)

    is_empty = lambda x: len(x) == 0
    peekleft = lambda d: d[0] if len(d) > 0 else None

    while not is_empty(deque1) or not is_empty(deque2):
        pl1, pl2 = peekleft(deque1), peekleft(deque2)
        winner = deque1 if pl2 is None or (pl1 is not None and pl1 < pl2) else deque2
        merged.append(winner.popleft())

    get_median = lambda x: (x[len(x) // 2] + x[len(x) // 2 - 1]) / 2 if len(x) % 2 == 0 else x[len(x) // 2]
    print(f"{merged} -> {get_median(merged)}")
    return get_median(merged)


def get_median_merge2(nums1, nums2):
    merged = nums1 + nums2
    merged.sort()
    get_median = lambda x: (x[len(x) // 2] + x[len(x) // 2 - 1]) / 2 if len(x) % 2 == 0 else x[len(x) // 2]
    return get_median(merged)


def get_median_merge3(nums1, nums2):
    n = len(nums1)
    m = len(nums2)
    i = 0
    j = 0
    curr = 0
    prev = 0

    # only step half way
    for _ in range(0, (n + m) // 2 + 1):
        prev = curr
        print(f"i={i}, j={j}, curr={curr}, prev={prev}")

        if i < n and j < m:
            # both have elems
            if nums1[i] > nums2[j]:
                curr = nums2[j]
                j += 1
            else:
                curr = nums1[i]
                i += 1
        elif i < n:
            # only nums1 has elems
            curr = nums1[i]
            i += 1
        else:
            # only nums2 has elems
            curr = nums2[j]
            j += 1

    print(f"final curr={curr}, prev={prev}")
    is_odd = (n + m) % 2 == 1
    if is_odd:
        return float(curr)
    else:
        return (float(curr) + float(prev)) / 2.0


nums1 = [0, 2, 3, 5]
nums2 = [1, 4]
result = get_median_merge3(nums1, nums2)
