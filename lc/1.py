from typing import List


def two_sum_simple(nums: List[int], target: int) -> List[int]:
    result = [-1, -1]
    for i in range(len(nums)):
        for j in range(len(nums)):
            not_same = i != j
            is_target = nums[i] + nums[j] == target
            if not_same and is_target:
                return [i, j]
    return result


def two_sum_false(nums: List[int], target: int) -> List[int]:
    result = [-1, -1]

    # assumption: there are len(nums)*2 unique combinations as (i+j) == (j+i)
    # this is wrong! counter argument: 1+1 != 2+0
    checked = [False for i in range(len(nums) * 2 - 1)]

    for i in range(len(nums)):
        for j in range(len(nums)):
            not_same = i != j
            not_checked = checked[i + j] == False
            is_target = nums[i] + nums[j] == target
            if not_same and not_checked and is_target:
                return [i, j]
            checked[i + j] = True
    return result


def two_sum(nums: List[int], target: int) -> List[int]:
    result = [-1, -1]
    hashmap = {}

    for i in range(len(nums)):
        hashmap[nums[i]] = i

    for i in range(len(nums)):
        complement = target - nums[i]
        if complement in hashmap and hashmap[complement] != i:
            return [i, hashmap[complement]]

    return result


def two_sum_optimized(nums: List[int], target: int) -> List[int]:
    result = [-1, -1]
    hashmap = {}
    for i in range(len(nums)):
        complement = target - nums[i]
        if complement in hashmap:
            return [i, hashmap[complement]]
        hashmap[nums[i]] = i
    return result


if __name__ == "__main__":
    nums = [2, 5, 5, 11]
    target = 10
    result = two_sum(nums, target)
    print(f"{nums} with target {target} --> {result}")
