from typing import Optional
import math


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def addTwoNumbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    temp = ListNode()
    current = temp
    p1, p2 = l1, l2
    carry = 0

    while p1 or p2 or carry:
        total = (p1.val if p1 else 0) + (p2.val if p2 else 0) + carry
        store = total % 10
        carry = total // 10

        if p1:
            p1 = p1.next
        if p2:
            p2 = p2.next
        current.next = ListNode(store)
        current = current.next

    return temp.next  # move 1 node, to get actual head


l1 = ListNode(2)
l1.next = ListNode(4)
l1.next.next = ListNode(3)

l2 = ListNode(5)
l2.next = ListNode(6)
l2.next.next = ListNode(4)

# sum
sum = addTwoNumbers(l1, l2)
print("solution: ")
while sum != None:
    print(sum.val, end="")
    sum = sum.next
