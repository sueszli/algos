class Solution:
    def distMoney(self, money: int, children: int) -> int:        
        money = money - children # constraint
        if money < 0:
            return -1

        satisfiable = money // 7
        carry = money % 7
        # all children get 8$
        if satisfiable == children and carry == 0:
            return children
        # avoid the last child getting 4 dollars, by borrowing 1$ from another one
        elif satisfiable == children - 1 and carry == 3:
            return children - 2
        # (a) too much money per child, so the last one will shoot beyond 8$
        # (b) just as much money per child as we previously computed
        else:
            return min(children - 1, satisfiable)
