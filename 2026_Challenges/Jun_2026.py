###################################################
# 2144. Minimum Cost of Buying Candies With Discount
# 01JUN26
######################################################
class Solution:
    def minimumCost(self, cost: List[int]) -> int:
        '''
        sort, and take the first 2, then the third for free
        '''
        cost.sort(reverse = True)
        ans = 0
        n = len(cost)
        for i in range(n):
            if (i + 1) % 3 == 0:
                continue
            ans += cost[i]
        return ans