###########################################
# 2561. Rearranging Fruits
# 03AUG25
############################################
class Solution:
    def minCost(self, basket1: List[int], basket2: List[int]) -> int:
        '''
        [4,2,2,2]
        [1,4,1,2]
        make arrays equal by swapping,
        but we can only swap at their indices
        first we need to check that we can do it
            this might actually be even harder than then current problem
        store counts of each fruit in map
        if counts1[x] + count2[x] is odd we can't do
        indices don't need to line up, we can pick any two indices (i from basket1 to j in bakset2)
        crux of the problem is to identify which fruits to swap
        https://leetcode.com/problems/rearranging-fruits/?envType=daily-question&envId=2025-08-02
        just use the minimum cost fruit to swapy with another fruit
            cost any way is just min(x,y), if we find the min, we can swap it
        '''
        #counts shoule be even first of all
        counts1 = Counter(basket1)
        counts2 = Counter(basket2)
        total_counts = counts1 + counts2
        #first check if we can do it
        for k,v in total_counts.items():
            if v % 2 == 1:
                return -1
            
        #look through all the fruits
        fruits_to_swap = []
        for k,v in total_counts.items():
            #needed for equal counts
            target = v // 2
            diff = counts1[k] - target
            #if diff > 0, surplus is basket 1, defecit in basket2
            #if diff < 0, defecit in basket 1, surplus in basket2
            for _ in range(abs(diff)):
                fruits_to_swap.append(k)

        fruits_to_swap.sort()
        min_val = min(total_counts.keys())
        min_cost = 0
        #we only need to swap half through fruits that are in either surplus or in defecit
        #its either a direct swap or indirect swap
        for i in range(len(fruits_to_swap) // 2):
            min_cost += min(fruits_to_swap[i],2*min_val)

        return min_cost


###########################################################
# 2106. Maximum Fruits Harvested After at Most K Steps
# 03AUG25
###########################################################
class Solution:
    def maxTotalFruits(self, fruits: List[List[int]], startPos: int, k: int) -> int:
        '''
        if i'm going in one direction, i should just pick up the fruits along the way
        but i only get k steps, when we reach a position, we get all the fruits
        optimal path only has one turn, left once, then right once, or right once then left once
        use prefix sums to get the ranges
        im stupid i dont need to offset, just roll up and use pref_sums to calculate the sums
        pref sums and suff sums get us the smounts, but how far from start should i go
            trying all x steps from start would be n*n
            no we just need to do steps from start
        '''
        max_position = 0
        for pos,amount in fruits:
            max_position = max(max_position, pos)
        
        pref_sum = [0]*(max_position+1)
        for pos,amount in fruits:
            pref_sum[pos] = amount
        
        for i in range(1,max_position+1):
            pref_sum[i] += pref_sum[i-1]
        
        ans = 0
        #try all k+1 steps from startPos to the right
        for x in range(k+1):
            right = min(startPos + x, max_position)
            left = max(0,startPos - (k - x))
            ans = max(ans, pref_sum[right] - pref_sum[left])

        #try all steps to the left
        for x in range(k+1):
            left = max(0,startPos - x)
            right = min(max_position, startPos + (k-x))
            ans = max(ans, pref_sum[right] - pref_sum[left])
        return ans

#phewww
class Solution:
    def maxTotalFruits(self, fruits: List[List[int]], startPos: int, k: int) -> int:
        '''
        if i'm going in one direction, i should just pick up the fruits along the way
        but i only get k steps, when we reach a position, we get all the fruits
        optimal path only has one turn, left once, then right once, or right once then left once
        use prefix sums to get the ranges
        im stupid i dont need to offset, just roll up and use pref_sums to calculate the sums
        pref sums and suff sums get us the smounts, but how far from start should i go
            trying all x steps from start would be n*n
            no we just need to do steps from start
        '''
        #fruits are sorted anyway
        max_pos = max(fruits[-1][0], startPos + k) + 2
        prefix = [0] * max_pos

        for pos, amount in fruits:
            prefix[pos + 1] += amount

        for i in range(1, max_pos):
            prefix[i] += prefix[i - 1]

        res = 0

        for steps_left in range(k + 1):
            left = max(startPos - steps_left, 0)
            right = max(startPos + (k - 2 * steps_left), 0)
            res = max(res, prefix[right + 1] - prefix[left])

        for steps_right in range(k + 1):
            right = startPos + steps_right
            left = max(startPos - (k - 2 * steps_right), 0)
            res = max(res, prefix[right + 1] - prefix[left])

        return res
        