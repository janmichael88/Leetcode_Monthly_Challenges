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
            for _ in range(abs(diff)):
                fruits_to_swap.append(k)

        fruits_to_swap.sort()
        min_val = min(total_counts.keys())
        min_cost = 0
        for i in range(len(fruits_to_swap) // 2):
            min_cost += min(fruits_to_swap[i],2*min_val)

        return min_cost


        
