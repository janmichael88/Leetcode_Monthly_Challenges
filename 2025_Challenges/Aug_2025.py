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
        
#can also do binary search
class Solution:
    def maxTotalFruits(
        self, fruits: List[List[int]], startPos: int, k: int) -> int:
        n = len(fruits)
        #pref sum by index, not x position
        sum_ = [0] * (n + 1)
        #list of increasing indices
        indices = [0] * n

        for i in range(n):
            sum_[i + 1] = sum_[i] + fruits[i][1]
            indices[i] = fruits[i][0]

        ans = 0
        #could also have donw k//2 floored
        for x in range(k +1):
            # move left x steps, then right (k - 2x) steps
            y = k - 2 * x
            left = startPos - x
            right = startPos + y
            #largest right >=
            start = bisect_left(indices, left)
            #smallest right <
            end = bisect_right(indices, right)
            ans = max(ans, sum_[end] - sum_[start])

            # move right x steps, then left (k - 2x) steps
            y = k - 2 * x
            left = startPos - y
            right = startPos + x
            start = bisect_left(indices, left)
            end = bisect_right(indices, right)
            ans = max(ans, sum_[end] - sum_[start])

        return ans
    
############################################
# 794. Valid Tic-Tac-Toe State
# 04AUG25
############################################
class Solution:
    def validTicTacToe(self, board: List[str]) -> bool:
        '''
        since x starts first: count(x) >= count(o)
        since players take turns, return false of count(x) - count(o) > 1
        if win(o), then win(x) cannot be true
        '''
        x_count,o_count = 0,0
        for row in board:
            for ch in row:
                x_count += ch == 'X'
                o_count += ch == 'O'
        
        if o_count > x_count or x_count-o_count>1:
            return False
        
        if self.can_win(board, 'O'):
            if self.can_win(board, 'X'):
                return False
            return o_count == x_count
        
        if self.can_win(board, 'X') and x_count!=o_count+1:
            return False

        return True

    
    def can_win(self,board,player):
        #check rows
        for i in range(len(board)):
            if board[i][0] == board[i][1] == board[i][2] == player:
                return True
            
        #check cols
        for i in range(len(board)):
            if board[0][i] == board[1][i] == board[2][i] == player:
                return True
        
        #check diags
        if board[0][0] == board[1][1] == board[2][2] or  \
            board[0][2] == board[1][1] == board[2][0] == player:
            return True
        return False
    
#################################################
# 3477. Fruits Into Baskets II
# 05AUG25
##################################################
class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        '''
        iterate fruits and look for the leftmost basket >= fruits for the current i
        '''
        n = len(fruits)
        used = [False]*n
        for i in range(n):
            for j in range(n):
                if not used[j] and baskets[j] >= fruits[i]:
                    used[j] = True
                    break
        
        #cout false
        return n - sum(used)
    
#no boolean array, count unused
class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        count = 0
        n = len(baskets)
        for fruit in fruits:
            unused = 1
            for i in range(n):
                if fruit <= baskets[i]:
                    baskets[i] = 0
                    unused = 0
                    break
            count += unused
        return count
    
################################################
# 3479. Fruits Into Baskets III
# 06AUG25
#################################################
#sqrt decomp
class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        '''
        the problem is the we need to take the left most, not just the the basket that is just >= the current fruit
        if sort and binary search for the first index where baskets[i] >= fruit we get an index
        but is there another index to the left that we could have chosen?
        square root decomposition with updates
        https://cp-algorithms.com/data_structures/sqrt_decomposition.html
        review, decompose array in root(n) blocks, where each block represents a group operator sum,product,max,min..etc
        for a query l,r, it should span some blocks, but with heads and tails (maybe) going into partial blocks
        so we can get sum by doing sum(head block) + curr_block + sum(tail block) in root(n) time
        we can also do updates by going into the block
        '''
        n = len(baskets)
        m = int(math.sqrt(n))
        sections = (n + m - 1) // m
        count = 0
        blocks = [0]*sections #store max for each block

        #stor max for each block
        for i in range(n):
            blocks[i//m] = max(blocks[i//m],baskets[i])
        
        #queries against fruit
        for fruit in fruits:
            unset = 1
            #seach block, still n*root(n)
            for sec in range(sections):
                #skip block if we cant put in fruit
                if blocks[sec] < fruit:
                    continue
                #otherwise we have a block that has basket >= fruit
                choose = False
                blocks[sec] = 0 #need to update new max after taking
                #scan block
                for i in range(m):
                    #index into baskets array
                    pos = sec*m + i
                    #valid basket, change to 0 to mark as unavailable
                    if pos < n and baskets[pos] >= fruit and not choose:
                        choose = True
                        baskets[pos] = 0
                    #update section
                    if pos < n:
                        blocks[sec] = max(blocks[sec],baskets[pos]) 
                #if we got here, we were able to place fruit successfully
                #we only want to use one basket for one fruit
                unset = 0
                break
            count += unset
        
        return count
    
#segment tree is littlre more confusing :(


##################################################
# 3363. Find the Maximum Number of Fruits Collected
# 07AUG25
###################################################
#yessss
class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        '''
        well theyve given us the (i,j) transitions already
        child at (0,0) can only down/right
        child at (0,n-1) can only go down/(left,right)
        child at (n-1,0) can only go (up,down)/right,

        kicker is that they only have n-1 moves
        so child 1 MUST walk the diagonal
        now how about the other2?
            they can go straight down or across, and they aren't allowed to cross the diagonal
        
        we can handle the 0,0 case indepdently
        '''
        n = len(fruits)
        ans = 0
        for i in range(n):
            ans += fruits[i][i]

        #dp for child at (0,n-1), remember they can't cross the main diagonal
        #make new fruits for only top half
        memo1 = {}
        fruits1 = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1,n):
                fruits1[i][j] = fruits[i][j]
        #now do dp
        def dp1(i,j,fruits,memo):
            if i < 0 or i >= n or j < 0 or j >= n:
                return float('-inf')
            if (i,j) == (n-1,n-1):
                return 0
            if (i,j) in memo:
                return memo[(i,j)]
            ans = fruits[i][j] + max(dp1(i+1,j-1,fruits,memo),dp1(i+1,j,fruits,memo),dp1(i+1,j+1,fruits,memo))
            memo[(i,j)] = ans
            return ans
        #print(dp1(0,n-1,fruits1,memo1))
        #child 2
        memo2 = {}
        fruits2 = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(0,i):
                fruits2[i][j] = fruits[i][j]
        
        def dp2(i,j,fruits,memo):
            if i < 0 or i >= n or j < 0 or j >= n:
                return float('-inf')
            if (i,j) == (n-1,n-1):
                return 0
            if (i,j) in memo:
                return memo[(i,j)]
            ans = fruits[i][j] + max(dp2(i-1,j+1,fruits,memo),dp2(i,j+1,fruits,memo),dp2(i+1,j+1,fruits,memo))
            memo[(i,j)] = ans
            return ans

        #print(dp2(n-1,0,fruits2,memo2))
        return ans + dp1(0,n-1,fruits1,memo1) + dp2(n-1,0,fruits2,memo2)
    
###############################################
# 808. Soup Servings
# 08AUG25
###############################################
import math
class Solution:
    def soupServings(self, n: int) -> float:
        '''
        notice the pours are:
        A   |   B
        100 |   0
        75  |   25
        50  |   50
        25  |   75
        n is very high in this case
        we want the probability that A is used before B + half the probability that both soupds are used up in the same turn
        both A and B are equal starting off
        each is a multiple of 25, so we can treat at n/25 servings, rounded up
        then we can use dp(i,j), we need full prob a is used up + half prob they run out in the same turn
        need to come with trivial n*n dp solution and examine what happens to the answe as n increases
        it eventuall approahes 1 when n is high enough
        '''
        memo = {}
        #n = math.ceil(n/25)
        dirrs = [(100,0),(75,25),(50,50),(25,75)]

        #could also do without ceiling
        #n = math.ceil(n/25)
        #dirrs = [(100,0),(75,25),(50,50),(25,75)]

        def dp(a,b):
            if a <= 0 and b <= 0:
                return 0.5
            if a <= 0:
                return 1.0
            if b <= 0:
                return 0.0
            
            if (a,b) in memo:
                return memo[(a,b)]
            
            ans = 0
            for aa,bb in dirrs:
                ans += dp(a-aa,b-bb)
            ans /= 4.0
            memo[(a,b)] = ans
            return ans
        
        for k in range(1,n+1):
            if dp(k,k) > 1 - 1e-5:
                return 1.0
        
        return dp(n,n)

###########################################
# 869. Reordered Power of 2
# 10AUG25
###########################################
#sorting signature
class Solution:
    def reorderedPowerOf2(self, n: int) -> bool:
        '''
        for example, digits are 61, re order to 16
        how many powers of two are there up to
        biggest number would be 9999999
        generate all powers of 2, and see if we can make any of them, and put them in sorted digit order, then see if sorted(n) is in ther
        '''
        temp = set()
        for i in range(30):
            num = "".join(sorted(str(2**i)))
            temp.add(num)
        
        n = "".join(sorted(str(n)))
        return n in temp

########################################
# 2497. Maximum Star Sum of a Graph
# 11AUG25
#######################################
class Solution:
    def maxStarSum(self, vals: List[int], edges: List[List[int]], k: int) -> int:
        '''
        just check all nodes for k edges
        and sort neighbors and tkae k max
        '''
        graph = defaultdict(set)
        for i,j in edges:
            if vals[i] > 0 : 
                graph[j].add(i)
            if vals[j] > 0 : 
                graph[i].add(j)
            
        stars = []
        for i,v in enumerate(vals):
            vv = [vals[j] for j in graph[i]]
            vv.sort(reverse=True)
            stars.append(v + sum(vv[0:k]))
            
        return max(stars)
    
############################################
# 2438. Range Product Queries of Powers
# 11AUG25
#############################################
class Solution:
    def productQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        '''
        powers array can be made use the binary represntation of n
        which means powers array cannot be more than 32
        answers array can be taken using brute force
        hint gave it away
        '''
        powers = []
        for i in range(32):
            if n & (1 << i):
                powers.append(1 << i)
        
        ans = []
        mod = 10**9 + 7
        for l,r in queries:
            product = 1
            for j in range(l,r+1):
                product *= powers[j]
                product %= mod
            ans.append(product)
        return ans
    
#using prefix product and mod mult inverse
class Solution:
    def productQueries(self, n: int, queries: List[List[int]]) -> List[int]:
        '''
        powers array can be made use the binary represntation of n
        which means powers array cannot be more than 32
        answers array can be taken using brute force
        hint gave it away
        in modular arithmetic, division by x is multiplacation by x^(-1) ,mod m
        since m = 10**9 + 7, we can compute x^(-1) as x^(m-2) mod m
        could have just accumulated products without mod, only in python because there's no limit in integer size
        binary decomp of n, gives the smallest powers of 2 that sum to n
        '''
        powers = []
        for i in range(32):
            if n & (1 << i):
                powers.append(1 << i)
        
        ans = []
        mod = 10**9 + 7
        pref_prod = [1]
        for p in powers:
            pref_prod.append((p*pref_prod[-1]) % mod)
        
        for l,r in queries:
            #need modular multiplactive inverse, only when using module
            inv = pow(pref_prod[l], mod - 2, mod)  # Fermat's little theorem
            product = (pref_prod[r + 1] * inv) % mod
            ans.append(product % mod)
        return ans