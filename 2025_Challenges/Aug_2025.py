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
class SegTree:
    def __init__(self, arr):
        '''
        this is a 1 index tree, if you want zero index tree
        left_child = 2 * idx + 1
        right_child = 2 * idx + 2
        then we can binary search on the segment tree
            - if max value in left interval is > fruits, lok left
            - if max value in left < fruits and max value in right >= fruits continue right
            - else no interval that meets the condition
        '''
        self.n = len(arr)
        self.tree = [0] * (4 * self.n)
        self.build(arr, 1, 0, self.n - 1)  # start at index 1

    def build(self, arr, idx, left, right):
        if left == right:
            self.tree[idx] = arr[left]
            return
        mid = (left + right) // 2
        self.build(arr, 2 * idx, left, mid)          # left child at 2 * idx
        self.build(arr, 2 * idx + 1, mid + 1, right) # right child at 2 * idx + 1
        self.tree[idx] = max(self.tree[2 * idx], self.tree[2 * idx + 1])
    
    def find_first_and_update(self,o,l,r,x):
        if self.tree[o] < x:
            return -1
        if l == r:
            self.tree[o] = -1
            return l
        m = l + (r - l) // 2
        i = self.find_first_and_update(o*2,l,m,x)
        if i == -1:
            i = self.find_first_and_update(o*2 + 1, m + 1,r,x)
        self.tree[o] = max(self.tree[2 * o], self.tree[2 * o + 1])
        return i


class Solution:
    def numOfUnplacedFruits(self, fruits: List[int], baskets: List[int]) -> int:
        '''
        first make segment tree, just use 4*n space for array
        sum of geometric series: 1 + 2 + 4....+2**(log(n))
        sum_{i=0}^{i=log(n)} 2**i < 2n - 1 < 4n
        '''
        m = len(baskets)
        if m == 0:
            return len(fruits) #can't fit them
        seg_tree = SegTree(baskets)
        count = 0
        for fruit in fruits:
            if seg_tree.find_first_and_update(1,0,m-1,fruit) == -1:
                count += 1
        
        return count


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
        if a bit is set in that position, i need to use that power of 2 at least
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
    
##########################################
# 1006. Clumsy Factorial
# 15AUG25
##########################################
#cheeze is real
import operator
class Solution:
    def clumsy(self, n: int) -> int:
        '''
        we just cycle through operations in order
        ['*',//,+,-]
        eval function?
        '''
        digits = [num for num in range(n,0,-1)]
        ops = ["*","//","+","-"]
        idx = 0
        ans = 1
        temp = []
        for i in range(len(digits)):
            temp.append(str(digits[i]))
            temp.append(ops[idx])
            idx += 1
            idx %= 4
        eval_string = "".join(temp[:-1])
        return eval(eval_string)
    
#############################################
# 837. New 21 Game (REVISITED)
#17AUG25
#############################################
#TLE
#need to go thorugh bottom up n*n
#then sliding window bottom up n
class Solution:
    def new21Game(self, n: int, k: int, maxPts: int) -> float:
        '''
        dp solution is similar to the soup servigins problem
        base case return full probvalue 
        then sum up probs and divie by number of events
        '''
        memo = {}
        def dp(points):
            if k <= points <= n:
                return 1.0
            if points > n:
                return 0.0
            if points in memo:
                return memo[points]
            
            ans = 0
            for num in range(1,maxPts+1):
                ans += dp(points+num)
            
            ans = ans / maxPts
            memo[points] = ans
            return ans
        
        return dp(0)
    
class Solution:
    def new21Game(self, n: int, k: int, maxPts: int) -> float:
        '''
        converting to bottom up
        '''
        dp = [0]*(n+1)
        dp[0] = 1.0

        for i in range(1,n+1):
            for p in range(1,maxPts+1):
                if i - p >= 0 and i - p < k:
                    dp[i] += dp[i-p] / maxPts
        
        return sum(dp[k:])
    
#sligind window
class Solution:
    def new21Game(self, n: int, k: int, maxPts: int) -> float:
        dp = [0] * (n + 1)
        dp[0] = 1
        s = 1 if k > 0 else 0
        for i in range(1, n + 1):
            dp[i] = s / maxPts
            if i < k:
                s += dp[i]
            if i - maxPts >= 0 and i - maxPts < k:
                s -= dp[i - maxPts]
        return sum(dp[k:])
    
########################################
# 679. 24 Game
# 18AUG25
########################################
#stupid parantheses
class Solution:
    def judgePoint24(self, cards: List[int]) -> bool:
        '''
        need to use the cards array and operations to make 24
        division is real division
        unary operator only works on 1 operand and op cannot be unary
        cannot concat numbers
        intelligently try all, ughh
        there are 4! permutations of cards
        i cant just place a single op between two cards, becaue i could have parantheses before and after
        i could have (a)op(b)op(c)op(d)
        what if we just used cards and ops first
        there are 3 spots, and for each spot we can use 4, so its
        4^3*(4!) = (4**3)*(4*3*2*1) = 1536
        now we can weave parentheses in here 
        because of () we don't need to consider order of operations
        recall a + b == b + a, same thing a*b = b*a
        but need a - b and b - a

        '''
        def rec(nums):
            if len(nums) == 1:
                return abs(nums[0] - 24) <= 1e-6

            for i in range(len(nums)):
                for j in range(i):  # j < i
                    a, b = nums[i], nums[j]
                    # all possible results of combining a and b
                    vals = [a + b, a - b, b - a, a * b]
                    #guarding againt divsion by 0, need to make sure for float points being near zero
                    if abs(b) > 1e-6:
                        vals.append(a / b)
                    if abs(a) > 1e-6:
                        vals.append(b / a)

                    # build next state without nums[i] and nums[j]
                    # we applied the operator to nums[i] and nums[j]
                    # just passing in states and return true not
                    next_nums = [nums[k] for k in range(len(nums)) if k != i and k != j]

                    for v in vals:
                        if rec(next_nums + [v]):
                            return True
            return False

        return rec(cards)
    
######################################################
# 2526. Find Consecutive Integers from a Data Stream
# 18AUG25
##################################################
class DataStream:

    def __init__(self, value: int, k: int):
        self.k = k
        self.value = value
        self.stream = Counter()


    def consec(self, num: int) -> bool:
        if num != self.value:
            self.stream = Counter()
            return False
        else:
            self.stream[num] += 1
            return self.stream[num] >= self.k


# Your DataStream object will be instantiated and called as such:
# obj = DataStream(value, k)
# param_1 = obj.consec(num)

######################################################
# 2087. Minimum Cost Homecoming of a Robot in a Grid
# 20AUG25
#####################################################
#TLE
class Solution:
    def minCost(self, startPos: List[int], homePos: List[int], rowCosts: List[int], colCosts: List[int]) -> int:
        '''
        this is just dp(i,j), backtracking solution taking too long :(
        '''
        rows = len(rowCosts)
        cols = len(colCosts)
        memo = {}
        seen = set()
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]


        def dp(i,j):
            if i < 0 or i >= rows or j < 0 or j >= cols:
                return float('inf')
            if [i,j] == homePos:
                return 0
            if (i,j) in memo:
                return memo[(i,j)]
            ans = float('inf')
            for di,dj in dirrs:
                ii = i + di
                jj = j + dj
                if (ii,jj) not in seen and 0 <= ii < rows and 0 <= jj < cols:
                    seen.add((ii,jj))
                    #up/down
                    if dj == 0:
                        ans = min(ans, dp(ii,jj) + rowCosts[ii])
                    #left/right
                    elif di == 0:
                        ans = min(ans, dp(ii,jj) + colCosts[jj])
                    seen.remove((ii,jj))

            memo[(i,j)] = ans
            return ans

        seen.add((startPos[0],startPos[1]))
        return dp(startPos[0],startPos[1])

class Solution:
    def minCost(self, startPos: List[int], homePos: List[int], rowCosts: List[int], colCosts: List[int]) -> int:
        '''
        path only has one turn, walk down/up to homePos row
        then walk left/right to homePos col
        '''
        sr, sc = startPos
        hr, hc = homePos
        ans = 0

        # move along rows
        if sr < hr:
            #walk down
            for r in range(sr + 1, hr + 1):
                ans += rowCosts[r]   # pay cost of the row we enter
        else:
            #walk up
            for r in range(sr - 1, hr - 1, -1):
                ans += rowCosts[r]

        # move along columns
        if sc < hc:
            #walk right
            for c in range(sc + 1, hc + 1):
                ans += colCosts[c]   # pay cost of the column we enter
        else:
            #walk left
            for c in range(sc - 1, hc - 1, -1):
                ans += colCosts[c]

        return ans