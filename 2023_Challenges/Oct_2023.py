####################################################################
# 2038. Remove Colored Pieces if Both Neighbors are the Same Color
# 02OCT23
#####################################################################
class Solution:
    def winnerOfGame(self, colors: str) -> bool:
        '''
        alice can only remove A pieces, if an A is surrounded by two other As and is not on the edge
        same thing for Bob, but with B pieces
        if a player cannot make a move on their turn, they lose and the other player wins
        in order for Alice to win, she has to have more moves than Bob
        count the number of moves and check if Alice has more moves than Bob
        
        count streaks of As and Bs and calculate the number of moves
        '''
        def countMoves(colors, char):
            moves = 0
            curr_streak = 0
            for c in colors:
                if c == char:
                    curr_streak += 1
                else:
                    if curr_streak > 2:
                        moves += curr_streak - 2
                    curr_streak = 0
            if curr_streak > 2:
                moves += curr_streak - 2
                curr_streak = 0
            return moves
        
        
        aliceMoves = countMoves(colors,'A')
        bobMoves = countMoves(colors, 'B')
        print(aliceMoves,bobMoves)
        return aliceMoves > bobMoves
    
class Solution:
    def winnerOfGame(self, colors: str) -> bool:
        '''
        just a review on two key intutions
        1. when one player removes a letter, it wil never create a new removal oppuruntiy for another player
            i.e deleting an A or B on alice or bob's turn doesn't give the opposing player any advantage
        2. the order in which removals happens is irrelevant
            if we have AAAAA, removing any of the middle As just brings it to one less A
        
        
        '''
        A,B,N = 0,0,len(colors)
        #dont go to the edges
        for i in range(1,N-1):
            if colors[i-1] == colors[i] == colors[i+1]:
                if colors[i] == 'A':
                    A += 1
                else:
                    B += 1
        
        return A > B
    
##############################################
# 1804. Implement Trie II (Prefix Tree)
# 02OCT23
##############################################
#yessss!
class Node:
    def __init__(self):
        self.children = defaultdict()
        self.word_counts = 0
        self.is_end = False
        #keep track of ending words
        self.words_ending = 0
        

class Trie:
    '''
    this is a design problem, think about what we want to support for the methods
    '''

    def __init__(self):
        self.trie = Node()

    def insert(self, word: str) -> None:
        curr = self.trie
        for ch in word:
            if ch in curr.children:
                curr = curr.children[ch]
            else:
                curr.children[ch] = Node()
                curr = curr.children[ch]
            #update counts
            curr.word_counts += 1
        
        curr.is_end = True #maark
        curr.words_ending += 1

    def countWordsEqualTo(self, word: str) -> int:
        #this means we could have multiple occurrences for a word in the tree
        #for the end of a word, store the counts
        curr = self.trie
        for ch in word:
            if ch in curr.children:
                curr = curr.children[ch]
            else:
                return 0
        
        if curr.is_end and curr.words_ending >= 0:
            return curr.words_ending
        return 0
        

    def countWordsStartingWith(self, prefix: str) -> int:
        #each node should have a count as well as marking the end of word
        #then we just descend the tree until the end and grab a count
        curr = self.trie
        for ch in prefix:
            if ch in curr.children:
                curr = curr.children[ch]
            else:
                return 0
        
        if curr.word_counts >= 0:
            return curr.word_counts
        return 0

    def erase(self, word: str) -> None:
        #this part is tought
        #remove one ocrrucne. just decrese by 1
        curr = self.trie
        for ch in word:
            if ch in curr.children:
                curr = curr.children[ch]
            else:
                return False
            curr.word_counts -= 1
        
        curr.words_ending -= 1


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.countWordsEqualTo(word)
# param_3 = obj.countWordsStartingWith(prefix)
# obj.erase(word)

######################################################
# 2001. Number of Pairs of Interchangeable Rectangles
# 03OCT23
######################################################
class Solution:
    def interchangeableRectangles(self, rectangles: List[List[int]]) -> int:
        '''
        two rectanlge are interchangable if their ratios of widht/heights are the same
        '''
        counts = Counter()
        ans = 0
        
        for w,h in rectangles:
            ratio = w/h
            ans += counts[ratio]
            counts[ratio] += 1
        
        return ans
    
class Solution:
    def interchangeableRectangles(self, rectangles: List[List[int]]) -> int:
        '''
        the real way would be to use gcd, then reduce fraction to lowest terms
        then use this fraction as the key
        '''
        
        def gcd(a,b):
            if b == 0:
                return a
            return gcd(b, a % b)
            
        counts = Counter()
        ans = 0
        
        for w,h in rectangles:
            GCD = gcd(w,h)
            #store key is tuple 
            w = w // GCD
            h = h // GCD
            ratio = (w,h)
            ans += counts[ratio]
            counts[ratio] += 1
        
        return ans
    
##########################################################
# 2083. Substrings That Begin and End With the Same Letter
# 03OCT23
##########################################################
class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        '''
        look at some examples
        abcba, examine subtrings starting and ending with each letter
        a: a, abcba,a
        b: b, bcb, b
        c: a
        
        in general if a char i, appears n times, then there are n*(n+1)/2 substrings that start and being with that letter
        '''
        ans = 0
        counts = defaultdict()
        
        for ch in s:
            counts[ch] = counts.get(ch,0) + 1
        
        for count in counts.values():
            ans += count*(count+1) // 2
        
        return ans

#######################################
# 229. Majority Element II
# 05OCT23
#######################################
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        '''
        take the idea from Majority Element I,where we keep track of the current candiate
        essentially we are finding majority nums[:i] for all i
        for the case when n//2:
            keep count of occurrences of current candidate and current candidate variables
            when count if zero, we assign new candidate as the current element
            otherwise +=1 if num is the cnadiate, else -1
        heres the snippet
        class Solution:
            def majorityElement(self, nums):
                count = 0
                candidate = None

                for num in nums:
                    if count == 0:
                        candidate = num
                    count += (1 if num == candidate else -1)

                return candidate
        
        now for the case n//3
        intuition
            there can be at most one majority element with count more than n//2
            there can be as most two elements with count more than n//3
            three for count more than n//4
        
        for some count n // k, there can be at most k-1, with k being >= 1
        we adopt the same intution for the n//2 case to the n // 3 case, but keep track of two candidates instead of 1 
        '''
        if not nums:
            return []
        
        cand1,cand2 = None,None
        count1,count2 = 0,0
        
        for num in nums:
            #matching candidates
            if num == cand1:
                count1 += 1
            elif num == cand2:
                count2 += 1
            #check counts
            elif count1 == 0:
                cand1 = num
                count1 += 1
            elif count2 == 0:
                cand2 = num
                count2 += 1
            else:
                count1 -= 1
                count2 -= 1
        
        #check
        ans = []
        count1 = 0
        count2 = 0
        for num in nums:
            if num == cand1:
                count1 += 1
            if num == cand2:
                count2 += 1
        N = len(nums)
        if count1 > N//3:
            ans.append(cand1)
        if count2 > N//3:
            ans.append(cand2)
        
        return ans
        
#################################################
# 1206. Design Skiplist
# 05OCT23
#################################################
'''
reading on skiplists
https://brilliant.org/wiki/skip-lists/#height-of-the-skip-list
https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2015/resources/mit6_046js15_lec07/
be sure to check out implementaion frmo video
actual implementation
https://leetcode.com/problems/design-skiplist/discuss/1573487/Clean-Python
'''
import random

class Node:
    def __init__(self, val = -1, right = None, bottom = None):
        #nodes for only going right and bottom
        self.val = val
        self.right = right
        self.bottom = bottom

class Skiplist:
    def __init__(self):
        self.head = Node()

    def flip_coin(self): #coin flip for promoting when element is added
        return random.randrange(0, 2)

    def search(self, target: int) -> bool:
        node = self.head

        #go right and down
        while node:
            while node.right and target > node.right.val:
                node = node.right
            if node.right and target == node.right.val:
                return True
            node = node.bottom

        return False

    def add(self, num: int) -> None:
        node = self.head
        #store all the nodes we went down on while search
        record_levels = []

        while node:
            while node.right and num > node.right.val:
                node = node.right

            record_levels.append(node) #all the nodes we went down on are in the array, with the most recent ones (i.e the bottom) are at the end of the array
            node = node.bottom
        #insertion prep
        new_node = None
        
        #while we don't have a new node or while we can promote (get a heads)
        while not new_node or self.flip_coin():
            #if we are at the top level
            if len(record_levels) == 0:
                #just jeep adding to head and point to isetself,
                #in the case where we keep getting heads and we have to promote
                self.head = Node(-1, None, self.head)
                prev_level = self.head
            #we need to promote and we have levles
            else:
                prev_level = record_levels.pop()
            #make new node
            new_node = Node(num, prev_level.right, new_node)
            #connect
            prev_level.right = new_node

    def erase(self, num: int) -> bool:
        #easy peeze insert
        node = self.head
        boolean = False

        while node:
            while node.right and num > node.right.val:
                node = node.right
            #erase all nodes with that num value doing down levels
            if node.right and num == node.right.val:
                node.right = node.right.right
                boolean = True
            node = node.bottom

        return boolean
    
#another way
'''
https://cw.fel.cvut.cz/old/_media/courses/a4b36acm/maraton2015skiplist.pdf
https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2015/resources/mit6_046js15_lec07/
https://leetcode.com/problems/design-skiplist/discuss/1082053/simple-solution-with-dynamic-levels-%2B-references
'''
import random


class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None
        self.down = None

class Skiplist:
    def __init__(self):
        # sentinel nodes to keep code simple
        node = ListNode(float('-inf'))
        node.next = ListNode(float('inf'))
        self.levels = [node]

    def search(self, target: int) -> bool:
        level = self.levels[-1]
        while level:
            node = level
            while node.next.val < target:
                node = node.next
            if node.next.val == target:
                return True
            level = node.down
        return False

    def add(self, num: int) -> None:
        stack = []
        level = self.levels[-1]
        while level:
            node = level
            while node.next.val < num:
                node = node.next
            stack.append(node)
            level = node.down

        heads = True
        down = None
        while stack and heads:
            prev = stack.pop()
            node = ListNode(num)
            node.next = prev.next
            node.down = down
            prev.next = node
            down = node
            # flip a coin to stop or continue with the next level
            heads = random.randint(0, 1)

        # add a new level if we got to the top with heads
        if not stack and heads:
            node = ListNode(float('-inf'))
            node.next = ListNode(num)
            node.down = self.levels[-1]
            node.next.next = ListNode(float('inf'))
            node.next.down = down
            #this is in reverse, 
            #top left node is actually at the bottom of the list
            self.levels.append(node)

    def erase(self, num: int) -> bool:
        stack = []
        level = self.levels[-1]
        while level:
            node = level
            while node.next.val < num:
                node = node.next
            if node.next.val == num:
                stack.append(node)
            level = node.down

        if not stack:
            return False

        for node in stack:
            node.next = node.next.next

        # remove the top level if it's empty
        while len(self.levels) > 1 and self.levels[-1].next.next is None:
            self.levels.pop()

        return True

# Your Skiplist object will be instantiated and called as such:
# obj = Skiplist()
# param_1 = obj.search(target)
# obj.add(num)
# param_3 = obj.erase(num)

#########################################
# 343. Integer Break (REVISTED)
# 06OCT23
#########################################
class Solution:
    def integerBreak(self, n: int) -> int:
        '''
        dp
        '''
        if n == 3:
            return 2
        memo = {}
        def dp(n):
            if n <= 2:
                return 1
            if n in memo:
                return memo[n]
            ans = 1
            for i in range(2,n+1):
                ans = max(ans, i*dp(n-i))
            
            memo[n] = ans
            return ans
        
        
        return dp(n)
    
#bottom up, be careful with boundar conditions
class Solution:
    def integerBreak(self, n: int) -> int:
        '''
        dp
        '''
        if n == 3:
            return 2
        
        dp = [0]*(n+1)
        #base case fill
        for i in range(2+1):
            dp[i] = 1
            
        for i in range(3,n+1):
            ans = 1
            for j in range(2,n+1):
                ans = max(ans, j*dp[i-j])
            
            dp[i] = ans
        
        return dp[-1]
    
#math
class Solution:
    def integerBreak(self, n: int) -> int:
        '''
        AM-GM inequality says that to maximize the product of a set of numbers with a fixed sum, all numbers should be equal
            a + b + c + d = SUM
            a*b*c*d
            using AM_GM
            (a + b + c + d)*4 >= (a*b*c*d)^(1/2)
            ((SUM)*4))^2 >= abcd , this is maximum when a = b = c = d and a+b+c+d = SUM
        
        so we have n = a*x, this is just x some number of times
        a = n/x
        product will be x**a = x**(n/x)
        f(x) = x**(1/x)*n, we want to know where this max product us
        derivative is actually hard to do
        find where this is max by taking derivative
        f'(x) = -n*x^(n/x - 2) *(log(x) - 1)
        first product converges to zero, so we are interest in (log(x) - 1) = 0, which happens at e
        we can use e since we need an interger, so we try using as many 3s
        only apply to n > 4, for n == 4, we do 2*2, insteaf of 3*1
        '''
        if n <= 3:
            return n - 1
        ans = 1
        while n > 4:
            ans *= 3
            n -= 3
        
        return ans*n
    
#logn
class Solution:
    def integerBreak(self, n: int) -> int:
        '''
        we know that we need to use 3, so just count hay many times we can use
        adjusting for edge cases ofr course
        
        cases:
        1. if n % 3 == 0, just break it into n//3 threes
        2. if n % 3 == 1, we will have a remainder of 1, buts its better to combine this one with one of the 3s to form sum4, which is 2*2
        3. if n % 3 == 2, this is optimale just break into n/3 threes and use a two
        '''
        if n <= 3:
            return n - 1
        
        if n % 3 == 0:
            return 3 ** (n // 3)
        
        if n % 3 == 1:
            return 3 ** (n // 3 - 1) * 4
        
        return 3 ** (n // 3) * 2
    
############################################################################
# 1420. Build Array Where You Can Find The Maximum Exactly K Comparisons
# 07OCT23
############################################################################
#nice try!
class Solution:
    def numOfArrays(self, n: int, m: int, k: int) -> int:
        '''
        we are given searh cost algorithm, it calculates the number of times we update max, and returns index
        create array that satisfies:
            n integers,
            each elements is in the range [1,m] 
            after applhying the algorithm, we get search cost equal to k
        
        if search cost is k, it means that while traversing the array we updated the maximum k times,
        dp(i,j,k) gives number of ways to build an array starting from index a, search cost k, and max integer used was b
        
        if i == N, we have found a way, return 1
        we can loop through from [1 to m], so we start at 1
        if k greater then search cost, return 0
        if j greater than m, return 0
        
        if i'm at some state (i,j,k) and i know the number of ways to to make a valid array
        i need to move i to i+1, but for moving i i can mainting the current max j or i an increment j by 1 and increase cost k by 1
        '''
        memo = {}
        
        def dp(a,b,c):
            if a == n:
                if c == k and b <= m:
                    return 1
                else:
                    return 0
            #prune
            if c > k:
                return 0
            if b > m:
                return 0
            
            if (a,b,c) in memo:
                return memo[(a,b,c)]
            
            ways = 1
            ways += max(b*dp(a+1,b,c)
            ways += (b+1)*dp(a+1,b+1,c+1)
            ways %= 10**9 + 7
            memo[(a,b,c)] = ways
            return ways

        return dp(0,1,0)

#not too bad
class Solution(object):
    def numOfArrays(self, n: int, m: int, k: int) -> int:
        '''
        we are given searh cost algorithm, it calculates the number of times we update max, and returns index
        create array that satisfies:
            n integers,
            each elements is in the range [1,m] 
            after applhying the algorithm, we get search cost equal to k
        
        if search cost is k, it means that while traversing the array we updated the maximum k times,
        dp(i,j,k) gives number of ways to build an array starting from index a, search cost k, and max integer used was b
        
        if i == N, we have found a way, return 1
        we can loop through from [1 to m], so we start at 1
        if k greater then search cost, return 0
        if j greater than m, return 0
        
        if i'm at some state (i,j,k) and i know the number of ways to to make a valid array
        i need to move i to i+1, but for moving i i can mainting the current max j or i an increment j by 1 and increase cost k by 1
        before adding a number that is not a new maximum, we need to get the number of ways of doing this
        if we are at state (i,j,k) and adding the same j
            we can pick any number in the range [1,j] so we multiply:
            j*dp(i+1,j,k)
        otherwise we introduct a new number, which cause the search cost the go up
        '''
        memo = {}
        
        def dp(a,b,c):
            if a == n:
                if c == k and b <= m:
                    return 1
                else:
                    return 0
            #prune
            if c > k:
                return 0
            if b > m:
                return 0
            
            if (a,b,c) in memo:
                return memo[(a,b,c)]
            
            ways = b*dp(a+1,b,c) #get current ways for this state
            #we are free to add another number from b to m+1
            for next_max in range(b+1,m+1):
                #add to it
                ways += dp(a+1,next_max,c+1)
                ways %= 10**9 +7
                
            ways %= 10**9 + 7
            memo[(a,b,c)] = ways
            return ways
        
        return dp(0,0,0)

#we dont need to wait to base case
#if we are equal to search cost, then we can place the max at on the remaining spots
class Solution:
    def numOfArrays(self, n: int, m: int, k: int) -> int:
        mod = int(10**9 + 7)
        @cache
        def dp(i, max_so_far, remain):
           # Original code: check at the last index
           # if i == n:
           #     if remain == 0:
           #         return 1
           #    
           #     return 0
           # proposed improvement: check when no more items remain
            if remain == 0:
                return max_so_far ** (n - i)
            if i == n: return 0
            ans = (max_so_far * dp(i + 1, max_so_far, remain)) % MOD
            for num in range(max_so_far + 1, m + 1):
                ans = (ans + dp(i + 1, num, remain - 1)) % MOD
                
            return ans
        
        MOD = 10 ** 9 + 7
        return dp(0, 0, k)


#bottom up
class Solution:
    def numOfArrays(self, n: int, m: int, k: int) -> int:
        '''
        '''
        dp = [[[0]*(k+1) for _ in range(m+1)] for _ in range(n+1)]
        mod = 10**9 + 7
        
        #base case fill
        for a in range(n+1):
            for b in range(m+1):
                for c in range(k+1):
                    if c == k and b <= m:
                        dp[a][b][c] = 1
        
        
        for a in range(n-1,-1,-1):
            for b in range(m-1,-1,-1):
                for c in range(k,-1,-1):
                    
                    ways = b*dp[a+1][b][c] % mod
                    if c < k:
                        for next_max in range(b+1,m+1):
                            ways += dp[a+1][next_max][c+1] % mod
                            ways %= mod
                    
                    ways %= mod
                    dp[a][b][c] = ways
        
        return dp[0][0][0]

#use different dp state
class Solution:
    def numOfArrays(self, n: int, m: int, k: int) -> int:
        '''
        if we notince there is a loop inside the recursive function.
        can we avoid this to improve the time complexity?
        first lets try coming up with a new dp state (i,maxNum,cost)
        so given a length i, with maxNum and current cost , how many ways can we build this array?
        the previous do answered the question: given this state, how many was can we finish
        
        answer if just dp(n,maxNum,k) for all values in range [1,m]
        base case is just i = 1, and is cost == 1, return 1 else 0
        
        transitions:
            if we dont add a new maximum, then we could have placed any of the nums, [1,max_num], to the array of sizes n-1 with max_num nad value cost - 1
                maxNum*dp(i-1,maxNum,cost)
            update cost cost+1
        
        '''
        mod = 10**9 + 7
        memo = {}
        def dp(i,max_num,cost):
            #single array, make sure we have cost
            if i == 1:
                return cost == 1
            
            if (i,max_num,cost) in memo:
                return memo[(i,max_num,cost)]
            
            #count so far
            ways_so_far = max_num*dp(i-1,max_num,cost) % mod
            #find new number of ways
            #arrived from here
            for num in range(1,max_num):
                ways_so_far += dp(i-1,num,cost-1) % mod
            
            ways_so_far %= mod
            memo[(i,max_num,cost)] = ways_so_far
            return ways_so_far
        
        #sum all dp states with 1 to m+1
        ways = 0
        for num in range(1,m+1):
            ways += dp(n,num,k) % mod
            ways %= mod
        
        return ways % mod

#improve time complextiy using prefix sums
class Solution:
    def numOfArrays(self, n: int, m: int, k: int) -> int:
        dp = [[[0] * (k + 1) for _ in range(m + 1)] for __ in range(n + 1)]
        prefix = [[[0] * (k + 1) for _ in range(m + 1)] for __ in range(n + 1)]
        MOD = 10 ** 9 + 7
        
        for num in range(1, m + 1):
            dp[1][num][1] = 1
            prefix[1][num][1] = prefix[1][num - 1][1] + 1

        for i in range(1, n + 1):
            for max_num in range(1, m + 1):
                for cost in range(1, k + 1):                    
                    ans = (max_num * dp[i - 1][max_num][cost]) % MOD
                    ans = (ans + prefix[i - 1][max_num - 1][cost - 1]) % MOD

                    dp[i][max_num][cost] += ans
                    dp[i][max_num][cost] %= MOD
                    
                    prefix[i][max_num][cost] = (prefix[i][max_num - 1][cost] + dp[i][max_num][cost]) % MOD

        return prefix[n][m][k]

##########################################
# 1458. Max Dot Product of Two Subsequences
# 08OCT23
##########################################
class Solution:
    def maxDotProduct(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        brute force would be to examine all subsequences of all lengths
        let dp(i,j) be the max dot prodcut using nums1[:i] and nums2[:j]
        say we are in state (i,j) i can include i+1 and j+1 and increase 
            nums1[i]*nums2[j] + dp(i-1,j-1)
        
        skip both, we just go to dp(i-1,j-1)
        now take 1 or the other
        if take take i and skip this j
        
        this is just LCS, the tricky bit is finding the rule for the special cases
        imagine inputs like:
                [-1, -4, -7]
                [6, 2, 52]
        if we just did knapsack then we would end up retruning 0 as the max product (because we wend up not taking any elements from nums1 or nums2)
        but in this case we cannot have non-empty subsequencces, i.e there must be elements in the array
        when all elements in nums1 are negative and all element in nums2 are positive (or vice versa)
        the answer will be negative, so we must return the high negative possible!
            choose the largest negative and smalelst positive
        '''
        
        #special cases
        if max(nums1) < 0 and min(nums2) > 0:
            return max(nums1)*min(nums2)
        
        if max(nums2) < 0 and min(nums1) > 0:
            return max(nums2)*min(nums1)
        memo = {}
        
        def dp(i,j):
            if i < 0 or j < 0:
                return 0
            if (i,j) in memo:
                return memo[(i,j)]
            #use i and j
            take = nums1[i]*nums2[j] + dp(i-1,j-1)
            ans = max(take,dp(i-1,j),dp(i,j-1))
            memo[(i,j)] = ans
            return ans
        
        return dp(len(nums1) - 1, len(nums2) - 1)

#bottom up
class Solution:
    def maxDotProduct(self, nums1: List[int], nums2: List[int]) -> int:
        '''
        bottom up
        '''
        
        #special cases
        if max(nums1) < 0 and min(nums2) > 0:
            return max(nums1)*min(nums2)
        
        if max(nums2) < 0 and min(nums1) > 0:
            return max(nums2)*min(nums1)
        
        dp = [[0]*(len(nums2) + 1) for _ in range(len(nums1) + 1)]
        
        for i in range(1,len(nums1)+1):
            for j in range(1,len(nums2)+1):
                take = nums1[i-1]*nums2[j-1] + dp[i-1][j-1]
                ans = max(take,dp[i-1][j],dp[i][j-1])
                dp[i][j] = ans
                
        return dp[len(nums1)][len(nums2)]

##############################
# 505. The Maze II (REVISTED)
# 08OCT23
##############################
#ez SSP
class Solution:
    def shortestDistance(self, maze: List[List[int]], start: List[int], destination: List[int]) -> int:
        '''
        this is djikstras 
            an (i,j) cell is conneceted to all is neighbors by an edge
            edge is just the distance the ball rolls until it hits a wall
        '''
        rows = len(maze)
        cols = len(maze[0])
        
        #function to generate next neighbors
        def getNeighs(x,y):
            dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
            for dx,dy in dirrs:
                neigh_x = x
                neigh_y = y
                dist = 0
                while 0 <= neigh_x + dx < rows and 0 <= neigh_y + dy < cols and maze[neigh_x+dx][neigh_y+dy] == 0:
                    neigh_x += dx
                    neigh_y += dy
                    dist += 1
                yield dist,neigh_x,neigh_y
                
        visited = [[False]*cols for _ in range(rows)]
        dist = [[float('inf')]*cols for _ in range(rows)]
        
        dist[start[0]][start[1]] = 0
        #makr visited while in loop
        pq = [(0,start[0],start[1])]
        
        while pq:
            edge_weight,x,y = heapq.heappop(pq)
            #not smaller
            if dist[x][y] < edge_weight:
                continue
            #mark
            visited[x][y] = True
            for neigh_edge,neigh_x,neigh_y in getNeighs(x,y):
                #skip visited edges
                if visited[neigh_x][neigh_y]:
                    continue
                    
                new_dist = dist[x][y] + neigh_edge
                if new_dist < dist[neigh_x][neigh_y]:
                    dist[neigh_x][neigh_y] = new_dist
                    heapq.heappush(pq, (new_dist, neigh_x,neigh_y))
        
        if dist[destination[0]][destination[1]] != float('inf'):
            return dist[destination[0]][destination[1]] 
        return -1

#can also use dfs, but TLES
class Solution:
    def shortestDistance(self, maze: List[List[int]], start: List[int], destination: List[int]) -> int:
        '''
        this is djikstras 
            an (i,j) cell is conneceted to all is neighbors by an edge
            edge is just the distance the ball rolls until it hits a wall
            
        we can also use dfs, just make sure to miniize all outgoing i to j nodes when there is a smaller distance
        '''
        rows = len(maze)
        cols = len(maze[0])
        
        #function to generate next neighbors
        def getNeighs(x,y):
            dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
            for dx,dy in dirrs:
                neigh_x = x
                neigh_y = y
                dist = 0
                while 0 <= neigh_x + dx < rows and 0 <= neigh_y + dy < cols and maze[neigh_x+dx][neigh_y+dy] == 0:
                    neigh_x += dx
                    neigh_y += dy
                    dist += 1
                yield dist,neigh_x,neigh_y
                
        visited = [[False]*cols for _ in range(rows)]
        dist = [[float('inf')]*cols for _ in range(rows)]
        
        dist[start[0]][start[1]] = 0
        #makr visited while in loop
        
        def dfs(x,y,maze,dist):
            for neigh_edge,neigh_x,neigh_y in getNeighs(x,y):
                new_dist = dist[x][y] + neigh_edge
                if new_dist < dist[neigh_x][neigh_y]:
                    dist[neigh_x][neigh_y] = new_dist
                    dfs(neigh_x,neigh_y,maze,dist)
                    
        dfs(start[0],start[1],maze,dist)
        
        if dist[destination[0]][destination[1]] != float('inf'):
            return dist[destination[0]][destination[1]] 
        return -1

#########################################################################
# 34. Find First and Last Position of Element in Sorted Array (REVISTED)
# 09OCT23
#########################################################################
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        '''
        this is just left bound and right bound
        '''
        if len(nums) == 0: 
            return [-1, -1]
        left = 0
        right = len(nums) - 1
        leftbound = -1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] < target:
                leftbound = mid + 1
                left = mid + 1
            else:
                right = mid - 1
        
        
        left = 0
        right = len(nums) - 1
        rightbound = -1
        
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] <= target:
                left = mid + 1
            
            else:
                rightbound = mid - 1
                right = mid  - 1
        
        if 0 <= leftbound < len(nums) and leftbound <= rightbound and nums[leftbound] == target:
            return [leftbound, rightbound]
        else:
            return [-1, -1]
    
#fucking binary search
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if len(nums) == 0: return [-1, -1]
        
        def searchLow(nums, target):
            head, tail = 0, len(nums) - 1
            while head <= tail:
                mid = (head + tail)//2
                if nums[mid] >= target:
                    tail = mid - 1
                else:
                    head = mid + 1
            return head
                
        def searchHigh(nums, target):
            head, tail = 0, len(nums) - 1
            while head <= tail:
                mid = (head + tail)//2
                if nums[mid] > target:
                    tail = mid - 1
                else:
                    head = mid + 1
            return tail
        
        start = searchLow(nums, target)
        end = searchHigh(nums, target)
        if 0 <= start < len(nums) and start <= end and nums[start] == target:
            return [start, end]
        else:
            return [-1, -1]

#########################################################
# 1287. Element Appearing More Than 25% In Sorted Array
# 09OCT23
#########################################################
class Solution:
    def findSpecialInteger(self, arr: List[int]) -> int:
        '''
        we can just do sliding window
        we need to find the elements where arr[i] to arr[i + nunmber of .25 of time] is the same
        '''
        N = len(arr)
        count = N // 4
        for i in range(N-count):
            if arr[i] == arr[i+count]:
                return arr[i]
        
        return -1

class Solution:
    def findSpecialInteger(self, arr: List[int]) -> int:
        '''
        another way, keep track of curr, prev and count
        '''
        N = len(arr)
        prev = -1
        needed = N / 4
        curr_count = 0
        for num in arr:
            if num == prev:
                curr_count += 1
            else:
                prev = num
                count = 1
            
            if curr_count > needed:
                return num
        
        return -1

#binary search
class Solution:
    def findSpecialInteger(self, arr: List[int]) -> int:
        '''
        if a value occurs more than 25% of the time, then it must be one one of three positions
        (n/4, n/2 (3/4)*2)
        i.e the numbers cover more than a quarter of the array!, we we need to find the one tht spans more than 1/4
        '''
        size = len(arr)
        #must be on of the numbers at these sports
        #and must span size//4
        candidates = [arr[size // 4], arr[size // 2], arr[-size // 4]]
        for target in candidates:
            i = self.firstElementIndex(arr, target)
            if arr[i + size // 4] == target:
                return target
        return None
        
    def firstElementIndex(self, arr, target):
        l, r = 0, len(arr)
        first = -1
        while l < r:
            m = (l + r) // 2
            if arr[m] >= target:
                r = m
            else:
                first = m + 1
                l = m + 1
            
        return first

####################################################
# 1196. How Many Apples Can You Put into the Basket
# 10OCT23
####################################################
class Solution:
    def maxNumberOfApples(self, weight: List[int]) -> int:
        '''
        sort and keep taking apples
        
        '''
        bag = 5000
        weight.sort()
        N = len(weight)
        i = 0
        while i < N:
            if bag - weight[i] >= 0:
                bag -= weight[i]
                i += 1
            else:
                return i
        
        return i

class Solution:
    def maxNumberOfApples(self, weight: List[int]) -> int:
        '''
        we can use min heap and keep taking the lowest
        '''
        heapq.heapify(weight)
        bag_weight = 0
        apples = 0
        
        while weight and weight[0] + bag_weight <= 5000:
            apples += 1
            bag_weight += heapq.heappop(weight)
        
        return apples

##############################################################
# 2009. Minimum Number of Operations to Make Array Continuous
# 10OCT23
##############################################################
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        '''
        define continuous has having two properties
            all elements are unique
            max(nums) - min(nums) == len(nums) - 1
        
        in one operation, i can replace any element in nums with any integer
        return minimum number of moves to make nums continuous
        sort the array, smallest is at beginning, largest is at the end
        given [1,2,3,5,6] sort
        [1,2,3,5,6]
        N = 5
        6 - 1 != 5 - 1
        
        say we are at some index i, compare that to the largest, which is at some index k
        if nums[k] - nums[i] != len(nums) - 1
        then make we need to fix nums[k] to make it so
        but we need to make sure that number isn't used in the array
        
        [1,2,3,5,6]
        we are at 1, we need to find some number cand_max, such that cand_max - 1 == N - 1
        cand_max = N
        more so, given some curr_num after sorting
        cand_max - curr_num == N - 1
        cand_max = N - 1 + curr_num
        i probably need to duplicate the numbers first
        
        if when looking for the max im not at the end of the array, it means there is a number larger than the cand_max,
        i need to change this eleement to cand_max
        
        turns out we are trying to find a continuous array, where each elements goes up by 1, if sorted
        if we are given array of size n, and some start left, then the numbers in the array should be [left, left + n]
        idea is to treat each number as a left bound, and finds its right bound, [left, left + n - 1]
        
        to find how many operations we need to ot this, we need to find the number of elements already in the range [left, left + n - 1]
        we can leave these elements unchange, we also need to remove duplicates
        which mean we need to change the n - (right - left) elements
        the answer is just the minimum for each left
        need to use right bound
            the smallest number that is just greater when looking for the right
            right bound is the insertion point
        '''
        N = len(nums)
        ans = N #notice how N is still the original nums
        #in the case where we need to replace all
        deduped_nums = sorted(set(nums))
        
        for i in range(len(deduped_nums)):
            left = deduped_nums[i]
            right = left + N - 1
            right_idx = bisect.bisect_right(deduped_nums,right)
            count_no_change = right_idx - i
            ans = min(ans, N - count_no_change)
        
        return ans
            
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        '''
        writing out upper bound
        '''
        
        def upper_bound(arr,target):
            left = 0
            right = len(arr)
            right_bound = 0
            while left < right:
                mid = left + (right - left) // 2
                if arr[mid] > target:
                    right = mid
                else:
                    right_bound = mid + 1
                    left = mid + 1
            
            return right_bound
            
        N = len(nums)
        ans = N #notice how N is still the original nums
        #in the case where we need to replace all
        deduped_nums = sorted(set(nums))
        
        for i in range(len(deduped_nums)):
            left = deduped_nums[i]
            right = left + N - 1
            right_idx = upper_bound(deduped_nums,right)
            count_no_change = right_idx - i
            ans = min(ans, N - count_no_change)
        
        return ans
            
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        '''
        we can use sliding window to find the the count of unchanged numbers in the range [left, left+n-1]
        notince that as i inreases so does left = nums[i]
        and since left increase, right increases
        so we can use two pointers and expand the window until we go just past the needed right bound
        '''
        N = len(nums)
        ans = N #notice how N is still the original nums
        #in the case where we need to replace all
        deduped_nums = sorted(set(nums))
        
        right_idx = 0
        
        for i in range(len(deduped_nums)):
            left = deduped_nums[i]
            while right_idx < len(deduped_nums) and deduped_nums[right_idx] < left + N:
                right_idx += 1
            
            count_no_change = right_idx - i
            ans = min(ans, N - count_no_change)
        
        return ans
            
#################################################
# 2251. Number of Flowers in Full Bloom
# 11OCT23
#################################################
#brute force
class Solution:
    def fullBloomFlowers(self, flowers: List[List[int]], people: List[int]) -> List[int]:
        '''
        return array ans of size len(people) where ans[i] is the number of flowers in full bloom
        flowers[i] gives [start bloom, end bloom] for the ith flower
        sweep line?
        hints1
        1. count(t) = number flowers in bloom at time t = number of flowers that have started blooming - number of of flowers that have already stopped blooming
        
        '''
        N = len(people)
        ans = [0]*N
        
        for i,time in enumerate(people):
            #count
            in_bloom = 0
            for start,end in flowers:
                if start <= time <= end:
                    in_bloom += 1
            
            ans[i] = in_bloom
        
        return ans
            
class Solution:
    def fullBloomFlowers(self, flowers: List[List[int]], people: List[int]) -> List[int]:
        '''
        return array ans of size len(people) where ans[i] is the number of flowers in full bloom
        flowers[i] gives [start bloom, end bloom] for the ith flower
        sweep line?
        hints1
        1. count(t) = number flowers in bloom at time t = number of flowers that have started blooming - number of of flowers that have already stopped blooming
        2. speed up the search binary search to find the counts
        sort start and ends seperately
        then binary search to find the number starting bllom flowers and ending bloom flowers
        '''
        N = len(people)
        ans = [0]*N
        starts = []
        ends = []
        for start,end in flowers:
            starts.append(start)
            ends.append(end)
        
        starts.sort()
        ends.sort()
        
        for i,time in enumerate(people):
            #count
            start_bloom = bisect.bisect_right(starts,time)
            end_bloom = bisect.bisect_left(ends,time)
            count = start_bloom - end_bloom
            ans[i] = count
        
        return ans

#using heap/pq
class Solution:
    def fullBloomFlowers(self, flowers: List[List[int]], people: List[int]) -> List[int]:
        '''
        we first sort on flowers based on start time, sort based on people increasinly, and min heap for end times
        the idea is that for a person with time t, and flower that bloomed with less than time t, is a possibility for that person to see it bloom
        but if the flower finished blooming with time < t, then this peron cannot possbily see it
        
        intution:
            for person i with time t, find all flowers <= time t, then of those flowers remove the ones who's bloom ended before time t
            for finding the ending time we can use a min heap
        
        intution:
            after sorting, move pointer i to iterate along flowers where we find flowers that started bloooming <= time t
            as we advance, we add the end times to the min heap
            from this heap, we remove will end times that are <= the current time
            the answer is the size of the heap
        
        use dictionary to mapp times to people
        '''
        flowers.sort(key = lambda x: x[0])
        sorted_people = sorted(people) #keep separetely
        end_times = []
        mapp = {} #person time to num flowers
        start = 0
        
        for person in sorted_people:
            #get possible flowers
            while start < len(flowers) and flowers[start][0] <= person:
                heapq.heappush(end_times, flowers[start][1])
                start += 1
                
            #remover flowers that possibly couldn't be seen
            while end_times and end_times[0] < person:
                heapq.heappop(end_times)
            
            mapp[person] = len(end_times)
        
        #remapp
        return [mapp[p] for p in people ]

#sorting and two pointers
class Solution:
    def fullBloomFlowers(self, flowers: List[List[int]], people: List[int]) -> List[int]:
        arr1 = sorted([x[0] for x in flowers])
        arr2 = sorted([x[1] for x in flowers])
        people = sorted(enumerate(people), key = lambda x: x[1])

        p1, p2, p3 = 0, 0, 0
        n = len(arr1)
        m = len(people)
        cur_flower = 0
        # cur_time = 0
        res = {}
        
        for org_idx, t in people:
            while p1 < n and arr1[p1] <= t:
                cur_flower += 1
                p1 += 1
            while p2 < n and arr2[p2] < t:
                cur_flower -= 1
                p2 += 1
            res[org_idx] = cur_flower

        return [res[i] for i in range(m)]

#line sweep
#differene arrays
class Solution:
    def fullBloomFlowers(self, flowers: List[List[int]], people: List[int]) -> List[int]:
        '''
        line sweep or difference array, typically used in range base problems
        for some range [start,end] we do diff[start] += 1 and diff[end] -= 1
        the idea is that each index repersents a change in count, in this case a change in the number of flowers
        then for a certain range we can do a prefix sum to find the total flowers in bloom, more so, we do pref_sum at time t on the differences array
        
        algo:
            1. make the differences array (line sweep)
            2. create pref sum of the values in the difference array and mapp them to the positions, these are the keys
            3. binary search over people and find the person in the positions array, answer is in the pref_sum array
        '''
        difference = {} #could also use SortedDict
        for start,end in flowers:
            difference[start] = difference.get(start,0) + 1
            difference[end+1] = difference.get(end+1,0) - 1
        
        #makre prefix sum
        prefix_sums = []
        positions = []
        curr_flowers = 0
        
        for key in sorted(difference.keys()):
            val = difference[key]
            positions.append(key)
            curr_flowers += val
            prefix_sums.append(curr_flowers)
        
        ans = []
        
        #binary search for the index (right bound) of perosn in the positions array
        for p in people:
            right_idx = bisect.bisect_right(positions,p) - 1
            ans.append(prefix_sums[right_idx])
        
        return ans