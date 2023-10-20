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

###############################################
# 1095. Find in Mountain Array
# 12OCT23
###############################################
#ughh too many calls to the API
# """
# This is MountainArray's API interface.
# You should not implement it, or speculate about its implementation
# """
#class MountainArray:
#    def get(self, index: int) -> int:
#    def length(self) -> int:

class Solution:
    def findInMountainArray(self, target: int, mountain_arr: 'MountainArray') -> int:
        '''
        we need to binary seach in a mountain array
        we actually dont have access to the mountain array, can only interact withit through the API
        return min indiex such that it == target
        similar to search in rotated array
        i can binary search for the peack index, then search both left and right parts and check for target and validate the minimum
        '''
        #find peack index
        left = 1
        right = mountain_arr.length()-2
        peak_idx = 0
        while left <= right:
            mid = left + (right - left) // 2
            #on the peak
            if mountain_arr.get(mid -1) < mountain_arr.get(mid) > mountain_arr.get(mid+1):
                peak_idx = mid
                break
            #if we are on the ascending side
            elif mountain_arr.get(mid -1) < mountain_arr.get(mid) < mountain_arr.get(mid+1):
                peak_idx = mid + 1
                left = mid + 1
            #descending side
            else:
                right = mid - 1
            
        #print(peak_idx)
        #now binary search for target from [0 to peak_idx] and [peak_idx to right]
        left = 0
        right = peak_idx
        cand1 = -1
        
        while left <= right:
            mid = left + (right - left) // 2
            #is answer
            if mountain_arr.get(mid) == target:
                cand1 = mid
                break
            elif mountain_arr.get(mid) < target:
                left = mid + 1
            else:
                right = mid - 1
        
        #right side, but remember this is decreasing
        left = peak_idx
        right = mountain_arr.length()-1
        cand2 = -1
        
        while left <= right:
            mid = left + (right - left) // 2
            #is answer
            if mountain_arr.get(mid) == target:
                cand2 = mid
                break
            elif mountain_arr.get(mid) < target:
                right = mid - 1
            else:
                left = mid + 1
        
        print(peak_idx, cand1,cand2)
        #validate
        if cand1 == -1 and cand2 == -1:
            return -1
        elif cand1 == -1:
            return cand2
        elif cand2 == -1:
            return cand1
        else:
            return min(cand1,cand2)

#yes!!!
# """
# This is MountainArray's API interface.
# You should not implement it, or speculate about its implementation
# """
#class MountainArray:
#    def get(self, index: int) -> int:
#    def length(self) -> int:

class Solution:
    def findInMountainArray(self, target: int, mountain_arr: 'MountainArray') -> int:
                #find peack index
        left = 1
        right = mountain_arr.length()-2
        peak_idx = 1 #set this to 1 not 0, peak cannot be at beignning or end anway
        while left <= right:
            mid = left + (right - left) // 2
            #we actually dont need to check all thre conidtions, just if we are on the ascending side
            if mountain_arr.get(mid) < mountain_arr.get(mid+1):
                peak_idx = mid + 1
                left = mid + 1
            #descending side
            else:
                right = mid - 1
            
        #print(peak_idx)
        #now binary search for target from [0 to peak_idx] and [peak_idx to right]
        left = 0
        right = peak_idx
        cand1 = -1
        
        while left <= right:
            mid = left + (right - left) // 2
            #is answer
            if mountain_arr.get(mid) == target:
                cand1 = mid
                break
            elif mountain_arr.get(mid) < target:
                left = mid + 1
            else:
                right = mid - 1
        
        #right side, but remember this is decreasing
        left = peak_idx
        right = mountain_arr.length()-1
        cand2 = -1
        
        while left <= right:
            mid = left + (right - left) // 2
            #is answer
            if mountain_arr.get(mid) == target:
                cand2 = mid
                break
            elif mountain_arr.get(mid) < target:
                right = mid - 1
            else:
                left = mid + 1

        #validate
        if cand1 == -1 and cand2 == -1:
            return -1
        elif cand1 == -1:
            return cand2
        elif cand2 == -1:
            return cand1
        else:
            return min(cand1,cand2)

############################################
# 1213. Intersection of Three Sorted Arrays
# 13OCT23
############################################
class Solution:
    def arraysIntersection(self, arr1: List[int], arr2: List[int], arr3: List[int]) -> List[int]:
        '''
        the arrays are already sorted, just keep three pointers and move the smallest
        '''
        i,j,k = 0,0,0
        ans = []
        
        #if we get to the end of one array first, we can stop, since it wont have whats in the other two
        while i < len(arr1) and j < len(arr2) and k < len(arr3):
            #all the same
            if arr1[i] == arr2[j] == arr3[k]:
                ans.append(arr1[i])
                i += 1
                j += 1
                k += 1
            
            elif arr1[i] < arr2[j]:
                i += 1
            elif arr2[j] < arr3[k]:
                j += 1
            else:
                k += 1
        
        return ans

class Solution:
    def arraysIntersection(self, arr1: List[int], arr2: List[int], arr3: List[int]) -> List[int]:
        ans = []
        j,k = 0,0
        for i in range(len(arr1)):
            while j < len(arr2) - 2 and arr2[j+1] <= arr1[i]:
                j += 1
            while k < len(arr3) - 2 and arr3[k+1] <= arr1[i]:
                k += 1
            
            if arr1[i] == arr2[j] == arr3[k]:
                ans.append(arr1[i])
        
        return ans

###########################################
# 746. Min Cost Climbing Stairs (REVISTED)
# 13OCT23
##########################################
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        '''
        let dp(i) be min cost of climbing stairs starting at i
        then dp(i) = {
            cost[i] + min(dp(i+1), dp(i+2))
        }
        
        min dp(0,1)
        '''
        
        N = len(cost)
        memo = {}
        
        def dp(i):
            if i >= N:
                return 0
            if i in memo:
                return memo[i]
            ans = cost[i] + min(dp(i+1),dp(i+2))
            memo[i] = ans
            return ans
        
        
        return min(dp(0),dp(1))

#bottom up
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        '''
        let dp(i) be min cost of climbing stairs starting at i
        then dp(i) = {
            cost[i] + min(dp(i+1), dp(i+2))
        }
        
        min dp(0,1)
        '''
        
        N = len(cost)
        dp = [0]*(N+2)
        
        for i in range(N-1,-1,-1):
            ans = cost[i] + min(dp[i+1],dp[i+2])
            dp[i] = ans
        
        
        return min(dp[0],dp[1])

########################################
# 544. Output Contest Matches
# 13OCT23
#########################################
class Solution:
    def findContestMatch(self, n: int) -> str:
        '''
        generate all pairs, then cut in half and nest the left and right parts
        i can use recursion,
        left and right to numbers
        '''
        def rec(left,right):
            if left >= right:
                return ""
            
            larger_rank = "({},{})".format(left,right)
            smaller_rank = rec(left+1,right - 1)
            #play them
            play = "{},{}".format(larger_rank,smaller_rank)
            return play
        
        
        print(rec(1,n))

class Solution:
    def findContestMatch(self, n: int) -> str:
        '''
        generate all pairs, then cut in half and nest the left and right parts
        i can use recursion,
        left and right to numbers
        i need to recurse on the array
        '''
        def rec(left,right):
            if right - left == 1:
                return [(left,right)]
            
            res = [(left,right)]
            return res + rec(left+1,right-1)
        
        def rec2(arr,left,right):
            if left >= right:
                return ""
            
            play = "{},{}".format(arr[left],arr[right])
            return "("+play+","+rec2(arr,left+1,right-1)+")"
        
        teams = rec(1,n)
        print(teams)
        print(rec2(teams,0,len(teams)-1))

class Solution:
    def findContestMatch(self, n: int) -> str:
        def helper(array):
            if len(array) == 1:
                return array[0]
            res = []
            for i in range(len(array)//2):
                res.append('('+array[i]+','+array[len(array)-1-i]+')')
            return helper(res)
        a = [str(num) for num in range(1,n+1)]
        return helper(a)

class Solution:
    def findContestMatch(self, n: int) -> str:
        '''
        iteratvively
        '''
        #seeds = map(str, range(1,n+1))
        seeds = [str(num) for num in range(1,n+1)]
        while n > 1:
            seeds = ["("+ seeds[i] + "," + seeds[n-1-i] + ")" for i in range(n//2)]
            n //= 2
        
        return seeds[0]

###########################################
# 2742. Painting the Walls
# 14OCT23
############################################
#nice try
class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        '''
        we are given costs and times and cost[i] and time[i] represents that cost and time to paint the ith wall
        
        we have two paitners
        paid:
            cost = cost[i]
        free painter:
            paint any wall in 1 unit time at cost of zero
            free painter can only be used if paid painters is already occupied? WTF??
            we can only use a free painter while we are using a paid painter
            rather, if free painter takes time t to paint a wall, i can use free painter to paint t walls
            
        if im at wall i using a paid patiner, cost must be cost[i] and time[i]
        so he paints and we move up i
        but given time[i], we can also use free painters to advance by time[i] with cost of 1
            or we can pick time[i] walls to paint
        
        so we paint this will cost[i] and use time[i] free painters
        sort cost increasinly
        and keep track of i
        '''
        paired = [(c,t) for c,t in zip(cost,time)]
        paired.sort()
        cost = [c for c,t in paired]
        time = [t for c,t in paired]
        N = len(cost)
        
        memo = {}
        
        def dp(i):
            if i >= N:
                return 0
            if i in memo:
                return memo[i]
            #use only paid, 
            #use only paid and free
            only_paid = cost[i] + dp(i+1)
            paid_free = cost[i] + dp(i+time[i]+2)
            ans = min(only_paid,paid_free)
            memo[i] = ans
            return ans
        
        return dp(0)

class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        '''
        need to keep track of the index 
        and number of walls painted so far
        when we paint wall i, we can also elimnate time[i] walls
        or we dont paint
        doesnt matter what walls we paint, as long as we paint them
        '''
        memo = {}
        N = len(cost)
        def dp(i,left):
            if left <= 0:
                return 0
            if i >= N:
                return float('inf')
            if (i,left) in memo:
                return memo[(i,left)]
            
            paint = cost[i] + dp(i+1,left - 1 - time[i])
            dont_paint = dp(i+1,left)
            ans = min(paint,dont_paint)
            memo[(i,left)] = ans
            return ans
        
        return dp(0,N)
            
#bottom up
class Solution:
    def paintWalls(self, cost: List[int], time: List[int]) -> int:
        '''
        bottom up
        '''
        N = len(cost)
        dp = [[0]*(N+1) for _ in range(N+1)]
        #prepopulate base cases
        for i in range(1,N+1):
            for left in range(1,N+1):
                if i >= N:
                    dp[i][left] = float('inf')
                    
                    
        for i in range(N-1,-1,-1):
            for left in range(1,N+1):
                paint = cost[i] + dp[i+1][max(0,left - 1 - time[i])]
                dont_paint = dp[i+1][left]
                ans = min(paint,dont_paint)
                dp[i][left] = ans
        
        return dp[0][N]
        
#################################################################
# 1269. Number of Ways to Stay in the Same Place After Some Steps
# 15OCT23
#################################################################
class Solution:
    def numWays(self, steps: int, arrLen: int) -> int:
        '''
        lets call (i,j) state as i being the position in the array and the number of steps j
        let dp(i,j) be the number of ways of getting to (i,j)
        so we need to stolve dp(0,steps)
        if im at (i,j), i either come from (i-1,j-1), (i+1,j-1), or (i,j-1)
        
        dp(i,j) = dp(i-1,j-1) + dp(i+1,j-1)  +dp(i,j-1)
        if (steps == 0) and (i == 0) its a valid way
        
        '''
        mod = 10**9 + 7
        memo = {}
        def dp(i,j):
            if j == 0:
                if i == 0:
                    return 1
                return 0
            
            if (i,j) in memo:
                return memo[(i,j)]
            #staying theree and reduce step count
            ans = dp(i,j-1)
            #if we can move left
            if i - 1 >= 0:
                ans += (dp(i-1,j-1) % mod)
            #if we can mvoe right
            if i + 1 <= arrLen - 1:
                ans += (dp(i+1,j-1) % mod)
            ans %= mod
            memo[(i,j)] = ans
            return ans
        
        return dp(0,steps) % mod

#bottom up
class Solution:
    def numWays(self, steps: int, arrLen: int) -> int:
        '''
        bottom up
        
        '''
        mod = 10**9 + 7
        dp = [[0]*(steps+1) for _ in range(arrLen+1)]
        
        #prepopulate base casses
        for i in range(arrLen+1):
            for j in range(steps+1):
                if j == 0:
                    if i == 0:
                        dp[i][j] = 1
                        
        dp[0][0] = 1
        
        #for steps start 1 away from 0
        for j in range(1,steps+1):
            #for position, we need to start from the end
            for i in range(arrLen):
                ans = dp[i][j-1]
                #if we can move left
                if i - 1 >= 0:
                    ans += (dp[i-1][j-1] % mod)
                #if we can mvoe right
                if i + 1 <= arrLen - 1:
                    ans += (dp[i+1][j-1] % mod)
                ans %= mod
                dp[i][j] = ans
        
        return dp[0][steps]

########################################
# 1243. Array Transformation
# 17OCT23
########################################
class Solution:
    def transformArray(self, arr: List[int]) -> List[int]:
        '''
        simulate
        
        '''
        N = len(arr)
        while True:
            isThereChange = False
            next_array = arr[:]
            for i in range(1,N-1):
                if arr[i-1] > arr[i] < arr[i+1]:
                    next_array[i] = arr[i] + 1
                    isThereChange = True
                if arr[i-1] < arr[i] > arr[i+1]:
                    next_array[i] = arr[i] - 1
                    isThereChange = True
            
            arr = next_array
            if isThereChange == False:
                return arr
                
#########################################
# 1361. Validate Binary Tree Nodes
# 17OCT23
#########################################
#gahh, it not just a simple hashmap problem
class Solution:
    def validateBinaryTreeNodes(self, n: int, leftChild: List[int], rightChild: List[int]) -> bool:
        '''
        we just need to make a tree,
        the graph can have no connected components and must not have no back edge
        rather if we find the parent of each node,
            a valid tree must have nodes with only parent and exactly one node with no parent
        '''
        parents = [-1]*n
        parent_mapp = defaultdict(list)
        for i in range(n):
            #i -> left and i -> right
            left = leftChild[i]
            right = rightChild[i]
            if left != -1:
                if parents[left] == -1:
                    parents[left] = i
                if left != i:
                    parent_mapp[left].append(i)
            if right != -1: 
                if parents[right] == -1:
                    parents[right] = i
                if right != i:
                    parent_mapp[right].append(i)
                
        #validate
        #there should only be one -1 and each must have only one parent
        def onlyOneRoot(arr):
            return arr.count(-1) == 1
        
        def onlyOneParent(mapp):
            for k,v in mapp.items():
                if len(v) > 1:
                    return False
            return True
        
        print(parents)
        print(parent_mapp)
        return onlyOneRoot(parents) and onlyOneParent(parent_mapp)
        
#this is just DFS,BFS problem
class Solution:
    def validateBinaryTreeNodes(self, n: int, leftChild: List[int], rightChild: List[int]) -> bool:
        '''
        find root, 
        dfs for cycle, and make sure we touch all nodes
        '''
        children = set(leftChild) | set(rightChild)
        root = -1
        for i in range(n):
            if i not in children:
                root = i
        
        if root == -1:
            return False
        
    
        seen = set()
        def cycleDetect(node):
            if node in seen:
                return True
            seen.add(node)
            left = leftChild[node]
            right = rightChild[node]
            for child in [left,right]:
                if child != -1:
                    if cycleDetect(child):
                        return True
            
            return False
        
        hasCycle = cycleDetect(root)
        return hasCycle == False and len(seen) == n

#bfs
class Solution:
    def validateBinaryTreeNodes(self, n: int, leftChild: List[int], rightChild: List[int]) -> bool:
        '''
        bfs variant
        '''
        children = set(leftChild) | set(rightChild)
        root = -1
        for i in range(n):
            if i not in children:
                root = i
        
        if root == -1:
            return False
        
    
        def cycleDetect(node):
            seen = set()
            q = deque([])
            q.append(node)
            while q:
                node = q.popleft()
                if node in seen:
                    return [True,seen]
                seen.add(node)
                left = leftChild[node]
                right = rightChild[node]
                for child in [left,right]:
                    if child != -1:
                        q.append(child)

            return [False,seen]
        
        hasCycle,seen = cycleDetect(root)
        return hasCycle == False and len(seen) == n

#can also do top sort
class Solution:
    def validateBinaryTreeNodes(self, n: int, leftChild: List[int], rightChild: List[int]) -> bool:
        '''
        for top sort, start with node that has zero indegree, this is the root
        then we just check if we can touch all nodes
        children cannot have more the 1 indegree
        '''
        indegree = [0]*n
        for i in range(n):
            if leftChild[i] != -1:
                indegree[leftChild[i]] += 1
            if rightChild[i] != -1:
                indegree[rightChild[i]] += 1
            
        #check no more than 1 indegree for each node
        for i in range(n):
            if indegree[i] > 1:
                return False
        
        q = deque([]) #only root has 0 indegree
        for i in range(n):
            if indegree[i] == 0:
                q.append(i)
            
        if len(q) > 1:
            return False
        
        seen = set()
        while q: 
            curr = q.popleft()
            seen.add(curr)
            for child in [leftChild[curr], rightChild[curr]]:
                if child != -1:
                    indegree[child] -= 1
                    if indegree[child] == 0:
                        q.append(child)
        
        return len(seen) == n

#union find???
class DSU:
    def __init__(self,n):
        self.parent = [i for i in range(n)]
        self.comps = n
        
    def find(self,node):
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
        
        return self.parent[node]
    
    def union(self,child,parent):
        child_par = self.find(child)
        parent_par = self.find(parent)
        
        #if childs parent doesn't point to itself, it means it was assigned
        #first assigned of a parent to child would be allowed
        if child != child_par:
            return False
        
        #can't have this in a tree
        if parent_par == child_par:
            return False
        
        self.comps -= 1
        self.parent[child_par] = parent_par
        return True
        
    
class Solution:
    def validateBinaryTreeNodes(self, n: int, leftChild: List[int], rightChild: List[int]) -> bool:
        '''
        key insights for union find solution
        while exmaining (parent,left) and (parent,right)
            1. if find(child) != child, it must mean that the child got assigned a parent earlier, and thus must have one parent
                otherwise we make the relationship
                reduce group size
            2. if parent and child alreayd belong to the same group, and we see them again, it must mean they are in a cycle
        
        all the while keep track of the component size
        initally there are N groups, 
        checkk if there is one 1 at the end
        '''
        dsu = DSU(n)
        for node in range(n):
            for child in [leftChild[node], rightChild[node]]:
                if child != -1:
                    if not dsu.union(child,node):
                        return False
                    
        return dsu.comps == 1

#############################################
# 2050. Parallel Courses III
# 18OCT23
#############################################
class Solution:
    def minimumTime(self, n: int, relations: List[List[int]], time: List[int]) -> int:
        '''
        find min number of months needed to complete all courses
        relations[i] is edge list (u to v)
        time gives amount of time needed to complete course
        rules
            1. can start taking a course at anyh time if prereqs are met
            2. any number of course can be taken at the same time
        
        need to start with courses that have no prereqs
        if courses all took the same time, then we would just take thme one by one
        use kahns to determine if i can take all the courses, then if i can take all the courses
        as we do kahns keep track of the total time taken by taking the max
        min months is really just defined as the longest bottlneck (by a class)
        min number of semesters is just the longest path
        
        intution:
            the minimum time to finish a course is defined as the maximum time to finish any one of its preruesists
            if we want to take course v, we have prev a list of [prev_courses] going into v
            then the min time needeed to complete course v woudl be time[v] + max(u[time] for u in prev_courses)
            then we just takte the max time for all course
        
        intution
            needed latest prerequsite time needed for a class, which is bottlenecked by the lognest pre-requisite
            define value of a path as the sum of values for eahc node,
                consider all paths going into this node, the answer for this node is just the max
            
            top sort from 0 indegree and keep track pf max
        '''
        #just use zero indexing
        graph = defaultdict(list)
        indegree = [0]*n
        for u,v in relations:
            graph[u-1].append(v-1)
            indegree[v-1] += 1
        
        #print(indegree)
        #print(graph)
        max_times = [0]*n
        q = deque([])
        for i in range(n):
            if indegree[i] == 0:
                q.append(i)
                max_times[i] = time[i]
        while q:
            curr_class = q.popleft()
            for neigh in graph[curr_class]:
                indegree[neigh] -= 1
                max_times[neigh] = max(max_times[neigh],max_times[curr_class] + time[neigh])
                if indegree[neigh] == 0:
                    q.append((neigh))
        
        return max(max_times)
            
#dp
class Solution:
    def minimumTime(self, n: int, relations: List[List[int]], time: List[int]) -> int:
        '''
        we can treat this as dp 
        let dp(node) be the minimum time needed to complete all courses start from node
        base case is when we hit a node with 0 indegree, we can just return the time from time
        '''
        #just use zero indexing
        graph = defaultdict(list)
        indegree = [0]*n
        outdegree = [0]*n
        for u,v in relations:
            graph[u-1].append(v-1)
            indegree[v-1] += 1
            outdegree[u-1] += 1
        
        memo = {}
        def dp(curr_class):
            if outdegree[curr_class] == 0: #class with no out degree means we just use its full time
            #if curr_class not in graph:
                return time[curr_class]
            if curr_class in memo:
                return memo[curr_class]
            ans = 0
            for neigh in graph[curr_class]:
                ans = max(ans, dp(neigh) + time[curr_class])
            
            memo[curr_class] = ans
            return ans
        
        
        ans = 0
        for i in range(n):
            ans = max(ans, dp(i))
        
        return ans
            
            
############################################
# 2355. Maximum Number of Books You Can Take
# 18OCT23
############################################
#nice try though
class Solution:
    def maximumBooks(self, books: List[int]) -> int:
        '''
        given books array of length n, books[i] is number of books on ith shelf
        want to take a contig secttion [l,r], and for i in range(l,r+1)
            if we take k books from shelf i, we must take > k + 1 books in shelf i+1
        
        i can use (l,r) as states, too big
        then use prefix array
        dp(i) gives the maximum books i can take using books[:i] inclusive
        if we are at i, we can only take up to books[i+1] - 1
        if we can take this many do so, otherwise skip
        '''
        memo = {}
        n = len(books)
        
        def dp(i):
            if i >= n:
                return 0
            if i in memo:
                return memo[i]
            
            ans = 0
            if i + 1 < n:
                for j in range(books[i+1]):
                    ans = max(ans,j + dp(i+1))
            memo[i] = ans
            return ans
        
        ans = 0
        for i in range(n):
            ans = max(ans,dp(i))
        
        return ans
                
#O(n^3)
class Solution:
    def maximumBooks(self, books: List[int]) -> int:
        '''
        for dp, we need two states (posotion i, and prev books taken)
        then we keep trying to take a book and minimize
        '''
        memo = {}
        n = len(books)
        
        def dp(i,prev_taken):
            if i >= n:
                return 0
            if (i,prev_taken) in memo:
                return memo[(i,prev_taken)]
            
            ans = 0
            #try taking books from books[i] to 0, only its greater then previous taken
            for books_taken in range(books[i], 0, -1):
                if books_taken > prev_taken:
                    ans = max(ans, books_taken + dp(i+1,books_taken))
            
            memo[(i,prev_taken)] = ans
            return ans
        
        
        ans = 0
        for i in range(n):
            ans = max(ans, dp(i,0))
        
        return ans

class Solution:
    def maximumBooks(self, books: List[int]) -> int:
        '''
        if we define a_i as the number of books we take from the ith shelf, there are two constrains
            1. a_i <= books[i]
            2. a_i <= a_{i+1} - 1, meaing we the books we take from the current shelf must be strictly greater than i + 1
        
        say we are it index i, and we take books[i], we then look to i - 1, and if books[i-1] >= books[i] - 1, we take books[i]-1 from books[i-1]
        try i-2, and take if books[i-1] >= books[i] - 2
        at some point we will find some index j, such that:
            * j < i and books[j] < books[i] - (i - j)
            * books[j] - j < books[i] - i
            this is the index that we cannot take as part of the arithmetic progreassion
            i.e the right most j, that satifies that the above two equations
        
        really we want a contiguous sequences that is strictly increasing and consective diff (from left to right) == 1
        need to be ablse to get sum for arithmetic sequence, ez peasy
        sequence of numbers of books should be in the range[j+1,i], and in this range it should be an arithmetic progression
        
        the number of books taken from shelves in range [l,r] is books[r] + (books[r] - 1) + (books[r] - 2) + ...+ (books[r] - (cnt - 1)]
        where cnt is th enumber of shelves or summnags, of which there shold only be [r - l + 1]
        and cnt must be positive, so cnt = min(books[r], r - l + 1), in cases where r-l + 1 is negative
        so the sum for a block of an arithemtic progression in the range [l,r] is defined as 
            1/2*(first element at l + second element at right)*cnt
            = (1/2)*(books[r] - (cnt - 1))*cnt
            
        sum of arithmetic sequence: general form
        S_{n} = n/2(2*a + (n-1)*2)
        n = number of terms to be added
        a = first term
        d = comon diffeertent
        
        we define calcSum(l,r) to find sum of arithmetic progression
        now on to dp
        let dp(i) be the max number of books we can take from all the shelves in range[0,i] taking exactly books[i]
        how can we find dp(i)?
            we already know that the number of books in the range [j+1,i] = calcSum(j+1,i)
            to find dp(i) we need to find j and add calcSum(j+1,i)
            dp(i) = dp(j) + calcSum(j+1,i)
            if such a j doest not exists, then dp(i) = calcSum(0,i), i.e its just the range from 0 to i
        now we need to find j for each i, how?
        define rightmost i as i_i and the next best j, as i_2
            it must be the case that i_2 < i_1 and books[i_2] - i_2 < books[i_1] - i_1
            and it must be the case that in the range [i_2 + 1, i_1] is arith progression
        then we just do the same for i_2
        
        motonic stack:
            stack will keep indices in order of books[i] - i with the index with the largest value being at the top of the stack
            when a new index comes we pop some elemnts from the stack, to keep it montonic, and push new index,
            need to maintain books[i] - i values of elements in ascending order
        
        '''
        N = len(books)
        
        #get arith preogression
        def calcSum(l,r):
            cnt = min(books[r], r - l + 1)
            ap = (2*books[r] - (cnt - 1))*cnt // 2
            return ap
        
        stack = []
        dp = [0]*N
        
        for i in range(N):
            #finding the best j for the current i, keeep track of indices
            while stack and books[stack[-1]] - stack[-1] >= books[i] - i:
                stack.pop()
            
            #if we cant find a j, uisg calcSum(0,i)
            if not stack:
                dp[i] = calcSum(0,i)
            #transition to find the best i
            else:
                j = stack[-1]
                dp[i] = dp[j] + calcSum(j+1,i)
            
            stack.append(i)
        
        return max(dp)
        
class Solution:
    def maximumBooks(self, books: List[int]) -> int:
        '''
        just solving it another way
        sum arithmetic progression in range [l,r]
        is books[r] + (books[r] - 1) + (books[r] -2) + ... + books[r] - (cnt - 1)
            last sum, books[r] - (cnt - 1) >= 0
            books[r] >= (cnt - 1), so cnt = min(books[r], r - l + 1)
        problem really is just finding all the strictly increasing arithmetic progressions
        if i is the right of the end of the arithmetic progression, we need to find the smallest j, where books[j] - j < books[i] - i does not hold
        let dp(i) be max sum of books we can take using books[:i] and taking books[i]
        we know the sum for some j is just the arithsum from (j+1,i)
            so we want max beteen (j+1,i), which is just dp(j)
        '''
        arith_sum = lambda x : (1 + x)*x // 2 if x >= 0 else 0
        N = len(books)
        dp = [0]*N
        stack = []
        for i in range(N):
            #use first shelf
            if i == 0:
                dp[i] = books[i]
            else:
                while stack and books[stack[-1]] > books[i] - (i - stack[-1]):
                    #find the next j, everything on the stack was good, need to make increasing
                    stack.pop()
                
                if stack:
                    dp[i] = dp[stack[-1]] + arith_sum(books[i]) - arith_sum(books[i] - (i - stack[-1])) #same as + calcSum(j+1,i), whole sum minus the parts up to j-1
                else:
                    #just caclSum(j+1,i)
                    dp[i] = arith_sum(books[i]) - arith_sum(books[i] - (i+1))
            
            stack.append(i)
        
        return max(dp)
                              
        