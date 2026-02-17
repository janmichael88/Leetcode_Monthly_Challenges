#############################################
# 3010. Divide an Array Into Subarrays With Minimum Cost I
# 01FEB26
#############################################
class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        '''
        cost of subarray is value of first element 
        need min possible sum of these subarray
        if we partition nums into 3 subarrays, there are 2 break points
        we can ge the first elements as
            (index0,i+1,j+1)
        enumerate all break proints (i,j) and take minimum sum
        '''
        ans = float('inf')
        n = len(nums)
        for i in range(1,n-1):
            for j in range(i+1,n):
                ans = min(ans, nums[0] + nums[i] + nums[j])
        
        return ans
    

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        '''
        single pass with variable swapping
        nums[0] is fixed
        we need to find to indices (i,j), where i < j and nums[i],nums[j] are the smallest
        '''
        min1 = float('inf')
        min2 = float('inf')
        n = len(nums)
        for i in range(1,n):
            if nums[i] < min1:
                min2 = min1
                min1 = nums[i]
            
            elif nums[i] < min2:
                min2 = nums[i]
        
        return nums[0] + min1 + min2
    
############################################################
# 3013. Divide an Array Into Subarrays With Minimum Cost II
# 02FEB26
############################################################
#TLE, yes!
class Solution:
    def minimumCost(self, nums: List[int], k: int, dist: int) -> int:
        '''
        we need k subarrays where the stating index of the second subarray and the starting index of the kth subarray is <= dist
        first subarray will alwasy have cost of nums[0]
        if we have subarray with index 0,i1,i2...ik
        we need ik - i1 to be <= dist
        '''
        ans = float('inf')
        n = len(nums)

        for i in range(1, n):
            left = i + 1
            right = min(i + dist, n - 1)

            # Need at least k-2 elements to choose from
            if right - left + 1 < k - 2:
                continue

            vals = nums[left:right + 1]
            vals.sort()

            cost = nums[0] + nums[i] + sum(vals[:k - 2])
            ans = min(ans, cost)

        return ans
    
from sortedcontainers import SortedList
from typing import List

class Solution:
    def minimumCost(self, nums: List[int], k: int, dist: int) -> int:
        '''
        instead of sorting on the fly, treat like sliding window with two balance structures
        we need to maintain them, similar to sliding window median and sliding windoo k smallest sum
        '''
        n = len(nums)
        ans = float('inf')

        small = SortedList()   # smallest k-2 elements
        large = SortedList()   # rest
        small_sum = 0

        def add(x):
            nonlocal small_sum
            #if we can take in the smaller
            if len(small) < k - 2:
                small.add(x)
                small_sum += x
            else:
                #if its bigger than smaller, we need to rebalance
                if x < small[-1]:
                    large.add(small.pop())
                    small.add(x)
                    small_sum += x - large[0] #update the sum
                else:
                    #add to the rest
                    large.add(x)

        def remove(x):
            nonlocal small_sum
            if x in small:
                small.remove(x)
                small_sum -= x
                if large:
                    y = large.pop(0)
                    small.add(y)
                    small_sum += y
            else:
                large.remove(x)

        # Initial window for i = 1
        for j in range(2, min(n, 1 + dist + 1)):
            add(nums[j])

        for i in range(1, n):
            if len(small) == k - 2:
                ans = min(ans, nums[0] + nums[i] + small_sum)

            # Slide window
            if i + 1 < n:
                remove(nums[i + 1])
            if i + dist + 1 < n:
                add(nums[i + dist + 1])

        return ans

from sortedcontainers import SortedList
from typing import List
class Solution:
    def minimumCost(self, nums: List[int], k: int, dist: int) -> int:
        '''
        we need to keep track of the sum of the smallest k-2 elements at anyone time
        we can use two sorted containers and maintain sums in the range nums[i+1:i+dist+1]
        as we increase i, lose one element on left and gain one element on right
        invariant:
            len(smaller) == k - 2, we must rebalance here
        '''
        n = len(nums)
        ans = float('inf')

        smaller = SortedList([]) #smallest k-2
        larger = SortedList([])
        smallest_sum = 0

        def add(x):
            nonlocal smallest_sum
            #if we can add
            if len(smaller) < k - 2:
                smaller.add(x)
                smallest_sum += x
            else:
                if x < smaller[-1]: #must put into smaller, extrude from larger
                    larger.add(smaller.pop())
                    smaller.add(x)
                    smallest_sum += x - larger[0] #change in sum, need to update on the fly
                else:
                    larger.add(x)
        
        def remove(x):
            nonlocal smallest_sum
            if x in smaller:
                smaller.remove(x)
                smallest_sum -= x
                if larger:
                    temp = larger.pop()
                    smaller.add(temp)
                    smallest_sum += temp
            else:
                larger.remove(x)
            
        #initial window for i = 1
        for j in range(2,min(n, 1 + dist + 1)):
            add(nums[j])
        
        for i in range(1,n):
            #check we have a sum
            if len(smaller) == k - 2:
                ans = min(ans, nums[0] + nums[i] + smallest_sum)
            
            #sliding window
            if i + 1 < n:
                remove(nums[i+1])
            if i + dist + 1 < n:
                add(nums[i + dist + 1])
        
        return ans

#######################################
# 3637. Trionic Array I
# 02FEB26
########################################
class Solution:
    def isTrionic(self, nums: List[int]) -> bool:
        '''
        so its 1 for inc and 0 for dec, else -1
        if its trionic then it should be 
        [1,1,1,0,0,1,1,1]
        if the array is trionic, the diff array can be reduced to the form [1,0,1]
        '''
        diffs = []
        n = len(nums)
        for i in range(1,n):
            diff = nums[i] - nums[i-1]
            if diff > 0:
                if diffs and diffs[-1] == 1:
                    continue
                else:
                    diffs.append(1)
            elif diff < 0:
                if diffs and diffs[-1] == 0:
                    continue
                else:
                    diffs.append(0)
            else: #if equal we can immediately escape
                return False
        
        return diffs == [1,0,1]
    
###################################################
# 3640. Trionic Array II
# 04DEC26
###################################################
#hints gave it away :(
#finite state machine
class Solution:
    def maxSumTrionic(self, nums: List[int]) -> int:
        '''
        is my solution from before, say for example we have this up/down signature
        1 for inc and 0 for dec, else -1
        if its trionic then it should be 
        [1,1,1,0,0,1,1,1]
        we need the max sum in this range left to right
        but it could be any subarray, we can solve that using kadanes
        hmmmm, just follow the hints and the transitions

        (start)*  (+)+  (-)+  (+)+
          dp0     dp1   dp2   dp3
        '''
        n = len(nums)
        NEG = float('-inf')

        dp0 = [NEG] * n
        dp1 = [NEG] * n
        dp2 = [NEG] * n
        dp3 = [NEG] * n #this representing the complete trionic array

        dp0[0] = nums[0]

        for i in range(1, n):
            diff = nums[i] - nums[i - 1]
            # Hint 6: dp0 carries on increasing
            #this is just kadanes
            if diff > 0:
                dp0[i] = max(dp0[i - 1] + nums[i], nums[i])
            else:
                dp0[i] = nums[i]

            # Hint 4: inc phase
            if diff > 0:
                dp1[i] = max(dp1[i - 1] + nums[i],dp0[i - 1] + nums[i]) #continue increasing, or staart increasing
                dp3[i] = max(dp3[i - 1] + nums[i],dp2[i - 1] + nums[i]) #final decreasing phase, continue final increase, or switch from dec to inc

            # Hint 5: dec phase
            if diff < 0:
                dp2[i] = max(dp2[i - 1] + nums[i],dp1[i - 1] + nums[i]) #continue decreasing, or swithc from increasing to decreasing

        return max(dp3)
    
######################################################
# 3379. Transformed Array
# 05FEB26
#####################################################
class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        '''
        allocate array, then calc using mod length
        '''
        n = len(nums)
        ans = [-1]*n

        for i in range(n):
            num = nums[i]
            if num > 0:
                j = (i + num) % n
                ans[i] = nums[j]
            elif num < 0:
                j = (i - abs(num)) % n
                ans[i] = nums[j]
            else:
                ans[i] = num
        
        return ans
    
class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        '''
        can just do i + nums[i]
        but that could be negative
        so add in n
        (i + num[i] + n) % n
        '''
        n = len(nums)
        ans = [-1]*n

        for i in range(n):
            num = nums[i]
            j = (i + nums[i] + n) % n
            ans[i] = nums[j]
        
        return ans
    
####################################################
# 3634. Minimum Removals to Balance Array
# 06FEB26
#####################################################
class Solution:
    def minRemoval(self, nums: List[int], k: int) -> int:
        '''
        an array is balanced if its max element is at most k times min
        i.e max(nums) <= k*min(nums)
        what if i sort the array
        [1,2,6,9]
        and use two pointers, problem is i dont know whether to remove the min or the max
        '''
        #uhhh, i dont know why it works lolol
        ans = float('inf')
        n = len(nums)
        nums.sort()
        right = 0
        for left in range(n):
            while right < n and nums[right] <= k*nums[left]:
                right += 1
            #right to left is balance, so the removls are n - (right - left)
            #update min
            ans = min(ans, n - (right - left))
        
        return ans
            
########################################################
# 3157. Find the Level of Tree with Minimum Sum
# 08FEB26
#########################################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def minimumLevel(self, root: Optional[TreeNode]) -> int:
        '''
        bfs all the way
        '''
        curr_level = 1
        min_level = 1
        min_sum = float('inf')
        q = deque([root])

        while q:
            N = len(q)
            curr_sum = 0
            for _ in range(N):
                curr = q.popleft()
                curr_sum += curr.val
                if curr.left:
                    q.append(curr.left)
                if curr.right:
                    q.append(curr.right)
            if curr_sum < min_sum:
                min_sum = curr_sum
                min_level = curr_level
            curr_level += 1
        
        return min_level
    
#######################################################
# 3719. Longest Balanced Subarray I
# 10FEB26
#######################################################
class Solution:
    def longestBalanced(self, nums: List[int]) -> int:
        '''
        must use dictint odds and evens
        '''
        ans = 0
        n = len(nums)
        for i in range(n):
            evens = set()
            odds = set()
            for j in range(i,n):
                if nums[j] % 2 == 0:
                    evens.add(nums[j])
                else:
                    odds.add(nums[j])
                if len(evens) == len(odds):
                    ans = max(ans, j - i +1)

        return ans
    
#########################################
# 3721. Longest Balanced Subarray II
# 11FEB26
###########################################
#good review on segment tree
#this problem is beautiful

#brute force with balance sum
class Solution:
    def longestBalanced(self, nums: List[int]) -> int:
        n = len(nums)
        result = 0
        for l in range(n):
            seen, B = set(), 0
            for r in range(l, n):
                x = nums[r]
                if x not in seen:
                    seen.add(x)
                    B += 1 if (x % 2) == 0 else -1
                if B == 0:
                    result = max(result, r - l + 1)
        
        return result

#this is probably the better intution to understand
#we can still use prefix sum, but we'd have to recompute prefsum after a new number comes in
#need segment tree
#no lazy propogation, updates are done on each value
class Solution:
    def longestBalanced(self, nums: List[int]) -> int:
        '''
        we can do better with balance array
        B(l,r) = #distinct_even(l..r) − #distinct_odd(l..r)
        the main idea is that for a fixed l a value affects b(l,r) only once at its first occurence between [l,r]
        we can ecnode as this
        balance[i] = 1, if first occurnece of even nums in [l,r]
        balance[i] -= 1 if first occurence of odd nums in [l.r]
        balance[i] = 0, otherwise
        then balance(l,r) = sum(balance[i] for i in range(l,r+1))
        we can also use balance whe moving l, from right to left
        we just need to check if we have already seen nums[l] at some index i > l,
            first update balance[i] = 0 and set balance[l] = +- 1
        
        we ccan fist initialize balance array to 0s and fill right to left
        then we need to store first[val] => indices of first occurences of vals in this subarray
        '''
        n = len(nums)

        balance = [0] * n  # first-occurrence markers for current l
        first = dict()  # val -> first occurence idx for current l

        result = 0
        for l in reversed(range(n)):
            x = nums[l]

            # If x already had a first occurrence to the right, remove that old marker.
            if x in first:
                balance[first[x]] = 0

            # Now x becomes first occurrence at l.
            first[x] = l
            if x % 2 == 0:
                balance[l] = 1
            else:
                balance[l] = -1

            # Find rightmost r >= l such that sum(balance[l..r]) == 0
            s = 0
            for r in range(l, n):
                s += balance[r]
                if s == 0:
                    result = max(result, r - l + 1)
        return result

#segment tree solution
class SegmentTree:
    """Segment Tree over array of size n"""

    def __init__(self, n: int):
        self.n = n
        self.size = 4 * n
        self.sum = [0] * self.size
        self.min = [0] * self.size
        self.max = [0] * self.size

    def _pull(self, node: int):
        """Helper to recompute information of node by it's children"""

        l, r = node * 2, node * 2 + 1

        self.sum[node] = self.sum[l] + self.sum[r]
        self.min[node] = min(self.min[l], self.sum[l] + self.min[r])
        self.max[node] = max(self.max[l], self.sum[l] + self.max[r])

    def update(self, idx: int, val: int):
        """Update value by index idx in original array"""

        def _update(node: int = 1, l: int = 0, r: int = self.n - 1):
            if l == r:
                self.sum[node] = val
                self.min[node] = val
                self.max[node] = val
                return

            m = l + (r - l) // 2
            if idx <= m:
                _update(node * 2, l, m)
            else:
                _update(node * 2 + 1, m + 1, r)

            self._pull(node)

        return _update()

    def find_rightmost_prefix(self, target: int = 0) -> int:
        """Find rightmost index r with prefixsum(r) = target"""

        def _exist(node: int, sum_before: int):
            return self.min[node] <= target - sum_before <= self.max[node]

        def _find(node: int = 1, l: int = 0, r: int = self.n - 1, sum_before: int = 0):
            if not _exist(node, sum_before):
                return -1
            if l == r:
                return l

            m = l + (r - l) // 2
            lchild, rchild = node * 2, node * 2 + 1

            # Check right half first
            sum_before_right = self.sum[lchild] + sum_before
            if _exist(rchild, sum_before_right):
                return _find(rchild, m + 1, r, sum_before_right)

            return _find(lchild, l, m, sum_before)

        return _find()


class Solution:
    def longestBalanced(self, nums: List[int]) -> int:
        n = len(nums)

        stree = SegmentTree(n)  # SegmentTree over balance array for current l
        first = dict()  # val -> first occurence idx for current l

        result = 0
        for l in reversed(range(n)):
            num = nums[l]
    
            # If x already had a first occurrence to the right, remove that old marker.
            if num in first:
                stree.update(first[num], 0)

            # Now x becomes first occurrence at l.
            first[num] = l
            if num % 2 == 0:
                stree.update(l, 1)
            else:
                stree.update(l,-1)

            # Find rightmost r >= l such that sum(w[l..r]) == 0
            r = stree.find_rightmost_prefix(target=0)
            if r >= l:
                result = max(result, r - l + 1)

        return result
######################################################
# 3713. Longest Balanced Substring I
# 12FEB26
######################################################
class Solution:
    def longestBalanced(self, s: str) -> int:
        '''
        just track counts 
        '''
        n = len(s)
        ans = 0
        for i in range(n):
            window = Counter()
            for j in range(i,n):
                ch = s[j]
                window[ch] += 1
                #validate the window
                counts = set(window.values())
                if len(counts) == 1:
                    if [window[ch]] == list(counts):
                        ans = max(ans,j-i+1)
        return ans
    
####################################################
# 3714. Longest Balanced Substring II
# 13FEB26
#####################################################
#like subarraysum == k
#pref_sum, balance, first occurence trick
class Solution:
    def longestBalanced(self, s: str) -> int:
        '''
        we can check all substrings like the last problem
        but now the alphabet is smaller
        we need indices i,j
        where 
        * counts(a) == counts(b) == counts(c)
        * counts(a) == counts(b)
        * counts(a) == counts(c)
        * counts(b) == counts(c)
        then just the singletons
        * counts(a) == len(substring)
        * counts(b) == len(substring)
        * counts(c) == len(substring)
        this is a lot of pref sum, why not just make it a bunch of smaller problems????
        '''
        #count streaks
        def case1(s,ch):
            longest_streak = 0
            curr_streak = 0
            for letter in s:
                if letter == ch:
                    curr_streak += 1
                else:
                    longest_streak = max(longest_streak,curr_streak)
                    curr_streak = 0
            return max(longest_streak,curr_streak)
        
        def case2(s,ch1,ch2):
            #this is just for pair counts, a char that isn't part of the pair would immediatlye break it
            #balance and first occurence trick
            balance = 0
            ans = 0
            
            # stores first index where each balance occurred
            first_seen = {0: -1}
            
            for i, ch in enumerate(s):
                
                if ch == ch1:
                    balance += 1
                elif ch == ch2:
                    balance -= 1
                else:
                    # third character breaks the segment
                    balance = 0
                    first_seen = {0: i}
                    continue
                
                if balance in first_seen:
                    ans = max(ans, i - first_seen[balance])
                else:
                    first_seen[balance] = i
                    
            return ans
        
        def case3(s):
            #same as case2, but now with all chars
            count_a,count_b,count_c = 0,0,0
            first_seen = {(0,0):-1}
            ans = 0
            for i,ch in enumerate(s):
                if ch == 'a':
                    count_a += 1
                elif ch == 'b':
                    count_b += 1
                else:
                    count_c += 1
                if (count_b - count_a, count_c - count_a) in first_seen:
                    ans = max(ans,i - first_seen[(count_b - count_a, count_c - count_a)])
                else:
                    first_seen[(count_b - count_a, count_c - count_a)] = i
            
            return ans

        
        ans = max([case1(s,ch) for ch in "abc"])
        combs = [['a','b'],['a','c'],['b','c']]
        for ch1,ch2 in combs:
            ans = max(ans,case2(s,ch1,ch2))
        
        return max(ans,case3(s))

##################################################
# 2303. Calculate Amount Paid in Taxes
# 16FEB26
###################################################
class Solution:
    def calculateTax(self, brackets: List[List[int]], income: int) -> float:
        '''
        intervals
        '''
        prev = 0
        tax = 0
        
        for upper, percent in brackets:
            if income <= prev:
                break
            
            taxable = min(income, upper) - prev
            tax += taxable * (percent / 100)
            prev = upper
        
        return tax
