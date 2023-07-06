#############################################
# 2305. Fair Distribution of Cookies
# 01JUL23
#############################################
#TLE
class Solution:
    def distributeCookies(self, cookies: List[int], k: int) -> int:
        '''
        we can parition cookies into k subsets
        we define unfairness of a parition schemes as the max total cookies obtained by a single child
        return minimum unfairness
        
        backtracking and add to bags
        i need to keep track of the i that i'm on, as well as the person who gets this bag
        let i represent the ith bag and j represnet the jth person
        '''
        bags = [0]*k
        N = len(cookies)
        self.ans = float('inf')
        
        def rec(i):
            if i == N:
                #print(bags)
                unfairness = max(bags)
                self.ans = min(self.ans,unfairness)
                return
            #print(bags)
            for j in range(k):
                bags[j] += cookies[i]
                rec(i+1)
                bags[j] -= cookies[i]
        
        rec(0)
        return self.ans
    
#cache states as tuples, fuckkk
class Solution:
    def distributeCookies(self, cookies: List[int], k: int) -> int:
        '''
        we can parition cookies into k subsets
        we define unfairness of a parition schemes as the max total cookies obtained by a single child
        return minimum unfairness
        
        backtracking and add to bags
        i need to keep track of the i that i'm on, as well as the person who gets this bag
        let i represent the ith bag and j represnet the jth person
        '''
        bags = [0]*k
        N = len(cookies)
        memo = {}
        
        def rec(i,bags):
            if i == N:
                return max(bags)
            if (i,tuple(bags)) in memo:
                return memo[(i,tuple(bags))]
            
            ans = float('inf')
            for j in range(k):
                bags[j] += cookies[i]
                ans = min(ans,rec(i+1,bags))
                bags[j] -= cookies[i]
            
            memo[(i,tuple(bags))] = ans
            return ans
        
        return rec(0,bags)
        
#always read the constraints
class Solution:
    def distributeCookies(self, cookies: List[int], k: int) -> int:
        '''
        we can parition cookies into k subsets
        we define unfairness of a parition schemes as the max total cookies obtained by a single child
        return minimum unfairness
        
        backtracking and add to bags
        i need to keep track of the i that i'm on, as well as the person who gets this bag
        let i represent the ith bag and j represnet the jth person
        
        we need to optimize the backtracking approach otherwise we will time out, stop early technque
        the problem is that we can only access
        say that we have three cookies and three children and we have already given the first two cookies to child 0
        should we continue?
            NO. why? becuase it would lead to an invalid distributiong
            we need to introduce a new paramter, zero count that represetns the number of children without a cookie
            if we ever fewer undistribute cookies thatn zero count, it menas that some child will always end up with no cookie
            so return float('inf')
            
        we also don't need to cach anything
        '''
        bags = [0]*k
        N = len(cookies)
        
        def rec(i,no_cookies,bags):
            if i == N:
                return max(bags)
            #if we don't have enough cookies
            if N - i < no_cookies:
                return float('inf')
            
            ans = float('inf')
            for j in range(k):
                #a child gets a cookie
                no_cookies -= int(bags[j] == 0)
                bags[j] += cookies[i]
                ans = min(ans,rec(i+1,no_cookies,bags))
                
                bags[j] -= cookies[i]
                no_cookies += int(bags[j] == 0)
            
            return ans
        
        return rec(0,k,bags)
    
######################################################
# 1601. Maximum Number of Achievable Transfer Requests
# 02JUN23
####################################################
#no it actually works!
class Solution:
    def maximumRequests(self, n: int, requests: List[List[int]]) -> int:
        '''
        we have n builidings labels 0 to n-1, and reqeusts: to_i and from_i
        notes:
            all buildings are full and reqeust are ACHIEVABLE if
                the net change in emplyee transgers is zero
                this means the number of emlpoyees leaving == number employees moving in
                i.e indegree == outdegree
        return maximum number of achiebale requests
        hints? brute force, and when are subsets ok
        subsets are ok if indegree == out_degree
        brute force woule be to try all subsets of requets and check that indegree == outdegree
        '''
        
        N = len(requests)
        self.ans = 0
        
        def calc_balance(subset,requests):
            balance = [0]*n
            for i in subset:
                u,v = requests[i]
                balance[u] += 1
                balance[v] -= 1

            return all([num == 0 for num in balance])
        
        def rec(i,subsets,requests):
            if calc_balance(subsets,requests):
                self.ans = max(self.ans,len(subsets))
            if i == N:
                return
            rec(i+1, subsets+[i],requests)
            rec(i+1, subsets,requests)
        
        rec(0,[],requests)
        return self.ans

#backtracking without recursing
class Solution:
    def maximumRequests(self, n: int, requests: List[List[int]]) -> int:
        '''
        in my frist approach i called rec twice, insteaf we can backtrack and keep a count of the number of transfers
        only then when i == len(requestts) we check the indegree and update the answer
        we also don't need to keep track of the indices in requests to take
        for each request we take, advance i+1, and add to a request
        once we are done looking through all the request, we need to check for a valid configuration
        '''
        
        N = len(requests)
        indegree = [0]*n #should be zero
        ans = [0]
        

        
        def rec(i,count,ans):
            if i == N:
                if all([num == 0 for num in indegree]):
                    ans[0] = max(ans[0],count)
                
                return
            u,v = requests[i]
            indegree[u] -= 1
            indegree[v] += 1
            #take it
            rec(i+1, count+1,ans)
            indegree[u] += 1
            indegree[v] -= 1
            #need to recurse again
            rec(i+1,count,ans)
        rec(0,0,ans)
        return ans[0]
    
#bit masking
class Solution:
    def maximumRequests(self, n: int, requests: List[List[int]]) -> int:
        '''
        we can also do this itertively by examing all subsets using the knuth
        genereate all integers from 0 to 2^len(requests), then
        we also need to check if we have more requests than the max answer
        if the requests we are considering is < our answer so far, we know we can't make it any more maximum
        so we can skip this state
        '''
        
        N = len(requests)
        ans = 0
        for state in range(2**N):
            indegree = [0]*N
            #count bits
            bitCount = bin(state).count("1")
            
            #prune: can't optimze any higher
            if bitCount <= ans:
                continue
                
            #set indegree
            #subset = []
            for i in range(N):
                if state & (1 << i):
                    u,v = requests[i]
                    indegree[u] -= 1
                    indegree[v] += 1
            
            if all([num == 0 for num in indegree]):
                ans = bitCount
        
        return ans
################################################
# 1005. Maximize Sum Of Array After K Negations
# 03JUL23
################################################
#O(k*logN)
class Solution:
    def largestSumAfterKNegations(self, nums: List[int], k: int) -> int:
        '''
        we can only negate, it makes sense to first negate the negative signs to make them positive
        we want to negate the most negative numbers
        min heap and negate the top k times, then sum the heap
        '''
        heapq.heapify(nums)
        
        while k:
            top = heapq.heappop(nums)
            heapq.heappush(nums,-top)
            k -= 1
        
        return sum(nums)

#O(NlgN)
class Solution:
    def largestSumAfterKNegations(self, nums: List[int], k: int) -> int:
        '''
        using sorting
        1. sort the numbers in ascending order
        2. flip all the negativ numbers as long as k > 0
        3. find sum of new array and keep track of the minimum number
        4. for the return statement
            res is sum of new array
            check for parity of remainig k, if even, it does nothing
            if odd we flip the minimum number on more time and subtract twice its value
            becase we added it twice
        '''
        nums.sort()
        i = 0
        while i < len(nums) and i < k and nums[i] < 0:
            nums[i] = -nums[i]
            i += 1
        
        res = sum(nums)
        smallest = min(nums)
        remaining_k = k - i
        return res - (remaining_k % 2)*smallest*2
    
class Solution:
    def largestSumAfterKNegations(self, nums: List[int], k: int) -> int:
        '''
        notice that negating a negative number add twice its value to the sum
        we can fist negate all the negative numbers and modify the sum by incremanting by twice its absolute value
        then only push back if its bigger so we maintain all postive
        then check for the remainder of k operations
        '''
        res = sum(nums)
        heapq.heapify(nums)
        while k > 0 and nums[0] < 0:
            smallest = heapq.heappop(nums)
            res -= 2*smallest
            
            if -smallest > nums[0]:
                heapq.heappush(nums,-smallest)
            k -= 1
        
        #left over opeations if odd, even sum won't change
        if k % 2 == 1:
            smallest = heapq.heappop(nums)
            res -= 2*smallest
        
        return res
    
############################################
# 137. Single Number II (REVISTED)
# 05JUL23
#############################################
#bit shift hacks, insteaf mod2 use mod3, 
#mod2 from single number is essentially XOR
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        '''
        every element can appear three times except for 1, there could k elements that appear three times, 0 times if that makes sense LMAOOO
        if sorted it would be [a,a,a,b,b,b,c,d,d,d,e,e,e] for nums [a,b,d,e] repeated three times
        
        properties for bit manipulation
        A XOR B = (A+B) mod 2, rather modulo 2 addtion, here we are interested in modulo three additions
        if there is a count of three for a number count % 3 = 0, and the other number will not
        
        we need to do modulo three addition bit by bit
        get last bit by &1
        if we want ith bit >> i times and do num &1
        
        we need to compute the loner bit by bit using mod 3
        
        because the numbers can only appear three times (bit_sum % 3_ will either be 0 or 1, which means the loner number would be 1
        notice the constraints are 2*31 and not 2**32
        
        note on two's comleement, subatract 2**32 to get it, if the sign bit is set
        '''
        
        loner = 0
        for shift in range(32):
            bit_sum = 0 #count up bits at this positiosn
            for num in nums:
                bit_sum += (num >> shift) & 1
            
            #put back and compute longer
            loner_bit = bit_sum % 3
            loner = loner | (loner_bit << shift)
        
        #don't forget about twos compleemnt
        #do not mistake sign bit for the most siginificant bit
        if loner >= (1 << 31):
            loner = loner - (1 << 32)
        
        return loner
    
#equation for bit mask


###########################################################
# 1493. Longest Subarray of 1's After Deleting One Element
# 05JUL23
###########################################################
#dp??
class Solution:
    def longestSubarray(self, nums: List[int]) -> int:
        '''
        i can use dp, need to keep track of number of zeros ive delete
        dp(i,deletions_left) be the longest subarray using nums[:i]
        then take the max
        three cases
        1. is a 1, so extend
        2. is a 0, and we have deletion, extend
        3. is a 0 and no deletion, carry over, i.e push up
        '''
        memo = {}
        N = len(nums)
        
        def dp(i,deletions):
            if i == N:
                return 0
            if (i,deletions) in memo:
                return memo[(i,deletions)]
            
            case1 = case2 = case3 = 0
            if nums[i] == 1:
                case = 1 + dp(i+1,deletions)
            if nums[i] == 0 and deletions == 1:
                case2 = 1 + dp(i+1,0)
            if nums[i] == 0 and deletions == 0:
                case3 = max(case1,case2)
            
            ans = max(case1,case2,case3)
            memo[(i,deletions)] = ans
            return ans
        
        
        dp(0,1)
        print(memo)

#need global dp
class Solution:
    def longestSubarray(self, nums: List[int]) -> int:
        '''
        i can use dp, need to keep track of number of zeros ive delete
        dp(i,deletions_left) be the longest subarray using nums[:i]
        then take the max
        three cases
        1. is a 1, so extend
        2. is a 0, and we have deletion, extend
        3. is a 0 and no deletion, carry over, i.e push up

        Since 
Only in the case where all elements are one we will have to delete one (result will be totalSum - 1)
Otherwise we can always delete zero and in case of deletion we can have two possibilites
1) Affect our result that is might increase the length of 1 (for ex del 0 in 11011)
2) Does not affect our result (for ex 1111000) 
So whenever we encounter zero 
  .) if it is the first zero encountered from right then try deleting which might give increase the length of 1s (for example zero b/w 1's in 0011011)
     or skip it (for example first zero on rightmost side is a better option 1111111000000101) So try both possibility and take the max.
  .) if it is the second zero encountered then return 0 we are not allowed deleting this.
        '''
        memo = {}
        N = len(nums)
        #we need to update for ueach subarray on the fly, insteaf of optimize for all the whole problem we want the max
        self.ans = 0
        
        def dp(i,deletions):
            if i == N:
                return 0
            if (i,deletions) in memo:
                return memo[(i,deletions)]
            
            #we had already deleted
            if deletions == 0:
                curr = 0
                if nums[i] == 1:
                    curr = 1 + dp(i+1,deletions)
                    self.ans = max(self.ans,curr)
                    memo[(i,deletions)] = curr
                    return curr
                memo[(i,deletions)] = 0
                return 0
            else: #we have a deletion
                curr = 0
                if nums[i] == 1:
                    curr = 1 + dp(i+1,deletions)
                    self.ans = max(curr,self.ans)
                    memo[(i,deletions)] = curr
                    return curr
                
                #otherwise its the frist zero encounter
                nodelete = dp(i+1,deletions)
                delete = dp(i+1,0)
                self.ans = max(self.ans, nodelete,delete)
                memo[(i,deletions)] = delete #return result of deleting becasue in previous states, we did not have a delete zero, so we need this for expansion
                return delete
            return 0
        
        #coner case, all ones delete at least 1
        if sum(nums) == len(nums):
            return len(nums) - 1
        dp(0,1)
        return self.ans

#sliding window, bleagh it works
class Solution:
    def longestSubarray(self, nums: List[int]) -> int:
        '''
        maitain sliding window where there is at most one zero in it
        hard part is that we need to delete at least 1
        '''
        zeros = 0
        ans = 0
        left,right = 0,0
        N = len(nums)
        
        if sum(nums) == N:
            return N - 1
        
        while right < N:
            #expand
            while right < N and zeros <= 1:
                zeros += (nums[right] == 0)
                right += 1
            
            #ans = max(ans,(right - left - 1))
            cand = nums[left:right]
            ans = max(ans,cand.count(1))
            
            #shrink
            while left < N and zeros > 1:
                zeros -= (nums[left] == 0)
                left += 1
            
        
        return ans
    
class Solution:
    def longestSubarray(self, nums: List[int]) -> int:
        '''
        deleting a zero in the sequence really just means we can only have a subarray with only one zero in it
        if there are two zeros in it, we need to shrink
        '''
        zeros = 0
        left = 0
        ans = 0
        N = len(nums)
        
        for right in range(N):
            zeros += (nums[right] == 0)
            
            #too many zeros we shrink
            while zeros > 1:
                zeros -= (nums[left] == 0)
                left += 1
        
            
            ans = max(ans, right - left)
    
        return ans
    
from contextlib import suppress

class Solution:
    def longestSubarray(self, nums: List[int]) -> int:
        '''
        can length of consecutive ones, supress value erros when going out of bounds
        '''
        def lengths_of_one_intervals():
            i = 0
            with suppress(ValueError):
                while True:
                    #look for next occruence of zero in nums[i:]
                    j, i = i, nums.index(0, i) + 1
                    yield i - j - 1
            yield len(nums) - i
            
        #default of the empty iterable
        return max((a + b for a, b in pairwise(lengths_of_one_intervals())), default=len(nums) - 1)

############################################
# 209. Minimum Size Subarray Sum
# 06JUL23
############################################
#sliding window is trivial
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        '''
        once we have found a subarray sum >= target, it doesn't make sense to expand it anymore
        so we just shrink it
        '''
        if sum(nums) < target:
            return 0
        
        N = len(nums)
        ans = N
        
        curr_sum = 0
        left = 0
        
        for right in range(N):
            curr_sum += nums[right]
            
            while left < N and curr_sum >= target:
                ans = min(ans, right - left + 1)
                curr_sum -= nums[left]
                left += 1
        
        return ans
    
#binary seach for the valid sum
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        '''
        binary search on prefix sum
        for each index i, find the subarray who's sum is just greater than target
        dont forget bound conditions
        '''
        if sum(nums) < target:
            return 0
        N = len(nums)
        pref_sum = [0]*(N+1)
        for i in range(N):
            pref_sum[i+1] = pref_sum[i] + nums[i]
        
        ans = N
        
        #binary search for all is
        for i in range(N):
            #get the sum for this nums[i:]
            #curr_sum = pref_sum[-1] - pref_sum[i]
            left = i
            right = N
            while left < right:
                mid = left + (right - left) // 2
                curr_sum = pref_sum[mid] - pref_sum[left+1]
                #too big
                if curr_sum < target:
                    left = mid + 1
                else:
                    right = mid
            
            if i < N:
                ans = min(ans, left-i + 1)
        
        return ans
    

####################################
# 465. Optimal Account Balancing
# 02JUL23
###################################
