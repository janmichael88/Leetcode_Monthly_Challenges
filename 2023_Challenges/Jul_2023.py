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
            
            #put back and compute loner
            loner_bit = bit_sum % 3
            loner = loner | (loner_bit << shift)
        
        #don't forget about twos compleemnt
        #do not mistake sign bit for the most siginificant bit
        if loner >= (1 << 31):
            loner = loner - (1 << 32) #twos complement, exponent is one more than the places used for bits
        
        return loner
    
#equation for bit mask
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        '''
        from single number II:
        A XOR B = A(~B) + (~A)B
        from the bit by bit approach, we only want to if 1 appears (i.e count % 3 = 1, for any of the bits)
        a number can only appear in the array 0 or 3 times
            i.e if an integer appears three times, it should not be in the bitmask, and if it appears 1 time, it should 
        
        XOR does modulo 2, so we need seenZero, seenOnce, seenTwice
            if any bit in seenzero is set, it means that bit has appeared 0 times in all integers so far, since we are doing module 3, it is seen thrice
            if any bit in seenOnce is set it menas that bit has appeared 1 mod 3 times, same thing with seenTwice
        
        seenZero initally set to all 1's since we haven't seen anything yet
        we don't really need seenZero, if a bit is  not set in seenOnce and not set in seenTwice, it must be set in seenZero, pigeonhole
        
        now for each num in nums:
        for seenOnce:
            should not have been set in seenTwice, if it was previously seen once, it should be removed from seenONce and set to seenTwice
            if not it should be added to seenOnce, we can XOR with num
            i.e seenOnce = (seenOnce XOR num) AND (NOT seenTwice)
            
        for seenTwice:
            it should be previsouly seenOnce, so this put should be set, but if we have already ipdate seenOnce fo this num, then it should not been in seenOnce. it the bit was set in seenOnce, then for this num it is the first occrucne it sohuld not be mistaken for a second
            in other words, for the second ocurrence, it must be removed from seenOnce while updateing it using the seenOnce equaiotns
            thus it should not be in seenOnce while updating seenTwice
            
            if it was previously seen twice, it should be removed from seenTwice
            if not ti should be added to seenTwice, which can be done by XORING seenTwice with num
            either of them shoould be set, but not both
            seenTwice = (seenTwice XOR num) AND (NOT seenOnce)
        
       crux of the problem
        if a bit appears for the first time, add it to seenOnce, it will not be added to seenTwice because of its precense in seenonce
        if a bit appears a second time, remove itfrom seenOnce and set in seenTwice
        if a bit appears a third time, it wont be added to seenOnce because its alreayd present in seenTwice. after it will be removed from seenTwice
        
        one we are done, we will have seenOnce set at all the bits where the nums appeared only once return seenOnce
        '''
                # Initialize seen_once and seen_twice to 0
        seen_once = seen_twice = 0

        # Iterate through nums
        for num in nums:
            # Update using derived equations
            seen_once = (seen_once ^ num) & (~seen_twice)
            seen_twice = (seen_twice ^ num) & (~seen_once)

        # Return integer which appears exactly once
        return seen_once
    
#Boolean Algebra and Karnaugh Map


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
#fuck this shit...., nice try though
class Solution:
    def minTransfers(self, transactions: List[List[int]]) -> int:
        '''
        brute force seems doable given the input conditions
        given transactinos
        [[0,1,10],[2,0,5]]
        
        after first transaction
        0: -10
        1: 10
        2: 0
        after second transctions
        
        0:-5
        1:10
        2:-5
        
        if i get the final state after completing all transactions
        then i can just find the optimal way of setting verything to zero
        now the question becomes, given the final state, find the minimum number of transactions so that no value is negative
        if i'm at person i, and starting_debt[i] >= abs(sum(all other debts that are negative)) then we can settle the debt
        i dont want to touch people with zero debt because they have nothing to contribute
        debts can also be settled no matter what
        
        notes:
            we would never need to check if all were == 0 at the starting condition, this could never be the case
            we can settle all debts in n transactions, in the case where we have an addtional entity serving as the institution
            we can settle all debts in n-1 transactions if one of the people acts as the institution
        
        find all debts that are non zero, than backtrack to euqlize
        
        
        '''
        starting_debts = [0]*12
        for first, second, amount in transactions:
            starting_debts[first] -= amount
            starting_debts[second] += amount
            
        #only grab non zero entries
        starting_debts = [debt for debt in starting_debts if debt != 0]

        def rec(i,starting_debts):
            if i == len(starting_debts):
                return 0
            if starting_debts[i] < 0 or starting_debts[i] == 0:
                return rec(i+1,starting_debts)
            
            curr_debt = starting_debts[i]
            ans = float('inf')
            for j in range(len(starting_debts)):
                #make sure its not the same person
                if j != i and starting_debts[j] < 0:
                    #transfer
                    old_debt = starting_debts[j]
                    curr_debt -= old_debt
                    starting_debts[j] = 0
                    ans = min(ans,1 + rec(i+1,starting_debts))
                    starting_debts[j] = -old_debt
                    curr_debt += old_debt
            
            return ans
        
        return rec(0,starting_debts)
    
#TLE
class Solution:
    def minTransfers(self, transactions: List[List[int]]) -> int:
        '''
        brute force seems doable given the input conditions
        given transactinos
        [[0,1,10],[2,0,5]]
        
        after first transaction
        0: -10
        1: 10
        2: 0
        after second transctions
        
        0:-5
        1:10
        2:-5
        
        if i get the final state after completing all transactions
        then i can just find the optimal way of setting verything to zero
        now the question becomes, given the final state, find the minimum number of transactions so that no value is negative
        if i'm at person i, and starting_debt[i] >= abs(sum(all other debts that are negative)) then we can settle the debt
        i dont want to touch people with zero debt because they have nothing to contribute
        debts can also be settled no matter what
        
        notes:
            we would never need to check if all were == 0 at the starting condition, this could never be the case
            we can settle all debts in n transactions, in the case where we have an addtional entity serving as the institution
            we can settle all debts in n-1 transactions if one of the people acts as the institution
        
        find all debts that are non zero, than backtrack to euqlize
        
        
        '''
        starting_debts = [0]*12
        for first, second, amount in transactions:
            starting_debts[first] -= amount
            starting_debts[second] += amount
            
        #only grab non zero entries
        starting_debts = [debt for debt in starting_debts if debt != 0]
        
        def rec(i):
            if i == len(starting_debts):
                return 0
            if starting_debts[i] == 0:
                return rec(i+1)
            cost = float('inf')
            for j in range(i+1,len(starting_debts)):
                starting_debts[j] += starting_debts[i]
                cost = min(cost,1 + rec(i+1))
                starting_debts[j] -= starting_debts[i]
            
            return cost
        
        return rec(0)
    
#need to optimize, and only carry debts over on allowed states
class Solution:
    def minTransfers(self, transactions: List[List[int]]) -> int:
        '''
        we need to slightly optimize
        if the current balance and the balance we wish to move are oppozite, we can clear one of the debts
        
        '''
        starting_debts = [0]*12
        for first, second, amount in transactions:
            starting_debts[first] -= amount
            starting_debts[second] += amount
            
        #only grab non zero entries
        starting_debts = [debt for debt in starting_debts if debt != 0]
        
        def rec(i):
            if i == len(starting_debts):
                return 0
            if starting_debts[i] == 0:
                return rec(i+1)
            curr_balance = starting_debts[i]
            cost = float('inf')
            for j in range(i+1,len(starting_debts)):
                next_balance = starting_debts[j]
                if curr_balance*next_balance > 0:
                    continue
                starting_debts[j] += curr_balance
                cost = min(cost,1 + rec(i+1))
                starting_debts[j] -= curr_balance
                
                #can slighlight prune
                #when curr_balance == next_balance, thsi is the best case, because onfe of the debts has been cleared
                if next_balance == curr_balance:
                    break
            
            return cost
        
        return rec(0)

class Solution:
    def minTransfers(self, transactions: List[List[int]]) -> int:
        '''
        intuition:
            for a group of n person with a total balance of 0, only n-1 transfers are needed
        we can degenerate the problem to:
            how many subgroupes the balance list can be divided into such that the sum of balances in each group is zero
            the number of subgroups would give the min transsactions
            so we have turned a minmization problem into a counting problem
        
        we can use bit masks to represent the states of the subgroups
        balance array [3,-3,1,-1], represetned by the bit mask(1111) menaing all are in this group
        then we recurse and try removing a person recursively
                (1111)
        (0111), (1011), (1101), (1110)
        
        we need to solve the problem for each of these subgroups
        but what we solving? we want the sums to in these subgroups to be 0
        then it would be max(rec(subgroup) for all subgroups) + (1111) sum is 0
        
        important:
        Once we obtain the optimal solution to the subproblems, an important step is still missing: 
        if the total balance of the current group is zero, it means that the sum of each subproblem is not zero
        
        Therefore, the non-zero part of the subproblem, plus the balance of the additional person in the current problem, make up an additional group whose sum is zero
        
        Thus, the optimal solution to the current problem is the maximum optional solution to its subproblems plus 1. However, if the total balance of the current group is not zero, this property does not hold.
        '''
        balance_map = collections.defaultdict(int)
        for a, b, amount in transactions:
            balance_map[a] += amount
            balance_map[b] -= amount
        
        balance_list = [amount for amount in balance_map.values() if amount]
        n = len(balance_list)
        print(balance_list)
        
        memo = [-1] * (1 << n)
        memo[0] = 0
        
        def dfs(total_mask):
            if memo[total_mask] != -1:
                return memo[total_mask]
            balance_sum, answer = 0, 0

            # Remove one person at a time in total_mask
            for i in range(n):
                cur_bit = 1 << i
                if total_mask & cur_bit:
                    balance_sum += balance_list[i]
                    answer = max(answer, dfs(total_mask ^ cur_bit))

            # If the total balance of total_mask is 0, increment answer by 1.
            memo[total_mask] = answer + (balance_sum == 0)
            return memo[total_mask]
        
        return n - dfs((1 << n) - 1)

                    

##########################################
# 2024. Maximize the Confusion of an Exam
# 07JUL23
#########################################
#fuckkk
class Solution:
    def maxConsecutiveAnswers(self, answerKey: str, k: int) -> int:
        '''
        given string of T's and F's, and using at most k operations where i can flip a T to F or F to T
        return the maximum consecutive Ts or Fs in the array
        
        TTFF
        
        kep track of maxi length at each position
        1200
        then use sliding window on the consec arrays making sure the we can only use k 0's in the window
        use consective arrays to figure out size
        
        '''
        N = len(answerKey)
        Ts = [0]*N
        Ts[0] = 1 if answerKey[0] == 'T' else 0
        for i in range(1,N):
            if answerKey[i] == 'T':
                Ts[i] = Ts[i-1] + 1
            else:
                Ts[i] = 0
        
        Fs = [0]*N
        Fs[0] = 1 if answerKey[0] == 'F' else 0
        for i in range(1,N):
            if answerKey[i] == 'F':
                Fs[i] = Fs[i-1] + 1
            else:
                Fs[i] = 0
        
        #sliding window on T's array
        ans = 0
        left = right = 0
        allowed_k = k
        #print(Ts)
        while right < N:
            while right < N and allowed_k >= 0:
                allowed_k -= Ts[right] == 0
                right += 1
            #print(Ts[left:right-1])
            ans = max(ans, len(Ts[left:right-1]))
            while left < N and allowed_k < k:
                allowed_k += Ts[left] == 0
                left += 1
        
        #do Fs
        left = right = 0
        allowed_k = k
        #print(Ts)
        while right < N:
            while right < N and allowed_k >= 0:
                allowed_k -= Fs[right] == 0
                right += 1
            #print(Ts[left:right-1])
            ans = max(ans, len(Fs[left:right-1]))
            while left < N and allowed_k < k:
                allowed_k += Fs[left] == 0
                left += 1
        
        return ans
    
#what the invariant for the sliding window, it could be anything really
class Solution:
    def maxConsecutiveAnswers(self, answerKey: str, k: int) -> int:
        '''
        sliding window, check on both Fs and Ts
        we need to keep a count of both F and Ts and if the min count of F or T is bigger than K, its not valid
        '''
        N = len(answerKey)
        #check Ts
        ans = 0
        counts = Counter()
        left = 0
        for right in range(N):
            counts[answerKey[right]] += 1
            
            while min(counts['T'],counts['F']) > k:
                counts[answerKey[left]] -= 1
                left += 1
            
            ans = max(ans,right - left + 1)
        
        return ans
    
class Solution:
    def maxConsecutiveAnswers(self, answerKey: str, k: int) -> int:
        '''
        we don't need to keep shrinking the window for every invalid subarray 
        if we have already found a valid window of length maxlength and we next find an invalid one, we dont need to keep shrking it one by one
        i.e we just need to find a window os max_size + 1
        just remove the left most answer in the window to keep the window at max_size
        '''
        max_size = 0
        counts = Counter()
        N = len(answerKey)
        max_size = 0
        
        for right in range(N):
            counts[answerKey[right]] += 1
            
            if min(counts['F'],counts['T']) <= k:
                max_size += 1
            
            #remove leftmost
            else:
                counts[answerKey[right - max_size]] -= 1
        
        return max_size
    
class Solution:
    def maxConsecutiveAnswers(self, A: str, k: int) -> int:
        max_len = left = 0
        k = {'T': k, 'F': k}
        for right in range(len(A)):
            k[A[right]] -= 1
            # while k['T'] < 0 and k['F'] < 0:
            #     k[A[left]] += 1
            #     left += 1
            # max_len = max(max_len, right - left + 1)
            if k['T'] < 0 and k['F'] < 0:
                k[A[left]] += 1
                left += 1
            else:
                max_len = max(max_len, right - left + 1)
        return max_len
    
########################################
# 2551. Put Marbles in Bags
# 09JUL23
#########################################
#dp gets TLE
class Solution:
    def putMarbles(self, weights: List[int], k: int) -> int:
        '''
        we need to put marbles into k bags such that
        1. no bag is empty
        2. if marble[i] and marble[j] are in a bag, then all marbles between i and j should also be in that same bag
        3. if a bag consists of all the marbles with an index from i to j inclusive, then cost is
            wieghts[i] + weights[j]
            
        score is the sum of all the costs of k bags
    
        
        return diff between maxi and min score bag distributions

        i can also say, split weights into k parititions
        for each partition scheme, find the differeence between the max and minimum scores on each paritition scheme
        
        
        '''
        #dp gets TLE, we try all possible splits and find the min and max
        memo_min = {}
        memo_max = {}
        N = len(weights)
        
        def dp_min(i,k):
            if i == N and k == 0:
                return 0
            if k < 0:
                return float('inf')
            
            if (i,k) in memo_min:
                return memo_min[(i,k)]
            
            cost = 0
            ans = float('inf')
            #try all splits
            for left in range(i+1,N):
                for right in range(left):
                    cost = weights[left] + weights[right] + dp_min(left+1,k-1)
                    ans = min(ans,cost)
            
            memo_min[(i,k)] = ans
            return ans
        
        
        def dp_max(i,k):
            if i == N and k == 0:
                return 0
            if k < 0:
                return float('-inf')
            
            if (i,k) in memo_min:
                return memo_max[(i,k)]
            
            cost = 0
            ans = float('-inf')
            #try all splits
            for left in range(i+1,N):
                for right in range(left):
                    cost = weights[left] + weights[right] + dp_max(left+1,k-1)
                    ans = max(ans,cost)
            
            memo_max[(i,k)] = ans
            return ans
        
        return dp_max(0,k) - dp_min(0,k)
        
#this is a variant of parition in to k subarrays
class Solution:
    def putMarbles(self, weights: List[int], k: int) -> int:
        '''
        check this out
        https://leetcode.com/problems/put-marbles-in-bags/discuss/3136136/Intuitive-approach-and-then-optimization-Greedy
        u have two option at any index

either cut at that index and add a[i]+a[i+1] to the answer ->solve(i+1,k-1) +a[i]+a[i+1]
skip that index -> solve(i+1,k)
but the catch is at 0th and n-1th index will ba added anyways irrespective or the manner how u r dividing / cutting the array so at 0th index
a[i] is added
and if u decide to keep only single element in bag then that element will be added twice so the same with the 0th element as well.

 if(i==0)
    {
       
        return dp1[i][k]= a[i]+min(solve1(a,i+1,k-1)+a[i]+a[i+1],solve1(a,i+1,k));
       
    }
    
    and when i=n-1 and k==1 then a[n-1] is returned


        '''
        N = len(weights)
        memo_min = {}
        memo_max = {}
        
        def dp_min(i,k):
            if k == 0:
                return float('inf')
            if i == N-1 and k == 1:
                return weights[i]
            if i == N-1 and k > 0:
                return float('inf')
            if (i,k) in memo_min:
                return memo_min[(i,k)]
            
            if i == 0:
                ans = weights[i] + min(dp_min(i+1,k-1) + weights[i] + weights[i+1], dp_min(i+1,k))
                memo_min[(i,k)] = ans
                return ans
            else:
                ans = min(dp_min(i+1,k-1) + weights[i] + weights[i+1], dp_min(i+1,k))
                memo_min[(i,k)] = ans
                return ans
            
            
        def dp_max(i,k):
            if k == 0:
                return float('-inf')
            if i == N-1 and k == 1:
                return weights[i]
            if i == N-1 and k > 0:
                return float('-inf')
            if (i,k) in memo_max:
                return memo_max[(i,k)]
            
            if i == 0:
                ans = weights[i] + max(dp_max(i+1,k-1) + weights[i] + weights[i+1], dp_max(i+1,k))
                memo_max[(i,k)] = ans
                return ans
            else:
                ans = max(dp_max(i+1,k-1) + weights[i] + weights[i+1], dp_max(i+1,k))
                memo_max[(i,k)] = ans
                return ans
            
            
        return dp_max(0,k) - dp_min(0,k)
    
#true solution is actually greedy
class Solution:
    def putMarbles(self, weights: List[int], k: int) -> int:
        '''
        we need to put marbles into k bags such that
        1. no bag is empty
        2. if marble[i] and marble[j] are in a bag, then all marbles between i and j should also be in that same bag
        3. if a bag consists of all the marbles with an index from i to j inclusive, then cost is
            wieghts[i] + weights[j]
            
        score is the sum of all the costs of k bags
    
        
        return diff between maxi and min score bag distributions

        i can also say, split weights into k parititions
        for each partition scheme, find the differeence between the max and minimum scores on each paritition scheme
        if we parition the array into k groups, we always make k-1 splitting points, and we only neeed to examine the boundary points
        
        we need to fin the sum of the largest k-1 paris and the sum of the smallest k-1 pairs
        '''
        pairs = sorted([weights[i - 1] + weights[i] for i in range(1, len(weights))])
        return sum(pairs[len(pairs) - k + 1:]) - sum(pairs[:(k - 1)])
    
class Solution:
    def putMarbles(self, weights: List[int], k: int) -> int:   
        # We collect and sort the value of all n - 1 pairs.
        n = len(weights)
        pair_weights = [0] * (n - 1)
        for i in range(n - 1):
            pair_weights[i] = weights[i] + weights[i + 1]
        pair_weights.sort()
        
        # Get the difference between the largest k - 1 values and the 
        # smallest k - 1 values.
        answer = 0
        for i in range(k - 1):
            answer += pair_weights[n - 2 - i] - pair_weights[i]
            
        return answer
    
'''
formal derivations
max_score = weights[0] + weights[n-1] + \sum_{i = n-k}^{n-1}
min_score = weights[0] + weights[n-1] + \sum_{i=0}^{k-2}
ans = max_score = min_score
ans = \sum_{i= n-k}^{n-1} - \sum_{i=0}^{k-2}
'''

########################################
# 2272. Substring With Largest Variance
# 09JUL23
########################################
#almonst, just kadanes repreated 
# general kadane's does not work because we have negative values
class Solution:
    def largestVariance(self, s: str) -> int:
        '''
        we define variance of a string as
            largest difference between the number of occruences of any 2 characters in a string
        
        return the largesrt variance of all possible substrings s
        well we know its dp
        hint1, solve if the string only had 2 distinct characters
        hint2, replace all occruence of the first char by +1 and second char by-1
        hint3, try all combindatinos
        aababbb
    
        [1,1,-1,1,-1,-1,-1]
        max sum subarray or min sub subarray
        decode the string into to chars (u,v) then find maximim sum
        '''
        if len(s) == len(set(s)):
            return 0
        
        counts = Counter(s)
        ans = 0
        N = len(s)
        for i in range(26):
            for j in range(26):
                first = chr(ord('a') + i)
                second = chr(ord('a') + j)
                #must both appar and cannot by the same
                if i != j and counts[first] > 0 and counts[second] > 0:
                    #create aux array
                    temp = [0]*N
                    for i in range(N):
                        if s[i] == first:
                            temp[i] = 1
                        elif s[i] == second:
                            temp[i] = -1
                        else:
                            temp[i] = 0
                    #find maximum, two cases
                    dp_max = [0]*N
                    dp_max[0] = temp[0]


                    for i in range(1,N):
                        dp_max[i] = max(dp_max[i-1] + temp[i],temp[i])

                    #find max on both
                    ans = max(ans,max(dp_max))
        return ans
                
#need modified version of Kadanes
class Solution:
    def largestVariance(self, s: str) -> int:
        '''
        we define variance of a string as
            largest difference between the number of occruences of any 2 characters in a string
        
        return the largesrt variance of all possible substrings s
        well we know its dp
        hint1, solve if the string only had 2 distinct characters
        hint2, replace all occruence of the first char by +1 and second char by-1
        hint3, try all combindatinos
        aababbb
    
        [1,1,-1,1,-1,-1,-1]
        max sum subarray or min sub subarray
        decode the string into to chars (u,v) then find maximim sum
        
        the issue with kadanes is that it allows so a subarray to have 0 occruence of a major or minor elements
        however, a valid substring MUST contain one major and one minor
        so we only update global max when minor_count > 0
        i.e no element with negative value would be allowed with kadane's algo
        
        and reset local max to - when there is at least one minor in the remainin substring
        
        recall that we need a step local_max = max(local_max,0) in regular Kadane's
        but we cannot simply reset the sum to 0 here in this problem
            doing so would reset boht major count and minor count to 0
            if there are no more minors in the remaning traversal, the minor count will remain - and we would never update global
            to avoid this we reset local_max to 0 only when there is at least one minor in the remaning s
            to do this we can use an additional variable rest_minor to keep track of the minors in the remaining string
        '''
        counts = Counter(s)
        ans = 0
        N = len(s)
        for i in range(26):
            for j in range(26):
                first = chr(ord('a') + i)
                second = chr(ord('a') + j)
                #major and minor cannot be the same and must appear in s
                if first == second or counts[first] == 0 or counts[second] == 0:
                    continue
                
                major_count = 0
                minor_count = 0
                
                #get remaning of s
                rest_minor = counts[second]
                for ch in s:
                    if ch == first:
                        major_count += 1
                    if ch == second:
                        minor_count += 1
                        rest_minor -= 1
                    
                    #only update the variance (local max) if we have at least aminor
                    if minor_count > 0:
                        ans = max(ans, major_count - minor_count)
                    
                    #dist care the previous string if there is at leat one remaining minor
                    if major_count < minor_count and rest_minor > 0:
                        major_count = 0
                        minor_count = 0
        
        return ans
    
class Solution:
    def largestVariance(self, s: str) -> int:
        '''
        same thing as Kadanes' the only problem is that we have to maintain a substring with an a and a b
        the presence of another char other than a or b would break the subarray sum
        so we have to watch for that
        '''
        counts = Counter(s)
        ans = 0
        for a in counts.keys():
            for b in counts.keys():
                if a == b:
                    continue
                #keep tracking of remaining
                remaining_a = counts[a]
                remaining_b = counts[b]
                variance = 0
                has_a = has_b = False
                for ch in s:
                    if ch != a and ch != b:
                        continue
                    if ch == a:
                        variance += 1
                        remaining_a -= 1
                        has_a = True
                    else:
                        variance -= 1
                        remaining_b -= 1
                        has_b = True
                    
                    #special cases, 
                    ## abbb case: cannot reset after abb as we can build longer substring abbb, this is the locl max rest part
                    if variance < 0 and remaining_a > 0 and remaining_b > 0:
                        variance = 0
                        has_a = has_b = False
                        
                    if has_a and has_b:
                        ans = max(ans,variance)
            
        
        return ans
    
########################################################
# 1481. Least Number of Unique Integers after K Removals
# 10JUL23
#########################################################
#jesus fuck
class Solution:
    def findLeastNumOfUniqueInts(self, arr: List[int], k: int) -> int:
        '''
        remove the least frequent ones first
        '''
        counts = Counter(arr)
        min_counts = [(v,k) for k,v in counts.items()]
        heapq.heapify(min_counts)
        
        while len(min_counts) > 0 and k > 0:
            curr_count, curr_num = heapq.heappop(min_counts)
            curr_count -= 1
            k -= 1
            if curr_count > 0:
                heapq.heappush(min_counts, (curr_count,curr_num))
            
        
        return len(min_counts)
    
class Solution:
    def findLeastNumOfUniqueInts(self, arr: List[int], k: int) -> int:
        '''
        exclude k smallest
        '''
        counts = Counter(arr)
        #sort base on counts
        s = sorted(arr,key = lambda x: (counts[x],x)) #important to break ties
        #remove k smallest
        return len(set(s[k:]))
    

#######################################
# 529. Minesweeper
# 11JUL23
######################################
#im thkning too hard just do on the board and follow the rules
class Solution:
    def updateBoard(self, board: List[List[str]], click: List[int]) -> List[List[str]]:
        '''
        first find the mines and for neighboring cells of that mine, 
            if they are 'E', change them to a '1'
            then bfs, adding more mines until we are out of mines
            pre-allocate grid where we are storing numbers
        
        then bfs again from click
        '''
        #if we hit a mine, change it and return board
        if board[click[0]][click[1]] == 'M':
            board[click[0]][click[1]] = 'X'
            return board
        rows = len(board)
        cols = len(board[0])
        dirrs = []
        for dx in [1,0,-1]:
            for dy in [1,0,-1]:
                if (dx,dy) != (0,0):
                    dirrs.append((dx,dy))
        
        mines = []
        for i in range(rows):
            for j in range(cols):
                if board[i][j] == 'M':
                    mines.append((i,j))
        
        bomb_counts = board[:][:]
        #modify this with the bomb scores
        for x,y in mines:
            for dx,dy in dirrs:
                neigh_x = x + dx
                neigh_y = y + dy
                #bounds
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                    #if already has number
                    if '1' <= bomb_counts[neigh_x][neigh_y] <= '8':
                        bomb_counts[neigh_x][neigh_y] = str(int(bomb_counts[neigh_x][neigh_y]) + 1)
                    #first bomb
                    else:
                        bomb_counts[neigh_x][neigh_y] = '1'
        
        #put mines back
        for x,y in mines:
            bomb_counts[x][y] = 'M'
        
        #bfs from this first clock
        seen = set()
        q = deque([(click[0],click[1])])
        
        while q:
            x,y = q.popleft()
            seen.add((x,y))
            #if its a number on bomb_counts, reveal that number in the board and continue
            if '1' <= bomb_counts[x][y] <= '8':
                board[x][y] = bomb_counts[x][y]
                print(x,y)
                continue
            #unrevealed empty
            if board[x][y] == 'E':
                board[x][y] = 'B'
                #dont continue, search from here
            #only look 4 directions here
            for dx,dy in [(1,0),(-1,0),(0,-1),(0,1)]:
                neigh_x = x + dx
                neigh_y = y + dy
                #bounds
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                    if (neigh_x,neigh_y) not in seen:
                        q.append((neigh_x,neigh_y))
        
        return board
    
#bfs
class Solution:
    def updateBoard(self, board: List[List[str]], click: List[int]) -> List[List[str]]:
        '''
        im  a dumb dumb, just count mines surrounding at the current cell, and update or don't update accorindling
        gahhhhhhh
        '''
        #get starts as i,j
        i,j = click
        #if we hit a mine, change it and return board
        if board[i][j] == 'M':
            board[i][j] = 'X'
            return board
        
        rows = len(board)
        cols = len(board[0])
        dirrs = []
        for dx in [1,0,-1]:
            for dy in [1,0,-1]:
                if (dx,dy) != (0,0):
                    dirrs.append((dx,dy))
                    
        q = deque([(i,j)])
        
        while q:
            x,y = q.popleft()
            #if we are unrevealed
            if board[x][y] == 'E':
                #get mine count
                mine_count = 0
                for dx,dy in dirrs:
                    neigh_x = x + dx
                    neigh_y = y + dy
                    #bounds
                    if 0 <= neigh_x < rows and 0 <= neigh_y < cols and board[neigh_x][neigh_y] == 'M':
                        mine_count += 1
                
                #if we have a mine count
                if mine_count > 0:
                    board[x][y] = str(mine_count)
                #othereise reveal and add neights
                else:
                    board[x][y] = 'B'
                    for dx,dy in dirrs:
                        neigh_x = x + dx
                        neigh_y = y + dy
                        #bounds
                        if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                            q.append((neigh_x,neigh_y))
        
        return board
    
#dfs
class Solution:
    def updateBoard(self, board: List[List[str]], click: List[int]) -> List[List[str]]:
        '''
        im  a dumb dumb, just count mines surrounding at the current cell, and update or don't update accorindling
        gahhhhhhh
        '''
        #get starts as i,j
        i,j = click
        #if we hit a mine, change it and return board
        if board[i][j] == 'M':
            board[i][j] = 'X'
            return board
        
        rows = len(board)
        cols = len(board[0])
        dirrs = []
        for dx in [1,0,-1]:
            for dy in [1,0,-1]:
                if (dx,dy) != (0,0):
                    dirrs.append((dx,dy))
                    
        q = deque([(i,j)])
        
        def dfs(x,y):
            if board[x][y] == 'E':
                #get mine count
                mine_count = 0
                for dx,dy in dirrs:
                    neigh_x = x + dx
                    neigh_y = y + dy
                    #bounds
                    if 0 <= neigh_x < rows and 0 <= neigh_y < cols and board[neigh_x][neigh_y] == 'M':
                        mine_count += 1
                
                #if we have a mine count
                if mine_count > 0:
                    board[x][y] = str(mine_count)
                #othereise reveal and add neights
                else:
                    board[x][y] = 'B'
                    for dx,dy in dirrs:
                        neigh_x = x + dx
                        neigh_y = y + dy
                        #bounds
                        if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                            dfs(neigh_x,neigh_y)
        
        dfs(i,j)
        return board


#####################################
# 802. Find Eventual Safe States
# 12JUL23
#####################################
class Solution:
    def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
        '''
        terminal node is anode that has no out going edges
        a node is afe if every possible path starting from that nodes leads to a terminal node
        return all safe nodes in ascending order
        
        brute force would be to dfs from each node and check that we always lade on a safe node ON EVERY PATH PATH
        any node that leads to a cycle cannot be a safe node
        better brute force would be to dfs on each node and check for cycle, which is still long
        
        we can use dfs to add the nodes in a the cyle while doing dfs on each node
        then retravese the nodes and see if that node was part of a cycle, which means it is not a safe node
        '''
        visited = set()
        in_cycle = set()
        
        #make adjaceny list
        adj_list = defaultdict(list)
        N = len(graph)
        for i in range(N):
            #recal its a DAG
            for neigh in graph[i]:
                adj_list[i].append(neigh)
        
        #detect cycles
        def has_cycle(node,visited,in_cycle):
            if node in in_cycle:
                return True
            if node in visited:
                return False
            #make viste and in current cycle
            visited.add(node)
            in_cycle.add(node)
            for neigh in adj_list[node]:
                if has_cycle(neigh,visited,in_cycle):
                    return True
            
            #remove from in cycle to allow
            in_cycle.remove(node)
            return False
        
        for i in range(N):
            has_cycle(i,visited,in_cycle)
        
        safe = []
        for i in range(N):
            if i not in in_cycle:
                safe.append(i)
        
        return safe
    
class Solution:
    def eventualSafeNodes(self, graph: List[List[int]]) -> List[int]:
        '''
        we need to use kahns algorithm and start with the nodes that have no out going edges
        recall terminal node = nodes with no outgoing edges
        safe node = for every possible path starting from this node, leads to a terminal node
        terminal nodes are naively safe
        any node part of cycle cannot be safe, why?
            would back to itself at the end of a path
        
        intuition:
            a node is safe if all outgoing edges are to nodes that are safe
            and since the all the neighbors are safe, no neighbor should lead to a cycle
        
        we know that terminal nodes are safe, and so nodes that have ONLY outgoing edges to terminal nodes are also safe
        so we start from terminal nodes
            reverse edges and create new graph, visit terminal nodes, remove edges, then go to the next ones
        
        a node is safe if all its incoming edges come from previously safe nodes
            if we erase the outging edges from the safe node and discover a node with no incoming edges, it is a new safe node
            KAHNS! topological sort
            repeatedly visting nodes with indegress 0 and deleting all the edges associted with it, leading to a decreement of indegree to neighboring nodes
            continue process until no eleement with zero indegree can be found
        
        the nodes in the reverse graph would never be visited again
            since the nodes have no out going edges
            bascially every node in the original network that has a path to the cycle will never be visited by Kahns
        
        takeaway, if we start from terminal nodes, we visit the next of neighbors for each terminal nodes but remove an edge
        since we only add back into the q if indegree is zero, nodes in the cycle would never be visited because the would have indegree of 1
        
        '''
        N = len(graph)
        in_degree = [0]*N
        adj_list = defaultdict(list)
        
        #reverse edges, we need to start from terminal nodes, remove edges, and find new safe nodes
        #only nodes in the cyle would remain unvisted in the traversal
        for i in range(N):
            for neigh in graph[i]:
                adj_list[neigh].append(i)
                in_degree[i] += 1
        
        safe_nodes = [False]*N
        
        q = deque([])
        
        #add in terminal nodes
        for i in range(N):
            if in_degree[i] == 0:
                q.append(i)
                
        while q:
            curr = q.popleft()
            safe_nodes[curr] = True
            
            for neigh in adj_list[curr]:
                #remove edge
                in_degree[neigh] -= 1
                #new safe node
                if in_degree[neigh] == 0:
                    q.append(neigh)
        
        return [i for i in range(N) if safe_nodes[i]]
    
#########################################
# 863. All Nodes Distance K in Binary Tree
# 11JUL23
##########################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        '''
        remake the tree into a graph, then dfs from target getting paths lenghts up to size k
        '''
        graph = defaultdict(list)
        
        def dfs(node):
            if not node:
                return
            if node.left:
                graph[node.val].append(node.left.val)
                graph[node.left.val].append(node.val)
            if node.right:
                graph[node.val].append(node.right.val)
                graph[node.right.val].append(node.val)
            
            dfs(node.left)
            dfs(node.right)
            
        dfs(root)
        #dfs from target
        ans = []
        seen = set()
        
        def dfs2(node,k,seen):
            seen.add(node)
            if k == 0:
                ans.append(node)
                return
            for neigh in graph[node]:
                if neigh not in seen:
                    dfs2(neigh,k-1,seen)
        
        dfs2(target.val,k,seen)
        return ans
    
#BFS
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        '''
        we can also do bfs from the target node
        '''
        graph = defaultdict(list)
        
        def dfs(node):
            if not node:
                return
            if node.left:
                graph[node.val].append(node.left.val)
                graph[node.left.val].append(node.val)
            if node.right:
                graph[node.val].append(node.right.val)
                graph[node.right.val].append(node.val)
            
            dfs(node.left)
            dfs(node.right)
            
        dfs(root)
        #dfs from target
        ans = []
        seen = set()
        

        q = deque([(target.val,k)])
        
        while q:
            node, curr_k = q.popleft()
            seen.add(node)
            if curr_k == 0:
                ans.append(node)
                continue #no no eed to go down anymore
            
            for neigh in graph[node]:
                if neigh not in seen:
                    q.append((neigh,curr_k - 1))
        
        return ans
    
#we can also modify parent pointers
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        '''
        we can also do bfs from the target node
        '''
        graph = defaultdict(list)
        
        def add_parent(node,parent):
            if node:
                node.parent = parent
                add_parent(node.left,node)
                add_parent(node.right,node)
        
        add_parent(root,None)
        ans = []
        seen = set()
        def dfs(node,k,seen):
            if not node or node.val in seen:
                return
            seen.add(node.val)
            if k == 0:
                ans.append(node.val)
                return
            dfs(node.parent,k-1,seen)
            dfs(node.left,k-1,seen)
            dfs(node.right,k-1,seen)
        
        dfs(target,k,seen)
        return ans
    
########################################
# 545. Boundary of Binary Tree
# 12JUL23
########################################
#fuck, this problem, almost had it....

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def boundaryOfBinaryTree(self, root: Optional[TreeNode]) -> List[int]:
        '''
        dfs, flags for root and in left boundary
        lists for left, right and leaves
        '''
        root_boundary = []
        left_boundary = []
        leaves = []
        right_boundary = []
        
        def dfs(node,is_root,is_left_boundary,is_right_boundary):
            if not node:
                return
            
            if is_left_boundary:
                left_boundary.append(node.val)
            
            if is_right_boundary:
                right_boundary.append(node.val)
            
            elif not node.left and not node.right:
                leaves.append(node.val)
            
            if is_root:
                root_boundary.append(root.val)
                if node.left:
                    dfs(node.left,False,True,False)
                if node.right:
                    dfs(node.right,False,False,True)
            
            else:
                dfs(node.left,is_root,is_left_boundary,is_right_boundary)
                dfs(node.right,is_root,is_left_boundary,is_right_boundary)
        
        dfs(root,True,False,False)
        print(root_boundary,left_boundary,leaves,right_boundary)

#function helper mania
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def boundaryOfBinaryTree(self, root: Optional[TreeNode]) -> List[int]:
        '''
        the [root] + [left_boundary] + [leaves] + [right_boundary] is simlilar to the pre-order traversal, just minus the middle nodes
        and the right boundary is also reversed, 
        we can use flag variables to mark whether they belong a certain part of the boundary
        flag 0 = root, flag 1 = left, flag 2 = right, flag 3 = others
        then when going left and right define left_child_flag and right_child flag
        for left_child_flag:
            if curr is a left boundary node, let will alwasy be in the left boundary
            if curr node is roote node, left child will alwasy be left boundary
            if curr node is right boundary node
                if there is no right child, the left always acts as the right boundar
                if there is a right child, left child acts as middle
        
        for right_child_flag:   
            if curr node is right boundary, right child will alwasy be right boundary
            if curr node is root, right child will always be right boundary
            if curr node ie left boundary:
                if there is no left child, right child is still left boundary
                if there is a left child, child acts as middle node
        '''
        left_boundary = []
        right_boundary = []
        leaves = []
        
        def is_leaf(node):
            return not node.left and not node.right
        def is_right_boundary(flag):
            return flag == 2
        def is_left_boundary(flag):
            return flag == 1
        def is_root(flag):
            return flag == 0
        def left_child_flag(curr,flag):
            if is_left_boundary(flag) or is_root(flag):
                return 1
            elif is_right_boundary(flag) and not curr.right:
                return 2
            else:
                return 3
        def right_child_flag(curr,flag):
            if is_right_boundary(flag) or is_root(flag):
                return 2
            elif is_left_boundary(flag) and not curr.left:
                return 1
            else:
                return 3
            
        def preorder(node,left_boundary,right_boundary,leaves,flag):
            if not node:
                return
            if is_right_boundary(flag):
                right_boundary.append(node.val)
            elif is_left_boundary(flag) or is_root(flag):
                left_boundary.append(node.val)
            elif is_leaf(node):
                leaves.append(node.val)
            preorder(node.left,left_boundary,right_boundary,leaves,left_child_flag(node,flag))
            preorder(node.right,left_boundary,right_boundary,leaves,right_child_flag(node,flag))
            
        preorder(root,left_boundary,right_boundary,leaves,0)
        #print(left_boundary,right_boundary,leaves)
        return left_boundary + leaves + right_boundary[::-1]

#bfs wont work
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def boundaryOfBinaryTree(self, root: Optional[TreeNode]) -> List[int]:
        '''
        easier to divide into three parts, 
        findinv leaves is easy,
        we just need to find the left boundary and right boundary
        use bfs to find the left and right bounds, then dfs to find the leaves
        '''
        root_vals = []
        left_boundary = []
        right_boundary = []
        leaves = []
        
        q = deque([(root,True)])
        
        while q:
            N = len(q)
            for i in range(N):
                curr,is_root = q.popleft()
                if is_root:
                    root_vals.append(curr.val)
                elif i == 0:
                    left_boundary.append(curr.val)
                elif i == N-1:
                    right_boundary.append(curr.val)
                
                if curr.left:
                    q.append((curr.left,False))
                if curr.right:
                    q.append((curr.right,False))
        #pop the last items from left if right
        if left_boundary:
            left_boundary.pop()
        if right_boundary:
            right_boundary.pop()
            
        #find leaves
        def find_leaf(node,leaves):
            if not node:
                return
            if not node.left and not node.right:
                leaves.append(node.val)
                return
            find_leaf(node.left,leaves)
            find_leaf(node.right,leaves)
            
        find_leaf(root,leaves)
        print(root_vals,left_boundary,leaves, right_boundary[::-1])

#traverse left, leaves, then right
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def boundaryOfBinaryTree(self, root: Optional[TreeNode]) -> List[int]:
        '''
        easier to divide into three parts, 
        findinv leaves is easy,
        we just need to find the left boundary and right boundary
        use bfs to find the left and right bounds, then dfs to find the leaves
        '''
        ans = []
        if not root:
            return [] 
        leaves = []
        #find leaves
        def find_leaf(node,leaves):
            if not node:
                return
            if not node.left and not node.right:
                leaves.append(node.val)
                return
            find_leaf(node.left,leaves)
            find_leaf(node.right,leaves)
            
        def is_leaf(node):
            return not node.left and not node.right
        
        if not is_leaf(root):
            ans.append(root.val)
        
        #left boundary first
        curr = root.left
        while curr:
            if not is_leaf(curr):
                ans.append(curr.val)
            if curr.left:
                curr = curr.left
            else:
                curr = curr.right
        
        #leaves
        find_leaf(root,ans)
        right_boundary = [] #add its reverse after
        curr = root.right
        while curr:
            if not is_leaf(curr):
                right_boundary.append(curr.val)
            if curr.right:
                curr = curr.right
            else:
                curr = curr.left
        
        ans += right_boundary[::-1]
        return ans


#######################
# 207. Course Schedule
# 13JUL23
#######################
#cycle detection
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        '''
        cycle detection algorithm, if there is a cycle anywhere we can't do it
        '''
        adj_list = defaultdict(list)
        for a,b in prerequisites:
            adj_list[a].append(b)
        
        #two sets for cycles, visited and on traversal
        visited = set()
        in_cycle = set()
        def has_cycle(node,visited,in_cycle):
            if node in in_cycle:
                return True
            if node in visited:
                return False
            visited.add(node)
            in_cycle.add(node)
            for neigh in adj_list[node]:
                if has_cycle(neigh,visited,in_cycle):
                    return True
            
            in_cycle.remove(node)
            return False
        
        
        for i in range(numCourses):
            has_cycle(i,visited,in_cycle)

        
        return len(in_cycle) == 0
    
#top sort
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        '''
        we can use top sort
        start with classes that have 0 indegree, meaning that we can take them first
        then take neighbors, remove edge, and add back into q is in degree is now zero
        '''
        N = numCourses
        taken = [False]*N
        adj_list = defaultdict(list)
        in_degree = [0]*N
        
        for a,b in prerequisites:
            adj_list[a].append(b)
            in_degree[b] += 1
        
        q = deque([])
        for i in range(N):
            if in_degree[i] == 0:
                q.append(i)
        
        while q:
            node = q.popleft()
            taken[node] = True
            for neigh in adj_list[node]:
                in_degree[neigh] -= 1
                if in_degree[neigh] == 0:
                    q.append(neigh)
        
        return sum(taken) == N

###########################################
# 1018. Binary Prefix Divisible By 5
# 13JUL23
###########################################
#bit shift and set
class Solution:
    def prefixesDivBy5(self, nums: List[int]) -> List[bool]:
        '''
        i number is divisible by 5 of it ends in 0 or 5
        i could generate the integer in O(1) time from the previous and check if mod 5 == 0
        i.e dp(i) = 
        '''
        N = len(nums)
        ans = [False]*N
        #check first
        ans[0] = nums[0] % 5 == 0
        prev = nums[0]
        for i in range(1,N):
            #get new digits
            prev = prev << 1
            #set bit
            prev = prev | nums[i]
            ans[i] = prev % 5 == 0
        
        return ans
    
############################################################
# 1218. Longest Arithmetic Subsequence of Given Difference
# 14JUL23
###########################################################
#brute force
class Solution:
    def longestSubsequence(self, arr: List[int], difference: int) -> int:
        '''
        pure brute force wouild be to try each arr[i], then look at all j in range(i+1,N)
        and if arr[j] - arr[i] = difference*(num elemetns in subsequence)
        '''
        #brute force
        N = len(arr)
        ans = 1
        for i in range(N):
            curr_size = 1
            for j in range(i+1,N):
                if arr[j] - arr[i] == curr_size*difference:
                    curr_size += 1
            
            ans = max(ans,curr_size)
    
        return ans
#gahh
class Solution:
    def longestSubsequence(self, arr: List[int], difference: int) -> int:
        '''
        need to keep track of the position im in as well as the difference
        dp(i,diff) = longest subsequence ending at i with difference diff
            if [i+1] - i == diff, we can extend:
                dp(i+1,diff) + 1
            otherwise we have to break the sequence here and move up
            the answer at this position is the max
        '''
        
        memo = {}
        N = len(arr)
        
        def dp(i,diff):
            #dont go out of bounrs
            if i == N-1:
                return 1
            if (i,diff) in memo:
                return memo[(i,diff)]
            next_diff = arr[i+1] - arr[i]
            extend = 0
            if next_diff == diff:
                extend = 1 + dp(i+1,next_diff)
            dont_take = dp(i+1,diff)
            take_but_no_extend = dp(i+1,next_diff)
            ans = max(extend,dont_take,take_but_no_extend)
            memo[(i,diff)] = ans
            return ans
        
        return dp(0,difference)
    
#we need the position where it last ended with diff, TLE/MLE, too many states
class Solution:
    def longestSubsequence(self, arr: List[int], difference: int) -> int:
        '''
        need to keep track of the position im in as well as the difference
        dp(i,diff) = longest subsequence ending at i with difference diff
            if [i+1] - i == diff, we can extend:
                dp(i+1,diff) + 1
            otherwise we have to break the sequence here and move up
            the answer at this position is the max
            and we also need the last index where it ended with diff!
        '''
        
        memo = {}
        N = len(arr)
        
        def dp(i,last_index,diff):
            if i == N:
                return 0
            if (i,last_index,diff) in memo:
                return memo[(i,last_index,diff)]
            
            no_take = dp(i+1,last_index,diff)
            take = 0
            if last_index == -1 or arr[i] - arr[last_index] == diff:
                take = 1 + dp(i+1,i,diff)
            
            ans = max(take,no_take)
            memo[(i,last_index,diff)] = ans
            return ans
        
        return dp(0,-1,difference)
    

#another way would be to pass in the last index position, as well as the last element added to that subsequence
class Solution:
    def longestSubsequence(self, arr: List[int], difference: int) -> int:
        '''
        need to keep track of the position im in as well as the difference
        dp(i,diff) = longest subsequence ending at i with difference diff
            if [i+1] - i == diff, we can extend:
                dp(i+1,diff) + 1
            otherwise we have to break the sequence here and move up
            the answer at this position is the max
            and we also need the last index where it ended with diff!
        '''
        
        memo = {}
        N = len(arr)
        
        def dp(i,last_element,diff):
            if i == N:
                return 0
            if (i,last_element,diff) in memo:
                return memo[(i,last_element,diff)]
            
            no_take = dp(i+1,last_element,diff)
            take = 0
            if last_element == float('-inf') or arr[i] - last_element == diff:
                take = 1 + dp(i+1,arr[i],diff)
            
            ans = max(take,no_take)
            memo[(i,last_element,diff)] = ans
            return ans
        
        return dp(0,float('-inf'),difference)

#binary search with dp
class Solution:
    def longestSubsequence(self, arr: List[int], difference: int) -> int:
        '''
        we can use binary search to quickly find the next occurence of the element + difference that should be included in the subsequence
        say we are at index i
            we need to find all indicies j, where j > i and arr[j] - arr[i] == difference
            we can use binary search to speed up 
            then recurse on index j by adding 1 to it
        i.e dp(i) = longest_subseuqnece with difference i starting i
        dp(i) = {
            for all j, where j > i:
                if arr[j] - arr[i] = difference:
                    1 + rec(j)
        }
        
        
        '''
        #mapp elements to ther index
        elements_to_idx = defaultdict(list)
        for i,num in enumerate(arr):
            elements_to_idx[num].append(i)
        
        N = len(arr)
        memo = {}
        
        def dp(i):
            #find next element in subsequence
            next_num = arr[i] + difference
            if next_num not in elements_to_idx:
                return 1
            if i in memo:
                return memo[i]
            #find next index
            j = bisect.bisect_left(elements_to_idx[next_num],i)
            #edge case, when differnece is zero
            if difference == 0:
                j += 1
            #no such index
            if j == len(elements_to_idx[next_num]):
                return 1
            ans = 1 + dp(elements_to_idx[next_num][j])
            memo[i] = ans
            return ans
        
        longest = 1
        for i in range(N):
            longest = max(longest,dp(i))
        
        return longest
            
class Solution:
    def longestSubsequence(self, arr: List[int], difference: int) -> int:
        '''
        pure brute force wouild be to try each arr[i], then look at all j in range(i+1,N)
        and if arr[j] - arr[i] = difference*(num elemetns in subsequence)
        
        instead of doing brute force, recorde the count of the longest subsequence ending with this curren difference
        we just need to check if arr[i] - difference is already in the array
        since arr[i] - difference would have been the last element added in the subsequence
        '''
        last_seen_element = {}
        
        for num in arr:
            prev = num - difference
            curr_length = last_seen_element.get(prev,0)
            last_seen_element[num] = curr_length + 1
        
        return max(last_seen_element.values())

##########################################################
# 1751. Maximum Number of Events That Can Be Attended II
# 15JUL23
#########################################################
#yes!
class Solution:
    def maxValue(self, events: List[List[int]], k: int) -> int:
        '''
        notes:
            end dates are inclusive, can only attend on event at a time, if we choose this event we must stay for the entire event
            you cannot attend two events where one starts and the other ends
        sort on start, 0/1 knapsack take the event or don't take
        need to keep track of the ith event that we are on
        use binary search to find the next event that i can take
        keep track of the time and the current and events k
        
        '''
        N = len(events)
        events.sort(key = lambda x : x[0])
        start_times = [x[0] for x in events]
        
        print(start_times)
        
        memo = {}
        
        def dp(time,i,k):
            if i == N:
                return 0
            if k == 0:
                return 0
            if (time,i,k) in memo:
                return memo[(time,i,k)]
            #if we can take
            take = 0
            if events[i][0] >= time:
                next_time = events[i][1]
                #look for the smallest index j that is just greater than next time to take it
                next_index = bisect.bisect_right(start_times,next_time)
                if next_index <= N:
                    take = events[i][2] + dp(next_time,next_index,k-1)
                    
            dont_take = dp(time,i+1,k)
            ans = max(take,dont_take)
            memo[(time,i,k)] = ans
            return ans
        
        return dp(0,0,k)
    
#we don't need to keep track of the times just the index
class Solution:
    def maxValue(self, events: List[List[int]], k: int) -> int:
        '''
        notes:
            end dates are inclusive, can only attend on event at a time, if we choose this event we must stay for the entire event
            you cannot attend two events where one starts and the other ends
        sort on start, 0/1 knapsack take the event or don't take
        need to keep track of the ith event that we are on
        use binary search to find the next event that i can take
        keep track of the time and the current and events k
        
        '''
        N = len(events)
        events.sort(key = lambda x : x[0])
        start_times = [x[0] for x in events]
        
        memo = {}
        
        def dp(i,k):
            if i == N:
                return 0
            if k == 0:
                return 0
            if (i,k) in memo:
                return memo[(i,k)]
            #if we can take
            take = 0
            next_time = events[i][1]
            #look for the smallest index j that is just greater than next time to take it
            next_index = bisect.bisect_right(start_times,next_time)
            if next_index <= N:
                take = events[i][2] + dp(next_index,k-1)
                    
            dont_take = dp(i+1,k)
            ans = max(take,dont_take)
            memo[(i,k)] = ans
            return ans
        
        return dp(0,k)
    
#bottom up
class Solution:
    def maxValue(self, events: List[List[int]], k: int) -> int:
        '''
        notes:
            end dates are inclusive, can only attend on event at a time, if we choose this event we must stay for the entire event
            you cannot attend two events where one starts and the other ends
        sort on start, 0/1 knapsack take the event or don't take
        need to keep track of the ith event that we are on
        use binary search to find the next event that i can take
        keep track of the time and the current and events k
        
        '''
        N = len(events)
        events.sort(key = lambda x : x[0])
        start_times = [x[0] for x in events]
        
        dp = [[0]*(k+1) for _ in range(N+1)]
        
        
        for i in range(N-1,-1,-1):#start from last n
            #start from fist k
            for count in range(1,k+1):
                take = 0
                next_time = events[i][1]
                #look for the smallest index j that is just greater than next time to take it
                next_index = bisect.bisect_right(start_times,next_time)
                if next_index <= N:
                    take = events[i][2] + dp[next_index][count-1]

                dont_take = dp[i+1][count]
                ans = max(take,dont_take)
                dp[i][count] = ans
        
        return dp[0][k]
    
#precomputet binary searches before hand
class Solution:
    def maxValue(self, events: List[List[int]], k: int) -> int:
        '''
        if wanted to, we could precompute the next indices for each event
        
        '''
        N = len(events)
        events.sort(key = lambda x : x[0])
        start_times = [x[0] for x in events]
        next_index_map = []
        for start in start_times:
            next_index_map.append(bisect.bisect_right(start_times,start))
        
        dp = [[0]*(k+1) for _ in range(N+1)]
        
        
        for i in range(N-1,-1,-1):#start from last n
            #start from fist k
            for count in range(1,k+1):
                take = 0
                #look for the smallest index j that is just greater than next time to take it
                next_index = next_index_map[i]
                if next_index <= N:
                    take = events[i][2] + dp[next_index][count-1]

                dont_take = dp[i+1][count]
                ans = max(take,dont_take)
                dp[i][count] = ans
        
        return dp[0][k]

#recursive precompute
class Solution:
    def maxValue(self, events: List[List[int]], k: int) -> int:        
        events.sort()
        n = len(events)
        starts = [start for start, end, value in events]
        next_indices = [bisect_right(starts, events[cur_index][1]) for cur_index in range(n)]
        dp = [[-1] * n for _ in range(k)]
        
        def dfs(cur_index, count):
            if count == k or cur_index == n:
                return 0
            if dp[count][cur_index] != -1:
                return dp[count][cur_index]
            next_index = next_indices[cur_index]
            dp[count][cur_index] = max(dfs(cur_index + 1, count), events[cur_index][2] + dfs(next_index, count + 1))
            return dp[count][cur_index]
        
        return dfs(0, 0)

###############################################
# 1644. Lowest Common Ancestor of a Binary Tree II
# 15JUL23
###############################################
#two pass works
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        '''
        traverse tree checking for p and q, return null if not in true
        otherwise adopt the same algorithm from LCA I
        '''
        self.found_p = False
        self.found_q = False
        
        def dfs(node):
            if not node:
                return
            if node == p:
                self.found_p = True
            if node == q:
                self.found_q = True
            
            dfs(node.left)
            dfs(node.right)
            
        dfs(root)
        
        if not self.found_p or not self.found_q:
            return None
        
        def lca(node,p,q):
            if node == None:
                return None
            if node == p or node == q:
                return node
            left = lca(node.left,p,q)
            right = lca(node.right,p,q)
            #if there something to return on left or right, this must be the LCA
            if left != None and right != None:
                return node
            #if we returned nothing, retuing nothing
            if left == None and right == None:
                return None
            return left if left else right
        
        return lca(root,p,q)
    
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        '''
        traverse tree checking for p and q, return null if not in true
        otherwise adopt the same algorithm from LCA I
        '''
        self.found_p = False
        self.found_q = False
        
        def dfs(node):
            if not node:
                return
            if node == p:
                self.found_p = True
            if node == q:
                self.found_q = True
            
            dfs(node.left)
            dfs(node.right)
            
        dfs(root)
        
        if not self.found_p or not self.found_q:
            return None
        
        def lca(node,p,q):
            if node is None or node == p or node == q:
                return node
            left = lca(node.left,p,q)
            right = lca(node.right,p,q)
            
            if left and right:
                return node
            elif left:
                return left
            else:
                return right
        return lca(root,p,q)
    
#actual solution, call lca, than check for the exitence of the other node in the the subtree that we returned
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        def dfs(node, target):
            if node == target:
                return True
            if node is None:
                return False
            return dfs(node.left, target) or dfs(node.right, target)

        def LCA(node, p, q):
            if node is None or node == p or node == q:
                return node
            left = LCA(node.left, p, q)
            right = LCA(node.right, p, q)
            if left and right:
              return node
            elif left:
              return left
            else:
              return right

        ans = LCA(root, p, q)
        if ans == p:  # check if q is in the subtree of p
            return p if dfs(p, q) else None
        elif ans == q:  # check if p is in the subtree of q
            return q if dfs(q, p) else None
        return ans
    
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        '''
        global dfs answer and counting conditions
        '''
        self.ans = None
        
        def dfs(node,p,q):
            if not node:
                return False
            left = dfs(node.left,p,q)
            right = dfs(node.right,p,q)
            
            mid = node == p or node == q
            if left + right + mid >= 2:
                self.ans = node
                
            return left or mid or right
        
        dfs(root,p,q)
        return self.ans

    
####################################
# 1125. Smallest Sufficient Team
# 16JUL23
####################################
#wtf.....???
class Solution:
    def smallestSufficientTeam(self, req_skills: List[str], people: List[List[str]]) -> List[int]:
        '''
        we have List<string> required skills and List<List<string>> people where people[i] has those skills
        sufficent team is a team that contains all required skills
        return any suffcient team that is as small as possible
        
        this is bitmask dp
        say we have bit maske (0000), which means no one is on this time
        we can add the ith person to this team, but for this team pair, we need to know if we have the required skills
        pair it with a required skills mask
        (0000) and (000), when the required skills mask is all ones, we have valid team
        then we only add this person the team is they can contribute a skill that is not the current part of the team
        '''
        M = len(req_skills)
        N = len(people)
        
        #mapp skills to id, which is the position
        skills_to_id = {}
        
        for i in range(M):
            skills_to_id[req_skills[i]] = i
        memo = {}
        
        def count_set_bits(mask):
            count = 0
            while mask:
                mask = mask & (mask-1)
                count += 1
            
            return count
        
        def dp(skills_mask,team_mask):
            #base case we have all skills present, return team mask and number of set ones
            if skills_mask == (1 << M) - 1:
                return [team_mask,team_mask]
            #retrieve
            if (skills_mask,team_mask) in memo:
                return memo[(skills_mask,team_mask)]
            #get curr count
            curr_set_bits = count_set_bits(team_mask)
            ans = [skills_mask,team_mask]
            for i in range(N):
                for skill in people[i]:
                    j = skills_to_id[skill]
                    #if this ith person has this skill needed in the mask, add it
                    if skills_mask & (1 << j) == 0:
                        next_skills_mask = skills_mask | (1 << j)
                        next_team_mask = team_mask | (1 << i)
                        child_ans = dp(next_skills_mask,next_team_mask)
                        if count_set_bits(child_ans[1]) < curr_set_bits:
                            curr_set_bits = child_ans[1]
                            ans = [next_skills_mask,next_team_mask]
            memo[(skills_mask,team_mask)] = ans
            return ans
        
        dp(0,0)
        for k,v in memo.items():
            print(k,v)
                

#this is a covering set problem
class Solution:
    def smallestSufficientTeam(self, req_skills: List[str], people: List[List[str]]) -> List[int]:
        '''
        we need to use a mask of skills from the req_skills for each people[i]
        we can also reformulate the questions: find the smallest team such that the bitwise OR fo the bitmasks representing the skills of
        the current members on the team is 2**(len(req_skills)) - 1
        
        we let dp(skillsMask) be the bitmask represetning the team that poasses all the skills from this skillsmask
        the value of dp(skillsMask) is the mask the represents the indices of the people on this team such that the size is minimum
        dp(0) = 0, base case
        
        for a given skillsMaks != 0, there must bet at least on person in a team, since we dont know, we just set it to a large value
        essential, all skills maksk will be as larges as possible giiving us the chance to minimize
        to make it easier we use skills mask for each person
        skillsMaskPerson[i] denot the bitmask respreetning the skill set of the ith person
            precompute before hand
            
            In [42]: skills = '111111'

        In [43]: has =    '100101'

        In [44]: int(skills,2) & ~int(has,2)
        Out[44]: 26

        In [45]: bin(26)
        Out[45]: '0b11010'

        In [46]: '011010'
            
        notes, althrough the other team members may possess sills from skillMaskPerson[i] it is not nesscary to examine all pairings
        however, THEY MUST have the skills from skillSmaks not presnt in skillsMaskPerson[i]
        
        the set smallerSkillsMask = skillsMask \ skillsMaskPerson[i], where i denotes the difference in skills
            this contains the required skills that the ith person does not possess
        
        i,e set(skills_person[i]) differecen set(skills person[j]) = skills_mask & -skillsMaskPerson[i]
        but we can also do it manually, where we check bit by bit
        
        we update dp(skillsMask) with dp(smallerSkills_mask) OR 2**i 
            this add the ith perosn to the team with the skills not covered by smallerSillsMask
            only update when smaller_skills_maks != skillsMask
            
        one more question:
            how to know where we actually call dp(skillsMask) of if we need to update for a previous computed asnwer
            we store -1, indicatting we don't have an answer, which we comptue
            otherwirse we write into dp[skillsMask] != -1 the new mask
        
        '''
        M = len(req_skills)
        N = len(people)
        
        #mapp skills to id, which is the position
        skills_to_id = {}
        for i in range(M):
            skills_to_id[req_skills[i]] = i
            
        #get masks of each people[i] holding skills
        skillsMaskPerson = [0]*N
        for i in range(N):
            mask = 0
            for skill in people[i]:
                mask = mask | (1 << skills_to_id[skill])
            skillsMaskPerson[i] = mask
            
        memo = [-1]*(1 << M)
        memo[0] = 0
        
        #B. kernighan
        def count_set_bits(mask):
            count = 0
            while mask:
                mask = mask & (mask-1)
                count += 1
            
            return count
        
        def dp(skills_mask):
            if memo[skills_mask] != -1:
                return memo[skills_mask]
            for i in range(N):
                #find skills not covered by person i
                needed_skills = skills_mask & ~skillsMaskPerson[i]
                #make sure they are not the same
                if skills_mask != needed_skills:
                    team_mask = dp(needed_skills) | (1 << i) #includign person i
                    if memo[skills_mask] or count_set_bits(team_mask) < count_set_bits(memo[skills_mask]):
                        memo[skills_mask] = team_mask

            return memo[skills_mask]
        
        ans_mask = dp((1 << M)-1)
        
        ans = []
        for i in range(N):
            if (ans_mask >> i) & 1 == 1:
                ans.append(i)
        
        return ans
    
class Solution:
    def smallestSufficientTeam(self, req_skills: List[str], people: List[List[str]]) -> List[int]:
        '''
        we can also treat this is 0/1 knapack with states (i,skills_mask)
        we either choose to take this person and update the skills_mask
        of we don't take this person
        
        '''
        M = len(req_skills)
        N = len(people)
        
        #mapp skills to id, which is the position
        skills_to_id = {}
        for i in range(M):
            skills_to_id[req_skills[i]] = i
            
        #get masks of each people[i] holding skills
        skillsMaskPerson = [0]*N
        for i in range(N):
            mask = 0
            for skill in people[i]:
                mask = mask | (1 << skills_to_id[skill])
            skillsMaskPerson[i] = mask
            
        memo = {}
        
        def dp(i,needed_skills):
            if i >= N:
                if not needed_skills:
                    #if we dont need skills we dont need a team
                    return [] 
                return None
            
            if (i,needed_skills) in memo:
                return memo[(i,needed_skills)]
            
            take = dp(i+1, needed_skills & ~skillsMaskPerson[i])
            no_take = dp(i+1,needed_skills)
            
            #if bother are none no anser
            if take is None and no_take is None:
                memo[(i,needed_skills)] = None
            elif no_take is None:
                #add this person when taking
                memo[(i,needed_skills)] = [i] + take
            elif take is None:
                memo[(i,needed_skills)] = no_take #must be the other answer
            else:
                if len(no_take) < 1 + len(take):
                    ans = no_take
                else:
                    ans = [i] + take
                memo[(i,needed_skills)] = ans
            
            return memo[(i,needed_skills)]
        
        return dp(0,(1 << M) - 1)

#BFS state exploration and stop at minimum
class Solution:
    def smallestSufficientTeam(self, req_skills: List[str], people: List[List[str]]) -> List[int]:
        '''
        for each skill, add the people that know this skill by index
        intution:
            the idea is to select the rarest skill in terms of people (skills that only few know)
            nodes will be map of people belong to a pariticular skill
            starting node is the empty team and the table what maps all skills to certian people
            then select the skill that few people know
            create the next state of skills map:
                go through each person (from the rare_skill_set) and select the skills they dont have
                the next state are the skills we dont have yet
            
            once we have covered the skills (COVERING) return the team
        '''
        #for each skills, find the peoplle that can do that skill
        skills_people_table = [set() for i in range(len(req_skills))]
        #fast lookup
        skills_map = {skill : i for i, skill in enumerate(req_skills)}
        for i in range(len(people)):
            for skill in people[i]: 
                skills_people_table[skills_map[skill]].add(i)
        
        #print(skills_people_table)
        #add to queue, the current skills to people mapping and empty team
        #we need to contsruct this
        q = deque()
        q.append((skills_people_table, []))
        while q:
            curr_table, curr_team = q.popleft()
            #add one at a time
            rare_skill_people_set = min(curr_table, key=len)
            #print(curr_table,rare_skill_people_set)
            #for person in range(len(people)): #if we just checked for every person in the list
            for person in rare_skill_people_set: 
                next_table = []
                for skill_people_set in curr_table:
                    #if this person wasn't part of the current skills to person mapping, then it means we need to include this person
                    #to cover the skills
                    if person not in skill_people_set:
                        next_table.append(skill_people_set)
                #empty table means we have covered all the skills
                #we only had a next table if we were missing some kills, if we arent missing any skills then this is our team
                if not next_table: 
                    return curr_team + [person]
                #print(next_table)
                q.append((next_table, curr_team + [person]))

#another dp way
class Solution:
    def smallestSufficientTeam(self, req_skills: List[str], people: List[List[str]]) -> List[int]:
        memo = {}
        M = len(req_skills)
        N = len(people)
        #mapp skills to id, which is the position
        skills_to_id = {}
        for i in range(M):
            skills_to_id[req_skills[i]] = i
            
        #get masks of each people[i] holding skills
        skillsMaskPerson = [0]*N
        for i in range(N):
            mask = 0
            for skill in people[i]:
                mask = mask | (1 << skills_to_id[skill])
            skillsMaskPerson[i] = mask
        need_skills = (1 << M) - 1
        
        def solve(i, team_skills):
            if team_skills == need_skills: 
                return 0
            if i == len(people): 
                return (1 << 61) - 1
            if (i,team_skills) in memo:
                return memo[(i,team_skills)]
            
            pick = (1 << i) | solve(i + 1, team_skills | skillsMaskPerson[i])
            skip = solve(i + 1, team_skills)
            ans = pick if pick.bit_count() < skip.bit_count() else skip
            memo[(i,team_skills)] = ans
            return ans
        
        final_mask = solve(0,0)
        ans = [i for i in range(N) if final_mask & 1 << i]
        return ans
    
#bottom up
class Solution:
    def smallestSufficientTeam(self, req_skills: List[str], people: List[List[str]]) -> List[int]:
        memo = {}
        M = len(req_skills)
        N = len(people)
        #mapp skills to id, which is the position
        skills_to_id = {}
        for i in range(M):
            skills_to_id[req_skills[i]] = i
            
        #get masks of each people[i] holding skills
        skillsMaskPerson = [0]*N
        for i in range(N):
            mask = 0
            for skill in people[i]:
                mask = mask | (1 << skills_to_id[skill])
            skillsMaskPerson[i] = mask
        need_skills = (1 << M) - 1
        
        dp = [[0]*(need_skills+1) for i in range(N+1)]
        #base cases
        for i in range(N+1):
            for team_skills in range(need_skills+1):
                if team_skills == need_skills:
                    dp[i][team_skills] = 0
                elif i == N:
                    dp[i][team_skills] = (1 << 61) - 1
        
        for i in range(N-1,-1,-1):
            for team_skills in range(need_skills):
                pick = (1 << i) | dp[i + 1][team_skills | skillsMaskPerson[i]]
                skip = dp[i + 1][team_skills]
                ans = pick if pick.bit_count() < skip.bit_count() else skip
                dp[i][team_skills] = ans
        
        final_mask = dp[0][0]
        ans = [i for i in range(N) if final_mask & 1 << i]
        return ans
    
#additional backtracking
#https://leetcode.com/problems/smallest-sufficient-team/discuss/334630/Python-Optimized-backtracking-with-explanation-and-code-comments-88-ms
class Solution:
    def smallestSufficientTeam(self, req_skills: List[str], people: List[List[str]]) -> List[int]:
        
        # Firstly, convert all the sublists in people into sets for easier processing.
        for i, skills in enumerate(people):
            people[i] = set(skills)
        
        # Remove all skill sets that are subset of another skillset, by replacing the subset with an
        # empty set. We do this rather than completely removing, so that indexes aren't 
        # disrupted (which is a pain to have to sort out later).
        for i, i_skills in enumerate(people):
            for j, j_skills in enumerate(people):
                if i != j and i_skills.issubset(j_skills):
                    people[i] = set()
        
        # Now build up a dictionary of skills to the people who can perform them. The backtracking algorithm
        # will use this.
        skills_to_people = collections.defaultdict(set)
        for i, skills in enumerate(people):
            for skill in skills:
                skills_to_people[skill].add(i)
            people[i] = set(skills)
        
        # Keep track of some data used by the backtracking algorithm.
        self.unmet_skills = set(req_skills) # Backtracking will remove and readd skills here as needed.
        self.smallest_length = math.inf # Smallest team length so far.
        self.current_team = [] # Current team members.
        self.best_team = [] # Best team we've found, i,e, shortest team that covers skills/
        
		# Here is the backtracking algorithm.
        def meet_skill(skill=0):
			# Base case: All skills are met.
            if not self.unmet_skills:
				# If the current team is smaller than the previous we found, update it.
                if self.smallest_length > len(self.current_team):
                    self.smallest_length = len(self.current_team)
                    self.best_team = self.current_team[::] # In Python, this makes a copy of a list.
                return # So that we don't carry out the rest of the algorithm.
                        
            # If this skill is already met, move onto the next one.
            if req_skills[skill] not in self.unmet_skills:
                return meet_skill(skill + 1)
				# Note return is just to stop rest of code here running. Return values
				# are not caught and used.
            
            # Otherwise, consider all who could meet the current skill.
            for i in skills_to_people[req_skills[skill]]:
                
				# Add this person onto the team by updating the backtrading data.
                skills_added_by_person = people[i].intersection(self.unmet_skills)
                self.unmet_skills = self.unmet_skills - skills_added_by_person
                self.current_team.append(i)
                
				# Do the recursive call to further build the team.
                meet_skill(skill + 1)
                
                # Backtrack by removing the person from the team again.
                self.current_team.pop()
                self.unmet_skills = self.unmet_skills.union(skills_added_by_person)
        
		# Kick off the algorithm.
        meet_skill()        
        return self.best_team 



###########################################
# 445. Add Two Numbers II (REVISTED)
# 17JUL23
############################################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        '''
        i can reverse the lists, then created the new ll and reverse again
        '''
        def reverse(node):
            if not node or not node.next:
                return node
            #reverse rest of list
            reversed_rest = reverse(node.next)
            node.next.next = node
            node.next = None
            return reversed_rest
        
        def reverse2(node):
            prev = None
            curr = node
            while curr:
                next_node = curr.next
                curr.next = prev
                prev = curr
                curr = next_node
            
            return prev
        
        
        ll1 = reverse(l1)
        ll2 = reverse(l2)
        
        dummy = ListNode(-1)
        curr = dummy
        carry = 0
        
        curr1 = ll1
        curr2 = ll2
        
        while curr1 or curr2:
            v1 = curr1.val if curr1 else 0
            v2 = curr2.val if curr2 else 0
            entry = v1 + v2 + carry
            carry,val = divmod(entry,10)
            new_node = ListNode(val = val)
            curr.next = new_node
            curr = curr.next
            curr1 = curr1.next if curr1 else curr1
            curr2 = curr2.next if curr2 else curr2
        
        if carry == 1:
            curr.next = ListNode(val=1)
        print(dummy.next,carry)
        return reverse(dummy.next)
            
#instead of reversing lists
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        s1 = []
        s2 = []

        while l1:
            s1.append(l1.val)
            l1 = l1.next
        while l2:
            s2.append(l2.val)
            l2 = l2.next

        total_sum = 0
        carry = 0
        ans = ListNode()
        while s1 or s2:
            if s1:
                total_sum += s1.pop()
            if s2:
                total_sum += s2.pop()

            ans.val = total_sum % 10
            carry = total_sum // 10
            head = ListNode(carry)
            head.next = ans
            ans = head
            total_sum = carry

        return ans.next if carry == 0 else ans
    
##################
# 146. LRU Cache
# 18JUL23
##################
#queue solution
class LRUCache:
    '''
    we can also use a queue, where the key at the front is the least recently used and the key at the back is most recently used
    '''

    def __init__(self, capacity: int):
        self.cache = {}
        self.q = deque([])
        self.size = capacity

    def get(self, key: int) -> int:
        if key in self.cache:
            #remove from q and put back
            self.q.remove(key)
            self.q.append(key)
            return self.cache[key]
        return -1


    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            #remove from q and put back
            self.q.remove(key)
            self.q.append(key)
        
        else:
            self.cache[key] = value
            self.q.append(key)
        
        #if too big
        if len(self.cache) > self.size:
            least_recent_key = self.q.popleft()
            del self.cache[least_recent_key]
            
            

import collections
class LRUCache:
    '''
    put evicts the least recently used key and puts in the new (key,value) if the capacity is too high
        if they have the same min count, we need to evict the older one
    the crux of the problem is keeping track of the least recently used keys
    count map! this might TLE though
    '''

    def __init__(self, capacity: int):
        self.size = capacity
        self.cache = collections.OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        #move this key to the end because we used this key
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        #move key to end
        if key in self.cache:
            self.cache.move_to_end(key)
        
        self.cache[key] = value
        if len(self.cache) > self.size:
            self.cache.popitem(False)
            


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

################################################
# 435. Non-overlapping Intervals (REVISTED)
# 19JUL23
###################################################
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        '''
        ---
        ----
        -----
        we have one interval that contains the other intervals
        so really want to find the the maximum number of overlapping intervals
        to make them non overlapping i need to remove 2
        if i have n overlapping intervals where they are essentially nested, i need to remove n-1 intervals
        
        sort array on start and set the last interval's ending point as end
        1. if end > the start of the next interval, there is an overlap here, but we dont increment the count
            we need to keep track of the smaller ends to include in out window, by including the smaller ends we increase the number of intervals we can fit
        2. end <= the start of the next interval, we cant possibly have an overlap here
            so increment count, and set end to the end of the next interval
        '''
        if not intervals:
            return 0
        intervals.sort()
        curr_end = intervals[0][1] 
        unpcaptured_intervals = 1 #maximize this, we will alwasy have one uncapture intervals, i.e a single interval is still uncaptured
        for start,end in intervals[1:]:
            #if we want to include this interval to be captured, adjust the end to be the smaller
            if curr_end > start:
                curr_end = min(curr_end, end)
            elif start >= curr_end:
                #we cant possible capture this interval
                unpcaptured_intervals += 1
                curr_end = end        
        return len(intervals) - unpcaptured_intervals
    
#top down dp
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        '''
        we can also use dp to solve this problem
        let dp(i) be the maximum number of tasks we can take using intervals[i:]
        dp(i) = {
            then find the next interval we can take at index j
            take = 1 + dp(j)
            no_take = dp(i+1)
            max(take,no_take)
            
            we can just binary search to find the starttime just greater than the current end time for this task
        }
        '''
        N = len(intervals)
        intervals.sort()
        starts = []
        ends = []
        for start,end in intervals:
            starts.append(start)
            ends.append(end)
        
        end_mapp = {}
        for end in ends:
            end_mapp[end] = bisect.bisect_left(starts,end)
            
        memo = {}
        
        def dp(i):
            if i == N:
                return 0
            if i in memo:
                return memo[i]
            curr_start = starts[i]
            curr_end = ends[i]
            #find the next task we can take, i.e the jth task is just before the end
            #bisect left, not we also could have precompute the array for each start
            no_take = dp(i+1)
            #we also could have precomputed this for all ends
            j = end_mapp[curr_end]
            #j = bisect.bisect_left(starts,curr_end)
            take = dp(j) + 1
            ans = max(take,no_take)
            memo[i] = ans
            return ans
        
        #dp(0) returens the maximum number of tasks i can do, which means i need to get rid of the rest 
        #to find the minimum number of intervals to remove
        return N - dp(0)

#another greedy
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        '''
        if we sort on end times, then to avoid overlap we greedily take the one with the earlier end time
        want to find the maximum number of overlapping intervals
        if we choose the intervals with the earlier end times we can maximize the number of overlapping intervals
        
        because we sorted on the end times, ends > curr _end
        if start >= end, we can certinal take this interval, because it doesnt cause an overlap, update curr_end end
        if end < k, there is an overlap so inncrement the count
        '''
        ans = 0 #count of overlaps
        curr_end = float('-inf')
        intervals.sort(key = lambda x: x[1])
        for start,end in intervals:
            if start >= curr_end:
                curr_end = end
            else:
                ans += 1
        
        return ans
    
##########################################
# 735. Asteroid Collision (REVISTED)
# 20JUL23
##########################################
#dang it
class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        '''
        just simulate, add asteroids to the stack until
        if they are in opposite side, simulate collections
        '''
        stack = []
        for ast in asteroids:
            while stack and stack[-1]*ast < 0 and abs(ast) > stack[-1]:
                stack.pop()
            
            if not stack:
                stack.append(ast)
            
            elif stack and abs(stack[-1]) == abs(ast):
                stack.pop()
            elif stack and abs(stack[-1]) > abs(ast):
                continue
            else:
                stack.append(ast)
        return stack


#case work?
class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        '''
        just simulate, add asteroids to the stack until
        if they are in opposite side, simulate collections
        '''
        if not asteroids:
            return []
        
        results =  []
        
        #starting with the second one
        for ast in asteroids:
            #check if we need pop off, or explistion
            while results and results[-1] > 0 and ast < 0 and abs(ast) > results[-1]:
                results.pop()
                
            #check if we need to append, exploed,or both
            #if empty add it
            if not results:
                results.append(ast)
            #iff the last asteroid in our stack is positive but same size they both go away
            elif results[-1] > 0 and ast < 0 and abs(ast) == results[-1]:
                results.pop()
            #iff last asteroid is negative, and the asteroid is going the other way
            elif results[-1] < 0 or ast > 0:
                results.append(ast)
        return results

class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        '''
        just add asteroids to the right and clean up the last two
        '''
        ans = []
        for ast in asteroids:
            ans.append(ast)
            while len(ans) > 1 and ans[-2] > 0 and ans[-1] < 0:
                right = ans.pop()
                left = ans.pop()
                if abs(left) != abs(right):
                    if abs(left) > abs(right):
                        ans.append(left)
                    else:
                        ans.append(right)
        
        return ans

##########################################
# 553. Optimal Division
# 18JUL23
##########################################
#WOOOHOOO
class Solution:
    def optimalDivision(self, nums: List[int]) -> str:
        '''
        inputs are small enough to allow brute force
        i can generate the string and use pythons eval literal function
        
        kinda a stupid problem because it really should be an valid literal epression
        
        the answer is just:
            nums[0] / "/".join(nums[1:])
            
        but just be careful of special cases
        
        not sure why this works, but looking at the examples i just guessed
        proof, sorta of
        say we have [a,b,c,d] and we want to maximize
        we can write the answer as p/q and in order to maximize p/q we minimize q
        repeated division in successsion will always give smaller numbers ( as long as the numbers are > 1)
        we could do b/c/d or (b/c)/d or b/(c/d)
        
        and we must have a number in the top

        X1/X2/X3/../Xn will always be equal to (X1/X2) * Y, no matter how you place parentheses. 
        i.e no matter how you place parentheses, X1 always goes to the numerator and X2 always goes to the denominator. 
        Hence you just need to maximize Y. And Y is 
        maximized when it is equal to X3 *..*Xn. So the answer is always X1/(X2/X3/../Xn) = (X1 *X3 *..*Xn)/X2
        '''
        first = nums[0]
        second = nums[1:]
        #reformat
        if len(second) > 1:
            return str(first)+"/"+"("+"/".join([str(num) for num in second])+")"
        elif len(second) == 1:
            return str(first)+"/"+"/".join([str(num) for num in second])
        return str(first)
    
#in one line
class Solution:
    def optimalDivision(self, nums: List[int]) -> str:
        return "/".join(map(str, nums)) if len(nums) <= 2 else f'{nums[0]}/({"/".join(map(str, nums[1:]))})'
    
'''
better proof
[a/b/c/d/e] = a*(1/b)*(1/c)*(1/d)*(1/e) = k
no matter how we split up using parentheses, product will always be k
in which case, hold a consant then use the rest, so we just get a / (1(b*c*d*e)) = a / (b/c/d/e)
'''
    
#now do the actual dp, start with brute force recursion
