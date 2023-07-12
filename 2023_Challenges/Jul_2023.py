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