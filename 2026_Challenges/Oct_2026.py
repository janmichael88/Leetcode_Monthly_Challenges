#################################################
# 2333. Minimum Sum of Squared Difference
# 10OCT26
#################################################
#diff counter
class Solution:
    def minSumSquareDiff(self, nums1: list[int], nums2: list[int], k1: int, k2: int) -> int:
        '''
        we can only dp +/- 1 to each element k1 or k2 times per array
        we want to minimuzize the sum squared differenece
        that means we want to minimize the individual squared difference
        which means we want to raise the smaller number and lower the larger number (the gap)
        but to which elements do we act on?
            can do this greedily
        but we can repeatedly do this k times on an element if we want to
        store as absolulte diffs, no we need to try to reduce each diff as small as possible
        heap?, yeah but k1 and k2 are too big
        i need to efficiently reduce the largest difference
        i need to be careful about the ordering
        '''
        diffs = [abs(u - v) for u, v in zip(nums1, nums2)]
        k = k1 + k2
        total = sum(diffs)

        if sum(diffs) <= k:
            return 0

        counts = Counter(diffs)

        for diff in range(max(diffs), 0, -1):
            cnt = counts[diff]
            if cnt == 0:
                continue

            if k >= cnt:
                k -= cnt
                counts[diff - 1] += cnt
                counts[diff] = 0
            else:
                counts[diff] -= k
                counts[diff - 1] += k
                k = 0


            if k == 0:
                break

        return sum(diff * diff * cnt for diff, cnt in counts.items())

#heap solution, tryick case handling repeated k operations
class Solution:
    def minSumSquareDiff(self, nums1: list[int], nums2: list[int], k1: int, k2: int) -> int:
        '''
        heap
        '''
        heap = [ -abs(x - y) for x, y in zip(nums1, nums2)]
        s = -sum(heap)
        
        #can bring them all down to zero
        if k1 + k2 >= s: 
            return 0
        #combine
        delta = k1 + k2
        heapify(heap)
        n = len(nums1)
        
        while delta > 0:
            d = -heappop(heap)
            #check if we can reduce each diff in heap by 1
            gap = max(delta//n, 1) if heap else delta
            #lower the differ
            d -= gap
            #push back
            heappush(heap, -d)
            #use up operations
            delta -= gap

        return sum(e*e for e in heap)

#############################################################
# 2897. Apply Operations on Array to Maximize Sum of Squares
# 13OCT26
#############################################################
class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        '''
        make each number as big as possible using bit wise ops
        if i pick two indices (i,j)
        nums[i] = nums[i] & nums[j]
        nums[j] = nums[i] | nums[j]
        make each number as big as possible and take the top k
        but whats a valid transformation given the operations?
            we transfer bits from one element to another

        try counting bits

        think like this too:
        Intuition: Think about this like gravity.

Look at one bit of i and j: the pairs (0,0), (1,1) and (0,1) are unchanged, and (1,0) becomes (0,1). 
So the operation does one thing: in every column where i has a chip and j has a gap, the chip falls from i to j. 
Chips never change columns.
        '''
        mod = 10**9 + 7
        #count current bits
        bit_counts = [0]*32
        
        for num in nums:
            for b in range(32):
                if num & (1 << b):
                    bit_counts[b] += 1

        ans = 0
        for _ in range(k):
            curr = 0
            for b in range(32):
                if bit_counts[b] > 0:
                    #contribute it
                    curr = curr | (1 << b)
                    bit_counts[b] -= 1
            ans = (ans + curr*curr) % mod

        return ans % mod

##################################################
# 3727. Maximum Alternating Sum of Squares
# 13OCT26
##################################################
class Solution:
    def maxAlternatingSum(self, nums: List[int]) -> int:
        '''
        indices contribute positively and odds contribute negativel
        numbers are all sqrd so it doest matter
        once we square them, we need to alternate going up as large as possibile, then down as small as possible
        two pointers
        '''
        n = len(nums)
        arr = [num*num for num in nums]
        arr.sort()
        op = 1
        l,r = 0,n-1
        ans = 0
        while l <= r:
            if op == 1:
                ans += arr[r]
                r -= 1
            else:
                ans -= arr[l]
                l += 1
            op *= -1
        return ans

#split half the elements
class Solution:
    def maxAlternatingSum(self, nums: List[int]) -> int:
        n = len(nums)
        for i in range(n):
            nums[i] *= nums[i]
        nums.sort()
        return -sum(nums[: n // 2]) + sum(nums[n // 2 :])