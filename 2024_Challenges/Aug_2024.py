##################################
# 2678. Number of Senior Citizens
# 01AUG24
###################################
class Solution:
    def countSeniors(self, details: List[str]) -> int:
        '''
        first ten chars are phone numbers
        next char is geneder
        next to chars are age
        10+1 
        ''' 
        age_idx = 10 + 1
        ans = 0
        for d in details:
            if int(d[age_idx:age_idx + 2]) > 60:
                ans += 1
        
        return ans
    
#doing char by char insteaf of converting the whole thing
class Solution:
    def countSeniors(self, details: List[str]) -> int:
        '''
        first ten chars are phone numbers
        next char is geneder
        next to chars are age
        10+1 
        ''' 
        age_idx = 10 + 1
        ans = 0
        for d in details:
            tens = ord(d[age_idx]) - ord('0')
            ones = ord(d[age_idx + 1]) - ord('0')
            
            age = tens*10 + ones
            if age > 60:
                ans += 1
        
        return ans
    
###################################################
# 2134. Minimum Swaps to Group All 1's Together II
# 02AUG24
##################################################
class Solution:
    def minSwaps(self, nums: List[int]) -> int:
        '''
        there could be multiple arrangments for which a ciruclar array is divided accordigly, we want the minimum
        if there are k's, then each subarray of length k should all 1
        we want the the subarray wit the smallest number of zeros
        we dont want to keep computing the number of zeros, so we can use sliding window
        '''
        ones = sum([num == 1 for num in nums])
        if ones == 0:
            return 0
        nums_doubled = nums + nums
        ans = float('inf')
        left = 0
        count_zeros = 0
        
        for right in range(len(nums_doubled)):
            count_zeros += nums_doubled[right] == 0
            if right - left + 1 == ones:
                ans = min(ans,count_zeros)
                count_zeros -= nums_doubled[left] == 0
                left += 1
            
        return ans
    
#we dont need to concat, jsut use mod N
class Solution:
    def minSwaps(self, nums: List[int]) -> int:
        '''
        no need to concat, just use mod N
        '''
        ones = sum([num == 1 for num in nums])
        if ones == 0:
            return 0
        ans = float('inf')
        left = 0
        count_zeros = 0
        N = len(nums)
        
        for right in range(2*N):
            count_zeros += nums[right % N] == 0
            if right - left + 1 == ones:
                ans = min(ans,count_zeros)
                count_zeros -= nums[left % N] == 0
                left += 1

        return ans
    
###########################################
# 1062. Longest Repeating Substring
# 02JUL24
############################################
#cheeze
class Solution:
    def longestRepeatingSubstring(self, s: str) -> int:
        '''
        generate all possible substrings
        '''
        seen = set()
        N = len(s)
        ans = 0
        for i in range(N):
            for j in range(i+1,N+1):
                substring = s[i:j]
                if substring in seen:
                    ans = max(ans,len(substring))
                else:
                    seen.add(substring)
        
        return ans
    
#using rolling hash
#doesnt quite work collisiosn???
class Solution:
    def longestRepeatingSubstring(self, s: str) -> int:
        '''
        need to use rolling hash instead of getting subsstring
        hash_value = (hash_value + (c - 'a' + 1) * p_pow) % m;
        p_pow = (p_pow * p) % m;
        '''
        print(len(s))
        seen = set()
        N = len(s)
        ans = 0
        mod = 10**9 + 7
        base = 31
        for i in range(N):
            curr_hash = 0
            p_pow = 1
            for j in range(i,N):
                dig = ord(s[j]) - ord('a')
                curr_hash = (curr_hash + (dig + 1)*p_pow) % mod
                p_pow = (p_pow*base) % mod
                if curr_hash in seen:
                    ans = max(ans,j-i+1)
                else:
                    seen.add(curr_hash)
        print(len(seen))
        return ans


class Solution:
    def longestRepeatingSubstring(self, s: str) -> int:
        
        def check_length(length):
            seen = set()
            base = 26
            mod = 2**31 - 1
            current_hash = 0
            base_power = 1
            for i in range(length):
                current_hash = (current_hash * base + ord(s[i])) % mod
                base_power = (base_power * base) % mod
            
            seen.add(current_hash)
            
            for start in range(1, len(s) - length + 1):
                current_hash = (current_hash * base - ord(s[start - 1]) * base_power + ord(s[start + length - 1])) % mod
                if current_hash in seen:
                    return True
                seen.add(current_hash)
            
            return False
        
        left, right = 1, len(s)
        ans = 0
        while left <= right:
            mid = (left + right) // 2
            if check_length(mid):
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return ans

##########################################
# 1508. Range Sum of Sorted Subarray Sums
# 04AUG24
##########################################
class Solution:
    def rangeSum(self, nums: List[int], n: int, left: int, right: int) -> int:
        '''
        say we have three numbers. a,b,c
        a + b + c + (a + b) + (a + b + c) + (b + c)
        sums are sorted increasingly
        a + b + c + (a + b) + (b+c) + (a+b+c)
        use pref sums, then find all (i,j) sums to get subarrays
        
        '''
        pref_sum = [0]
        for num in nums:
            pref_sum.append(pref_sum[-1] + num)
        
        N = len(nums)
        all_sums = []
        for i in range(N):
            for j in range(i,N):
                sub_sum = pref_sum[j+1] - pref_sum[i]
                all_sums.append(sub_sum)
        
        all_sums.sort()
        ans = 0
        mod = 10**9 + 7
        for i in range(left,right+1):
            ans += all_sums[i-1]
            ans %= mod
        
        return ans
    
#dont need to use pref sum
class Solution:
    def rangeSum(self, nums: List[int], n: int, left: int, right: int) -> int:
        '''
        say we have three numbers. a,b,c
        a + b + c + (a + b) + (a + b + c) + (b + c)
        sums are sorted increasingly
        a + b + c + (a + b) + (b+c) + (a+b+c)
        use pref sums, then find all (i,j) sums to get subarrays
        
        '''
        N = len(nums)
        all_sums = []
        for i in range(N):
            sub_sum = 0
            for j in range(i,N):
                sub_sum += nums[j]
                all_sums.append(sub_sum)
        
        all_sums.sort()
        ans = 0
        mod = 10**9 + 7
        for i in range(left,right+1):
            ans += all_sums[i-1]
            ans %= mod
        
        return ans
    

###########################################
# 2053. Kth Distinct String in an Array
# 05AUG24
###########################################
class Solution:
    def kthDistinct(self, arr: List[str], k: int) -> str:
        '''
        count them up, then going left to right, check if count(s) > 1
        return the kth one
        '''
        counts = Counter(arr)
        for s in arr:
            if counts[s] == 1:
                k -= 1
            if k == 0:
                return s
        
        return ""