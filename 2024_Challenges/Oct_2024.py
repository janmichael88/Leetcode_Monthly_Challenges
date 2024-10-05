###############################################
# 1497. Check If Array Pairs Are Divisible by k
# 01OCT24
###############################################
class Solution:
    def canArrange(self, arr: List[int], k: int) -> bool:
        '''
        we need pairs of the form
        (a_i + a_j) % k == 0
        destribute
        (a_i % k + a_j % k) %k = 0
        we can just find each number's % k and storm them
        (mod_i + mod_j) % k = 0
        
        this implies that 
        mod_j = k - mod_i
        
        if mod is 0, we need to pair with another elements that is also mod = 0
        first we store numbers % k
        compute each number num as 
            mod = ((num % k) + k) % k to handle negative numbers
            
        tricky math problem
        '''
        counts = Counter()
        for num in arr:
            comp = ((num % k) + k) % k
            counts[comp] += 1
        
        for num in arr:
            comp = ((num % k) + k) % k
            if comp == 0:
                if counts[comp] % 2 == 1:
                    return False
            #mods and mods comps must have equal counts
            elif counts[comp] != counts[k - comp]:
                return False
        
        return True
    
class Solution:
    def canArrange(self, arr: List[int], k: int) -> bool:
        '''
        in python we don't need to worry about taking the mod of negative numbers
        we can also keep an array of size k
        instead of checking all numbers in arr
        just check all mod_counts from 1 to k

        example
        if a number leaves a remainder of 2 when divided by k, then the other number in the pair 
        should leave a remainder that 'completes' the pair - usually, that means its remainder should be k - 2
        i.e remainder, and k - remainder
        '''
        mod_count = [0]*k
        for num in arr:
            mod_count[num % k] += 1
        
        #special case for mod k == 0
        if mod_count[0] % 2 == 1:
            return False
        
        for i in range(1,k):
            if mod_count[i] != mod_count[k-i]:
                return False
        
        return True
    
class Solution:
    def canArrange(self, arr: List[int], k: int) -> bool:
        # Custom comparator to sort based on mod values.
        arr = sorted(arr, key=lambda x: (k + x % k) % k)

        # Assign the pairs with modulo 0 first.
        start = 0
        end = len(arr) - 1
        while start < end:
            if arr[start] % k != 0:
                break
            if arr[start + 1] % k != 0:
                return False
            start = start + 2

        # Now, pick one element from the beginning and one element from the
        # end.
        while start < end:
            if (arr[start] + arr[end]) % k != 0:
                return False
            start += 1
            end -= 1

        return True

########################################
# 1590. Make Sum Divisible by P (REVISTED)
# 04OCT24
########################################
class Solution:
    def minSubarray(self, nums: List[int], p: int) -> int:
        '''
        we need to remove a subarray to make sum divivibsle by p
        say we have subarray [a,b,c,d,e]
        we want (a+b+c+d+e) % p == 0
        
        if (a+b+c+d+e) % p == k, where k != 0
        then we need to find the smallest subarray == k where sum(subarray) % p == k
        
        more fomrally:
        we need a subarray who's sum % p == sum(nums) % p
        we can say: 
        let k = sum(nums) % p
        (prefSum_i - prefSum_j) % p = k
        k is some multiple of pf
        prefSum_i - prefSum_j = k + m*p
        prefSum_i = prefSum_j + k + m*p
        prefSum_i = (prefSum_j - k) % p
        
        to correct for negatvies we just add p and most it again
        
        '''
        SUM = sum(nums)
        if SUM % p == 0:
            return 0
        N = len(nums)
        k = SUM % p
        ans = N
        
        mapp = {}
        mapp[0] = -1
        curr_sum = 0
        for i,num in enumerate(nums):
            curr_sum = (curr_sum + num + p) % p
            #find its complement
            comp = (curr_sum - k + p) % p
            if comp in mapp:
                ans = min(ans, i - mapp[comp])
            
            mapp[curr_sum] = i
        
        return -1 if ans == N else ans
        
#################################################
# 2491. Divide Players Into Teams of Equal Skill
# 04OCT24
#################################################
#phew
class Solution:
    def dividePlayers(self, skill: List[int]) -> int:
        '''
        need to divide into len(skill) / 2 teams
        such that the skill of total skill of each team is equal
        '''
        SUM = sum(skill)
        N = len(skill)
        teams = N // 2
        
        #check that we can't make teams of even skill
        if SUM % teams != 0:
            return -1
        
        skill_of_team = SUM // teams
        counts = Counter(skill)
        
        chemistry = 0
        for i in range(1,(skill_of_team // 2)+1):
            #same i and skill - i
            if i == skill_of_team - i and counts[i] != 0 and counts[skill_of_team - i] != 0:
                if counts[i] % 2 != 0:
                    return -1
                chemistry += ((i)*(skill_of_team - i))*(counts[i] // 2)
                counts[i] = 0
                counts[skill_of_team - i] = 0
            elif counts[i] == counts[skill_of_team - i] and counts[i] != 0 and counts[skill_of_team - i] != 0:
                if counts[i] != counts[skill_of_team - i]:
                    return -1
                chemistry += (i*(skill_of_team - i))*counts[i]
                counts[i] = 0
                counts[skill_of_team - i] = 0
        
        #validate after getting answer
        for k,v in counts.items():
            if v != 0:
                return -1
        return chemistry
    
#sort solution, each pair should add up to the precompute score
class Solution:
    def dividePlayers(self, skill: List[int]) -> int:
        '''
        need to divide into len(skill) / 2 teams
        such that the skill of total skill of each team is equal
        '''
        SUM = sum(skill)
        N = len(skill)
        teams = N // 2
        
        #check that we can't make teams of even skill
        if SUM % teams != 0:
            return -1
        
        skill_of_team = SUM // teams
        skill.sort()
        chemistry = 0
        left = 0
        right = N - 1
        
        while left < right:
            if skill[left] + skill[right] != skill_of_team:
                return -1
            chemistry += skill[left]*skill[right]
            left += 1
            right -= 1
        
        return chemistry
    
class Solution:
    def dividePlayers(self, skill: List[int]) -> int:
        '''
        we can use a count array,
        then retraverse the skill array and find compelements and add accordingly
        '''
        SUM = sum(skill)
        N = len(skill)
        teams = N // 2
        
        #check that we can't make teams of even skill
        if SUM % teams != 0:
            return -1
        
        skill_of_team = SUM // teams
        counts = [0]*1001
        for s in skill:
            counts[s] += 1
        
        chemistry = 0
        for s in skill:
            comp = skill_of_team - s
            if counts[comp] == 0:
                return -1 #cant do it
            chemistry += s*comp
            counts[comp] -= 1
        
        return chemistry // 2 #we are only counting pairs, but we double counted
    
#############################################
# 567. Permutation in String (REVISTED)
# 05OCT24
#############################################
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        '''
        hashamp validation
        if we had a countmap of s1's chars, then there needs to be a subtring in s2, that contains all the chars of s1's count mapp
        we can use sliding window
        '''
        counts_s1 = Counter(s1)
        M = len(s1)
        N = len(s2)
        
        if N < M:
            return False
        
        count1 = Counter()
        count2 = Counter()
        
        for i in range(M):
            count1[s1[i]] += 1
            count2[s2[i]] += 1
        
        for i in range(N - M):
            #if they match, return
            if count1 == count2:
                return True
            #add to new char to s2, which is ust i + M
            count2[s2[i + M]] += 1
            #remove the leftmost char, which is at i
            count2[s2[i]] -= 1
        
        return count1 == count2