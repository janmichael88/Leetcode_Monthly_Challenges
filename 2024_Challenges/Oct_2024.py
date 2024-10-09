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
    
########################################
# 3163. String Compression III
# 05OCT24
########################################
class Solution:
    def compressedString(self, word: str) -> str:
        '''
        we can only compress up to size 9
        '''
        comp = ""
        curr_char = ""
        curr_count = 0
        
        for ch in word:
            if not curr_char:
                curr_char = ch
                curr_count += 1
            elif ch == curr_char and curr_count < 9:
                curr_count += 1
            else:
                comp += str(curr_count)+curr_char
                curr_char = ch
                curr_count = 1
        if curr_count:
            comp += str(curr_count)+curr_char 
            
        return comp
    
#another cool way, advance in steps of length
class Solution:
    def compressedString(self, word: str) -> str:
        comp = ""
        i = 0
        while i < len(word):
            current_char = word[i]
            length = 0
            while i + length < len(word) and word[i + length] == current_char and length < 9:
                length += 1
            comp += str(length) + current_char
            i += length
        return comp

##########################################
# 1813. Sentence Similarity III
# 06OCT24
##########################################
class Solution:
    def areSentencesSimilar(self, sentence1: str, sentence2: str) -> bool:
        '''
        the idea is that sentence1 must have a prefix and or suffix of sentence2 only!
        we also need to check that sentence1 has a prefix and/or suffix of setence 1
        we can use queue for both and check that they are non empty
        
        i.e one of the sentences must be empty after clearing the queue
        '''
        q1 = deque(sentence1.split(" "))
        q2 = deque(sentence2.split(" "))
        
        while q1 and q2 and q1[0] == q2[0]:
            q1.popleft()
            q2.popleft()
            
        while q1 and q2 and q1[-1] == q2[-1]:
            q1.pop()
            q2.pop()
            
        return not q1 or not q2
        
class Solution:
    def areSentencesSimilar(self, s1: str, s2: str) -> bool:
        '''
        this problem fucking sucks....
        i need to pick the smaller of the two strings and compare to the larger one
        assume s1 is the larger of the two strings
        '''

        s1_words = s1.split(" ")
        s2_words = s2.split(" ")
        start, ends1, ends2 = 0, len(s1_words) - 1, len(s2_words) - 1

        # If words in s1 are more than s2, swap them and return the answer.
        if len(s1_words) > len(s2_words):
            return self.areSentencesSimilar(s2, s1)

        # Find the maximum words matching from the beginning.
        while start < len(s1_words) and s1_words[start] == s2_words[start]:
            start += 1

        # Find the maximum words matching in the end.
        while ends1 >= 0 and s1_words[ends1] == s2_words[ends2]:
            ends1 -= 1
            ends2 -= 1

        # if ends1 crosses, it means prefixes and suffix match so return true
        return ends1 < start
    
########################################################
# 2696. Minimum String Length After Removing Substrings
# 07OCT24
########################################################
class Solution:
    def minLength(self, s: str) -> int:
        '''
        stack problem,
        whenever we have stack of length 2 or greater, remove AB
        '''
        stack = []
        for ch in s:
            stack.append(ch)
            while len(stack) >= 2 and (stack[-2:] == ['A','B'] or stack[-2:] == ['C','D']):
                stack.pop()
                stack.pop()
            
        
        return len(stack)
    
class Solution:
    def minLength(self, s: str) -> int:
        '''
        we can use two pointers and temporary char array
        we read each char s into write ptr
        and if there's a match, we move write ptr back
        '''
        char_arr = list(s)
        write = 0
        
        for read in range(len(char_arr)):
            char_arr[write] = char_arr[read]
            if write - 1 >= 0 and (char_arr[write - 1 : write + 1] == ['A', 'B'] or char_arr[write - 1 : write + 1] == ['C','D']):
                write -= 1
            else:
                write += 1
        return write
    
###########################################
# 2559. Count Vowel Strings in Ranges
# 07OCT24
###########################################
class Solution:
    def vowelStrings(self, words: List[str], queries: List[List[int]]) -> List[int]:
        '''
        need a way to access count of vowels fast
        prefix sum!
        keep array that stores number of strings staring or ending with vowel
        '''
        pref_counts = [0]
        vowels = 'aeiou'
        
        for w in words:
            if w[0] in vowels and w[-1] in vowels:
                pref_counts.append(pref_counts[-1] + 1)
            else:
                pref_counts.append(pref_counts[-1])
        
        ans = []
        for l,r in queries:
            ans.append(pref_counts[r+1] - pref_counts[l])
        
        return ans

########################################################################
# 1963. Minimum Number of Swaps to Make the String Balanced (REVISITED)
# 08OCT24
#########################################################################
class Solution:
    def minSwaps(self, s: str) -> int:
        '''
        keep track of opening and closing
        try closing valid bracket pairs if we can,
        antyhign left over needs to be swapped
        
        only works becaue the original can be balanced!
        
        '''
        stack = []
        for ch in s:
            if ch == '[':
                stack.append(ch)
            else:
                if stack:
                    stack.pop()
                else:
                    stack.append(ch)
        
        return len(stack)//2
        
class Solution:
    def minSwaps(self, s: str) -> int:
        '''
        keep track of balance
        whenver we are unbalanced, we can swap
        then the new balance becomes 1
        '''
        balance = 0
        swaps = 0
        
        for ch in s:
            if ch == '[':
                balance += 1
            else:
                balance -= 1
            
            #rebalancing leaves it as 1
            #because we made up the balance, but not we swapped, so its unbalanced by 1
            if balance < 0:
                swaps += 1
                balance = 1
        
        return swaps