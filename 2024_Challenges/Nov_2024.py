###############################################
# 1957. Delete Characters to Make Fancy String
# 03NOV24
###############################################
class Solution:
    def makeFancyString(self, s: str) -> str:
        '''
        build on the fly
        '''
        ans = ""
        
        for ch in s:
            if len(ans) >= 2 and ans[-1] == ans[-2] == ch:
                continue
            ans += ch
        
        return ans
    
##############################################
# 2490. Circular Sentence
# 03NOV24
###############################################
class Solution:
    def isCircularSentence(self, sentence: str) -> bool:
        '''
        check i and i + 1
        for the last word, we need to check n - 1 with zero
        '''
        words = sentence.split(" ")
        N = len(words)
        for i in range(N):
            if i == N  - 1:
                if words[i][-1] != words[0][0]:
                    return False
            else:
                if words[i][-1] != words[i+1][0]:
                    return False
        
        return True
    
#cheese way on space
class Solution:
    def isCircularSentence(self, sentence: str) -> bool:
        '''
        we know the sentence is valid, so insteaf of splitting, we check on spaces
        and if index i is a space, we check i - 1 and i + 1
        '''
        N = len(sentence)
        for i in range(N):
            if sentence[i] == ' ' and sentence[i-1] != sentence[i+1]:
                return False
        
        return sentence[-1] == sentence[0]
    

########################################
# 2955. Number of Same-End Substrings
# 03NOV24
########################################
class Solution:
    def sameEndSubstringCount(self, s: str, queries: List[List[int]]) -> List[int]:
        '''
        same ending substrings are the number of substrings that have the same chars at start and ends
        if there are t chars in a substrin, then there are t*(t-1)/2 same ending substrings
        for each index i, keep track of the frequence count
        then use partial sums for all 26 chars
        '''
        n = len(s)
        pref_counts = [Counter()]
        
        for i in range(n):
            curr_count = Counter()
            curr_count[s[i]] += 1
            for k,v in pref_counts[-1].items():
                curr_count[k] += v
            
            pref_counts.append(curr_count)
        
        
        ans = []
        for l,r in queries:
            temp = 0
            for i in range(26):
                char = chr(ord('a') + i)
                count = pref_counts[r+1][char] - pref_counts[l][char]
                temp += count*(count + 1) // 2
            
            ans.append(temp)
        
        return ans

#ussing array instead of count object
class Solution:
    def sameEndSubstringCount(self, s: str, queries: List[List[int]]) -> List[int]:
        '''
        the counting idea comes from the combindation formula
        if we have k occruences of some character i
        we can place these any number of these occurence at some start and some end
        so its just
        k choose 2
        k*(k-1)/2 + k
        k*(k+1)/2
        
        insteaf of using counter objects we can just an array of size 26
        '''
        n = len(s)
        pref_counts = [[0]*26 for _ in range(n+1)]
        
        for i in range(n):
            pref_counts[i+1][ord(s[i]) - ord('a')] += 1
        
        for i in range(1,n+1):
            for j,count in enumerate(pref_counts[i-1]):
                pref_counts[i][j] += count
        
        ans = []
        for l,r in queries:
            temp = 0
            for i in range(26):
                count = pref_counts[r+1][i] - pref_counts[l][i]
                temp += count*(count + 1) // 2
            
            ans.append(temp)
        
        return ans
                
#binary search???
class Solution:
    def sameEndSubstringCount(self, s: str, queries: List[List[int]]) -> List[int]:
        '''
        for each char in s, store its indices in a hashmap
        then for each query, look for its left and right positions for each character
        we want to find the first position of the character that is at or after the starting index
        the first poistion of the character that is beyong the ending range
        '''
        mapp = defaultdict(list)
        for i,s in enumerate(s):
            mapp[s].append(i)
        
        
        ans = []
        for l,r in queries:
            count = 0
            #do this for each character
            for indices in mapp.values():
                #find leftmost and right most
                #for left <=
                left = bisect.bisect_left(indices,l)
                #for right just greater than
                right = bisect.bisect_right(indices,r)
                num = right - left
                count += (num)*(num + 1) // 2
            
            ans.append(count)
        
        return ans

##########################################
# 1513. Number of Substrings With Only 1s
# 04NOV24
###########################################
class Solution:
    def numSub(self, s: str) -> int:
        '''
        count the streaks
        if we have a streak of ones of length k, then there are k*(k+1) / 2 of subtrings
        find the streaks and compute
        '''
        ans = 0
        curr_streak = 0
        
        for ch in s:
            if ch == '0':
                ans += (curr_streak)*(curr_streak + 1) // 2
                curr_streak = 0
            else:
                curr_streak += 1
        
        ans += (curr_streak)*(curr_streak + 1) // 2
        return ans % (10**9 + 7)
    
#accumulating variant
class Solution:
    def numSub(self, s: str) -> int:
        '''
        count the streaks
        if we have a streak of ones of length k, then there are k*(k+1) / 2 of subtrings
        find the streaks and compute
        '''
        ans = 0
        curr_streak = 0
        
        for ch in s:
            if ch == '1':
                curr_streak += 1
            else:
                curr_streak = 0
            
            ans = (ans + curr_streak) % (10**9 + 7)
        
        return ans % (10**9 + 7)
    
#################################################################
# 2914. Minimum Number of Changes to Make Binary String Beautiful
# 05NOV24
##################################################################
class Solution:
    def minChanges(self, s: str) -> int:
        '''
        we can loop through each cahr in the string and keep track of the current sequence'ss length
        if we reeach the end of a sequence and its length is even, we can move on to the next sequence,
            basically count all streaks of zero's or ones
        
        if its an odd length, we will flip the last bit of that sequence to make it even
        flipping the last bit will an additional bit to the next sequcne, so count here is 1 when flip
        
        prrof by contradiction
        assume there exists a better solution that requires few flips by flipping some bit other than the lasst
        call this S1, which is of size k and is odd
        call another sequence S2
        S1 = b1 b2 b3 ... bk
        S2 is at b_{k+1}
        case 1 
            flip b_{k}
            it then S1 = b1 .. b_{k-1}
            and b_{k} becomes part of S2
            cosst is 1 flip
        
        case 2
            flip any number of bits where i < k
            this cost more than one flip
            but we already set S1 was optimal
        
        so assumption that there exists a better solution is false
        '''
        curr_char = s[0]
        streak_count = 0
        flips = 0
        
        for ch in s:
            if ch == curr_char:
                streak_count += 1
                continue
            
            if streak_count % 2 == 0:
                streak_count = 1
            else:
                streak_count = 0
                flips += 1
            
            curr_char = ch
        
        return flips
    
class Solution:
    def minChanges(self, s: str) -> int:
        '''
        the string is even length
        why does it need to be even though?
        whoops its not splitting, i can change any character in s to a 0 or a 1
        its beautiful if we can partition it into on or more substrings such that
        each had substring has even length and it only contains 1's or 0's
        check each block of size 2
        since each part consists of an even number of the same chars, we just check each block of size 2

        atomic unit of beautiful string is size 2
        if size 2 isn't beautfil, we need to make change to eithe the left or right pair

        even length means and possibilty of making it beautiful means that any substring that is beauitfil must be 00 or 11
        so we just count disimlar pairs
        '''
        n = len(s)
        ans = 0
        for i in range(0,n,2):
            if s[i] != s[i+1]:
                ans += 1
        
        return ans
    
#####################################################
# 3011. Find if Array Can Be Sorted
# 06NOV24
#####################################################
class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        '''
        we are only allowed to swap two numbers if they have the same number of bits
        N^2 is allowed given the inputs
        for each number compute the number of set bits
        split the array into segments where each segment has the same number of bits
        largest element in prev segment should be smaller than current
        '''
        segments = []
        #from left to right largest num in prev must be smaller than the smallest num in next
        i = 0
        n = len(nums)
        
        
        while i < n:
            curr_segment = []
            while len(curr_segment) == 0 or (i < n and self.count_bits(curr_segment[-1]) == self.count_bits(nums[i])):
                curr_segment.append(nums[i])
                i += 1
            
            segments.append(curr_segment)
        
        m = len(segments)
        i = 0
        #print(segments)
        while i + 1 < m:
            prev = segments[i]
            next_ = segments[i+1]
            if max(prev) > min(next_):
                return False
            i += 1
        
        return True
        
        
    def count_bits(self,num):
        count = 0
        while num:
            count += 1
            num = num & (num - 1)
        
        return count