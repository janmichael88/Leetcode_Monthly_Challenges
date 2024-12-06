############################################
# 2109. Adding Spaces to a String
# 03DEC24
#############################################
class Solution:
    def addSpaces(self, s: str, spaces: List[int]) -> str:
        '''
        slice the string at left and right
        prepend 0 and just slice
        '''
        spaces = [0] + spaces + [len(s)]
        words = []
        for i in range(len(spaces) - 1):
            temp = s[spaces[i]: spaces[i+1]]
            words.append(temp)
        
        return " ".join(words)

class Solution:
    def addSpaces(self, s: str, spaces: List[int]) -> str:
        '''
        we can also just keep two pointers 
        one into spaces array meaning we can get to a space index
        and another into string
        '''
        ans = []
        space_idx = 0
        
        for i in range(len(s)):
            #in bounds and is the index we need a space for
            if space_idx < len(spaces) and i == spaces[space_idx]:
                ans.append(" ")
                space_idx += 1
            
            ans.append(s[i])
        
        return "".join(ans)

##########################################################
# 2825. Make String a Subsequence Using Cyclic Increments
# 04DEC24
#########################################################
class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        '''
        we select any set of indices in str1 and promot the char at that index (cyclically)
        check if we can make str2 a subsequence of str1 using the operation at most once
        checking if str2 isSubseq takes linear time
        
        each char in s can either be s[i] or (s[i] + 1) % 26 so check both
        '''
        idx_2 = 0
        
        for ch in str1:
            promoted_idx = (ord(ch) - ord('a') + 1) % 26
            promoted_ch = chr(ord('a') + promoted_idx)
            if ch == str2[idx_2] or promoted_ch == str2[idx_2]:
                idx_2 += 1
                if idx_2 == len(str2):
                    return True
        
        return idx_2 == len(str2)
            

    def is_sub(self,s,t):
        #check if t is subseq of s
        t_idx = 0
        for s_idx in range(len(s)):
            if t[t_idx] == s[s_idx]:
                t_idx += 1
        
        return t_idx == len(t)

##############################################
# 2337. Move Pieces to Obtain a String
# 05DEC24
##############################################
class Solution:
    def canChange(self, start: str, target: str) -> bool:
        '''
        we can move L to the left if there's a blank space
        we can move R to the right if there's a blank space
        using any number of moves, return true if we can reach target
        if we have target like:
        "L______RR", L could be in any of the positions "_"
        
        for both start and target, try to move L's left as possible and R's right as possible
        then check if equal
        relative order or L and R must stay the same
        omg store indices and L's and R's for each start and target
        then compare the indices
        L pieces are the start must be to the left of the L pieces in target
        R pieces at the start must be to the right of R pieces in target
        '''
        start_q = deque([])
        end_q = deque([])
        n = len(start)
        #lengthg of start and targer are equal
        for i in range(n):
            if start[i] != '_':
                start_q.append((start[i],i))
            if target[i] != '_':
                end_q.append((target[i],i))
        
        if len(start_q) != len(end_q):
            return False
        while start_q:
            start_char,start_idx = start_q.popleft()
            end_char, end_idx = end_q.popleft()
            
            if start_char != end_char:
                return False
            if start_char == 'L' and start_idx < end_idx:
                return False
            if start_char == 'R' and start_idx > end_idx:
                return False
        
            
        return True
                
#using array
class Solution:
    def canChange(self, start: str, target: str) -> bool:
        '''
        not using queue, just build as array
        '''
        start_q = []
        end_q = []
        n = len(start)
        #lengthg of start and targer are equal
        for i in range(n):
            if start[i] != '_':
                start_q.append((start[i],i))
            if target[i] != '_':
                end_q.append((target[i],i))
        
        if len(start_q) != len(end_q):
            return False
        
        for s,e in zip(start_q,end_q):
            
            start_char,start_idx = s
            end_char, end_idx = e
            
            if start_char != end_char:
                return False
            if start_char == 'L' and start_idx < end_idx:
                return False
            if start_char == 'R' and start_idx > end_idx:
                return False
        
            
        return True