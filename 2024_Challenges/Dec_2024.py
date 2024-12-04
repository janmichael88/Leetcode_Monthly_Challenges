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
