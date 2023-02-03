###########################################
# 1071. Greatest Common Divisor of Strings
# 01FEB23
###########################################
#yessssssssss
class Solution:
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        '''
        for two strings s and t, t can divide s if s = t + t + ... t,
        rather t is just a concatenation of s
        return the largest string x such that x divides both str1 and str2
        
        brute force would be to check all possible substrings of str2, and check that it can be a substring of str1
        
		slow but it works
        '''
        def test_prefix(s,t):
            #s is str1 and t is what we are testing
            while t != "" and s.startswith(t):
                s = s.replace(t,"")
            
            return s == ""
        
        ans = ""
        for i in range(len(str2)+1):
            cand = str2[:i]
            #test both ways, it must divide both
            if test_prefix(str1,cand) and test_prefix(str2,cand):
                ans = cand

        
        return ans


class Solution:
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        '''
        we know that for a string to be a divisor of another string, it must be a prefix in both strings
        so we can just check all prefixes of one in the other
        
        in order for it to evenly divide, the smallest string can at most go into the larger string
        so we only need to generate all prefixies starting with the smaller string
        
        we need to find the base string in each str1 and str2, if the base string can divide str1 and str2
        it is a valid candidate
        the base string concatenated n times will yeild str1 and the base stringe concatenated m times will yeild str2
        for any valid interger n and m
        
        we actually don't need to compare potential prefixes as bases, just get their lengths
        the prefix can be any size from 0 to min(len(str1),len(str2))
        
        this is actually a really good review
        
        '''
        size1 = len(str1)
        size2 = len(str2)
        
        def valid_divisor(k):
            #must evenly divide both strings
            if size1 % k != 0 or size2 % k != 0:
                return False
            #count the number of parts for each string
            parts1,parts2 = size1 // k, size2 // k
            #find the base string
            base = str1[:k]
            #check that we can make conctenations
            return str1 == base*parts1 and str2 == base*parts2
        
        #we want the longest string, so start with the largest length
        #dont go past 0
        for size in range(min(size1,size2),0,-1):
            if valid_divisor(size):
                return str1[:size]
        
        return ""

#from chatgpt
class Solution:
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        '''
        just compute the gcd of len(str1) and len(str2)
        and return that prefix with that length
        '''
        def gcd_of_strings(str1, str2):
            if len(str1) < len(str2):
                str1, str2 = str2, str1
            if str1 == str2:
                return str1
            if str1[:len(str2)] != str2:
                return ""
            return gcd_of_strings(str1[len(str2):], str2)

        
        return gcd_of_strings(str1,str2)


#iteratively
class Solution:
	def gcd_of_strings(str1, str2):
	    if len(str1) < len(str2):
	        str1, str2 = str2, str1
	    if str1 == str2:
	        return str1
	    for i in range(len(str2), 0, -1):
	        if len(str2) % i == 0 and str1[:i] * (len(str2) // i) == str2:
	            if str1[:i] * (len(str1) // len(str2)) == str1:
	                return str1[:i]
	    return ""




#using gcd on lengths
class Solution:
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        '''
        since both strings need to be concatentations of each other, we need to check:
        str1 + str2 == str2 + str1
        
        otherwise we need to find the gcd of the two lengths and check
        
        contradiction:
            it is not possible for the GCD string to be shorter than the gcdbase
            
        if there is a shorter string that divides both str1 and str2, then gcdBase is also a divisible string, so a divisible string shorter than gcdBase can never be the GCD string
        '''
        def gcd(x,y):
            if y == 0:
                return x
            else:
                return gcd(y,  x % y)
            
        
        #check if the have the non-zero GCD string
        if str1 + str2 != str2 + str1:
            return ""

        k = gcd(len(str1),len(str2))
        return str1[:k]

################################################
# 953. Verifying an Alien Dictionary (REVISTED)
# 02FEB23
################################################
class Solution:
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        '''
        just compare each ith word to the i + 1 word
        '''
        ordering = {ch:i for i,ch in enumerate(order)}
        N = len(words)
        
        
        for i in range(1,N):
            first_word = words[i-1]
            second_word = words[i]
            #pointers into the words
            j = 0
            k = 0
            while j < len(first_word) and k < len(second_word):
                #get chars
                first_char = first_word[j]
                second_char = second_word[k]
                #less than, first occurent leave loopd
                if ordering[first_char] < ordering[second_char]:
                    break
                #violation
                elif ordering[first_char] > ordering[second_char]:
                    return False
                else:
                    j += 1
                    k += 1
            #in the case we have fall through after breaking
            #this fires only after a break in the before while loop
            else:
                #size
                if len(first_word) > len(second_word):
                    return False

        return True

class Solution:
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        '''
        also could have used comparator in the sorting method
        '''
        ordering = {ch:i for i,ch in enumerate(order)}
        
        #function
        order_words = lambda word : [ordering[ch] for ch in word ]
        
        #sort
        sorted_words = sorted(words, key =  order_words)
        return sorted_words == words

#####################################
# 1254. Number of Closed Islands
# 02FEB23
#####################################
#bleaghhhh
class Solution:
    def closedIsland(self, grid: List[List[int]]) -> int:
        '''
        0's are land, and 1's are water
        count number of closed islans
        
        i could get all the islands, then just check that each island is closed
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        islands = []
        dirrs = [[1,0],[-1,0],[0,1],[0,-1]]
        seen = set()
        
        def dfs(x,y,curr_island,seen):
            #add to currnet island
            curr_island.append([x,y])
            #add to seen
            seen.add((x,y))
            for dx,dy in dirrs:
                neigh_x = x + dx
                neigh_y = y + dy
                #bounds check
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                    #not seen and is land
                    if (neigh_x,neigh_y) not in seen and grid[neigh_x][neigh_y] == 0:
                        dfs(neigh_x,neigh_y,curr_island,seen)
                        
                        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 0 and (i,j) not in seen:
                    curr_island = []
                    dfs(i,j,curr_island,seen)
                    islands.append(curr_island)
        
        closed_islands = 0
        corner_cells = [[0,0],[rows-1,0],[0,cols-1],[rows-1,cols-1]]
        for isl in islands:
            #exclude corner cells
            for cell in isl:
                if cell in corner_cells:
                    break
            else:
                closed_islands += 1
        
        return closed_islands


#flood fill the the connected 0s to an edge with 1ones
#then dfs and count connected components
class Solution:
    def closedIsland(self, grid: List[List[int]]) -> int:
        '''
        flood fill edge containing 0s with ones, then just dfs and count connected components
        we don'y neet a seen set, we can just mutate the grid in place
        careful with xy, and ij
        '''
        rows = len(grid)
        cols = len(grid[0])
        dirrs = [[1,0],[-1,0],[0,1],[0,-1]]
        
        
        def dfs(x,y):
            #fill with 1
            grid[x][y] = 1
            for dx,dy in dirrs:
                neigh_x = x + dx
                neigh_y = y + dy
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                    if grid[neigh_x][neigh_y] == 0:
                        dfs(neigh_x,neigh_y)
                        
        #first fill in edges 
        for i in range(rows):
            for j in range(cols):
                if i == 0 or j == 0 or i == rows -1 or j == cols - 1:
                    if grid[i][j] == 0:
                        dfs(i,j)
        
        
        ans = 0
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 0:
                    dfs(i,j)
                    ans += 1
        
        return ans

class Solution:
    def closedIsland(self, grid: List[List[int]]) -> int:
        '''
        if i hit aborder this is not a captured island
        '''
        m, n = len(grid), len(grid[0])
        res = 0
        
        def dfs(x, y):
            if x in (0, m-1) or y in (0, n-1):
                self.isIsland = False 
            for i, j in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                if 0 <= i < m and 0 <= j < n and grid[i][j] == 0:
                    grid[i][j] = 1 
                    dfs(i, j)
                    
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0:
                    self.isIsland = True
                    dfs(i, j)
                    res += self.isIsland
                    
        return res 

class Solution:
    def closedIsland(self, grid: List[List[int]]) -> int:
        '''
        if i hit aborder this is not a captured island
        '''
        rows, cols = len(grid), len(grid[0])
        captured_islands = 0
        
        def dfs(x, y):
            #touches edge
            if x in (0, rows-1) or y in (0, cols-1):
                #just doing it a different way, non self variable
                #touches an edge means it is not a captured island
                nonlocal is_Captured
                is_Captured = False 
            for i, j in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                if 0 <= i < rows and 0 <= j < cols and grid[i][j] == 0:
                    grid[i][j] = 1 
                    dfs(i, j)
                    
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 0:
                    is_Captured = True
                    dfs(i, j)
                    captured_islands += is_Captured
                    
        return captured_islands 