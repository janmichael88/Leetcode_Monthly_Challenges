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

#######################################
# 6. Zigzag Conversion (REVISTED)
# 03FEB23
#######################################
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        '''
        zigzag pattern goes top to bottom
        left to right, but with alternatinv top and bottom
        
        allocate num rows as list
        then walk the zigzag pattern in the rows
        
        then rebuild
        '''
        #cornercase
        if numRows == 1:
            return s
        N = len(s)
        rows = [[] for _ in range(numRows)]
        
        i = 0
        row_ptr = 0
        curr_direction = 1 #initially going down
        #we only reverse direction on row_ptr == 0 and row_ptr == numRows
        
        while i < N:
            
            if row_ptr == 0:
                curr_direction = 1
            if row_ptr == numRows - 1:
                curr_direction = -1
            
            rows[row_ptr].append(s[i])
            row_ptr += curr_direction
            i += 1
            
        ans = ""
        for r in rows:
            ans += "".join(r)
        
        return ans

#official solution
#solving for rows and columns, by examing secitons
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        '''
        we can make a 2d grid, then walk the zig zag pattern,
        we know there are at least numRows. but how many columns?
        there will be at leas (numRows -1) columns in each sectino
        columns = ceil(n / (2 * numRows - 2)) * (numRows - 1) 
        
        grid is : numRows×numCols, where 
        numCols = ceil(n / (2 * numRows - 2)) * (numRows - 1)
        
        each sectino at most numRows + numRows - 2 characters
        sections =  ceil(n / (2 * numRows - 2))


        we first do down the column
        back up two rows, then over 1 column
        then diagonally up, then just repeat
        

        '''
        if numRows == 1:
            return s
        
        N = len(s)
        sections = ceil(N / (2* numRows - 2.0))
        numCols = sections*(numRows-1)
        
        grid = [[" "]*(numCols) for _ in range(numRows)]
        
        curr_row,curr_col = 0,0
        i = 0
        
        while i < N:
            #move down
            while curr_row < numRows and i < N:
                grid[curr_row][curr_col] = s[i]
                curr_row += 1
                i += 1
            
            #back up 2 rows, then 1 over
            curr_row -= 2
            curr_col += 1
            
            #up 1, right 1
            while curr_row > 0 and curr_col < numCols and i < N:
                grid[curr_row][curr_col] = s[i]
                i += 1
                curr_row -= 1
                curr_col += 1
            
        answer = ""
        for row in grid:
            for ch in row:
                if ch != " ":
                    answer += ch
        
        return answer

##################################
# 567. Permutation in String (REVISTED)
# 05FEB23
##################################
#count hashmap comparsion, could also have don't sorting
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        '''
         return true if s2 contains a permutation of s1, or false otherwise.
        first part of the question is misleadin
        s1's permutations must be a substring in s2
        
        permutation if s1 must exist as a substring in s2
        
        checkk all substrings of length(s1)
        but define count mapp first
        '''
        
        counts_s1 = Counter(s1)
        N = len(s1)
        M = len(s2)
        
        for i in range(M-N+1):
            substring = s2[i:i+N]
            #get count map
            temp = Counter(substring)
            if temp == counts_s1:
                return True
        
        return False

#sliding window is the true solution
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        '''
        this stems from the fact the we don't neeed to generate a new hashmap for every substring in s2
        we can create the hashmap once for the first window in s2
        then later when we slide in the window, we remove a char and add a char
        
        tricks:
            we can loop over the entire string s2 and check if current char in counter
            then we need to make sure the last character is removed
                we do this by first checking if we have at least len(s1), then decrement
                as soon as all the counts hit zero, we have a valid perm in s2
        '''
        counts_s1 = Counter(s1)
        
        for i in range(len(s2)):
            #remove
            if s2[i] in counts_s1:
                counts_s1[s2[i]] -= 1
            #add back in
            if i >= len(s1) and  s2[i-len(s1)] in counts_s1:
                counts_s1[s2[i-len(s1)]] += 1
            
            if all([counts_s1[i] == 0 for i in counts_s1]):
                return True
        
        return False


###############################################
# 438. Find All Anagrams in a String (REVISTED)
# 05FEB23
###############################################
#similart to #567
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        ans = []
        
        #get inital counts for 0
        counts_p = Counter(p)
        N = len(s)
        
        for i in range(N):
            if s[i] in counts_p:
                counts_p[s[i]] -= 1
                
            #add back in
            if i >= len(p) and s[i-len(p)] in counts_p:
                counts_p[s[i - len(p)]] += 1

                
            #zero counts meanins this is a valid index
            if all([count == 0 for ch,count in counts_p.items()]):
                #want starting index
                ans.append(i-len(p)+1)
        
        return ans


###################################################
# 1105. Filling Bookcase Shelves
# 05FEB23
###################################################
#fuccckkk, close one
#almost had it
class Solution:
    def minHeightShelves(self, books: List[List[int]], shelfWidth: int) -> int:
        '''
        return the minimum possible height that the total boooksehlf can be after placing shelves in this manner;
            * choose some of the books to place on this shelf such that the sum of their thickenss <= shelfWidth
            * then build another level of the sehlf of the bookcase so that the ttal heigh of the bookcase increase by the maximum height 
                of the books at this level
            * repeat until there are no more books to place
            
        dp using prefixes
        dp(i) will be the anser to the problem for books[i:]
        
        so we want dp(len(books))
        
        dp(i) be the min height needed to fit books[i:]
        if i < 0:
            return float('inf')
            
        if the book we are about to add fits on the shelfWidth and its height is bounded by the largest book on the current shelf
        then dp(i) = dp(i-1)
        
        otherwise we need to start a new row, in which case the height increases by the largest book on this current row
        dp(i) = dp(i-1) + max(of largest book on this row)
        
        keep track of idex, laregest book on shelf, and currshelfwidth
        '''
        memo = {}
        N = len(books)
        
        def dp(i,largest_height,curr_row):
            if i < 0:
                return largest_height
            if (i,largest_height,curr_row) in memo:
                return memo[(i,largest_heigh,curr_row)]
            
            first = float('inf')
            second = float('inf')
            #1. if we can fit this book on the current row
            if books[i][0] + curr_row <= shelfWidth:
                first = dp(i-1,max(largest_height, books[i][1]),books[i][0] + curr_row )
            #2. we have to make a new row 
            if books[i][0] + curr_row > shelfWidth:
                second = largest_height + dp(i-1,books[i][1],0)
            #take the minimum
            ans = min(first,second)
            memo[(i,largest_height,curr_row)] = ans
            return ans
        
        return dp(N-1,0,0)

#alsmot had it, in the case, out curr_row goes negative, just return a large int
#otherwise its knapsack
class Solution:
    def minHeightShelves(self, books: List[List[int]], shelfWidth: int) -> int:
        memo = {}
        N = len(books)
        
        def dp(idx: int, curr_height: int, curr_width: int):
            if curr_width < 0:
                return float("inf")

            if idx == len(books):
                return curr_height
            
            if (idx,curr_height,curr_width) in memo:
                return memo[(idx,curr_height,curr_width)]

            thickness, height = books[idx]
            same_shelf = dp(idx + 1, max(curr_height, height), curr_width - thickness)
            change_shelf = curr_height + dp(idx + 1, height, shelfWidth - thickness)
            ans = min(same_shelf, change_shelf)
            memo[(idx,curr_height,curr_width)] = ans
            return ans
        
        
        return dp(0, 0, shelfWidth)

#similar to line spacing alignment problem for MS word
class Solution:
    def minHeightShelves(self, books: List[List[int]], shelfWidth: int) -> int:
        '''
        instead of keeping three states (index,curr_heigh,and curr width)
        we can just use i
        dp(i) represents the the min height using books [:i]
        then we need to just examine j for [i,len(books)] and decide what this answer should be
        we can only update if adding this book doesn't exceed the shelfwidth
        '''
        N = len(books)
        memo = {}
        
        def dp(i):
            if i == N:
                return 0
            if i in memo:
                return memo[i]
            res = float('inf')
            curr_row_width = 0
            curr_max_height = 0
            #try all j after i
            for j in range(i,N):
                jth_thickness, jth_height = books[j]
                #if we can fit this book on this row
                if curr_row_width + jth_thickness <= shelfWidth:
                    curr_row_width += jth_thickness
                    #update max height
                    curr_max_height = max(curr_max_height,jth_height)
                    #update res
                    res = min(res,curr_max_height + dp(j+1))
                #we have to broek, eitherwise we cant fit on this shelf, because its books[:i]
                else:
                    break
                    
            memo[i] = res
            return res
        
        
        return dp(0)

#rewriting bottom up
class Solution:
    def minHeightShelves(self, books: List[List[int]], shelfWidth: int) -> int:
        N = len(books)
        dp = [0]*(N+1)
        
        for i in range(N-1,-1,-1):
            if i == N:
                dp[i] = 0
            else:
                res = float('inf')
                curr_row_width = 0
                curr_max_height = 0
                #try all j after i
                for j in range(i,N):
                    jth_thickness, jth_height = books[j]
                    #if we can fit this book on this row
                    if curr_row_width + jth_thickness <= shelfWidth:
                        curr_row_width += jth_thickness
                        #update max height
                        curr_max_height = max(curr_max_height,jth_height)
                        #update res
                        res = min(res,curr_max_height + dp[j+1])
                    #we have to broek, eitherwise we cant fit on this shelf, because its books[:i]
                    else:
                        break
                
                dp[i] = res
        
        return dp[0]
            



#########################################
# 1470. Shuffle the Array
# 06FEB23
##########################################
#just get i and i + n
class Solution:
    def shuffle(self, nums: List[int], n: int) -> List[int]:
        '''
        for and index i, we want [i,i + n]
        '''
        ans = []
        for i in range(n):
            ans.append(nums[i])
            ans.append(nums[i+n])
        
        return ans

class Solution:
    def shuffle(self, nums: List[int], n: int) -> List[int]:
        '''
        pre allocate results array
        '''
        ans = [0]*(2*n)
        
        for i in range(n):
            ans[2*i] = nums[i]
            ans[2*i + 1] = nums[i+n]
        
        return ans
            
class Solution:
    def shuffle(self, nums: List[int], n: int) -> List[int]:
        '''
        O(1) solution uses bit shift operators
        intuition:
            the largest value in nums array is 10**3, which is 1111101000 in binary
            each element would take at most 10 bits in a 32 bit interger
                the remaining bits are 0 and left unused
            
            this suggests the diea that in the remaining emptyy unused bits we can store extra inforamtion
            one possible solutino is toring two numbers togehter
                1 in the first 10, the other in the next 10
            in bit config
                00[bits for fist num].[bits in second num]
            
           
         idea store the last n numbers with thefirt n numbers in the nums array
            i.e the first n numbers will have their correspsoing first part and second part in one 32 signed in
            then we pass a second time, pulling apart the numbers in the spots 0 to n
            and place them at their respective spots (2*i and 2*i + 1)
            
        storing, two numbers together
        let a be the first number, and b the second number
        e can left shift b by 1- bits and take bitwise OR with a
            when we take any bitwise-OR with -, it results in the same bit
            
        the first 10 bits in are 0, so when we take bitwise OR, the result's first 10 bits will have a's 10 bits
        and the next 1- bits of a are 0, so the results next 10 will stroed b's 10 bits there
        
        the final result has bits of both a and b
        i.e, rigt shift, and take XOR
        
        extracting
            the new int will have a in the first 10 bits, and b in the second 10 bits, so can we pull it apart
            we can retreive it by taking bitwise AND with an all ones in the first 1- bits
            i.e num & 000000000111111111
        '''
        #for the first n numbers, encode them first and second
        #first 10 bits is the second number, next 10 bits is the first number
        for i in range(n,2*n):
            second_num = nums[i] << 10
            nums[i-n] = nums[i-n] | second_num
        
        #get the 1s mask in the first 10 bit position
        mask = (2 << 9) - 1
        #we start by putting all numbers for the end
        #i.e start fomr the end of the left half, if we started from the front of the left half, we might have over-written some of the number
        for i in range(n-1,-1,-1):
            second_num = nums[i] >> 10
            first_num = nums[i] & mask
            #put into position starting from end
            nums[2*i + 1] = second_num
            nums[2*i] = first_num
        
        return nums

#################################
# 904. Fruit Into Baskets
# 07FEB23
#################################
#closeeeeee
class Solution:
    def totalFruit(self, fruits: List[int]) -> int:
        '''
        we have row of trees, with each tree containing 1 fruit
            each fruit is a unique number
        we only have two baskets tht can only hold a single type of fruit
        we can start from any tree of our choice
            we must pick exactly one freuit from every t ree while moving to the right
        one we reach a tree that we cannot fit in any of my baskets, we must stop
        
        return maximum number of fruites i can pick
        
        sliding window!
        size of the window is the number of fruits, but we only want to have 2 unique fruits in that window
        
        '''
        N = len(fruits)
        in_basket = set()
        max_fruits = 0
        
        right = 0
        
        while right < N:
            #if we need to exapnd
            left = right
            while right < N and len(in_basket) != 2:
                in_basket.add(fruits[right])
                right += 1
            #we have a valid answer here
            max_fruits = max(max_fruits,right - left + 1)
            
            #reset baskset
            while left < N and len(in_basket) > 2:
                in_basket.remove(fruits[left])
                left += 1
            #advance
            right += 1
            
        
        return max_fruits

#we need to keep track of the counts in the basket, just a set won't work
class Solution:
    def totalFruit(self, fruits: List[int]) -> int:
        '''
        almost had it, 
        instead of keeping a set to show the basket, we need to use a count map
        then expand until our size of count map becomes > 2
        then remove
        '''
        basket = Counter()
        left = 0
        right = 0
        N = len(fruits)
        ans = 0
        
        while right < N:
            #add to basket
            basket[fruits[right]] += 1
            right += 1 
            
            #if we have too many fruits
            while len(basket) > 2:
                #remove the left side fuite
                basket[fruits[left]] -= 1
                #if this has a zero count, delete it
                if basket[fruits[left]] == 0:
                    del basket[fruits[left]]
                
                #advance 
                left += 1
            
            #valid subarray where sliding window is in the constraint
            ans = max(ans, right - left)
        
        return ans

class Solution:
    def totalFruit(self, fruits: List[int]) -> int:
        '''
        unoptimized sliding window
        anchor the left side of the sliding window, try all left in range(N)
        then we try to expand each possible sliding window
        
        we can stop early if the fruit we are trying to add is  not in the current basket, and we are already at capacity
        '''
        ans = 0
        N = len(fruits)
        
        for i in range(N):
            basket = set()
            j = i
            
            while j < N:
                if fruits[j] not in basket and len(basket) == 2:
                    break
                
                basket.add(fruits[j])
                j += 1
            
            ans = max(ans, j - i) #no plus 1 becase we have gone one more beyong the window
            
        return ans

class Solution:
    def totalFruit(self, fruits: List[int]) -> int:
        # Hash map 'basket' to store the types of fruits.
        basket = {}
        left = 0
        
        # Add fruit from the right index (right) of the window.
        for right, fruit in enumerate(fruits):
            basket[fruit] = basket.get(fruit, 0) + 1

            # If the current window has more than 2 types of fruit,
            # we remove one fruit from the left index (left) of the window.
            if len(basket) > 2:
                basket[fruits[left]] -= 1

                # If the number of fruits[left] is 0, remove it from the basket.
                if basket[fruits[left]] == 0:
                    del basket[fruits[left]]
                left += 1
        
        # Once we finish the iteration, the indexes left and right 
        # stands for the longest valid subarray we encountered.
        return right - left + 1

##################################
# 45. Jump Game II
# 08FEB23
##################################
#push dp
class Solution:
    def jump(self, nums: List[int]) -> int:
        '''
        first of all, its always possible to reach nums[n-1]
        this is dp
        if dp(i) represents the minimum jumps needed to get to n-1
        then its just 1 more step away,
        so try all jumps
        '''
        memo = {}
        N = len(nums)
        
        def dp(i):
            if i == N-1:
                return 0
            if i in memo:
                return memo[i]
            
            ans = float('inf')
            for jump in range(1,nums[i] + 1):
                if i + jump < N:
                    child = 1 + dp(i+jump)
                    ans = min(ans,child)
            
            memo[i] = ans
            return ans
        
        return dp(0)

class Solution:
    def jump(self, nums: List[int]) -> int:
        '''
        there exists a greedy solution
        subtelties
            since we are guaranteed to reach the last index, the starting range of each jump is always larger than the previous jump
        inutition
            we jump as far as we can from the ith position
            it doesn't make sense for us to go to a position < nums[i] + nums[i]
            greedy approach that tries to reach each index using the least number of jumps
        
        need to keep track of the furthest starting index of the current jump
        and the furhterst reachable iundex from the current jump
        
        once we have finished iterating over the range of the current jump (we reach index end)
        the next step is to continue iterating over the reachable indexes that are larger than end,
        which is represented as [end+1,far]
        
        in short, we can reach an index using j umps, we will never consider reaching it using more than j jumps
        Algorithm
            Initialize curEnd = 0, curFar = 0 and the number of jumps as answer = 0.

            Interate over nums, for each index i, the farthest index we can reach from i is i + nums[i]. We update curFar = max(curFar, i + nums[i]).

            If i = curEnd, it means we have finished the current jump, and should move on to the next jump. Increment answer, and set curFar = curEnd as the furthest we can reach with the next jump. Repeat from step 2.
            
        important part is assigning curr_end to curr_far to indicate the finishing of a current jump
        
        The main idea is based on greedy. Let's say the range of the current jump is [curBegin, curEnd], curFarthest is the farthest point that all points in [curBegin, curEnd] can reach. Once the current point reaches curEnd, then trigger another jump, and set the new curEnd with curFarthest, then keep the above steps, as the following:

This is an implicit bfs solution. i == curEnd means you visited all the items on the current level. Incrementing jumps++ is like incrementing the level you are on. And curEnd = curFarthest is like getting the queue size (level size) for the next level you are traversing.
        '''
        N = len(nums)
        jumps = 0
        curr_end = 0
        curr_furthest = 0
        
        for i in range(N):
            #if we have gotten to the end, we know we can get here using this number of jumps
            if i == N-1:
                return jumps
            #find the furthest range for this position
            curr_furthest = max(curr_furthest, i + nums[i])

            #if on this jump we reach the end, we need to use a new jump
            if i == curr_end:
                jumps += 1
                curr_end = curr_furthest
                
        return jumps

##########################################
# 2306. Naming a Company
# 09FEB23
###########################################
#TLE
class Solution:
    def distinctNames(self, ideas: List[str]) -> int:
        '''
        brute force is to examin all (i,j) pairs
        '''
        set_ideas = set(ideas)
        N = len(ideas)
        names = set()
        
        for i in range(N):
            for j in range(N):
                if i != j:
                    word1 = list(ideas[i])
                    word2 = list(ideas[j])
                    #swap
                    word1[0],word2[0] = word2[0],word1[0]
                    #join
                    if  "".join(word1) not in set_ideas and "".join(word2) not in set_ideas:
                        cand = "".join(word1)+" "+"".join(word2)
                        names.add(cand)
        
        return len(names)

class Solution:
    def distinctNames(self, ideas: List[str]) -> int:
        '''
        hints
            * group ideas sharing same suffix (all chars except the first), and notice that a pair of ideas from the same group is invalid 
            * but what is they come from differnt groups?, they are valid!
            * the first letter of the idea in the first group must not be the first letter of an idea in the second group
            * effeciiently cound the valid pairings for an idea of we already know how many ideas starting with letter x are within a group that doest no contain any ideas starting with letter y for all letters a and y
            
        notice, if two words are paird together, and have the same first letter, their concatenation will not yeilf a valid campany name
        inutition:
            in a hashmap, grou suffixes together key'd by their first letter
        
        if we have two starting letters x and y
            and a suffix is in both suffixes for x and y, swapping cannot result in a valid compnay name
            it must be the case then that the suffix must not be in either groups with first letter x and first letter y
            
        therefaire we need to try every pair of letters (a through z and find the unique suffixes)
        the number of unique suffixes in the two groups indicates one valid pairing, we can swap the pairing to get another one
        so times 2
        '''
        #group suffixes by index
        groups = defaultdict(set)
        
        for word in ideas:
            suffix = word[1:]
            first_letter = word[0]
            groups[first_letter].add(suffix)
        
        valid_pairings = 0
        for i in range(26):
            for j in range(i+1,26):
                first_letter = chr(ord('a') + i)
                second_letter = chr(ord('a') + j)
                #using set intserction notation
                
                #get nummber of common suffixes
                common_suffixes = len(groups[first_letter] & groups[second_letter])
                
                #unique for each group
                count_first_unique = len(groups[first_letter]) - common_suffixes
                count_second_unique = len(groups[second_letter]) - common_suffixes
                
                #cartesian product but twice
                valid_pairings += 2*count_first_unique*count_second_unique
        
        return valid_pairings

#####################################
# 1162. As Far from Land as Possible
# 10FEB23
#####################################
#TLE
class Solution:
    def maxDistance(self, grid: List[List[int]]) -> int:
        '''
        0 is water, 1 is land
        find a water cell such that its distance to the nearest land cell is maximized
        if no such land or water exists, return -1
        use manhat distance
        
        brute force, for all zero cells, get distances for all ones cells
            for this zero cell, find the distances to all ones, then take the min distance
            update globally on the max
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        zeros = []
        ones = []
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    ones.append((i,j))
                else:
                    zeros.append((i,j))
        
        ans = 0
        for zero in zeros:
            min_dist = float('inf')
            for one in ones:
                #get dist
                curr_dist = abs(zero[0] - one[0]) + abs(zero[1] - one[1])
                min_dist = min(min_dist,curr_dist)
            
            ans = max(ans,min_dist)
        
        if ans == 0 or ans == float('inf'):
            return -1
        else:
            return ans if ans != float('inf') else -1

#damn it
class Solution:
    def maxDistance(self, grid: List[List[int]]) -> int:
        '''
        bfs for each one cell
        keep aux 2d array 
            each (i,j) cell in this 2d array stored the maximum distance from a 1
            keep seen set to mark each (i,j) cell
            when we take a bfs step to a zero cell we have not visited, increment step count by 1 and update the max distance in the aux 2d array
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        dists = [[-1]*cols for _ in range(rows)]
        
        q = deque([])
        seen = set()
        dirrs = [(0,1),(0,-1),(1,0),(-1,0)]
        
        #q up
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    q.append((i,j,0))
                    
        while q:
            curr_x,curr_y,steps = q.popleft()
            #visit
            seen.add((curr_x,curr_y))
            for dx,dy in dirrs:
                neigh_x = curr_x + dx
                neigh_y = curr_y + dy
                #bounds
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                    #is zero and unvisited
                    if grid[neigh_x][neigh_y] == 0 and (neigh_x,neigh_y) not in seen:
                        #add to q
                        q.append((neigh_x,neigh_y,steps+1))
                        #maximize
                        dists[neigh_x][neigh_y] = max(steps+1, dists[neigh_x][neigh_y])
        
        #traverse and maximize
        ans = -1
        for row in dists:
            ans = max(ans,max(row))
        
        return ans

#multi point bfs
class Solution:
    def maxDistance(self, grid: List[List[int]]) -> int:
        '''
        bfs from each one cell and get the maximum number of levels
        because we bfs from a 1 to a zero, we know that this is the minimum number of steps from any 1 to a zero
        once we have coverd all 0 cells, we return the number of levels we had to do bfs on to
        inutition:
            start backwards
            
           Essentially we will start with all the 1's, and at each step, we will iterate in all four directions (up, left, right, down) for each 1. 
           The moment we reach a water cell 0, we can say that the number of steps we have taken so far is the minimum distance of this water cell from any land cell in the matrix. 
           This way, we will iterate over the whole matrix until all cells are covered; the number of steps we would need to cover the last water cell is the maximum distance we need. 
        '''
        rows = len(grid)
        cols = len(grid[0])
        dirrs = [(0,1),(0,-1),(1,0),(-1,0)]
        
        visited = [[False]*cols for _ in range(rows)]
        
        
        #push all land cells
        q = deque([])
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    q.append((i,j))
        

        level = -1
        #cant just do single point bfs
        #need visibilty of currently iteration level i am on
        while q:
            N = len(q)
            for _ in range(N):
                curr_x,curr_y = q.popleft()
                #mark
                visited[curr_x][curr_y] = True
                for dx,dy in dirrs:
                    neigh_x = curr_x + dx
                    neigh_y = curr_y + dy
                    #bounds
                    if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                        #is zero and unvisited
                        if grid[neigh_x][neigh_y] == 0 and not visited[neigh_x][neigh_y]:
                            #add to q
                            q.append((neigh_x,neigh_y))
            level += 1
        
        return -1 if level == 0 else level

class Solution:
    def maxDistance(self, grid: List[List[int]]) -> int:
        ROWS, COLS = len(grid), len(grid[0])
        queue = deque()
        
        for r,c in product(range(ROWS), range(COLS)): # prepopulate BFS queue with sources
            if grid[r][c]: queue.append((r,c))

        if not queue or ROWS * COLS == len(queue): # no land or all land
            return -1
        
        visited = set()
        level = 0
        while queue: # count number of BFS layers
            for _ in range(len(queue)):
                r,c = queue.popleft()
                for dr,dc in (0,1),(1,0),(0,-1),(-1,0):
                    nr,nc = r+dr,c+dc
                    if ROWS > nr >= 0 <= nc < COLS and not grid[nr][nc] and (nr,nc) not in visited:
                        visited.add((nr,nc))
                        queue.append((nr,nc))
            level += 1
        return level - 1

class Solution:
    def maxDistance(self, grid: List[List[int]]) -> int:
        '''
        this is also the 01 matrix problem and can be solved using dp
        
        intuition:
            for every 0 cell, we want the distance to the nearest land cell
            check all four directions
                take minimum of all four plus 1, if not a 1
        to find min distance for a cell, we need min distance of all the neighbor cells
        the catch is that we cannot have the min distance for all the neighbors in a single traversal
        if we traverse top left to bottom ight, we will have the min distance of the upper left cells as those cells would have already been traverse
        so we need two traversals
            first traversal, we do top left to bottom right and store min distance for cells using disance fo cell sin the up and left direction
            second from bottom right and top left storing min distance on the remaning right and down directions
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        max_distance = rows + cols + 1 #cannot be larger than this
        dists = [[0]*cols for _ in range(rows)]
        
        #first pass check up and to the left, opposting of the goind down and to the right
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    dists[i][j] = 0
                else:
                    #if we can go up
                    up = left = max_distance
                    if i > 0:
                        up = min(dists[i][j], dists[i-1][j] + 1,max_distance)
                    #if we can left
                    if j > 0:
                        left = min(dists[i][j],dists[i][j-1] + 1,max_distance)
                    dists[i][j] = min(up,left)
        
        
        #go in revere
        for i in range(rows-1,-1,-1):
            for j in range(cols-1,-1,-1):
                #check right and down
                right = down = max_distance
                if i < rows -1:
                    right = min(dists[i+1][j] +1,max_distance) 
                if j < cols - 1:
                    down = min(dists[j][j+1] + 1,max_distance)
                
                dists[i][j] = min(right,down)
        
        print(dists)
        
        ans = -1
        for i in range(rows):
            for j in range(cols):
                ans = max(ans,dists[i][j])
        
        return ans if ans != 0 else -1

#########################################
# 1129. Shortest Path with Alternating Colors
# 11FEB23
#########################################
#close
class Solution:
    def shortestAlternatingPaths(self, n: int, redEdges: List[List[int]], blueEdges: List[List[int]]) -> List[int]:
        '''
        given two edges, where each edge is of a different color
        return an array that gives the minimum path length from node 0 to node x, such that the edges in the shortest path alternate in color
        bfs from the start node, and keep track of last color taken in path and the current distance
        then just fill in the ans array
        '''
        red = defaultdict(list)
        blue = defaultdict(list)
        
        for u,v in redEdges:
            red[u].append(v) #its directed
        
        for u,v in blueEdges:
            blue[u].append(v)
        
        seen = set()
        ans = [float('inf')]*n
        
        q = deque([(0,0,None)]) #entry is (node,dist,lastedge taken)
        #1 is red, 0 is blue
        
        while q:
            curr_node,dist,last_edge = q.popleft()
            #visit
            seen.add((curr_node,last_edge))
            #update
            ans[curr_node] = min(dist,ans[curr_node])
            #1. no edge taken yet, so check both red and blue
            if not last_edge:
                #red edge
                for neigh in red[curr_node]:
                    if (neigh,1) not in seen:
                        q.append((neigh,dist+1,1))
                #blue edge
                for neigh in blue[curr_node]:
                    if (neigh,0) not in seen:
                        q.append((neigh,dist+1,0))
            
            #2.take red edge
            if last_edge == 0:
                for neigh in red[curr_node]:
                    if (neigh,1) not in seen:
                        q.append((neigh,dist+1,1))
            
            #take blue edge
            if last_edge == 1:
                for neigh in blue[curr_node]:
                    if (neigh,0) not in seen:
                        q.append((neigh,dist+1,0))
        
        for i in range(len(ans)):
            if ans[i] == float('inf'):
                ans[i] = -1
        return ans
                
                

class Solution:
    def shortestAlternatingPaths(self, n: int, redEdges: List[List[int]], blueEdges: List[List[int]]) -> List[int]:
        '''
        recall bfs gives shortest distance at first occruence of visited
        however, because we can reach a node via two differetn edges, each node can be visited at most twice
            once through a red edge and another time through a blue edge
        
        we alsot may need to return to the same node using a different color edge than the one we used on the first visit
        we can then procceed to cover other neighbors who not have been covered during the first visit due to similar color constraints
        
        algo:
            create adj list (recall its directed), but in addition to the node to neigh relatino ship, store 0 for red and 1 for blue
            the only catch here is that in the visited set for for (node,color) instead of just node
            update unto ans array of size n
        '''
        adj_list = defaultdict(list)
        for u,v in redEdges:
            adj_list[u].append((v,0))
        for u,v in blueEdges:
            adj_list[u].append((v,1))
        
        seen_states = set()
        ans = [float('inf')]*n
        
        q = deque([(0,0,-1)]) #entry is node, steps,and undefined state
        
        while q:
            curr_node,steps,edge = q.popleft()
            #update and add
            seen_states.add((curr_node,edge))
            ans[curr_node] = min(steps,ans[curr_node])
            for neigh_node,neigh_edge in adj_list[curr_node]:
                if (neigh_node,neigh_edge) not in seen_states and neigh_edge != edge:
                    q.append((neigh_node,steps+1,neigh_edge))
        
        
        for i in range(len(ans)):
            if ans[i] == float('inf'):
                ans[i] = -1
        
        return ans

###################################################
# 2477. Minimum Fuel Cost to Report to the Capital
# 12FEB23
####################################################
#bleghhh, kinda close
class Solution:
    def minimumFuelCost(self, roads: List[List[int]], seats: int) -> int:
        '''
        ideas:
            if i had unlimited capacity in each seat,
            then the answer would be the sum of the lengths of all paths coming from zero
            but i don't have unlimited capacity
            
            say we have a path length of size 5, and seat capacity 1
            fuel would be sum of 1 to 5, this is the cruz of the problem
            
        two parts
            1. find lengths of each path to zero
            2. for each path, determine the minimum fuel needed (this problem in itself could be another LC problem)
            3. for each subtree determin the min number of fuel needed,
            
        say path length if 5 and seats is 2
        min fuel needed is 2 + 4 + 5
        so for a given path length k, min fuek needed is:
            sections = k // seats
            fuel = 0
            start = seats
            for i in range(sections):
                fuel += start*(i+1)
            
            #then we want the remainder, each person would have to travel the remainder
            remainder = k % seats
            for i in range(remainder):
                fuel += n - i
            return fuel
        '''
        graph = defaultdict(list)
        for u,v in roads:
            graph[u].append(v)
            graph[v].append(u)
            
        #keep in mind, when i calculate min fuel, this is going to take up extra time
        def calc_min_fuel(k,seats):
            fuel = 0
            sections,remainder = divmod(k,seats)
            to_add = k
            for i in range(sections):
                fuel += to_add
                to_add -= seats
            #now add in remainders
            return fuel + remainder
        
        #dp, sum up the answer for each tree and return to the parent
        def dp(node,parent):
            ans = 0
            for neigh in graph[node]:
                if neigh != parent:
                    ans +=1 +  calc_min_fuel(dp(neigh,node),seats) 
            return ans
        
        return dp(0,None) + 1

#dfs global return, don't want global answer from top down return call
class Solution:
    def minimumFuelCost(self, roads: List[List[int]], seats: int) -> int:
        '''
        intution:
            it makes sense for a rep to only go from l+1 to l, insteaf of l to l+1 back to l
            we try to put as many represntatives as possible in the same car to save fuel
        
        consider node and parent, and r is the number of representatives in subtree node, that must traverse through parent
        worst case is that all r represntative just go, which requires r fuel, to cross the edge from node to parent
        best way is to put the r reps into a car, such that <= seats
        this would required (ceil(r/seats)) care and an equal amount of fuel
            take celing
                ceiling is used to round up the reaminder of reps, in which case, the last group should be <= seats
            For example, if you have 10 representatives in a subtree and the capacity is 3, then you would need ceil(10 / 3) = 4 cars.
            regardless of how the representatives arrive at node, there will definitely be at least ceil(r / seats) cars. 
            This is because all of the representatives in the subtree of node except for the one at node would arrive by using at least ceil((r - 1) / seats)
        
        we begin by moving all reps in a node's subtree to that node
        then calc min fuel getting to that node
        dfs to count number of reps in a subtree
        '''
                
        self.min_fuel = 0 #global variable    
        
        graph = defaultdict(list)
        for u,v in roads:
            graph[u].append(v)
            graph[v].append(u)
            
        def dfs(node,parent,seats):
            #function returns number of reps in a subtree
            reps = 1
            for child in graph[node]:
                if child != parent:
                    reps += dfs(child,node,seats)
            #we want the min fuel
            #0 has no parent, and we want min fuel to 0
            if node != 0:
                self.min_fuel += math.ceil(reps/seats)
            
            return reps
        
        dfs(0,-1,seats)
        return self.min_fuel

#bfs
class Solution:
    def minimumFuelCost(self, roads: List[List[int]], seats: int) -> int:
        '''
        we can also use bfs
        we need to traverse from the leaf noces to the root, Kahn's algorithm
            we move from l + k to l + k -1 ....l i.e bottom to top
        we also need to keep track of the in and out degree for each node, and another array to store the reps in eacn node
        
        start by inserting all leaf nodes into the q, leaf nodes should be a degree at most 1
        compute the min fuel needed to ferry a rep using ceiling (resp[node]/seats)
        
        we do not visite any child of node again while perfoming bfs
        we only push a need node if when taking that edge, reduces the degree, and that the degree[node] == 1
        
        '''
        graph = defaultdict(list)
        n = len(roads) + 1
        degree = [0]*n #edges per node
        reps = [1]*n #there is at least 1 rep at each node
        
        for u,v in roads:
            graph[u].append(v)
            graph[v].append(u)
            #increment degree
            degree[u] += 1
            degree[v] += 1
        
        #q up leaves
        q = deque([])
        for i in range(n):
            if degree[i] == 1:
                q.append(i)
                
        min_fuel = 0
        while q:
            curr_node = q.popleft()
            
            #get min fuel neeeded for this node
            min_fuel += math.ceil(reps[curr_node]/seats)
            for neigh in graph[curr_node]:
                #we no longer have an out going edge from this neigh
                degree[neigh] -= 1
                #increment the reps, rememebr we are moving bottom up
                reps[neigh] += reps[curr_node]
                #add the new leaf
                if degree[neigh] == 1 and neigh != 0: #is not root
                    q.append(neigh)
        
        return min_fuel
        
##############################################
# 1523. Count Odd Numbers in an Interval Range
# 13FEB23
##############################################
class Solution:
    def countOdds(self, low: int, high: int) -> int:
        '''
        brute force would be to check parity for all numbers between low and high and the increment a counter
        better brute force would be to start with the next larget odd number >= low, then move in steps of two
        
        intution:
            say we start with an odd integer x, next odd integer should be n + x
            there is exactly one even integer between every two odd integers, and hence all odd integers are equally space with a gap on one integer
            count of odd integers between x and greater integer where x is odd would be (y-x) / 2 + 1
            
            proof:
                if x is odd, then there are y -x /2 integers between x and y (not including y)
                we can write this as :
                    (y -x ) /2
                    the point here is that the number always leaves out the last intger as part of the count
                    so we need to add 1
        
        now the only part is trying to fin the next largest odd integer >= low
            if low is even, its following number would be odd
            so check if low is odd
                if not we can just incremnt it, that would be our next tarting
        '''
        #make sure low is odd
        if not (low & 1):
            low += 1
        
        #make sure to check if low passes high after incrementing
        if low > high:
            return 0
        return (high - low) // 2 + 1

class Solution:
    def countOdds(self, low: int, high: int) -> int:
        '''
        another way is to think about how many odds exists in the range [0,high]
        then we need to remove the counts from [0,low-1]
        
        starting from 0, the counts of any odd up to n is
            ceil(n/2)
        '''
        return math.ceil(high/2) - math.ceil((low - 1) / 2)

##################################
# 280. Wiggle Sort (REVISTED)
# 13FEB23
##################################
class Solution:
    def wiggleSort(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        '''
        first sort then wiggle
        [1,2,3,4,5,6]
        [1,6,2,5,3,4]
        take from each end
        '''
        sorted_nums = deque(sorted(nums))
        #then just build a wiggle sort
        i = 0
        while len(sorted_nums) > 2:
            nums[i] = sorted_nums.popleft()
            i += 1
            nums[i] = sorted_nums.pop()
            i += 1
        
        while sorted_nums:
            nums[i] = sorted_nums.popleft()
            i += 1

class Solution:
    def wiggleSort(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        '''
        sort and swap
        for every odd index i, swap with i + 1
        a <= b <= c <= d <= e.
        a <= c >= b <= d <= e
        a <= c >= b <= e >= d
        '''
        nums.sort()
        N = len(nums)
        for i in range(1,N-1,2):
            nums[i],nums[i+1] = nums[i+1],nums[i]

class Solution:
    def wiggleSort(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        '''
        for every index i, if i is even nums[i] should be smaller than or equal to nums[i+1]
        if its larger, we swap nums[i] with nums[i+1]
        
        if i is orrd nums[i] should be greater than or equal to nums[i+1]
        if nums[i] < nums[i+1] we swap
        
        i.e we preserve the wiggle invariant through the array, and check of any violations
        '''
        N = len(nums)
        for i in range(N-1):
            if (i % 2 == 0 and nums[i] > nums[i+1]) or (i % 2 == 1 and nums[i] < nums[i+1]):
                nums[i],nums[i+1] = nums[i+1],nums[i]

##########################################
# 726. Number of Atoms
# 13FEB23
##########################################
#close one
class Solution:
    def countOfAtoms(self, formula: str) -> str:
        '''
        we just need to count the atoms and return in sorted order by chemical abbreviation and their count
        similar to valid parentheses, i can use a stack
        closing bracket means we need to evaluate what is currently on the stack
        hardest part is trying to figure out what to put on the stack because a chemical abbreviation can be a single or double letter
        '''
        N = len(formula)
        i = 0
        stack = []
        while i < N:
            #first check double abbreviation
            if i < N-1 and formula[i].isalpha() and formula[i+1].isalpha():
                stack.append(formula[i:i+2])
                i += 2
            #if single abbreviation
            elif formula[i].isalpha():
                stack.append(formula[i])
                i += 1
            #if number, make sure to advance until we get to the end of the number
            elif '0' <= formula[i] <= '9':
                num = ""
                while '0' <= formula[i] <= '9':
                    num += formula[i]
                    i += 1
                #add to stack
                stack.append(int(num))
            # char is (
            elif formula[i] == '(':
                stack.append(formula[i])
                i += 1
            #else it ), but could be no number after or out of bounds
            else:
                #we already know we need to process what's on the stack, lets see if another number if after the )
                num = "1"
                i += 1
                while i < N and  '0' <= formula[i] <= '9':
                    num += formula[i]
                    i += 1
                num = int(num)
                #go backwards in the stack and if its type int, multiple
                j = len(stack) - 1
                while j > 0 and stack[j] != '(': #not the last closing
                    if isinstance(stack[j],int):
                        stack[j] *= num
                    j -= 1
                #we have arrived at the last 
                del stack[j]
        
        print(stack)

#####################################
# 989. Add to Array-Form of Integer
# 15FEB13
######################################
class Solution:
    def addToArrayForm(self, num: List[int], k: int) -> List[int]:
        '''
        convert and add then reconvert
        '''
        digits = []
        while k:
            digits.append(k % 10)
            k = k // 10
        
        #reverse
        digits = digits[::-1]
        
        ans = []
        carry = 0
        #this is just add binary now
        i = len(num) - 1
        j = len(digits) - 1
        
        while i >= 0 or j >= 0:
            first = num[i] if i >= 0 else 0
            second = digits[j] if j >= 0 else 0
            
            curr_digit = first + second + carry
            ans.append(curr_digit % 10)
            carry = curr_digit // 10
            i -= 1
            j -= 1
        
        #revers again
        if carry:
            ans.append(carry)
        
        return ans[::-1]

class Solution:
    def addToArrayForm(self, num: List[int], k: int) -> List[int]:
        '''
        we can just carry from the right
        get entire addend and keep adding right
        implementing carry each time
        '''
        N = len(num)
        #fist overload the right most spot
        num[-1] += k
        
        for i in range(N-1,-1,-1):
            carry, num[i] = divmod(num[i],10)
            #apply carry if we can before advancing
            if i > 0:
                num[i-1] += carry

        
        while carry:
            num = [carry % 10] + num
            carry = carry // 10
        
        return num

class Solution:
    def addToArrayForm(self, num: List[int], k: int) -> List[int]:
        '''
        since k will alwyas have a smaller number of digits than in num, we can do early termination
        
        '''
        N = len(num)
        for i in range(N-1,-1,-1):
            if not k:
                break
            k,num[i] = divmod(num[i] + k,10)
        
        while k:
            num = [k % 10] + num
            k = k // 10
        
        return num

