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

#using stack of Counter objects
class Solution:
    def countOfAtoms(self, formula: str) -> str:
        '''
        we can use a stack
        for every opening parenthesesis, we need to evalute a new expression by adding to the top of the stack new hashmap
        for a closing parentehse, evaluate what is currently on the stack and add the previous one
        
        '''
        N = len(formula)
        stack = [Counter()] #counter holds counts of atoms for each expression
        i = 0
        
        while i < N:
            #new expression
            if formula[i] == '(':
                stack.append(Counter())
                i += 1
            #closing parantheses, evalute current exrpession and add to previous expression
            elif formula[i] == ')':
                top = stack.pop()
                #advance 
                i += 1
                i_start = i
                while i < N and formula[i].isdigit():
                    i += 1
                #convert to int
                multiplier = int(formula[i_start:i] or 1) #need at least 1
                #scale to previous expression
                for name,v in top.items():
                    stack[-1][name] += v*multiplier
            
            #otherwise its a name of a compound
            else:
                i_start = i
                i += 1
                while i < N and formula[i].islower():
                    i += 1
                #get the name
                name = formula[i_start:i]
                #in the case where its a digit after the anem
                i_start = i
                while i < N and formula[i].isdigit():
                    i += 1
                #again get the multiplier
                multiplier = int(formula[i_start:i] or 1)
                #add back in
                stack[-1][name] += multiplier
        
        ans = ""
        for name,count in sorted(stack[-1].items()):
            if count == 1:
                count = ""
            curr_element_count = name + str(count)
            ans += curr_element_count
        
        return ans

class Solution:
    def countOfAtoms(self, formula: str) -> str:
        '''
        we can also use a deque
        '''
        s = deque(formula)
        stack = []
        temp = []
        
        while s:
            #if we have a valid chemical name
            if s[0].isupper():
                name = s.popleft()
                while s and s[0].islower():
                    name += s.popleft()
                #get number
                num = self.extract_number(s)
                stack.append([name,num]) #entry is [chemical,num atomes]
            
            elif s[0] == '(':
                stack.append(s.popleft())
            
            elif s[0] == ')':
                #remove closing
                s.popleft()
                #number is shold be followed by closing 
                num = self.extract_number(s)
                
                while stack and stack[-1] != '(':
                    temp.append(stack.pop()) #aux stack for current evaluation
                stack.pop() #remove '('
                while temp:
                    name,count = temp.pop()
                    stack.append([name,count*num])
            
        counts = Counter()
        for name,count in stack:
            counts[name] += count
        
        ans = ""
        for name,count in sorted(counts.items()):
            if count == 1:
                count = ""
            curr_element_count = name + str(count)
            ans += curr_element_count
        
        return ans
            
                
    def extract_number(self,s):
        #s is a dequre of chars
        n = 0
        while s and s[0].isdigit():
            n = n*10 + int(s.popleft())
        #edge case when taking empty string
        n = max(1,n)
        return n

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

#####################################################
# 2357. Make Array Zero by Subtracting Equal Amounts
# 16FEB23
#####################################################
#well the hint gave it away
class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        possible = set()
        for num in nums:
            if num != 0:
                possible.add(num)
        
        return len(possible)

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        '''
        its just the number of non zero unique values in the array
        example [1,5,0,3,5]
        we have to take 1
        [0,4,0,2,4]
        we take 2
        [0,2,0,0,2]
        we take 2
        
        if we keep taking the samllest value x, every x will be reduces to zero
        then take another smallest y, should reduce every y to zero
        for every number not equal to x or y, the number becomes n - x - y
        
        [a,b,c,d,e]
        we have a <= b <= c <= d <= e
        start with a
        [0, b-a, c-a , d-a, e-a]
        
        next smallest is b - a
        [0,0,c-b,d-b,e-b]
        
        next smallest is c - b
        [0,0,0,d-c,e-c]
        
        next smallest is d-c
        [0,0,0,0,e-d]
        
        the numbers a,b,c,d,e were all unique, so it just the number of unique non zeros
        '''
        nums = set(nums)
        nums.discard(0)
        return len(nums)

###########################################
# 1376. Time Needed to Inform All Employees
# 16FEB23
###########################################
#nice try
class Solution:
    def numOfMinutes(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        '''
        through manager list, we can generate graph
        with headID as the root
        for each of the subordinates, we want the minimum time, then just take the minimum time at each level
        nope, this would work for a complete binary tree, but that might not always be the case
        what if we assum informtime was a weight edge
        take minimum at each node?
        
        if employee i has no suboridnates, informTime[i] == 0
        
        #store for each node, the time needed to be informed, which is just the total time
        #answer is the max time a leaf node needs to be informed
        '''
        graph = defaultdict(list)
        
        for i in range(n):
            if manager[i] == -1:
                continue
            else:
                graph[i].append(manager[i])
                graph[manager[i]].append(i)
        
        #single headnode, no time at all
        if not graph:
            return 0
        
        times = [0]*n
        
        def dfs(node,parent):
            #increment times
            times[node] += informTime[parent if parent != None else node]
            for neigh in graph[node]:
                if neigh != parent:
                    dfs(neigh,node)
                    
        
        dfs(headID,None)
        
        #take max of leaves
        ans = 0
        for i in range(n):
            if informTime[i] == 0:
                ans = max(ans,times[i])
        return ans

class Solution:
    def numOfMinutes(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        '''
        im dumb, this is a tree structure, so we can only do down, the edges must be directe
        dp(node) = max(dp(child) + informTime[node] for child in graph)
        we want the maximum answer for each node
        '''
        graph = defaultdict(list)
        for i in range(len(manager)):
            #its ok to leave -1 as a node, because we'd never go back to it anywa
            graph[manager[i]].append(i)
            
            
        def dp(node):
            ans = 0
            for neigh in graph[node]:
                child = informTime[node] + dp(neigh)
                ans = max(ans,child)
            
            return ans
        
        return dp(headID)

#bfs
class Solution:
    def numOfMinutes(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        '''
        bfs
        '''
        graph = defaultdict(list)
        for i in range(len(manager)):
            #its ok to leave -1 as a node, because we'd never go back to it anywa
            graph[manager[i]].append(i)
            
            
        #we can just do bfs and take max at each answer
        ans = 0
        
        q = deque([(headID,0)])
        
        while q:
            curr_emp,curr_time = q.popleft()
            ans = max(ans,curr_time)
            for neigh in graph[curr_emp]:
                neigh_time = informTime[curr_emp] + curr_time
                q.append((neigh,neigh_time))
        
        return ans

######################################
# 226. Invert Binary Tree
# 18FEB23
######################################
#dfs
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        '''
        dfs, just swap left and right
        invoke to modify then return root
        '''
        def dfs(node):
            if not node:
                return None
            node.left,node.right = node.right,node.left
            dfs(node.left)
            dfs(node.right)
        
        dfs(root)
        return root

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        q = deque([root])
        
        while q:
            curr = q.popleft()
            
            if not curr:
                continue
            curr.left, curr.right = curr.right,curr.left
            q.append(curr.left)
            q.append(curr.right)
        
        return root

####################################
# 305. Number of Islands II
# 17FEB23
####################################
#CLOSE ONEEEEE 101/161
class Solution:
    def numIslands2(self, m: int, n: int, positions: List[List[int]]) -> List[int]:
        '''
        say dp(i,j) returns whether (i,j) is already part of an island
        if i turn (i,j) into an island, i need to check all neighbors if this (i,j) is already part of is land
        if any of those return false, its not part of of an island an i can increse the count
        but the issue comes when i have to sepearate island, and marking this (i,j) makes it an island, 
        then just reduce the count
        '''
        dp = [[False]*n for _ in range(m)]
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        curr_islands = 0
        ans = []
        
        for x,y in positions:
            #inititally increment
            curr_islands += 1
            #add island
            dp[x][y] = True
            for dx,dy in dirrs:
                neigh_x = x + dx
                neigh_y = y + dy
                #bounds
                if 0 <= neigh_x < m and 0 <= neigh_y < n:
                    #adjust count
                    curr_islands -= dp[neigh_x][neigh_y]
            
            ans.append(curr_islands)
        
        return ans
                
#its union find
#fuckkkk, still close
class DSU:
    def __init__(self,size,):
        #size is going to m*n
        self.islands = [False]*size
        self.parent = [i for i in range(size)]
        self.size = [0]*size
        self.num_islands = 0
        
    def mark(self,idx):
        #alwyas consider making making a new island
        self.islands[idx] = True
        self.num_islands += 1
        
    def find(self,idx):
        if self.parent[idx] != idx:
            self.parent[idx] = self.find(self.parent[idx])
        
        return self.parent[idx]
    
    def union(self,idx1,idx2):
        #if these two make an island, we need to join them
        if self.islands[idx1] == True and self.islands[idx2] == True:
            #the joining of thes two would automatically make one less island
            self.num_islands -= 1
            parent1 = self.find(idx1)
            parent2 = self.find(idx2)
            #now make these point to the same parent
            if parent1 == parent2:
                return
            if self.size[parent1] > self.size[parent2]:
                self.size[parent1] += 1
                self.parent[parent2] = parent1
            
            elif self.size[parent2] > self.size[parent1]:
                self.size[parent2] += 1
                self.parent[parent1] = parent2
            else:
                self.size[parent1] += 1
                self.parent[parent2] = parent1
        else:
            return
            
            

class Solution:
    def numIslands2(self, m: int, n: int, positions: List[List[int]]) -> List[int]:
        '''
        turns out we need to use union find
        say we have the UF api, for the parent ID, it will i*rows + j
        but what about the union operation?
            i first need to check if any of the up,down,left, and right operations are already part of islands
            check them, and find their parent ids
            if they are already part of island, then everything should point to that new islanad
            only perform the union operation if i can find a valid parent?
        '''
        size = m*n
        dsu = DSU(size)
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        ans = []
        
        for i,j in positions:
            #mark initially
            dsu.mark(i*m + j)
            #check the fucking neighbors
            for dx,dy in dirrs:
                neigh_x = i + dx
                neigh_y = j + dy
                #bounds
                if 0 <= neigh_x < m and 0 <= neigh_y < n:
                    dsu.union(i*m + j, neigh_x*m + neigh_y)
            #print(dsu.find(i*m + j))
            ans.append(dsu.num_islands)
        
        return ans

#note, need to use i*cols + j to convert an (i,j) tuple to a unique index
class DSU:
    def __init__(self,size,):
        self.parent = [-1]*size
        self.rank = [0]*size
        self.count = 0
        
    def addLand(self,x):
        if self.parent[x] >= 0:
            return
        #otherwise mark
        self.parent[x] = x
        self.count += 1 #new island
        
    def isLand(self,x):
        return self.parent[x] >= 0
    
    def getCount(self,):
        return self.count
    
    #find method
    def find(self,x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        
        return self.parent[x]
    
    def union(self,x,y):
        parent_x = self.find(x)
        parent_y = self.find(y)
        
        if parent_x == parent_y:
            return
        
        elif self.rank[parent_x] > self.rank[parent_y]:
            #size adjustmnet
            self.rank[parent_x] += self.rank[parent_y]
            self.rank[parent_y] = 0
            self.parent[parent_y] = parent_x
        
        elif self.rank[parent_y] > self.rank[parent_x]:
            self.rank[parent_y] += self.rank[parent_x]
            self.rank[parent_x] = 0
            self.parent[parent_x] = parent_y
        else:
            self.rank[parent_x] += self.rank[parent_y]
            self.rank[parent_y] = 0
            self.parent[parent_y] = parent_x
        
        self.count -= 1
        
            
            
        
class Solution:
    def numIslands2(self, m: int, n: int, positions: List[List[int]]) -> List[int]:
        '''
        almost had it, there are just few things different in the UF api
            initialize parents all to -1
            add method that all returns the count of the islands
            when adding land, first check if parents != -1, otherwise have it point to itself
        '''
        size = m*n
        dsu = DSU(size)
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        ans = []
        
        for i,j in positions:
            idx = i*n + j
            dsu.addLand(idx)
            #check the fucking neighbors
            for dx,dy in dirrs:
                neigh_x = i + dx
                neigh_y = j + dy
                #bounds
                if 0 <= neigh_x < m and 0 <= neigh_y < n:
                    neigh_idx = neigh_x*n + neigh_y
                    if dsu.isLand(neigh_idx):
                        dsu.union(idx,neigh_idx)
            ans.append(dsu.getCount())
        
        return ans

#######################################
# 35. Search Insert Position (REVISTED)
# 20FEB23
########################################
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        '''
        this is just binary search
        just find the lower bound
        
        recursively
        '''
        
        def search(left,right,target):
            print(left,right)
            if left >= right:
                return left
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] >= target:
                return search(left,mid,target)
            else:
                return search(mid+1,right,target)
            
        
        return search(0,len(nums),target)
            

class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:

        left = 0
        right = len(nums)
        
        while left < right:
            mid = left + (right - left) // 2
            #found it
            if nums[mid] == target:
                return mid
            elif nums[mid] >= target:
                right = mid
            else:
                left = mid + 1
        
        return left

####################################################
# 2287. Rearrange Characters to Make Target String
# 20FEB23
####################################################
#fucking easy
class Solution:
    def rearrangeCharacters(self, s: str, target: str) -> int:
        '''
        count up chars in s and count up chars in target
        in order for their to be enough chars to make up target, there needs to exsist some multiplicity of say some ch in target also in s
        
        brute force is to simulate
        
        '''
        ans = 0
        count_s = Counter(s)
        count_t = Counter(target)
        
        copies = float('inf')
        
        for ch,count in count_t.items():
            #we are limited by the amount we can take
            copies = min(copies, count_s[ch] // count)
        
        return copies

#####################################################
# 540. Single Element in a Sorted Array (revisited)
# 21FEB23
#####################################################
#binary search
#https://leetcode.com/problems/single-element-in-a-sorted-array/discuss/1587270/C%2B%2BPython-7-Simple-Solutions-w-Explanation-or-Brute-%2B-Hashmap-%2B-XOR-%2B-Linear-%2B2-Binary-Search
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        '''
        the array is sorted, every element is written only twice, except for 1, which is written once
        well the inputs are wrong, len(nums) should never be less than 3
        i need to use the fact that there is only 1 single element, and the rest are written twice
        [1,1,2,3,3,4,4,8,8]
        
        pick 3
        [1,1,2,3] [4,4,8,8]
        looke left and look right
        looking left we see another 3, we know 3 cannot be the single lemeent
        now compare the sizes
        it has to be on the left size
        [1,1,2,3]
        pick 1
        look left and look right
        cannot be left, go right
        [2,3]
        pick 2
        
        starting array must be odd lengthed
        advance to the part of the array that is odd-lenghted, odd lengthed will contain the right answer
        there are 4 cases to cover
        '''
        left = 0
        right = len(nums)
        while left < right:
            
            mid = left + (right - left) // 2
            
            isHalfEven = (mid - left) % 2 == 0 #in order to keep the starting bounds at left = 0 and right = len(nums), we need to find odd/eveness using the lower half
            
            #lower half is even length, go right
            if mid+1 < len(nums) and nums[mid] == nums[mid+1]:
                if isHalfEven: 
                    left = mid + 2
                #left half not even, answer resides on left
                else: 
                    right = mid - 1
            #checek left
            elif mid and nums[mid] == nums[mid-1]:
                #must be on left size
                if isHalfEven: 
                    right = mid - 2
                #must be on right side
                else: 
                    left = mid + 1
            #we guessed right and return the middle
            else: 
                return nums[mid]
        
        return nums[left]

#using upper middle to find mid, but changing initial right bounds to len(nums) - 1
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:

        left = 0
        right = len(nums) - 1
        while left < right:
            
            mid = left + (right - left) // 2
            
            isHalfEven = (right - mid) % 2 == 0 #this should be lower half, not lengths of each half, 
            #think about about this would change if we wanted the upper half
            
            #lower half is even length, go right
            if mid+1 < len(nums) and nums[mid] == nums[mid+1]:
                if isHalfEven: 
                    left = mid + 2
                #left half not even, answer resides on left
                else: 
                    right = mid - 1
            #checek left
            elif mid and nums[mid] == nums[mid-1]:
                #must be on left size
                if isHalfEven: 
                    right = mid - 2
                #must be on right side
                else: 
                    left = mid + 1
            #we guessed right and return the middle
            else: 
                return nums[mid]
        
        return nums[right]

#even indices only
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        '''
        we only need use binary search at the even indices, it the single element in the array should occur at the first even index that does
        not have its pair
        i.i nums[i] != nums[i+1] for i in range(0,N,2)
        
        after the single element, the pattern changes to being odd index4s followed by their pair
            i.e if we have shot post the even index that does not have a pair, or we are at an odd index, and nums[this odd index + 1] != nums[this odd index] then this must be our value
            
        this meants that the single elemnt (an even index) and all elements after it are even indeces not followed by their pair
        therefore, given any even idnex in the array we can easily determine whether the single element it to the left or to the right
        
        algo:
            set up lo, hi and mid, the usual way, we now need to check parity for the current mid index
            this is the index in question
            we need to make mid index even, so if its odd -= 1
            now of nums[mid] == nums[mid+1], we still havent found the first occurrence of this even index
            so we can elimnate the left side
            otherwise we eliminate the right side
        '''
        left = 0
        right = len(nums) - 1
        while left < right:
            mid = left + (right - left)  // 2
            #correct for eveness
            if mid & 1 == 1:
                mid -= 1
            #not on the left side, we still have yet to find the first even index not having its pair
            if nums[mid] == nums[mid + 1]:
                left += 2
            else: 
                right = mid
        
        return nums[left]

###############################################
# 1011. Capacity To Ship Packages Within D Days
# 22FEB23
##############################################
#damn it, almost had it
class Solution:
    def shipWithinDays(self, weights: List[int], days: int) -> int:
        '''
        this is similar to koko eating banans
        we want to find the least weight that can ship the packages in at most days (<=)
        
        we know that sum(weights) can be the largest answer
        brute force would be to check all integers in sum(wegiths) and stop when we can't do it in days
        
        '''
        lo = 0
        hi = sum(weights)
        N = len(weights)
        
        while lo < hi:
            mid = lo + (hi - lo) // 2
            #mid is the least capacity we are trying out
            curr_weight = 0
            curr_days = 1
            for i in range(N):
                curr_weight += weights[i]
                if curr_weight > mid:
                    curr_days += 1
                    curr_weight = weights[i]
            
            #if we got to the end, every possible weight after should work, larger weight than the current canddiate
            if curr_days <= days:
                hi = mid
            else:
                lo = mid + 1
        
        return lo

class Solution:
    def shipWithinDays(self, weights: List[int], days: int) -> int:
        '''
        this is similar to koko eating banans
        we want to find the least weight that can ship the packages in at most days (<=)
        
        we know that sum(weights) can be the largest answer
        brute force would be to check all integers in sum(wegiths) and stop when we can't do it in days
        
        optimzation:  smallest weight cannot be less than max(weights) / d
        '''
        #intereating, our lower bounds must be the maximum of the weight
        #the highest bound is just the sum
        #doesn't make
        #i'm stupid we need to start with the largest weight, i can't ship anything if i use 0 as the carrying capacity
        lo = max(weights) #doensn't work with 0 or min(weights)
        hi = sum(weights)
        N = len(weights)
        
        while lo < hi:
            mid = lo + (hi - lo) // 2
            #mid is the least capacity we are trying out
            curr_weight = 0
            curr_days = 1 #we need to use at least 1 day anyway if we fit all packages first time around
            for i in range(N):
                curr_weight += weights[i]
                if curr_weight > mid:
                    curr_days += 1
                    curr_weight = weights[i]
            
            #if we got to the end, every possible weight before should work
            if curr_days <= days:
                hi = mid
            else:
                lo = mid + 1
        
        return lo
                
##########################################
# 502. IPO
# 23FEB23
###########################################
class Solution:
    def findMaximizedCapital(self, k: int, w: int, profits: List[int], capital: List[int]) -> int:
        '''
        leetcode wants to work on projects to increase capital before IP
        but it has limite resources and can only finish at most k distinc projects before the IPO
        want to maximize total capital after fininshing at most k distinct projects
        
        profits[i] is what we can gain when finish project
        and captial[i] is what we need to at least start
        
        initally we have w capital
        finishing a project[i] results in total profit += profits[i] and w += capital[i]
        
        starting a project does not decrease current capital (kinda wierd) so once we gain capital we cannot ever lose it
        we are not tracking capital, but overall profit
        
        if k = 1, we would need to choose the project that yeilds that largest proift, constained by the amount of capital
        if no project is avilable, we dont have any options but to return 0
        
        greedily choose the most profitbale available project
        then our capital increases by this proift, and some new project that were unavailable might become available now
        if we choose a project other than the most profitbale one, our capital increases by a value less than the maximum
        so we need to greedily choose the project with the largest profit, but do this k times
        
        1. find new available projects after finishing the previous one  
        2. finding the most profitibale one
        
        when we increase in capital, we have more options to choose from (but how do we effeciently choose)
        and the smaller capital project becomes available sooner
        so we can sort projects by increasing capital and keep a poointer to the first unavilable project
        as we gain more money we move this pointer to unlock projects that require more cpaital
        
        algo:
            1. sort projects by increasing capital but paired with respective profit
            2. keep pointer to the first unavailable project since we are limited by current capital
                a. find the first project we can't do for this kth project
                b. add projects to max heap
                c. if there a project in the max heap, retreive its profit
        '''
        N = len(profits)
        projects = []
        for profit,capital in zip(profits,capital):
            projects.append((profit,capital))
        
        #sort on increaing capital
        projects.sort(key = lambda x: x[1])
        #entries are (profit,capital)
        
        max_profits = [] #this is a max heap
        i = 0
        for _ in range(k):
            #find the first project we can't do but also add profits into q
            while i < N and projects[i][1] <= w:
                heapq.heappush(max_profits, -(projects[i][0]))
                i += 1
            if max_profits:
                w += -heapq.heappop(max_profits)
        
        return w        
