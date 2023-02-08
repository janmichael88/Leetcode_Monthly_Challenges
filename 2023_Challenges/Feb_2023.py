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