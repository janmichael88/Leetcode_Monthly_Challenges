################################
# 256. Paint House (REVISITED)
# 01JUL22
################################
class Solution:
    def minCost(self, costs: List[List[int]]) -> int:
        '''
        if i paint the a house red, i can only paint it blue or green
        which means i absorb the cost of painting the ith house red and take the minimum of the next house from blue or green
        
        if i represent dp(i,color) by the answer to painting the ith house this color
        dp(i,color) = costs[i][color] + min(costs[i-1][other color],costs[i-1][other color])
        
        bases cases
        i == 0, return cost of that color
        '''
        memo = {}
        
        def dp(i,color):
            if i == 0:
                return costs[i][color]
            if (i,color) in memo:
                return memo[(i,color)]
            if color == 0:
                ans = costs[i][color] + min(dp(i-1,1),dp(i-1,2))
                memo[(i,color)] = ans
                return ans
            elif color == 1:
                ans = costs[i][color] + min(dp(i-1,0),dp(i-1,2))
                memo[(i,color)] = ans
                return ans
            elif color == 2:
                ans = costs[i][color] + min(dp(i-1,0),dp(i-1,1))
                memo[(i,color)] = ans
                return ans
            
        N = len(costs)
        return min(dp(N-1,0),dp(N-1,1),dp(N-1,2))
                

class Solution:
    def minCost(self, costs: List[List[int]]) -> int:
        '''
        dp, and just start from the base case working our way up to n
        space optimized is easy, just take prev and curr row
        then reassign after
        '''
        N = len(costs)
        dp = [[0]*3 for _ in range(N)]
        #base cases for first row
        dp[0][:] = costs[0][:]
        
        for i in range(1,N):
            for color in [0,1,2]:
                if color == 0:
                    ans = costs[i][color] + min(dp[i-1][1],dp[i-1][2])
                    dp[i][color] = ans
                elif color == 1:
                    ans = costs[i][color] + min(dp[i-1][0],dp[i-1][2])
                    dp[i][color] = ans
                elif color == 2:
                    ans = costs[i][color] + min(dp[i-1][0],dp[i-1][1])
                    dp[i][color] = ans
        
        return min(dp[N-1])

###############################
# 376. Wiggle Subsequence (REVISITED)
# 03JUL22
###############################
#nice try...
class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        '''
        a wiggle sequence if a sequence where successive differences are alternating
        length 1 is trivially a wiggle sequence
        length 2 with unequal eleemnts is a wiggle
        
        this is dp
        let dp(i) represent the longest subsequence starting with nums[i]
        we can extend if the the next elements has diff that is opposite this current diff
        we also need to record the previous as increasing or decreasing
        dp(i,state)
            then code state as 0 for increasing and 1 for decreasing
        then dp(i) for all i in range(len(nums))
        
        '''
        N = len(nums)
        memo = {}
        
        def dp(i,increasing):
            if i == 0:
                return 1
            if (i,increasing) in memo:
                return memo[(i,increasing)]
            first = 1
            second = 1
            for j in range(i+1,N):
                #get the sign of diff
                sign_diff = nums[j] - nums[i]
                #curretn if negative
                if sign_diff < 0 and inc == 1:
                    first = 1 + dp(i,inc)
                    memo[(j,0)] = first
                elif sign_diff > 0 and inc == 0:
                    second = 1 + dp(i,inc)
                    memo[(j,1)] = 1 + dp(i,inc)
            
            ans = max(first,second)
            memo[(i,increasing)] = ans
            return ans
        
        ans = 1
        for i in range(N):
            ans = max(ans,dp(i,0))
            ans = max(ans,dp(i,1))
        
        return ans

#top down, keeping track of i and increaisng from i
class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        memo = {}
        N = len(nums)
        def dp(i,increasing):
            #base case is when i gets to the last point, we are left at the point where the loop does not execute
            #so base case is when i == N
            if i == N:
                return 0
            if (i,increasing) in memo:
                return memo[(i,increasing)]
            ans = 0
            for j in range(i+1,N):
                if (increasing and nums[j] > nums[i]) or (not increasing and nums[j] < nums[i]):
                    ans = max(ans,1+dp(j, not increasing)) #flip
            
            memo[(i,increasing)] = ans
            return ans
        
        if N < 2:
            return N
        ans = 1 + max(dp(0,True),dp(0,False))
        return ans

#bottom up
class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        '''
        translating bottom up, each dp has state, i which is the index and whether we are increeasing or decresing rom there
        dp(i,j) represents the longest wiggle subsequence starting from index i, and if we are increasing/decreasing from index i
        '''
        N = len(nums)
        dp = [[0]*2 for _ in range(N+1)]
        
        for i in range(N-1,-1,-1):
            for j in range(i+1,N):
                diff = nums[j] - nums[i]
                if diff > 0: #we have an increasing here
                    #we need to maximize at each step
                    dp[i][0] = max(dp[i][0],dp[j][1] + 1)
                elif diff < 0: #we have decreasing here
                    dp[i][1] = max(dp[i][1],dp[j][0] + 1)
            
        
        if N < 2:
            return N
        return max(dp[0][0],dp[0][1]) + 


class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        '''
        we can use linear dynamic programming, after recognizing that there is a greedy way to solve the subpoblrems
        
        an up position, at i: nums[i] > nums[i-1]
        a down position, at i: nums[i] < nums[i-1]
        an equals position, nums[i] == nums[i-1]
        
        going backwards from the array
        
		also note that we can consolidate space by just keeping the previous up and down
        '''
        N = len(nums)
        dp = [[0]*2 for _ in range(N)]
        
        dp[N-1][0] = 1
        dp[N-1][1] = 1
        
        for i in range(N-2,-1,-1):
            #increasing
            if nums[i] > nums[i+1]:
                dp[i][1] = dp[i+1][0] + 1
                dp[i][0] = dp[i+1][0]
                
            elif nums[i] < nums[i+1]:
                dp[i][0] = dp[i+1][1] + 1
                dp[i][1] = dp[i+1][1]
            elif nums[i] == nums[i+1]:
                dp[i][0] = dp[i+1][0]
                dp[i][1] = dp[i+1][1]
        
        return max(dp[0])
                
class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        '''
        we can use linear dynamic programming, after recognizing that there is a greedy way to solve the subpoblrems
        
        an up position, at i: nums[i] > nums[i-1]
        a down position, at i: nums[i] < nums[i-1]
        an equals position, nums[i] == nums[i-1]
        
        going backwards from the array
        

        '''
        N = len(nums)
        up = 1
        down = 1
        
        if N < 2:
            return N
        
        for i in range(N-2,-1,-1):
            if nums[i] > nums[i+1]:
                up = down + 1
            elif nums[i] < nums[i+1]:
                down = up + 1
        
        return max(up,down)

class Solution:
    def wiggleMaxLength(self, nums: List[int]) -> int:
        '''
        we can use a stack to keep track of the last alternating increassing wiggle or decreasing wiggle
        '''
        stack = [0] #trivially length 1 is wiggle
        N = len(nums)
        for i in range(1,N):
            if nums[i] > nums[i-1] and stack[-1] != 1:
                stack.append(1)
            elif nums[i] < nums[i-1] and stack[-1] != -1:
                stack.append(-1)
        
        return len(stack)

##########################
# 388. Longest Absolute File Path
# 04JUL22
###########################
class Solution:
    def lengthLongestPath(self, input: str) -> int:
        '''
        this is just dfs, we can use dfs to build up a path
        we need to build up a path
        the number of \t gives us the the depth that we are at
        
        for each depth store the current path length and update max along the way if there is a file
        '''
        max_length = 0
        path_lengths = {0:0}
        #split on new lines to get folder structure
        for line in input.split('\n'):
            #parse out \n lines and parse out \t
            #get name of current line
            name = line.lstrip('\t')
            #get the depth for this line, which is just the number of \t in the string, we could also count this
            depth = len(line) - len(name)
            #if there is file, get the new length
            if '.' in name:
                max_length = max(max_length,path_lengths[depth] + len(name))
            #othewise it's a new directory
            else:
                path_lengths[depth+1] = path_lengths[depth] + len(name) + 1
        
        return max_length

#another way
class Solution:
    def lengthLongestPath(self, input: str) -> int:
        '''
        this is just dfs, we can use dfs to build up a path
        we need to build up a path
        the number of \t gives us the the depth that we are at
        
        for each depth store the current path length and update max along the way if there is a file
        
        for each directory name, trying storing a list of its files
        

        '''
        ans = 0
        depths = {-1:0}
        for split in input.split('\n'):
            #get the depth
            depth = split.count('\t')
            #update depths, do hold the lognest path here
            depths[depth] = depths[depth-1] + len(split) - depth #remove the tab characters
            #if there is afile, get its length
            if '.' in split:
                ans = max(ans,depths[depth] + depth)
        
        return ans

#using stack and getting path generation
class Solution:
    def lengthLongestPath(self, input: str) -> int:
        '''
        another way
        
        algo:
            for each dir or file, we store its current total length (including parent and '/' and depth, i.e how many '\t' to reach this subdir)
            if stack ie empty add new tuple
            if deepest dir or file in stack is at the same or deper dpeth of current path
                pop from stack until stack[-1] is hsalloer than depth of path
            add tuple and cumulat length
            if name has . then its file and maximuze
            
        note in this way, we are generating all paths and taking the lognest, we could save space and only record sizes
        '''
        stack = []
        ans = 0
        for path in input.split('\n'):
            #print(path.split('\t'))
            p = path.split('\t') #we have turn path into a list where \t have been reduced to empty stings
            #get depth
            depth = len(p) - 1
            #get its name
            name = p[-1]
            while stack and stack[-1][1] >= depth:
                stack.pop()
            if not stack:
                stack.append((name,depth))
            else:
                new_name = stack[-1][0]+"\\"+name
                stack.append((new_name,depth))
            
        print(stack)
        for path,depth in stack:
            #print(path)
            if '.' in path:
                ans = max(ans,len(path))
        
        return ans

class Solution:
    def lengthLongestPath(self, s: str) -> int:
        paths, stack, ans = s.split('\n'), [], 0
        for path in paths:
            p = path.split('\t')
            depth, name = len(p) - 1, p[-1]
            l = len(name)
            while stack and stack[-1][1] >= depth: 
            	stack.pop()
            if not stack: 
            	stack.append((l, depth))
            else: 
            	stack.append((l+stack[-1][0], depth))
            if '.' in name: 
            	ans = max(ans, stack[-1][0] + stack[-1][1])   
        return ans

############################
# 393. UTF-8 Validation
# 06JUL22
############################
class Solution:
    def validUtf8(self, data: List[int]) -> bool:
        '''
        return whether data is a valid UTF-encoding
        a character in UTF-8 can be from 1 to 4 bytes long
        for a 1 byte character, the first bit must be 0, followed by its unicode
        for an n bytes char, the first n bits are all ones, the n + 1 bit is 0, followed by n-1 bytes with the most significant 2 bits being 10
        
        1 byte -> 0xxxxxxx
        2 byte -> 110xxxxx 10xxxxxx
        3 byte -> 1110xxxx 10xxxxxx 10xxxxxx
        4 byte -> 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
        
        notes, the data can contain data for multiple characters, all of which can be valid UTF-8 characters and hence that the integers in the array can be larger than 255 as wekk
        the integers in the array can be larger than 255 as well
        the highest number that can be represented by 8 bits is 255
        so what do we do if an interger in the array exceeds 255, say 476, in this cas we only have to consider the 8 least signifiant bits of each integer
        
        example = [197,130,1]
        we can get the binary rep for each number
        11000101 10000010 00000001
        
        the first byte, has its most signifiant 2 bits set to 1, so thi is a valid 2 byte character
        if this is a valid 2 byte character, then he next byte must followe 10xxxxxx
        which indeed it does, so [197,130] form a valid 2-byte UTF-8 character
        the last one is 1, which follows the rule of a 1 byte utf8 char
        
        notes:
            the first byte tells us the length of the UTF8 char and hence the number of bytes we have to process in all in order to completely process a single UTF8 char in the array before moving on to another one
            This is the first rule in the problem statement and it clearly says that "A valid UTF-8 character can be 1 - 4 bytes long."
            
        algo:
            1. start processing the integers in the givven array one by one
            2. for every integer obtain the string format for its binary rep - we only need the least significant 8 bits
            3. There are two scenarios
                a. we are in hte middle of processing some UTF8 encoded char. in this case we simply need to check if the frist two bits of the string and see of they are 10 (i.e the 2 most signigicnat bits of the integer being 1 and 0)
                b. the other case is that we already processed some valid utf8 char and we have to start processing a new one
                in shi case we have to look at a prefix of thes tring rep and look at the numbers of 1' that we encounter
                this will tell us the ize of the next utf8 char
            4. we keep on prcessing the integers of the array in this way until we either end up processing all of them or we find an invalid scenario
        '''
        # Number of bytes in the current UTF-8 character
        n_bytes = 0

        # For each integer in the data array.
        for num in data:

            # Get the binary representation. We only need the least significant 8 bits
            # for any given number.
            bin_rep = format(num, '#010b')[-8:]

            # If this is the case then we are to start processing a new UTF-8 character.
            if n_bytes == 0:

                # Get the number of 1s in the beginning of the string.
                for bit in bin_rep:
                    if bit == '0': break
                    n_bytes += 1

                # 1 byte characters
                if n_bytes == 0:
                    continue

                # Invalid scenarios according to the rules of the problem.
                if n_bytes == 1 or n_bytes > 4:
                    return False
            else:
                # Else, we are processing integers which represent bytes which are a part of
                # a UTF-8 character. So, they must adhere to the pattern `10xxxxxx`.
                if not (bin_rep[0] == '1' and bin_rep[1] == '0'):
                    return False

            # We reduce the number of bytes to process by 1 after each integer.
            n_bytes -= 1

        # This is for the case where we might not have the complete data for
        # a particular UTF-8 character.
        return n_bytes == 0   
                
###############################
# 97. Interleaving String (REVISITED)
# 07JUL22
###############################
class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        '''
        we let dp(i,j) be the answer to the question
            is this interleaving after processing s1[:i] and s2[:j]
            so then if we are interleaving up this point, then the string must be interleaving if s1[i+1] or s2[j+1]
            is equal to the point in s3, which we call k
            
        dp(i,j,k ) = {
            first_match = s1[i] == s3[k] and dp(i+1,j,k+1)
            second_match = s2[j] == s3[k] and dp(i,j+1,k+1)
            
            the answer is if either first or second is true
        
        
        }
        
        base cases, when either i or j adanvances all the way, we are then left with an empty string, so compare the remidner of the string with the remainder of s3
        '''
        
        if len(s1) + len(s2) != len(s3):
            return False
        
        memo = {}
        
        def dp(i,j,k):
            if i == len(s1):
                return s2[j:] == s3[k:]
            if j == len(s2):
                return s1[i:] == s3[k:]
            if (i,j) in memo:
                return memo[(i,j)]
            first = s1[i] == s3[k] and dp(i+1,j,k+1)
            second = s2[j] == s3[k] and dp(i,j+1,k+1)
            
            ans = first or second
            memo[(i,j)] = ans
            return ans
        
        return dp(0,0,0)

class Solution:
    def isInterleave(self, s1: str, s2: str, s3: str) -> bool:
        '''
        we can translate this to bottom up, by starting from the ends of the strings, these were the base cases
        '''
        if len(s1) + len(s2) != len(s3):
            return False
        
        dp = [[0]*(len(s2)+1) for _ in range(len(s1) + 1)]
        for i in range(len(s1),-1,-1):
            for j in range(len(s2),-1,-1):
                if i == len(s1):
                    dp[i][j] = s2[j:] == s3[i+j:]
                elif j == len(s2):
                    dp[i][j] = s1[i:] == s3[i+j:]
                else:
                    first = s1[i] == s3[i+j] and dp[i+1][j]
                    second = s2[j] == s3[i+j] and dp[i][j+1]
                    dp[i][j] = first or second
        
        return dp[0][0]

#turns out this is BFS on a 2d grid
#for s1 and s2 to be an interleaving of s3, there needs to be a path from (0,0) to (len(s1),len(s2))
#in which case, our moves can only be one step right, or one step down
class Solution:
	def isInterleave(self, s1, s2, s3):
	    r, c, l= len(s1), len(s2), len(s3)
	    if r+c != l:
	        return False
	    queue, visited = [(0, 0)], set((0, 0))
	    while queue:
	        x, y = queue.pop(0)
	        if x+y == l:
	            return True
	        if x+1 <= r and s1[x] == s3[x+y] and (x+1, y) not in visited:
	            queue.append((x+1, y)); visited.add((x+1, y))
	        if y+1 <= c and s2[y] == s3[x+y] and (x, y+1) not in visited:
	            queue.append((x, y+1)); visited.add((x, y+1))
	    return False

###############################################
# 30. Substring with Concatenation of All Words
# 07JUL22
###############################################
#hash map
class Solution:
    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        '''
        we want to return all the starting indices in s that is a concatenation of each word in words only once
        brute force way would be to generate all possible concatenations
        then check for possible concat for it existens in s and mark their indices
        
        intuition:  
            all words have the same length, so a valid substring in length s would be of length words[0]*len*(wrods)
            so we can check s in this range
            becaue words can have duplicates, we can use hash table for counting them up
            also define a helper function that takes an index and returns if a valid string substring starting at this index esists
            
        '''
        N = len(s)
        k = len(words)
        word_length = len(words[0])
        substring_size = word_length*k
        word_count = Counter(words)
        
        #check for a current valid subtring at this index
        def check(i):
            #make a copy of the hashmap each time
            remaining = word_count.copy()
            words_used = 0
            
            #check i range up to length of substrin size, but insteps of word
            for j in range(i,i+substring_size,word_length):
                #get current part
                sub = s[j:j+word_length]
                #take away from hasmap
                if remaining[sub] > 0:
                    remaining[sub] -= 1
                    words_used += 1
                else:
                    break
            
            #check valid substring from this index
            return words_used == k
        
        res = []
        for i in range(N - substring_size +1):
            if check(i):
                res.append(i)
        
        return res

################################
# 1473. Paint House III
# 08JUL22
################################
#top down memo
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        '''
        we are given a houses array, where houses[i] represents a color of the house
        we can only choose from colors [1,n]
        cost[i][j] is the cost to paint house i color j+1
        we also have m houses
        costs is m x n matrix (the cols are 1 indexed, really)
        
        intuition:
            some houses that have been painted last summer should not be painted again
            meaning, we can only paint the houses such that house[i] == 0
        
        we can change the problem, paint the houses at houses[i] == 0, such that we get target neighborhoods with minimum cost, this is an easier problem...
        
        hint:
            Define dp[i][j][k] as the minimum cost where we have k neighborhoods in the first i houses and the i-th house is painted with the color j.
        
        intuition 2:
            starting with the firt house, is it already painted, we skip
            otherwise chose any color from [1,n] and spend that cost
            we also need to count the number of neihborhoods, if after visiting a house, we compare it with the previous color, and if its the same, we don't incremnet the neighborhood count
            and recursively go in to the next house with update values
            after traversing all the hosues, if the neighborhoods == target, then we can compare the cost wit the minmium cost we have acheived so far
            
        track of:
            we need the current index for the house
            the color of the previous house
            the current number of neighborhoods
        
        algo:
            1. curr index is 0, count of neighborhoods is zero, prevhouse color starts at 0, ensures that we pick the first house
            2. if house at curr index is already painted (not a zero), we recursively move on to the next house, while updating neighborhood count
            3.if the house is not painted (is 0), iterate over the colors from 1 to n
                for each color try painting the house at this currindex that color, and recursviely move on the next house with the update valies
                stor the minimum after tyring all valies from 1 to n
            4. return the min cost and cache it
            5. base cases:
                if we have gone over all the houses and count == target, return 0
                otherwise, return the max cost
                if the count of neighborhoods > target, the answer is not possible so return max cost
        '''
        memo = {}

        
        def dp(curr_house, neighCount,prev_color):
            if curr_house == m:
                if neighCount == target:
                    return 0
                else:
                    return float('inf')
            
            #too many neighborhoods
            if neighCount > target:
                return float('inf')
            
            if (curr_house,neighCount,prev_color) in memo:
                return memo[(curr_house,neighCount,prev_color)]
            
            min_cost = float('inf')
            #already painted
            if houses[curr_house] != 0:
                #update values
                min_cost = dp(curr_house + 1, neighCount + (houses[curr_house] != prev_color), houses[curr_house])
            else:
                #minimize
                for color in range(1,n+1):
                    curr_cost = cost[curr_house][color-1]
                    curr_cost += dp(curr_house+1,neighCount + (color != prev_color),color)
                    #minimuze the local answer here, not outside the loop
                    min_cost = min(min_cost,curr_cost)
            
            #cache
            memo[(curr_house,neighCount,prev_color)] = min_cost
            return min_cost
        
        ans = dp(0,0,0)
        if ans == float('inf'):
            return -1
        else:
            return ans

#close one for bottom up...
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        '''
        we can translate this to bottom up dp starting from the end
        '''
        #reverse nesting from the recursino memoe
        #index into dp(curr_house,neighCount,prev_color)
        dp = [[[0]*(n+1) for _ in range(target+1)] for _ in range(m+1)]
        
        #for curr house go down to 1
        for curr_house in range(m,-1,-1):
            for neighCount in range(target,-1,-1):
                for prev_color in range(1,n+1):
                    #check for indexable
                    #print(dp[curr_house][neighCount][prev_color])
                    
                    #base cases, out of bounds and have targets or not
                    if curr_house == m:
                        if neighCount == target:
                            dp[curr_house][neighCount][prev_color] = 0
                        else:
                            dp[curr_house][neighCount][prev_color] = float('inf')
                    
                    #too many
                    if neighCount > target:
                        dp[curr_house][neighCount][prev_color] = float('inf')
                    
                    min_cost = float('inf')
                    #we need to index into the houses array
                    if houses[curr_house-1] != 0:
                        min_cost = dp[curr_house-1][neighCount + (houses[curr_house - 1] != prev_color)][houses[curr_house-1]]
                    else:
                        for color in range(1,n+1):
                            curr_cost = cost[curr_house-1][color-1]
                            curr_cost += dp[curr_house-1][neighCount + (color != prev_color)-1][color]
                            min_cost = min(min_cost,curr_cost)
                    dp[curr_house][neighCount][prev_color] = min_cost
        
        print(dp)

#actual bottom up,but reframe the subproblems
class Solution:
    def minCost(self, houses: List[int], cost: List[List[int]], m: int, n: int, target: int) -> int:
        '''
        for bottom up, we reframe the intuition just slighlty
        intuition:
            suppose we want to find the minimum cost to paint the first house houses with neigh_count neighborhods
            
        if the house at index house is already paintted, and the color of the house != color:
            do nothing, since it is already painted
        if the house at index house is not painted yet, or it is painted the same color color
            if the house is not pained, paint house with thie color
            if it is alredy painted, the min cost here is 0, since we are trying to minimize the cost for all colors
                if the color and prevcolor are not the same then the subproblem becomes (house -1, neigh count -1, prevcolor)
                if color and prevcolor are the same, then the subproblem will be (house -1, neigh count,color)
        
        the only case where we don't need to break the problem into subproblems (base cases) is when we only have the frist house
        for the first house
            the value of the neigh count will be 1 and we can assign the cost for each color from 1 to n according to the cost to paind the house ad index 0 for that color
        
        once we have found the cost for all possible combinations of houses,colors,neighcount, we can find the min cost to pain all the house with target number
        this will be the minimum answer in the array dp[m-1][target]
        
        algo:
            1. init base cases for house = 0, neighborhoods = 1. iterate over the colors from 1 to n and assing the corresponding cost if the hosue itnot painted, otherwise it if it is painted, no cost incurred
            2. iterover over hosue inde from hosue [1,m-1] and neighborhood counts from [1,min(house+1),target]
                if the house at idnex house is alredy painted and color is the same as color, we can continue
                init cost for current parames to maxcost
                iterate over color options for prev hosue color from [1,n] for each prevColor
                
                    if color and prev color are differnt,
                        min(currcost, dp[house-1][neigh-1][prevcolor-1]) to curr
                    if color and prev mathc
                        curr = min(curr, dp[house-1][neigh][color-1])
                assign cost to paind the curr house with the ans
            3. find the min answer at the last house with target number of neighboroods
            4. return min
        '''
        #of the form house, neighborhoods, and color
        dp = [[[0]*n for _ in range(target+1)] for _ in range(m)]
        #base cases
        for i in range(m):
            for j in range(target+1):
                dp[i][j][:] = [float('inf')]*n
        
        #initialize for house 0, neighborhoods will be 1
        for color in range(1,n+1):
            #no cost if same color
            if houses[0] == color:
                dp[0][1][color-1] = 0
            #not yet painted, assing cost with this color
            elif houses[0] == 0:
                dp[0][1][color-1] = cost[0][color-1]
        
        #traverse houses starting with the second1
        for house in range(1,m):
            #we are limited to a certain number of neighborhoods if using the hosues up to this house
            for neighborhoods in range(1,min(target,house+1)+1):
                #traverse through colors
                for color in range(1,n+1):
                    #house is already painted and different color
                    if houses[house] != 0 and color != houses[house]:
                        continue
                    #minmize along subroblems
                    currCost = float('inf')
                    for prevColor in range(1,n+1):
                        #new neighboord, because different color
                        if prevColor != color:
                            currCost = min(currCost,dp[house-1][neighborhoods-1][prevColor-1])
                        else:
                            currCost = min(currCost,dp[house-1][neighborhoods][color-1])
                    #if the house is already painted, cost has to be zero
                    if houses[house] != 0:
                        #take the cost to paint this hosue that color
                        costToPaint = 0
                    else:
                        costToPaint = cost[house][color-1]
                    
                    #cache
                    dp[house][neighborhoods][color-1] = costToPaint + currCost
                    
        ans = float('inf')
        #find min cost with m houses and target neighbordshoods by looking at the dp array for that problem taking the min
        for color in range(1,n+1):
            ans = min(ans, dp[m-1][target][color-1])
        
        return ans if ans != float('inf') else -1

##############################
# 1696. Jump Game VI (REVISITED)
# 09JUL22
##############################
#close one!!!
class Solution:
    def maxResult(self, nums: List[int], k: int) -> int:
        '''
        we let dp(i) be the max score i can get after jumping through nums[:i]
        then i want the answer for dp(len(nums)-1)
        then dp(i) = nums[i] + max{
                #for all previous jumps before, i could have landed on i from a previous step j#
                for j in range(i-k,i-1):
                    dp(j)
                    
                    
        }
        base case i < 0, return no points
        '''
        memo = {}
        
        def dp(i):
            if i < 0:
                return 0
            if i in memo:
                return memo[i]
            ans = float('-inf')
            for j in range(i-k,i):
                ans = max(ans,dp(j))
            ans += nums[i]
            memo[i] = ans
            return ans
        
        return dp(len(nums)-1)

class Solution:
    def maxResult(self, nums: List[int], k: int) -> int:
        '''
        the problem with dp is the i take O(Nk) for find the maximum for an answer to dp(i)
        we can keep a scores array and the recurrence is
        score[i] = max(score[i-k], ..., score[i-1]) + nums[i]
        
        base case where scores[0] = nums[0]
        
        we now want to find the maximum for the previous k values from i
        this is similar to Sliding window maximum, typical monotonic queue problem
        
        we store onto a q, (of size k), the possible values, and maintain property do be monotnically decreasing
        the largest k values will be at the left, the smaller ones at the right
        and we exchange the smaler values for even larger values
        
        '''
        N = len(nums)
        scores = [0]*N #scores[i] represents the max scoreing getting here, starting at 0
        scores[0] = nums[0]
        
        deq = deque([0])
        
        for i in range(1,N):
            #ensure k contains allwable previous max jump scores
            while deq and deq[0] < i - k:
                deq.popleft()
            #current max so far
            scores[i] = nums[i] + scores[deq[0]]
            #maintain property the deq is striclty decreasing
            while deq and scores[deq[-1]] <= scores[i]:
                deq.pop()
            deq.append(i)
        
        return scores[-1]

#pq
class Solution:
    def maxResult(self, nums: List[int], k: int) -> int:
        '''
        we can also use q max heap to find the max previous scores so far
        
        '''
        N = len(nums)
        scores = [0]*N #scores[i] represents the max scoreing getting here, starting at 0
        scores[0] = nums[0]
        max_heap = []
        #we entries on the heap are (-nums[i],i)
        heapq.heappush(max_heap,(-nums[0],0))
        
        for i in range(1,N):
            #maksure max score in heap is withink i-k
            while max_heap[0][1] < i - k:
                heapq.heappop(max_heap)
            scores[i] = nums[i] + scores[max_heap[0][1]]
            heapq.heappush(max_heap,(-scores[i],i))
        
        return scores[-1]

#using deque but compressing states to O(k)
class Solution:
    def maxResult(self, nums: List[int], k: int) -> int:
        '''
        we can compress the states
        notice that we do not need all the values from scores[0] to scores[i-1]
        we only need to loks at values from scores[i-k] to scores[i-1]
        
        in the deq array, push the current score as well as the index, then were are in O(k) constant space
        
        '''
        N = len(nums)
        score = nums[0]
        
        deq = deque([(0,score)])
        
        for i in range(1,N):
            #ensure k contains allwable previous max jump scores
            while deq and deq[0][0] < i - k:
                deq.popleft()
            #current max so far
            score = nums[i] + deq[0][1]
            #maintain property the deq is striclty decreasing
            while deq and deq[-1][1] <= score:
                deq.pop()
            deq.append((i,score))
        
        return score

#same thing with max_heap
class Solution:
    def maxResult(self, nums: List[int], k: int) -> int:
        n = len(nums)
        score = nums[0]
        priority_queue = []
        # since heapq is a min-heap,
        # we use negative of the numbers to mimic a max-heap
        heapq.heappush(priority_queue, (-nums[0], 0))
        for i in range(1, n):
            # pop the old index
            while priority_queue[0][1] < i-k:
                heapq.heappop(priority_queue)
            score = nums[i]-priority_queue[0][0]
            heapq.heappush(priority_queue, (-score, i))
        return score

#using segment tree
class Solution:
    def maxResult(self, nums: List[int], k: int) -> int:
        # implement Segment Tree
        def update(index, value, tree, n):
            index += n
            tree[index] = value
            while index > 1:
                index >>= 1
                tree[index] = max(tree[index << 1], tree[(index << 1)+1])

        def query(left, right, tree, n):
            result = -inf
            left += n
            right += n
            while left < right:
                if left & 1:
                    result = max(result, tree[left])
                    left += 1
                left >>= 1
                if right & 1:
                    right -= 1
                    result = max(result, tree[right])
                right >>= 1
            return result

        n = len(nums)
        tree = [0]*(2*n)
        update(0, nums[0], tree, n)
        for i in range(1, n):
            maxi = query(max(0, i-k), i, tree, n)
            update(i, maxi+nums[i], tree, n)
        return tree[-1]

##############################################
# 510. Inorder Successor in BST II (Revisited)
# 11JUL22
##############################################
"""
# Definition for a Node.
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None
"""

class Solution:
    def inorderSuccessor(self, node: 'Node') -> 'Optional[Node]':
        '''
        the kicker is we are not given the whole tree
        but we are given access to a parent node
        to find the inorder successor node, we go up to the parent
        
        if the right subtree of node is not null, then in order succ lies in this right subtree, go right and return its mins
        if there is no right subtree, then inorder succ is an ancestor:
            travel up using the parent until we get to a node which is a left child of its parent
        '''
        # the successor is somewhere lower in the right subtree
        if node.right:
            node = node.right
            while node.left:
                node = node.left
            return node
        
        # the successor is somewhere upper in the tree
        while node.parent and node == node.parent.right:
            node = node.parent
        return node.parent

###############################
# 390. Elimination Game
# 13JUL22
###############################
class Solution:
    def lastRemaining(self, n: int) -> int:
        '''
        rules
        alternating starting from left to righ at each step
        for the current step, remove the first number and every other number after that
        simulating is trivial....
        
        '''
        nums = list(range(1,n+1))
        step = 0
        while len(nums) > 1:
            print(nums)
            next_nums = []
            if step % 2 == 0:
                for i in range(1,len(nums),2):
                    next_nums.append(nums[i])
            else:
                for i in range(0,len(nums),2):
                    next_nums.append(nums[i])
            step += 1
            nums = next_nums
        
        return nums[0]

class Solution:
    def lastRemaining(self, n: int) -> int:
        '''
        math problem, turns out to be a variant of the josephus problem
        '''
        def helper(n, isLeft):
            if(n==1): return 1
            if(isLeft):
                return 2*helper(n//2, 0)
    # if started from left side the odd elements will be removed, the only remaining ones will the the even i.e.
    #       [1 2 3 4 5 6 7 8 9]==   [2 4 6 8]==     2*[1 2 3 4]
            elif(n%2==1):
                return 2*helper(n//2, 1)
    # same as left side the odd elements will be removed
            else:
                return 2*helper(n//2, 1) - 1
    # even elements will be removed and the only left ones will be [1 2 3 4 5 6 ]== [1 3 5]== 2*[1 2 3] - 1
            
        return helper(n, 1)

#####################################################
# 1182. Shortest Distance to Target Color (Revisited)
# 15JUL22
#####################################################
#using binary seach coded out
class Solution:
    def shortestDistanceColor(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        '''
        we can actually use binary seach, 
        first hash the colors, 1,2,3 to each of their indices
        then when given a query, since the array is icnreasing, return the the shortes distance
        '''
        mapp = defaultdict(list)
        for i,c in enumerate(colors): 
            mapp[c].append(i)
        
        results = []
        for target,color in queries:
            #no color to query
            if color not in mapp:
                results.append(-1)
                continue
                
            #check possible indices
            indices = mapp[color]
            left,right = 0,len(indices) - 1
            while left < right:
                mid = left + (right - left) // 2
                #too far
                if indices[mid] > target:
                    right = mid
                else:
                    left = mid + 1
            
            left_nearest = abs(indices[max(left-1,0)] - target)               
            right_nearest = abs(indices[min(left,len(indices)-1)] - target)
            results.append(min(left_nearest,right_nearest))
        
        return results

#using built in
class Solution:
    def shortestDistanceColor(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        '''
        we can actually use binary seach, 
        first hash the colors, 1,2,3 to each of their indices
        then when given a query, since the array is icnreasing, return the the shortes distance
        '''
        mapp = defaultdict(list)
        for i,c in enumerate(colors): 
            mapp[c].append(i)
        
        results = []
        for target,color in queries:
            #no color to query
            if color not in mapp:
                results.append(-1)
                
            #check possible indices
            indices = mapp[color]
            left,right = 0,len(indices) - 1
            while left < right:
                mid = left + (right - left) // 2
                #too far
                if indices[mid] > target:
                    right = mid
                else:
                    left = mid + 1
            
            left_nearest = abs(indices[max(left-1,0)] - target)               
            right_nearest = abs(indices[min(left,len(indices)-1)] - target)
            results.append(min(left_nearest,right_nearest))
        
        return results

#precomputing
class Solution:
    def shortestDistanceColor(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        '''
        we can pre compute, but there is another trick for trying to find a 'c' to the left or right of an 'i'
        we we two indices i,j that are both c, and i < j with no other c in
        then for every k in between i and j
            the shortest distance between k and c is on the left, k-i
            the shortest distance between k and c on the right is j-k
        
        two phases:
            * from left to right, and looking forwards to find the nearest target color in the left
            * from right to left and looking backwards to find the nearest target color in the right
        '''
        # initializations
        n = len(colors)
        rightmost = [0, 0, 0]
        leftmost = [n - 1, n - 1, n - 1]

        distance = [[-1] * n for _ in range(3)]

        # looking forward
        for i in range(n):
            color = colors[i] - 1
            for j in range(rightmost[color], i + 1):
                distance[color][j] = i - j
            rightmost[color] = i + 1

        # looking backward
        for i in range(n - 1, -1, -1):
            color = colors[i] - 1
            for j in range(leftmost[color], i - 1, -1):
                # if the we did not find a target color on its right
                # or we find out that a target color on its left is
                # closer to the one on its right
                if distance[color][j] == -1 or distance[color][j] > j - i:
                    distance[color][j] = j - i
            leftmost[color] = i - 1

        return [distance[color - 1][index] for index,color in queries]

#####################################
# 576. Out of Boundary Paths (REVISITED)
# 16JUL22
#####################################
#brute force recursion
class Solution:
    def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
        '''
        dfs keeping track if the current i,j we are on
        in this case we are allowed to go out of bounds, once we are out of bounds, increment global counter
        and return (we do not want to keep going)
        also don't forget to decrement move counter
        also we are allowed to revisit cells
        '''
        mod = 10**9 + 7
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        self.paths = 0
        
        def dfs(i,j,moves):
            #out of bounds
            if  i < 0 or i == m or j < 0 or j == n:
                self.paths += 1
                self.paths %= mod
                return
            #otherwise we would still be traveling and we dont want that
            if moves == 0:
                return
            for dx,dy in dirrs:
                dfs(i+dx,j+dy,moves-1)
        
        dfs(startRow,startColumn,maxMove)
        return self.paths

class Solution:
    def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
        '''
        dfs keeping track if the current i,j we are on
        in this case we are allowed to go out of bounds, once we are out of bounds, increment global counter
        and return (we do not want to keep going)
        also don't forget to decrement move counter
        also we are allowed to revisit cells
        '''
        mod = 10**9 + 7
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        def dfs(i,j,moves):
            #out of bounds
            if  i < 0 or i == m or j < 0 or j == n:
                return 1
            #otherwise we would still be traveling and we dont want that
            if moves == 0:
                return 0
            ans = 0
            for dx,dy in dirrs:
                ans += dfs(i+dx,j+dy,moves-1) % mod
                ans %= mod
            return ans
        
        return dfs(startRow,startColumn,maxMove)

class Solution:
    def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
        '''
        in the call tree we are repeatedly calling previous i,j,moves states
        so we can just cache and retrieve along the way
        '''
        mod = 10**9 + 7
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        memo = {}
        def dfs(i,j,moves):
            #out of bounds
            if  i < 0 or i == m or j < 0 or j == n:
                return 1
            #otherwise we would still be traveling and we dont want that
            if moves == 0:
                return 0
            if (i,j,moves) in memo:
                return memo[(i,j,moves)]
            ans = 0
            for dx,dy in dirrs:
                ans += dfs(i+dx,j+dy,moves-1) % mod
                ans %= mod
            memo[(i,j,moves)] = ans
            return ans
        
        return dfs(startRow,startColumn,maxMove)

#bottom up tabulatino
class Solution:
    def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
        '''
        we can translate this bottom up
        '''
        mod = 10**9 + 7
        dirrs = dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        #index into dp reverse nested
        dp = [[[0]*(maxMove+1) for _ in range(n+1)] for _ in range(m+1)]
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        #start with moves first
        for M in range(1,maxMove+1):
            #then rows
            for i in range(m):
                #then cols
                for j in range(n):
                    for dx,dy in dirrs:
                        #out of bounds
                        if  i + dx < 0 or i + dx == m or j + dy < 0 or j + dy == n:
                            dp[i][j][M] += 1
                        #otherwise grab from othe sub problems
                        else:
                            dp[i][j][M] += dp[i+dx][j+dy][M-1] % mod
        
        return dp[startRow][startColumn][maxMove] % mod

 #space saving O(MN)
 class Solution:
    def findPaths(self, m: int, n: int, maxMove: int, startRow: int, startColumn: int) -> int:
        '''
        if we look closer, we only every need the answer from the pervious M-1 move
        so we only need to keep states for current move and move - 1 (similar to other DP where we keep)
        
        neat trick, we can alter between the move states using the and operator with 1
        
        '''
        mod = 10**9 + 7
        dirrs = dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        #index into dp reverse nested
        dp = [[[0]*(2) for _ in range(n+1)] for _ in range(m+1)]
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        #start with moves first
        for M in range(1,maxMove+1):
            #then rows
            for i in range(m):
                #then cols
                for j in range(n):
                    for dx,dy in dirrs:
                        #out of bounds
                        if  i + dx < 0 or i + dx == m or j + dy < 0 or j + dy == n:
                            dp[i][j][M & 1] += 1
                        #otherwise grab from othe sub problems
                        else:
                            dp[i][j][M & 1] += dp[i+dx][j+dy][(M-1) & 1] % mod
        
        return dp[startRow][startColumn][maxMove & 1] % mod

#############################
# 17JUL22 (REVISITED)
# 629. K Inverse Pairs Array
#############################
#memo
class Solution:
    def kInversePairs(self, n: int, k: int) -> int:
        '''
        let dp(n,k) be the number of arragnments we can arrange [1..n] having v inverse pairs
        if we knew the answer at dp(n-1,k)
        then dp(n,k) = sum(dp(n-1,k-i) for i in range(min(k,n-1)))
        the transition is very hard to come up with, unfortnualye
        examine we have the array [1,2,4,3]
        n = 4, with k = 1
        from the array [1,2,3,4] we just moved 4 over and got 1 inversion, the number of times we move from right gives as the number of inversions
        we can add in a new number 5 to get
        [1,2,3,4,5]
        but we can make [1,2,3,5,4] 
        we shfted right 1 time, so we need to add this to the number of inversions for n-1
        so we can sum through all dp(n-1.k-i) where we examine i as the number of shifts to the right
        we need to found the upper limit in the sum somehow
        inution:
            if we knew the number of inverse pairs (say x) in some abritrary array b with n
            then we can add in the n+1 at position p from the right to get x + p = k 
            i.e at a solution of n-1, with counts up to k, count0,count1...countk
         we bound the summatino to the min(k,n-1) because i > k, and k - i < 0
         since no arrangement exists with negative number of inverse pairs
         
         to generate a new arrangment adding k-i new inverse pairs after adding the nth number we need to add this number at the ith positino from the right, and we are liminted to n-1 shifts
         
         
        '''
        mod = 10**9 + 7
        memo = {}
        def dp(n,k):
            if n == 0:
                return 0
            if k == 0:
                return 1
            if (n,k) in memo:
                return memo[(n,k)]
            ans = 0
            for i in range(min(k,n-1)+1):
                ans += dp(n-1,k-i)
                ans %= mod
            memo[(n,k)] = ans
            return ans
        
        return dp(n,k)

#translate
class Solution:
    def kInversePairs(self, n: int, k: int) -> int:
        mod = 10**9 + 7
        dp = [[0]*(k+1) for _ in range(n+1)]
        
        for i in range(1,n+1):
            for j in range(k+1):
                if j == 0:
                    dp[i][j] = 1
                else:
                    for p in range(min(j,i-1)):
                        dp[i][j] += dp[i-1][j-p]
                        dp[i][j] %= mod
        
        return dp[n][k]

#unforuntalye this TLE's because there is one small optimization we can do
#we repsent the (n,k) state at count(i,j) + sum_{k=0}^{k-1} dp[i][k]
#each state repersents the cumsum, than we can find the total number in constant time
#this is because we filling up the dp array by adding
class Solution:
    def kInversePairs(self, n: int, k: int) -> int:
        '''
        dp(i,j) = count(i,j) + sum_{k=0}^{j-1}
        count(i,j) refers to the number of arrangments with i elements and j inversinos
        to obtain elements from dp[i-1][j-i+1] to dp[i-1][j], we can just grab
        dp[i-1][j] - dp[i-1][j-i]
        
        Now, to reflect the condition \text{min}(j, i-1)min(j,i−1) used in the previous approaches, we can note that, we need to take the sum of only ii elements in the previous row, if ii elements exist till we reach the end of the array while traversing backwards.
        
        
        '''
        memo = {}
        mod = 10**9 + 7
        
        def inversions(n,k):
            if n== 0:
                return 0
            if k == 0:
                return 1
            if (n,k) in memo:
                return memo[(n,k)]
            if k - n >= 0:
                val = (inversions(n-1,k) + mod - inversions(n-1,k-n)) % mod
            else:
                val = (inversions(n-1,k) + mod) % mod
            memo[(n,k)] = (inversions(n,k-1) + val) % mod
            return memo[(n,k)]
        
        if k > 0:
            return (inversions(n,k) + mod - inversions(n,k-1)) % mod
        else:
            return (inversions(n,k) + mod) % mod

        
##############################
# 396. Rotate Function
# 16JUL22
##############################
class Solution:
    def maxRotateFunction(self, nums: List[int]) -> int:
        '''
        simulation would involve applying the rotation for all 0 to n-1 rotations
        then take the max...
        
        example
        [4,5,3,2,9]
        
        listing out the rotation functions
        F(0) = 4 * 0 + 5 * 1 + 3 * 2 + 2 * 3 + 9 * 4 = 53
        F(1) = 9 * 0 + 4 * 1 + 5 * 2 + 3 * 3 + 2 * 4 = 31
        F(2) = 2 * 0 + 9 * 1 + 4 * 2 + 5 * 3 + 3 * 4 = 44
        F(3) = 3 * 0 + 2 * 1 + 9 * 2 + 4 * 3 + 5 * 4 = 52
        F(4) = 5 * 0 + 3 * 1 + 2 * 2 + 9 * 3 + 4 * 4 = 50
        
        there is a trick to go from F(0) to F(1)
        intution:
            all the values except the last one (in this case the last 9) is mulitplies by 1 number +1 higher than the previous 
        
        F(0) = 4 * 0 + 5 * 1 + 3 * 2 + 2 * 3 + 9 * 4 = 53
        F(1) = 4 * 1 + 5 * 2 + 3 * 3 + 2 * 4 + 9 * 0 = 31
        F(2) = 4 * 2 + 5 * 3 + 3 * 4 + 2 * 0 + 9 * 1 = 44
        
        algo:
            1. calculate the sum of the entire array
            2. first calculare F(0) = \sum_{i=0}^{len(nums)} i*nums[i], call this sum_of_prods
            3. to get F(1) from F(0)
                perform sum_of_prods + arr_sum (equivalent o increment the multiplier of all values in the array)
                now from the patial sum, substract the last_val * n 
                the last element in F(0) is supposed to be multiplied by 0
                however in the previous step, we essentially increaed the multipliers of all the array values, so partial sum relect the last element*n
            4.reapt start from the last element in the array to thes eocnd elment in the array
        '''
        arr_sum = sum(nums)
        sum_of_prods = 0
        for i,num in enumerate(nums):
            sum_of_prods += i*num
            
        N = len(nums)
        max_val = sum_of_prods
        for i in range(1,N):
            #increment the sum_of_prods, adding in the arr_sum
            sum_of_prods += arr_sum
            #take off the last num
            sum_of_prods -= nums[N-i]*N
            max_val = max(max_val,sum_of_prods)
        
        return max_val

#also think in dot products
#<nums> dot  <0...N-1>
#numss stays constant but the indices rotate\
#there exists a recurrence to get the rotaion function
class Solution:
    def maxRotateFunction(self, nums: List[int]) -> int:
        '''
        F(k)    = 0 * Bk[0] + 1 * Bk[1] + ... + (n-1) * Bk[n-1]
        F(k-1)  = 0 * Bk-1[0] + 1 * Bk-1[1] + ... + (n-1) * Bk-1[n-1]
                = 0 * Bk[1] + 1 * Bk[2] + ... + (n-2) * Bk[n-1] + (n-1) * Bk[0]
                
        F(k) - F(k-1) = Bk[1] + Bk[2] + ... + Bk[n-1] + (1-n)Bk[0]
              = (Bk[0] + ... + Bk[n-1]) - nBk[0]
              = sum - nBk[0]
              
        F(k) = F(k-1) + sum - nBk[0]
        
        k = 0; B[0] = A[0];
        k = 1; B[0] = A[len-1];
        k = 2; B[0] = A[len-2];
        '''
        memo = {}
        arr_sum = sum(nums)
        sum_of_prods = 0
        
        for i,num in enumerate(nums):
            sum_of_prods += i*num
        N = len(nums)
        
        def rec(i):
            if i == 0:
                return sum_of_prods
            if i in memo:
                return memo[i]
            ans = rec(i-1) + arr_sum - nums[N-i]*N
            memo[i] = ans
            return ans
        
        res = sum_of_prods
        for i in range(N):
            res = max(res,rec(i))
        
        return res

#################################
# 674. Longest Continuous Increasing Subsequence
# 18JUL22
#################################
class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        '''
        its really asking longest strictly increasing substring
        typical linear scan counting the streaks taking the max along the way
        
        '''
        max_inc = 0
        curr_inc = 1
        N = len(nums)
        for i in range(N-1):
            if nums[i] < nums[i+1]:
                curr_inc += 1
            else:
                max_inc = max(max_inc,curr_inc)
                curr_inc = 1
        
        return max(max_inc,curr_inc)

##########################################
# 397. Integer Replacement
# 19JUL22
##########################################
class Solution:
    def integerReplacement(self, n: int) -> int:
        '''
        noticing a pattern with more of these math questions, usually the intutino would be to get simiulate
        even we have no choice, for odds, we need to bring it to a power of 2 by incrementing or decremeneting
        i can use recursion
        let dp(i) be the min number of moves getting to n == 1
        '''
        memo = {}
        def dp(i):
            if i == 1:
                return 0
            if i in memo:
                return memo[i]
            if i % 2 == 0:
                ans = 1 + dp(i//2)
                memo[i] = ans
                return ans
            else:
                ans = 1 + min(dp(i-1),dp(i+1))
                memo[i] = ans
                return ans
        
        return dp(n)

##############################
# 473. Matchsticks to Square (REVISITED)
# 12JUL22
###############################
#backtracking solution
class Solution(object):
    def makesquare(self, matchsticks):
        """
        :type matchsticks: List[int]
        :rtype: bool
        """
        '''
        if i can make a square, then the permiter must be 4*len_of_side
        so in order to make a square, the sum must be divisible by 4
        we can use backtracking and try adding matchsticks to each side
        then once we use up all match sticks, check all sides are the same
        
        we also need to reveser sort, examine the case [8,4,4,4]
        '''
        if not matchsticks:
            return False
        
        N = len(matchsticks)
        perim = sum(matchsticks)
        
        #check possible side
        side = perim // 4
        
        #check
        if side*4 != perim:
            return False
        
        matchsticks.sort(reverse=True)
        
        sides = [0]*4
        
        def backtrack(curr_match):
            if curr_match == N and all([sides[i] == sides[3] for i in range(3)]):
                return True
            #try adding all matchstricks to each side
            for i in range(4):
                if sides[i] + matchsticks[curr_match] <= side:
                    #add it
                    sides[i] += matchsticks[curr_match]
                    #check if we can go on
                    if backtrack(curr_match + 1):
                        return True
                    #backtrack
                    sides[i] -= matchsticks[curr_match]
            
            return False
        
        return backtrack(0)
        

class Solution:
    def makesquare(self, matchsticks: List[int]) -> bool:
        '''
        good review on how to take recursive states and turn them into subproblems
        imagine we have the numbers 3,3,4,4,5,5
        if we set the side to length 8, then we can pair in different ways
        (4, 4), (3, 5), (3, 5) -----------> 3 sides fully constructed.
        (3, 4), (3, 5), (4), (5) ---------> 0 sides completely constructed.
        (3, 3), (4, 4), (5), (5) ---------> 1 side completely constructed.
        
        knowing what index of match sticks we use gives us many different states of complete sides -> not a good way to break into subproblems
        keeping track of what matching sticks remain, won't define the state
        we need to keep track of what matchsticks are available, and how sides have been compelted so far
        not we favor the state that leads to completing the most numbr of sides
        
        encode recursion state as matchsticks_used and sides_formed
        we can use a bit mask to store what match stick we have used -> would use an array but we want an effecient way to store the state
        also we cannot cache array types
        
        N = 15 matchsticks, 2**15 size
        
        also, we don't need to make 4 sides, if we get to the third side, then we just use up the remaning matchstikcs to see if their sum is equal to one of sides we already made
        
        note, this is a variant of the bin packing problem
        
        '''
        #no matches
        if not matchsticks:
            return False
        #globals
        L = len(matchsticks)
        perim = sum(matchsticks)
        possible_side = perim // 4
        
        #if we can't make from the beginning
        if possible_side*4 != perim:
            return False
        
        memo = {}
        
        #matches is a bit mask
        #1 means not taken, 0 means taken
        def dp(matches_taken,sides_done):
            #get sum of matchsticks for this mask up until now
            total = 0
            for i in range(L):
                #check bit position
                if not (matches_taken & (1 << i)):
                    #index back into matchsticks
                    total += matchsticks[i]
            #if some matchsticks have been used and sum is divisible by side, then we have completed a side
            if total > 0 and total % possible_side == 0:
                sides_done += 1
            #we only need to finish 3 sides
            if sides_done == 3:
                return True
            #retreive
            if (matches_taken,sides_done) in memo:
                return memo[(matches_taken,sides_done)]
            #store curr ans for this state
            ans = False
            #get available space in current side
            c = int(total/possible_side)
            rem = possible_side*(c+1) - total
            #pass over matchsticks
            for i in range(L):
                #if we can fit this matchstcik and it hasn't been taken
                if matchsticks[i] <= rem and matches_taken & (1 << i):
                    #if the recursino after considering this matchstick gives True, then we don't need to decsend any further
                    #set the ith bit, i.e take the match
                    if dp(matches_taken ^ (1 << i),sides_done):
                        ans = True
                        break
            #cache
            memo[(matches_taken,sides_done)] = ans
            return ans
        
        return dp((1 << L) - 1,0)
                
################################
# 400. Nth Digit
# 19JUL22
################################
class Solution:
    def findNthDigit(self, n: int) -> int:
        '''
        this is the same as concating all the digits in the inifinte sequence and return the nth digit
        need a way to express the number of digits in constant time (near constant)
        1,2,3...9, each have one digit
        10,11...99, each have two digits
        100,101..999 each have three digits
        
        we can keep going all the way the last digit that can fit into 2**32-1
        the first element in each of these groups is 1,10,100....., which can be written in the form 10**(digits-1) where digits is in the range[1,11]
        11 is big enough to include the 2**32 - 1 digit
        in each group there are 9,90,900,9000.....elemenets
        and the total number of digits in each groups are 1*9,2*90,3*900
        
        there are three properties for a digits group
            1. first = the element
            2. 9*first = the size of the group (the number of the elements)
            3. 9*first*digits, the number of digits in the group
            
        in our loop, we first check if n <= 9 (the size of the current group)
        if not then decrement by the number of digits
            i.e n = n-9 {from 1...9}, then test of the new n <= 2*90
            
        1. n means our return value is in the n-th digits group, here the order starts from 0
        2. n/digits = the digits that comes from the n/digits smalelst number in the group
        3. sinde theg roup starts with "first", first + n/digits = the number where the return value is one of the digits
        4. the digiet we want is the n%digits in this current number, starting from left
        5. so the answer is str(fist + n/digits)[n%digits]
        '''
        n -= 1 #zero indexing
        for group in range(1,11):
            #get the number of eleemnts in each grouping
            first_element_in_group = 10**(group-1)
            #store number of elements in group
            num_elements_in_group = 9*first_element_in_group
            #get number of digits in group
            num_digits_in_group = num_elements_in_group*group
            #if we are in this current nth digit group
            if n < num_digits_in_group:
                #find the element in this group
                element = first_element_in_group + n // group
                #get the digit from this element
                digit = str(element)[n % group]
                return digit
            #decrement by the number of digits to go to another group
            n -= num_digits_in_group

#just another way
class Solution(object):
    def findNthDigit(self, n):
        start, size, step = 1, 1, 9
        while n > size * step:
            n, size, step, start = n - (size * step), size + 1, step * 10, start * 10
        return int(str(start + (n - 1) // size)[(n - 1) % size])
                
######################################
# 792. Number of Matching Subsequences (REVISITED)
# 20JUL22
######################################
#brute force TLE
class Solution:
    def numMatchingSubseq(self, s: str, words: List[str]) -> int:
        '''
        brute force is chekc each word in words
        '''
        def isSub(t):
            if len(t) > len(s):
                return False
            i = 0
            j = 0
            while i < len(s) and j < len(t):
                if s[i] == t[j]:
                    j += 1
                i += 1
            return j == len(t)
        
        count = 0
        for w in words:
            if isSub(w):
                count += 1
        
        return count

#hashing iteraables 
#O(len(s) + total sum of all word lenghts in words)
class Solution:
    def numMatchingSubseq(self, s: str, words: List[str]) -> int:
        '''
        hash words by starting char and check letters as we advance along the string
        good review in turning string into iterable and check if we can still next on it
        '''
        count = 0
        heads = defaultdict(list)
        
        for word in words:
            it = iter(word)
            #key is next letter to be proccessed : next char after that word
            heads[next(it)].append(it)
        
        
        for ch in s:
            #get the current list of words waiting to be processd
            curr_words = heads[ch]
            #clear them
            heads[ch] = []
            
            #process these words
            while curr_words:
                #get the next words
                curr_word = curr_words.pop()
                #get next char
                next_char = next(curr_word,None)
                #still parts to process, add them back into map
                if next_char:
                    heads[next_char].append(curr_word)
                else:
                    count += 1
        
        return count

#there is also a binar search solution
class Solution:
    def numMatchingSubseq(self, s: str, words: List[str]) -> int:
        '''
        we can use binary search 
        precompute all places for each char in s, i.e mapp to their indices
        then we can use binary search to find the next char (we do this at most sum len(words) times)
        '''
        char_to_idxs = defaultdict(list)
        for i,char in enumerate(s):
            char_to_idxs[char].append(i)
        

        def isSub(word):
            curr = 0
            for ch in word:
                #find the the next largest index of of this char in s
                idxs = char_to_idxs[ch]
                left = 0
                right = len(idxs)
                #find nearest
                while left < right:
                    mid = left + (right - left) // 2
                    if idxs[mid] < curr:
                        left = mid + 1
                    else:
                        right = mid
                if left >= len(idxs):
                    return False
                curr = idxs[left] + 1
            
            return True
        
        count = 0
        for w in words:
            if isSub(w):
                count += 1
        
        return count

############################################
# 693. Binary Number with Alternating Bits
# 21JUL22
#############################################
class Solution:
    def hasAlternatingBits(self, n: int) -> bool:
        '''
        just reduce and see if they alternate
        '''
        prev_bit = n & 1
        #reduce
        n //= 2
        while n:
            curr_bit = n & 1
            if curr_bit == prev_bit:
                return False
            else:
                prev_bit = curr_bit
                n //= 2
        
        return True

#############################
# 92. Reverse Linked List II (Revisited)
# 21JUN22
##############################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        '''
        we can do this iteratively, make a new dummy Linked list and keep adding the nodes up to left
        then reverse the nodes between left and right
        then reconnect them
        '''
        if left == right:
            return head
        
        dummy = ListNode(0)
        dummy.next = head
        #always stay 1 back from the actual current node
        pre = dummy
        
        for i in range(left-1):
            pre = pre.next
            
        #reverse
        curr = pre.next
        nxt = curr.next
        
        #reverse everything in between
        for i in range(right-left):
            #temporaliy store the next
            temp = nxt.next
            #reverse direction
            nxt.next = curr
            curr = nxt
            nxt = temp
        
        #adjust pointers
        pre.next.next = nxt
        pre.next = curr
        return dummy.next

#to avoid swapping at the end, move both prev and curr pointers
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        dummy = ListNode(0)
        dummy.next = head
        
        prev = dummy
        curr = head
        
        for _ in range(left-1):
            prev = prev.next
            curr = curr.next
        
        for _ in range(right-left):
            temp = curr.next
            curr.next = temp.next
            temp.next = prev.next
            prev.next = temp
        
        return dummy.next

#################################
# 315. Count of Smaller Numbers After Self (REVISITED)
# 23JUL22
#################################
#building segment tree recursively
class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        '''
        lets try implementing segment tree using this article
        https://leetcode.com/articles/a-recursive-approach-to-segment-trees-range-sum-queries-lazy-propagation/
        the idea is to get the counts of numbers greater than nums[i] laying to the right of nums[i]
        in the article, they originally build out the whole segment tree using the sum query along an interval
        we can't just build out the whole tree from the start, because then we'd have know way of knowing what the counts to the rights r (if we we started off just putting them in buckets)
        so we need to update the tree and query on the fly
        node representation:
            each node in the tree represents the number of elemenets greater then nums[i]
            since we build the tree from right to left, this qives us our query answer
            so we can degenerate the node represenation into:
                number of elements greater then this value
        
        we query first the tree
        then update the tree
            we update going right to left
            
        '''
        offset = 10**4
        size = 2*offset + 1
        tree = [0]*(4*size)
        counts = []
        
        def querySegTree(treeIdx,lo,hi,i,j):
            #query is of the form, for this tree index, get counts from i to j, where i would be -10**4 to nums[i] - 1
            if lo > j or hi < i:
                return 0
            if i <= lo and j >= hi:
                return tree[treeIdx]
            mid = lo + (hi - lo) // 2
            if i > mid:
                return querySegTree(2*treeIdx + 2,mid+1,hi,i,j)
            else:
                return querySegTree(2*treeIdx + 1, lo,mid,i,j)
            
            left = querySegTree(2*treeIdx +1,lo,mid,i,mid)
            right = querySegTree(2*treeIdx + 2,mid+1,hi,mid+1,j)
            
            return left + right
        
        def updateTree(treeIdx,lo,hi,arrIdx,val):
            #update this nums count to 1
            if lo == hi:
                #print(treeIdx)
                tree[treeIdx] = val
                return
            mid = lo + (hi - lo) // 2
            if arrIdx > mid:
                updateTree(2*treeIdx + 2,mid+1,hi,arrIdx,val)
            elif arrIdx <= mid:
                updateTree(2*treeIdx + 1,lo,mid,arrIdx,val)
            #merge updates
            tree[treeIdx] = tree[2*treeIdx + 1] + tree[2*treeIdx + 2]
            
        #traverse nums right to left querying and updating
        for num in reversed(nums):
            count_smaller = querySegTree(0,0,size,0,num+offset-1)
            print(count_smaller)
            counts.append(count_smaller)
            updateTree(0,0,size,num+offset,1)
        
        return counts

#using fenwick tree, bit trick
class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        # implement Binary Index Tree
        def update(index, value, tree, size):
            index += 1  # index in BIT is 1 more than the original index
            while index < size:
                tree[index] += value
                index += index & -index

        def query(index, tree):
            # return sum of [0, index)
            result = 0
            while index >= 1:
                result += tree[index]
                index -= index & -index
            return result

        offset = 10**4  # offset negative to non-negative
        size = 2 * 10**4 + 2  # total possible values in nums plus one dummy
        tree = [0] * size
        result = []
        for num in reversed(nums):
            smaller_count = query(num + offset, tree)
            result.append(smaller_count)
            update(num + offset, 1, tree, size)
        return reversed(result)

#merge sort
class Solution:
    def countSmaller(self, nums: List[int]) -> List[int]:
        '''
        we can also use merge sort
        during the sorting process, the smaller elements to the right of a number will jump from its right to its left during the sorting process
        so if we can record the number of those elements during sorting, then the prbolem is solved
        
        algo:
            implement merger sort functions
            for each element i, the function needs to record the number of elements jumping from i's right to i's left during mersort
            
        '''
        n = len(nums)
        arr = [[v, i] for i, v in enumerate(nums)]  # record value and index
        result = [0] * n

        def merge_sort(arr, left, right):
            # merge sort [left, right) from small to large, in place
            if right - left <= 1:
                return
            mid = (left + right) // 2
            merge_sort(arr, left, mid)
            merge_sort(arr, mid, right)
            merge(arr, left, right, mid)

        def merge(arr, left, right, mid):
            # merge [left, mid) and [mid, right)
            i = left  # current index for the left array
            j = mid  # current index for the right array
            # use temp to temporarily store sorted array
            temp = []
            while i < mid and j < right:
                if arr[i][0] <= arr[j][0]:
                    # j - mid numbers jump to the left side of arr[i]
                    result[arr[i][1]] += j - mid
                    temp.append(arr[i])
                    i += 1
                else:
                    temp.append(arr[j])
                    j += 1
            # when one of the subarrays is empty
            while i < mid:
                # j - mid numbers jump to the left side of arr[i]
                result[arr[i][1]] += j - mid
                temp.append(arr[i])
                i += 1
            while j < right:
                temp.append(arr[j])
                j += 1
            # restore from temp
            for i in range(left, right):
                arr[i] = temp[i - left]

        merge_sort(arr, 0, n)

        return result

################################################
# 1059. All Paths from Source Lead to Destination (REVISTED)
# 24JUL22
###############################################
class Solution:
    def leadsToDestination(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        '''
        brute force would be to generate all paths and check that they ALL reach the end
        we can accomplish this using dfs
        what else can we say:
            the ending node but have outdirection of 0 (this is a directed edge graph)
            all the nodes must be in a connected component
            so if there is a cycle, we can return false, so first check for a cycle
            
        turns out that we do not need to check all paths, rather that if there is no cycle, and that if there is a leaf node encounter during traversal, that 
        it is the destination node
        
        we cannot simply keep a visited set for cycle detection, i.e we would have no way of reconciling a cross edge as back edge
        with one visited set, this would mean the cross edge is part of a cycle, which is incorrect
        we take the three coloring method for cycle detection from CLRS
        1 - white - not processed yet
        2 - gray - vertex is being processed (DFS has started, but not finished which means all decendatns in tree are not yet processed)
        3 - black - vertex and all decenatns have been processed
        '''
        adj_list = defaultdict(list)
        for start,end in edges:
            adj_list[start].append(end)
            
        states = [None]*n #initally all are white, neither visited nor processed
        def dfs(node):
            #check if black, i.e if gray and backward edge, this must be a loop
            if states[node] != None:
                return states[node] == 3
            #if leah node, this should be destination
            if len(adj_list[node]) == 0:
                return node == destination
            #we are processing, so color 3
            states[node] = 2
            
            for neigh in adj_list[node]:
                #if we get False from any recursive call, we don't do it form here
                if not dfs(neigh):
                    return False
            #we are done processing,
            states[node] = 3
            return True
        
        return dfs(source)

##############################
# 697. Degree of an Array
# 25JUL22
###############################
class Solution:
    def findShortestSubArray(self, nums: List[int]) -> int:
        '''
        an array with degree d, has some element x d times
        a subarray with degree d, also has some elemnt x times
        the number of occurences in this subarray would occur between the frist and last occruences of element x
        
        for each element, keep track of its first occruing index as well as it last occuring index
        as well as maintinaing counts
        
        then find the degree of the initial array
        then for each eleemnt x that occurs the max number of times is right[x] - left[x] + 1
        
        '''
        lefts = {}
        rights = {}
        counts = {}
        
        ans = len(nums)
        degree = 0
        for i, num in enumerate(nums):
            #stor its first occruecnes
            if num not in lefts:
                lefts[num] = i
            rights[num] = i
            counts[num] = counts.get(num,0) + 1
            degree = max(degree,counts[num])
        
        for num,count in counts.items():
            if count == degree:
                ans = min(ans,rights[num] - lefts[num] + 1)
        
        return ans

#############################
# 418. Sentence Screen Fitting
# 21JUL22
##############################
class Solution:
    def wordsTyping(self, sentence: List[str], rows: int, cols: int) -> int:
        '''
        brute force, fill in row by row
        rows*cols time complexity
        '''
        N = len(sentence)
        times = 0
        i = 0 #index into sentence 
        for _ in range(rows):
            c = 0 #times we fit into row, reset for every new row
            while c + len(sentence[i]) <= cols:
                c += len(sentence[i]) + 1
                i += 1
                #if we have used up all the words
                if i == N:
                    times += 1
                    i = 0
        return times

class Solution:
    def wordsTyping(self, sentence: List[str], rows: int, cols: int) -> int:
        '''
        we can cache
        we let dp(i) return the number of times the ith word appears at the beginning of the row and gives us the next index of the word to use
        doesnt quite work
        '''
        memo = {}
        N = len(sentence)
        def dp(i):
            if i in memo:
                return memo[i]
            c = 0
            times = 0
            next_i = i
            while c + len(sentence[i]) <= cols:
                c += len(sentence[i]) + 1
                next_i += 1
                if next_i == N:
                    times += 1
                    next_i = 0
            memo[i] = [next_i,times]
            return [next_i,times]
        
        ans = 0
        i = 0
        for _ in range(rows):
            ans += dp(i)[1]
            i = dp(i)[0]
        return ans

#cache
class Solution:
    def wordsTyping(self, sentence: List[str], rows: int, cols: int) -> int:
        n = len(sentence)

        @lru_cache(None)
        def dp(i):  # Return (nextIndex, times) if the word at ith is the beginning of the row
            c = 0
            times = 0
            while c + len(sentence[i]) <= cols:
                c += len(sentence[i]) + 1
                i += 1
                if i == n:
                    times += 1
                    i = 0
            return i, times

        ans = 0
        wordIdx = 0
        for _ in range(rows):
            ans += dp(wordIdx)[1]
            wordIdx = dp(wordIdx)[0]
        return ans

class Solution:
    def wordsTyping(self, sentence: List[str], rows: int, cols: int) -> int:
        '''
        in the brute force, we kept trying to fill in row by row, and word by word if it would fit
        the problem is we are recomputing an answer if we start a new line with a word we previosuly used to start a line with
        if we knew how many times we used up all the words with this ith word starting a new line, we could just increment the number of times and advance to the next word
        i.e we can cache the results of starting a new lin with any given word in the sentence
        
        the cached values will be the first word in the next line, and how many senences e'eve completed in that row
        
        the first time we have a sentence start with word i, we need to compute the number of sentences that have been completed on that line
        and the first on the next line
        we then cahce the results and use them again whenver we have a line that starts with that given word
        
        O(N*cols) where N is the number of words
        
        https://leetcode.com/problems/sentence-screen-fitting/discuss/443413/Detailed-Explanation-Python-Dynamic-Programming-Intuition
        '''
        x = 0 
        y = 0 
        
        word = 0
        ans = 0
        
        for w in sentence:
            if len(w) > cols:
                return 0
        
        memo = {}
        while y < rows:
            x = 0
            start_word = word
            
            # If the given starting word is not in our cache, compute the results for it
            if (start_word not in memo):
                comp_flag = 0
                # Count the number of completed sentences on this line, and what the starting word on the next line will be
                while x < cols: 
                    diff = cols - x 
                    if (len(sentence[word]) > diff):
                        break
                    else:
                        x += (len(sentence[word])+1)
                        word += 1
                        word = word%len(sentence)
                        if (word == 0):
                            comp_flag += 1 
                            ans += 1
                        # Cache the result
                        memo[start_word] = [word, comp_flag]
            else:
                # If the given word is in our cache, simply updated our answer with the # of completed sentences 
                word, com = memo[start_word] 
                ans += com
            y += 1
            
        return ans

###########################################################
# 236. Lowest Common Ancestor of a Binary Tree (REVISITED)
# 26JUL22
############################################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        '''
        we need to recurse onto the tree and when we find p and q return some flag
        the node that returns flags for both searches must be the lca
        
        '''
        self.lca = None
        
        def dfs(node):
            if not node:
                return False
            left = dfs(node.left)
            right = dfs(node.right)
            #if the current node is either p or q, represent as flag variable
            mid = node == p or node == q
            #if left,right,or mid is tree, this is the lca
            if mid + left + right >= 2:
                self.lca = node
            return mid or left or right
        
        dfs(root)
        return self.lca

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        '''
        we can save parent pointers as we descend until we hit either p or q
        the first common node we get during this traversal would the LCA
        save parent pointers into hashmap
        '''
        stack = [root]
        parents = {root:None} #child -> parent
        while p not in parents or q not in parents:
            node = stack.pop()
            
            if node.left:
                parents[node.left] = node
                stack.append(node.left)
            if node.right:
                parents[node.right] = node
                stack.append(node.right)
        
        ancestors = set()
        
        while p:
            ancestors.add(p)
            p = parents[p]
        
        while q not in ancestors:
            q = parents[q]
        
        return q

##############################
# 26JUL22
# 728. Self Dividing Numbers
###############################
class Solution:
    def selfDividingNumbers(self, left: int, right: int) -> List[int]:
        '''
        brute force, check each digit in each number in range left to right for self dividing property
        '''
        ans = []
        
        for num in range(left,right+1):
            curr_num = num
            while curr_num:
                digit = curr_num % 10
                if digit == 0 or num % digit != 0:
                    break
                else:
                    curr_num //= 10
            
            if curr_num == 0:
                ans.append(num)
        
        return ans

#############################
# 477. Total Hamming Distance
# 26JUL22
##############################
#brute force TLE
class Solution:
    def totalHammingDistance(self, nums: List[int]) -> int:
        '''
        we define hamming distance as the number of positions at which the correspoind bits are different
        to find the bits that are differnt i can do a^b
        then reduce this number counting the number of ones
        '''
        def numDiffBits(a,b):
            diff = a ^ b
            count = 0
            while diff:
                count += diff & 1
                diff >>= 1
            return count
        
        ans = 0
        N = len(nums)
        for i in range(N):
            for j in range(i+1,N):
                ans += numDiffBits(nums[i],nums[j])
        
        return ans

class Solution:
    def totalHammingDistance(self, nums: List[int]) -> int:
        '''
        generating all pairs of nums takes way too long
        lets for a bit position for all nums we count the number of bits ON for this position
        for this position lets call it count k
        then the number of off positions would be n - k
        
        we know for each unique pair, it contributes one unit of total hamming distance 
            at a bit position, one OFF and one ON
        
        rather for a bit position, if i have k 1's, then i should n-k 0's
        then the count of pairs contributing 1 unit of hamming distance is k*(n-k)
        
        '''
        N = len(nums)
        total_hamming = 0
        for i in range(32):
            num_ones = 0
            for num in nums:
                num_ones += (num >> i) & 1
            total_hamming += num_ones*(N-num_ones)
        
        return total_hamming

class Solution:
    def totalHammingDistance(self, nums: List[int]) -> int:
        N = len(nums)
        temp = map('{:032b}'.format, nums)
        total_hamming = 0
        for b in zip(*temp):
            ones = b.count('1')
            total_hamming += ones*(N-ones)
        return total_hamming

###############################
# 717. 1-bit and 2-bit Characters 
# 26JUL22
###############################
#cool idea
class Solution:
    def isOneBitCharacter(self, bits: List[int]) -> bool:
        '''
        we can either encode using a 0
        or using 10 or 11
        return true if the last character must be a one bit char,
        rather determing is the last bit must be 0
        
        i can just use recursion and pop off the digits and be encoded
        '''
        N = len(bits)
        def dp(i):
            if i > N:
                return False
            if i == N-1:
                return True
            #encode 1
            first = second = third = None
            if bits[i] == 0:
                first = dp(i+1)
            if bits[i:i+1] == [1,0]:
                second = dp(i+2)
            if bits[i:i+1] == [1,1]:
                third = dp(i+2)
            return first and second and third
            
        
        return dp(0)

class Solution:
    def isOneBitCharacter(self, bits: List[int]) -> bool:
        '''
        scan through array then just adavance on one char or two char
        '''
        N = len(bits)
        i = 0
        while i < N -1:
            if bits[i] == 0:
                i += 1
            else:
                i += 2
        
        return i == N-1 and bits[i] == 0

##############################
# 114. Flatten Binary Tree to Linked List (REVISITED)
# 27JUL22
##############################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        '''
        we can use morris traversal, 
        idea, for every node
            if there is a left substree, find the right most node
            make this right most nodes' right child, the current right child of the node we are on
            then make the right child of the current node we are on point to the left child
            basically we are moving everything to the right
        
        algo:
            we use a pointer for traversing the nodes of our tree stating fromt the root
            for every node check if it has a left, else we go right
            if node does have left cild, find right modst node
            once we find right most node, rewirete moving nodes left or right
            set left to null
        
        possibilty of touching each nodde at least twice
        '''
        if not root:
            return root
        node = root
        
        while node:
            if node.left:
                #find right modst
                rightmost = node.left
                while rightmost.right:
                    rightmost = rightmost.right
                
                #reconnect
                rightmost.right = node.right
                node.right = node.left
                node.left = None
            node = node.right

###################################
# 419. Battleships in a Board
# 27JUL22
###################################
#welp dfs works
class Solution:
    def countBattleships(self, board: List[List[str]]) -> int:
        '''
        traverse the board, then if we hit a cell that is an X, try looking in all four directions
        if we can go in a direction add the cells to a seen set
        this would result in N^3 time
        
        '''
        rows = len(board)
        cols = len(board[0])
        seen_X = set()
        
        dirrs = [[1,0], [-1,0],[0,1],[0,-1]]
        
        def dfs(i,j):
            seen_X.add((i,j))
            for dx,dy in dirrs:
                neigh_x = i + dx
                neigh_y = j + dy
                if 0 <= neigh_x < rows and 0 <= neigh_y < cols:
                    if board[neigh_x][neigh_y] == 'X' and (neigh_x,neigh_y) not in seen_X:
                        dfs(neigh_x,neigh_y)
        ans = 0
        for i in range(rows):
            for j in range(cols):
                if board[i][j] == 'X' and (i,j) not in seen_X:
                    dfs(i,j)
                    ans += 1
        
        return ans

class Solution:
    def countBattleships(self, board: List[List[str]]) -> int:
        '''
        we only need to cont the first cell of hte battle ship, since the ships and only by 1 by k cols or k rows by 1 col
        we identify a battleship startpoing as the stop left
        and we check of this X was previously part of a ship we had already counted
        '''
        rows = len(board)
        cols = len(board[0])
        
        ships = 0
        
        for i in range(rows):
            for j in range(cols):
                #empty sspace does not contribute to count
                if board[i][j] == '.':
                    continue
                #is this X was presiously part of a ship
                if i > 0 and board[i-1][j] == 'X':
                    continue
                if j > 0 and board[i][j-1] == 'X':
                    continue
                ships += 1
        
        return ships

###############################
# 424. Longest Repeating Character Replacement
# 28JUL22
###############################
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        '''
        we can pick any char in string s and change that char to any letter at most k times
        return the length of the the substring with longest repeat character 
        evidently this is a two pointer problem
        we need to maintain the frequency counts of the chars in our current window
        if size(current window) > k + max(current window count)?
        why, because we can change any character in the string at most k times, we might as well try to change the chars to the most frequent char in the window
        https://leetcode.com/problems/longest-repeating-character-replacement/discuss/363071/Simple-Python-two-pointer-solution
        '''
        ans = 0
        counts = Counter()
        left = 0
        N = len(s)
        for right in range(N):
            #add to the current window
            counts[s[right]] += 1
            #if we need to shrink the window because we have use to many changes
            while right - left + 1 - max(counts.values()) > k:
                counts[s[left]] -= 1
                left += 1
            #update the window size, because we are in at least k
            ans = max(ans,right -left + 1)
        
        return ans

class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        '''
        instead of shinking the window by advancing the left pointer, keep
        awlays keep track of the largest freq count
        then just move up the pointer
        '''
        ans = 0
        counts = Counter()
        left = 0
        N = len(s)
        max_freq = 0
        for right in range(N):
            counts[s[right]] += 1
            max_freq = max(max_freq,counts[s[right]])
            if right - left + 1 > k + max_freq:
                counts[s[left]] -= 1
                left += 1
            ans = max(ans, right - left + 1)

        return ans

###########################
# 916. Word Subsets (REVISITED)
# 30JUL22
###########################
#close one....
class Solution:
    def wordSubsets(self, words1: List[str], words2: List[str]) -> List[str]:
        '''
        i can make a count object of words2
        then travrese the words in word1 checking if it is univsrsal from the counts map
        we need to use up all of words2 count map for a word to be inversial
        '''
        counts_words2 = Counter()
        for w in words2:
            counts_words2[w] += 1
        
        ans = []
        for w in words1:
            temp = copy.deepcopy(counts_words2)
            for ch in w:
                if ch in temp:
                    temp[ch] -= 1
                    if temp[ch] == 0:
                        del temp[ch]
                else:
                    continue
            if len(temp) == 0:
                ans.append(w)
        
        return ans

class Solution:
    def wordSubsets(self, words1: List[str], words2: List[str]) -> List[str]:
        '''
        we can define word in words1 a superset if each count of letter in word is >= each count for all in words in words2
        so we would have to compare each word in words1 with each word in in words2
        whic would not work, 
        instead of comparing each word in words1 with each word in words2, we compare the max counts
        i.r reduce words2 into one
        '''
        words2_max = defaultdict(int)
        for w in words2:
            #get counts for this current word
            temp = Counter(w)
            for ch,count in temp.items():
                words2_max[ch] = max(words2_max.get(ch,0),count)
        
        ans = []
        for w in words1:
            temp = Counter(w)
            check = []
            for ch,count in temp.items():
                if ch in words2_max:
                    check.append(count >= words2_max[ch])
            if all(check):
                ans.append(w)
        
        return ans

################################
# 307. Range Sum Query - Mutable (REVISITED)
# 31JUL22
#################################
#segment tree recusring using built in node instead of arrays
"""
    The idea here is to build a segment tree. Each node stores the left and right
    endpoint of an interval and the sum of that interval. All of the leaves will store
    elements of the array and each internal node will store sum of leaves under it.
    Creating the tree takes O(n) time. Query and updates are both O(log n).
"""

#Segment tree node
class Node(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.total = 0
        self.left = None
        self.right = None
        

class NumArray(object):
    def __init__(self, nums):
        """
        initialize your data structure here.
        :type nums: List[int]
        """
        #helper function to create the tree from input array
        def createTree(nums, l, r):
            
            #base case
            if l > r:
                return None
                
            #leaf node
            if l == r:
                n = Node(l, r)
                n.total = nums[l]
                return n
            
            mid = (l + r) // 2
            
            root = Node(l, r)
            
            #recursively build the Segment tree
            root.left = createTree(nums, l, mid)
            root.right = createTree(nums, mid+1, r)
            
            #Total stores the sum of all leaves under root
            #i.e. those elements lying between (start, end)
            root.total = root.left.total + root.right.total
                
            return root
        
        self.root = createTree(nums, 0, len(nums)-1)
            
    def update(self, i, val):
        """
        :type i: int
        :type val: int
        :rtype: int
        """
        #Helper function to update a value
        def updateVal(root, i, val):
            
            #Base case. The actual value will be updated in a leaf.
            #The total is then propogated upwards
            if root.start == root.end:
                root.total = val
                return val
        
            mid = (root.start + root.end) // 2
            
            #If the index is less than the mid, that leaf must be in the left subtree
            if i <= mid:
                updateVal(root.left, i, val)
                
            #Otherwise, the right subtree
            else:
                updateVal(root.right, i, val)
            
            #Propogate the changes after recursive call returns
            root.total = root.left.total + root.right.total
            
            return root.total
        
        return updateVal(self.root, i, val)

    def sumRange(self, i, j):
        """
        sum of elements nums[i..j], inclusive.
        :type i: int
        :type j: int
        :rtype: int
        """
        #Helper function to calculate range sum
        def rangeSum(root, i, j):
            
            #If the range exactly matches the root, we already have the sum
            if root.start == i and root.end == j:
                return root.total
            
            mid = (root.start + root.end) // 2
            
            #If end of the range is less than the mid, the entire interval lies
            #in the left subtree
            if j <= mid:
                return rangeSum(root.left, i, j)
            
            #If start of the interval is greater than mid, the entire inteval lies
            #in the right subtree
            elif i >= mid + 1:
                return rangeSum(root.right, i, j)
            
            #Otherwise, the interval is split. So we calculate the sum recursively,
            #by splitting the interval
            else:
                return rangeSum(root.left, i, mid) + rangeSum(root.right, mid+1, j)
        
        return rangeSum(self.root, i, j)
                


# Your NumArray object will be instantiated and called as such:
# numArray = NumArray(nums)
# numArray.sumRange(0, 1)
# numArray.update(1, 10)
# numArray.sumRange(1, 2)

#iterative,using buildtin array
class NumArray(object):

    def __init__(self, nums):
        self.l = len(nums)
        self.tree = [0]*self.l + nums
        for i in range(self.l - 1, 0, -1):
            self.tree[i] = self.tree[i<<1] + self.tree[i<<1|1]
    
    def update(self, i, val):
        n = self.l + i
        self.tree[n] = val
        while n > 1:
            self.tree[n>>1] = self.tree[n] + self.tree[n^1]
            n >>= 1
        
    
    def sumRange(self, i, j):
        m = self.l + i
        n = self.l + j
        res = 0
        while m <= n:
            if m & 1:
                res += self.tree[m]
                m += 1
            m >>= 1
            if n & 1 ==0:
                res += self.tree[n]
                n -= 1
            n >>= 1
        return res

#good write up on segment tree in C++
#https://leetcode.com/problems/range-sum-query-mutable/discuss/1281195/Clean-Solution-w-Explanation-or-Segment-Tree-or-Beats-100


##########################
# 44. Wildcard Matching
# 30JUL22
##########################
#recursion
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        '''
        not as bad as you think, similar to other recursive string problems, where wejust keep passing
        a truncated string into the recursive caller
        cases:
            if the strings are equal, i.e p == s, then return True
            if pattern matches any string, p == '*', return True
            if p is empty or s is empty, return False
            if the current chars match (p[0] == s[0] or p[0] == '?') then compare the next ones functions rec(s[1:], p[1:])
            if the curr pattern is a start p[0] == '*':
                if start matches no characters, just call rec(s,p[1:])
                the start matches one ormore, call rec(s[1:])
            if p[0] != s[0]:
             return False
        
        clean up the input first
            for patterns like a****bc***cc
            we can simplify to a*bc*cc, so we can keep the depth of the recurion treee smal

        time complexity is O(S*P * (S+P))
        space complexity is (S*P)
        '''
        def remove_stars(s):
            new_s = ""
            for ch in s:
                if not new_s or ch != '*':
                    new_s += ch
                elif new_s[-1] != '*':
                    new_s += ch
            
            return new_s
        
        memo = {}
        
        def rec(s,p):
            #retrieve
            if (s,p) in memo:
                return memo[(s,p)]
            #match
            if p == s or p == '*':
                memo[(s,p)] = True
            #empty cases
            elif p == '' or s == '':
                memo[(s,p)] = False
            #first char match, truncate in caller
            elif p[0] == s[0] or p[0] == '?':
                memo[(s,p)] = rec(s[1:],p[1:])
            #single start
            elif p[0] == '*':
                memo[(s,p)] = rec(s,p[1:]) or rec(s[1:],p)
            else:
                memo[(s,p)] = False
            
            return rec(s,p)
        
        p = remove_stars(p)
        return rec(s,p)

#another way
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        @lru_cache(None)
        def dfs(i, j):
            if j == len(p):  # Reach full pattern
                return i == len(s)

            if i < len(s) and (s[i] == p[j] or p[j] == '?'):  # Match Single character
                return dfs(i + 1, j + 1)
            
            if p[j] == '*':
                return dfs(i, j + 1) or i < len(s) and dfs(i + 1, j)  # Match zero or one or more character
            
            return False

        return dfs(0, 0)

class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        '''
        bottom up, very similar to edit distance
        if we let dp(p_idx,s_idx) be the answer to if p[:p_idx] matches s[:s_idx]
        if the last characters are the same is there is match at this, then we can get the answer in constant time using the recurrence
        d[p_idx][s_idx] = d[p_idx - 1][s_idx - 1], if pattern at this char is '?'
        
        if the pattern char is '*' and there was match on the previous step d[p_idx-1][s_idx-1] then:
            the start at the end of the pattner still results in match
            the star could match as many chars as you wish
            d[p_idx-1][i] = True, for all i >= s_idx-1
            make sure to look at the pictures for this solution
        '''
        s_len = len(s)
        p_len = len(p)
        
        # base cases
        if p == s or set(p) == {'*'}:
            return True
        if p == '' or s == '':
            return False
        
        # init all matrix except [0][0] element as False
        d = [[False] * (s_len + 1) for _ in range(p_len + 1)]
        d[0][0] = True
        
        # DP compute 
        for p_idx in range(1, p_len + 1):
            # the current character in the pattern is '*'
            if p[p_idx - 1] == '*':
                s_idx = 1
                                        
                # d[p_idx - 1][s_idx - 1] is a string-pattern match 
                # on the previous step, i.e. one character before.
                # Find the first idx in string with the previous math.
                while not d[p_idx - 1][s_idx - 1] and s_idx < s_len + 1:
                    s_idx += 1
    
                # If (string) matches (pattern), 
                # when (string) matches (pattern)* as well
                d[p_idx][s_idx - 1] = d[p_idx - 1][s_idx - 1]
    
                # If (string) matches (pattern), 
                # when (string)(whatever_characters) matches (pattern)* as well
                while s_idx < s_len + 1:
                    d[p_idx][s_idx] = True
                    s_idx += 1
                                   
            # the current character in the pattern is '?'
            elif p[p_idx - 1] == '?':
                for s_idx in range(1, s_len + 1): 
                    d[p_idx][s_idx] = d[p_idx - 1][s_idx - 1] 
                                   
            # the current character in the pattern is not '*' or '?'
            else:
                for s_idx in range(1, s_len + 1): 
                    # Match is possible if there is a previous match
                    # and current characters are the same
                    d[p_idx][s_idx] = d[p_idx - 1][s_idx - 1] and p[p_idx - 1] == s[s_idx - 1]  
                                                               
        return d[p_len][s_len]