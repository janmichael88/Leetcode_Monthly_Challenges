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
