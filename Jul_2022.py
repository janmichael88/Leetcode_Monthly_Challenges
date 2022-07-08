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
