############################
# 338. Counting Bits
# 01MAR2022
############################
class Solution:
    def countBits(self, n: int) -> List[int]:
        '''
        i can count the ones in each num by shifting
        and using num & 1
        '''
        def count_ones(num):
            ones = 0
            while num:
                ones += num & 1
                num = num >> 1
            
            return ones
        
        
        ans = []
        for i in range(n+1):
            ans.append(count_ones(i))
        
        return ans

class Solution:
    def countBits(self, n: int) -> List[int]:
        
        def pop_count(x: int) -> int:
            count = 0
            while x != 0:
                x &= x - 1 # zeroing out the least significant nonzero bit
                count += 1
            return count
            
        ans = [0] * (n + 1)
        for x in range(n + 1):
            ans[x] = pop_count(x)
    
        return ans       

class Solution:
    def countBits(self, n: int) -> List[int]:
        '''
        we can use dp to find the count of currnet set bits in number i from number i-1
        we first need the transisiton function
        say we have x = 605 = 1001011101
        it differs by a number we have previosuly calculated
        x = 93 = 1011101
        0001011101 -> 605
        1001011101 -> 93
        count(x+b) = count(x) + 1
        
        if we have the numbers
        0 -> 0
        1 -> 1
        2 -> 10
        3 -> 11
        4 -> 100
        5 -> 101
        6 -> 110
        7 -> 111
        
        we can find the count for 2 and 3 just by adding 1 from 0 and 1
        we can find the count for 4 through 7, by adding one to the counts from the counts in 0 to 3
        
        we can write the transitino functon as
        P(x+b) = P(x) + 1. for b in range(0,2**b)
        
        rather dp(x) = dp(x-b) + 1 for b in all the bits up to x
        '''
        dp = [0]*(n+1)
        x = 0
        b = 1
        
        #for all bit positions <= n
        while b <= n:
            #check intervals [b,2b) or [b,n] from [0,b)
            #check all allowable bit positions for this b, and increment by 1
            #i.e move through the positions < b, and use the transition to generate the counts
            while x < b and x + b <= n:
                print(x,b)
                dp[x+b] = dp[x] + 1
                x += 1
            
            #reset x
            #why? we need to start back to the first bit position
            x = 0
            b <<= 1
        
        return dp

class Solution:
    def countBits(self, n: int) -> List[int]:
        '''
        we can also use the least significant bit, right most bit
        if we look at the relation between a number x and x//2
        x = 605 = 1001011101
        x = 302 = 100101110
        
        we just popped off the last bit 
        P(x) = P(x/2) + (x mod 2)
        '''
        ans = [0] * (n + 1)
        for x in range(1, n + 1):
            # x // 2 is x >> 1 and x % 2 is x & 1
            #not x mod 2 and x & 1 are the same
            ans[x] = ans[x >> 1] + (x % 2) 
        return ans 
        
class Solution:
    def countBits(self, n: int) -> List[int]:
        '''
        we can just get the last set bit using Brian Kernighan's trick
        P(x) = P(x & (x-1)) + 1
        '''
        ans = [0]*(n+1)
        for x in range(1,n+1):
            ans[x] = ans[x & (x-1)] + 1
        
        return ans

#recursive
class Solution:
    def countBits(self, n: int) -> List[int]:
        '''
        we can just get the last set bit using Brian Kernighan's trick
        P(x) = P(x & (x-1)) + 1
        '''
        memo = {}
        
        def dp(n):
            if n == 0:
                return 0
            if n == 1:
                return 1
            #otherwise fetch
            if n in memo:
                return memo[n]
            #res = dp(n >> 1) + (x % 2)
            res = dp(n & (n-1)) + 1
            memo[n] = res
            return res
        
        ans = [0]*(n+1)
        for i in range(1,n+1):
            ans[i] = dp(i)
        
        return ans

###############################
# 01MAR22
# 1490. Clone N-ary Tree
###############################
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children if children is not None else []
"""

class Solution:
    def cloneTree(self, root: 'Node') -> 'Node':
        '''
        i can use dfs to traverse the tree but keep cloned copy outside the rescursive function
        then move
        '''
        def dfs(node):
            if not node:
                return
            cloned_node = Node(val = node.val)
            for child in node.children:
                cloned_node.children.append(dfs(child))
            
            return cloned_node
        
        return dfs(root)

"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children if children is not None else []
"""

class Solution:
    def cloneTree(self, root: 'Node') -> 'Node':
        '''
        iterative dfs
        mainitin root and cloned in stack
        
        note for bfs just replace stack with deque
        '''
        if not root:
            return root
        
        cloned = Node(val = root.val)
        stack = [(root,cloned)]
        
        while stack:
            curr,copy = stack.pop()
            if not curr:
                continue
            for child in curr.children:
                copy_child = Node(child.val)
                copy.children.append(copy_child)
                stack.append((child,copy_child))
            
        return cloned

###############################
# 02MAR22
# 392. Is Subsequence
###############################
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        '''
        we just want to check if s is a substring of t
        just move pointers in s and t and return if we moved all the way to s
        '''
        #s is too big
        if len(s) > len(t):
            return False
        
        ptr_s = 0
        ptr_t = 0
        
        while ptr_t < len(t) and ptr_s < len(s):
            #match
            if s[ptr_s] == t[ptr_t]:
                ptr_s += 1
            ptr_t += 1
        
        return  ptr_s == len(s)

#recursive solution
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        '''
        we can use recursion 
        dp(i,j) answers the question if s[:i] is a subsequence of s[:j]
        if s[i] == t[j] then we are left so solve the subproblem on s[i+1:] and s[j+1]
        if s[i] != t[j], then we are forced to advance the pointer j into target and see if we can 
        find a subsequence s[:i] in t[:j+1]
        
        base cases:
            if we have advanced i, then it must be a subsequence
            if we have advance j, but still letters are unmatched
        '''
        memo = {}
        
        def dp(i,j):
            #base caes
            if i == len(s):
                return True
            if j == len(t):
                return False
            if (i,j) in memo:
                return memo[(i,j)]
            #recursive case
            if s[i] == t[j]:
                res = dp(i+1,j+1)
                memo[(i,j)] = res
                return res
            res = dp(i,j+1)
            memo[(i,j)] = res
            return res
        
        dp(0,0)
        print(memo)

class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        '''
        follow up question can be solved greedily using hashmap
        if we repeatedly have incoming chars s1,s2,s3....n
        we have to repeatedly search the target string each time a new char s comes in
        one way is to precompute a hash for each char in t mapping to its index
        
        now with the hashmap precomputed, we match greedily
        when searching indices, we can use binary search to avoid scaning linearly
        we want the index that is just greater than the current index
        if the index we find is >= len(s) it can't be dont because there are no more indices to choose from
        
        We use the pointer to check if an index is suitable or not. For instance, for the character a whose corresponding indices are [0, 3], we need to pick an index out of all the appearances as a match. Suppose at certain moment, the pointer is located at the index 1. Then, the suitable greedy match would be the index of 3, which is the first index that is larger than the current position of the target pointer.
        '''
        mapp = defaultdict(list)
        for i,char in enumerate(t):
            mapp[char].append(i)
            
        curr_idx = -1
        for char in s:
            if char not in mapp:
                return False
            idxs = mapp[char]
            candidate = bisect.bisect_right(idxs,curr_idx)
            #if i got to the end of the idxs list, there isn't an index just one greate
            if candidate != len(idxs):
                curr_idx = idxs[candidate]
            else:
                return False
            
        return True

###################################
# 03MAR22
# 413. Arithmetic Slices
###################################
#brute force, fails N^3
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        '''
        this is a revisit of a problem
        
        brute force would be to just check all possible array slices of length > 3
        check first first order differences, and scan all diffs
        
        '''
        ans = 0
        N = len(nums)
        for start in range(N-2):
            diff = nums[start+1] - nums[start]
            for end in range(start+2,N):
                #store slice
                curr_slice = nums[start:end+1]
                for i in range(len(curr_slice)-1):
                    if curr_slice[i+1] - curr_slice[i] != diff:
                        break
                else:
                    ans += 1
        
        return ans

class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        '''
        N^2 is a little different, but should still pass
        instead of checking every possible range and check differences we do something a little different
        
        define start and end pointers into nums of satisfibaility for arithmetic sequence
        we simply need to check that nums[end] and nums[end+1] have same difference as the previus diffs
        in the slice nums[start:end]
        
        i.e we just check the last pair to have the same diff
        '''
        ans = 0
        N = len(nums)
        for start in range(N-2):
            diff = nums[start+1] - nums[start]
            for end in range(start+2,N):
                #store slice
                if nums[end] - nums[end-1] == diff:
                    ans += 1
                else:
                    break
        
        return ans

#recursive case
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        '''
        finally we can use recursion to solve this problem
        let dp(i) be the number of arithmetic subsequences ending at i
        dp(i) = dp(i-1) + 1 if nums[i] - nums[i-1] == nums[i+1] - nums[i]
        we need to start from the end of the array though
        we reduce each subproblem by 1
        we need to update sum globally thought? we could probably write this as the final return call
        
        '''
        memo = {}
        def dp(i):
            if i < 2:
                return 0
            if i in memo:
                return memo[i]
            #check if adding new one increases the count
            if nums[i] - nums[i-1] == nums[i-1] - nums[i-2]:
                res = 1 + dp(i-1)
                memo[i] = res
                return res
            #we are forced to start over the count if there isn't a match at this current if when adding the new element at index i does not consitute a valid arithmetic sequence
            else:
                return 0
            
        #dp for all to get the count
        count = 0
        #we also could have reversed the order to get this to run faster
        for i in range(len(nums)):
            count += dp(i)
        
        return count
        

#dp
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        '''
        we can translate this to bottom up dp
        '''
        N = len(nums)
        dp = [0]*N
        for i in range(2,N):
            if nums[i] - nums[i-1] == nums[i-1] - nums[i-2]:
                dp[i] = 1 + dp[i-1]
            else:
                dp[i] = 0
        
        return sum(dp)
        
#no last pass for sum
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        '''
        we can translate this to bottom up dp
        '''
        N = len(nums)
        dp = [0]*N
        ans = 0
        for i in range(2,N):
            if nums[i] - nums[i-1] == nums[i-1] - nums[i-2]:
                dp[i] = 1 + dp[i-1]
                ans += dp[i]
            else:
                dp[i] = 0
        
        return ans

#constant space, no last pass for sum
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        '''
        we can translate this to bottom up dp
        '''
        N = len(nums)
        dp_curr = dp_prev = 0
        ans = 0
        for i in range(2,N):
            if nums[i] - nums[i-1] == nums[i-1] - nums[i-2]:
                dp_curr = 1 + dp_prev
                ans += dp_curr
            else:
                dp_curr = 0
            dp_prev = dp_curr
        
        return ans
      
#################################
# 03MAR22
# 267. Palindrome Permutation II
#################################
#good enough for interview
class Solution:
    def generatePalindromes(self, s: str) -> List[str]:
        '''
        we can just use recursion to generate all possible permutations
        then just check if the permutation is a palindrome
        '''
        N = len(s)
        ans = set()
        def rec(i,taken,path):
            if sum(taken) == N:
                cand = "".join(path)
                if cand == cand[::-1]:
                    ans.add(cand)
            #mark as taken
            for j in range(N):
                if not taken[j]:
                    taken[j] = True
                    path[i] = s[j]
                    rec(i+1,taken,path)
                    taken[j] = False

                            
        rec(0,[False]*N,[0]*N)

#make sure to check this post on similar combination, permutations problem
class Solution:
    def generatePalindromes(self, s: str) -> List[str]:
        '''
        we can just use recursion to generate all possible permutations
        then just check if the permutation is a palindrome
        
        pretty much had it, but we first need to check if a permuation can be a palindrom at all
        we can count up the chars for a permutation
        if the number of chars with off number of occruences exceeds 1, its indicates that no
        palindromic permuatis is possible for s
        
        after we have shown that the string s can yeild a palindromic permutation we must be judicious in the 
        permutation generation!
        
        one way is to generate half the string, and just append its reverse to the string
        
        Based on this idea, by making use of the number of occurences of the characters in ss stored in mapmap, we create a string stst which contains all the characters of ss but with the number of occurences of these characters in stst reduced to half their original number of occurences in ss.
        
        In case of a string ss with odd length, whose palindromic permutations are possible, one of the characters in ss must be occuring an odd number of times. 
        
        We keep a track of this character, chch, and it is kept separte from the string stst. We again generate the permutations for stst similarly and append the reverse of the generated permutation to itself, but we also place the character chch at the middle of the generated string.
        
        Key, we only want to generate a palindromic permutation, NOT generate every permutation
        
        we swap the current element with all the elements lying towards its right to generate the permutations
        
         Before swapping, we can check if the elements being swapped are equal. If so, the permutations generated even after swapping the two will be duplicates(redundant).
         
         here for the permutation scheme, we use the swapping trick
        '''
        #first check count requirment
             
        kv = Counter(s)
        mid = [k for k, v in kv.items() if v%2]
        #mid records count of odd counts, i cannot form a palindrom of the number of odd counts is > 1
        if len(mid) > 1:
            return []
        mid = '' if mid == [] else mid[0]
        #find the middle of the palindrome
        half = ''.join([k * (v//2) for k, v in kv.items()])
        #this is the only half we have to permute!
        #the remaining half should be on the other side of center
        half = [c for c in half]
        
        #using mid as cener point, generate permuations of the hafl elements
        #then create permutation + mid + permutation[::-1]
        def rec(end,temp):
            if len(temp) == end:
                curr = ''.join(temp)
                ans.append(curr + mid + curr[::-1])
            else:
                for i in range(end):
                    #the right half of the or statment is for making sure we don't duplicate
                    if visited[i] or (i>0 and half[i] == half[i-1] and not visited[i-1]):
                        continue
                    visited[i] = True
                    temp.append(half[i])
                    rec(end, temp)
                    visited[i] = False
                    temp.pop()
                    
        ans = []
        visited = [False] * len(half)
        rec(len(half), [])
        return ans


#using builtins, just don't forget about these
class Solution(object):
    def generatePalindromes(self, s):
        d = collections.Counter(s)
        m = tuple(k for k, v in d.iteritems() if v % 2)
        p = ''.join(k*(v/2) for k, v in d.iteritems())
        return [''.join(i + m + i[::-1]) for i in set(itertools.permutations(p))] if len(m) < 2 else []

##########################
# 04MAR22
# 799. Champagne Tower
##########################
class Solution:
    def champagneTower(self, poured: int, query_row: int, query_glass: int) -> float:
        '''
        we need to generate the final query after pouring poured
        once we create the query row, just get the glass number
        we need to simulate the overflow from a current glass to a glass below it
        
        #key intuition
        In general, if a glass has flow-through X, then Q = (X - 1.0) / 2.0 quantity of champagne will equally flow left and right. We can simulate the entire pour for 100 rows of glasses. A glass at (r, c) will have excess champagne flow towards (r+1, c) and (r+1, c+1).
        '''
        curr_row = [poured]
        #in the loop generate next row with on more
        for row in range(query_row):
            #calculate overflow
            next_row = [0]*(len(curr_row) + 1)
            for col in range(len(curr_row)):
                #get this cups overflow
                cup_overflow = (curr_row[col] - 1.0) / 2.0
                #if there is overflow it spills below, and left and right
                if cup_overflow > 0:
                    next_row[col] += cup_overflow
                    next_row[col+1] += cup_overflow
            
            curr_row = next_row
        
        return min(1,curr_row[query_glass])

#instead of not saving space, build out whole matrix
class Solution(object):
    def champagneTower(self, poured, query_row, query_glass):
        A = [[0] * k for k in xrange(1, 102)]
        A[0][0] = poured
        for r in xrange(query_row + 1):
            for c in xrange(r+1):
                q = (A[r][c] - 1.0) / 2.0
                if q > 0:
                    A[r+1][c] += q
                    A[r+1][c+1] += q

        return min(1, A[query_row][query_glass])

#recursive case
class Solution:
    def champagneTower(self, poured: int, query_row: int, query_glass: int) -> float:
        '''
        we can also definte the (i,j) element recursively
        dp(i,j) = {
        case 1: when we are at the beginning and end glasses of a row
            dp(i-1,j+1) or dp(i-1,j) we want the excess
        case 2: we take exess from both
            dp(i-1,j-1) + dp(i-1,j)
            
        base case, top. or (i,j) == 0 return poured
        }
        for each dp call we need to mainin the volume in the current glass and how much is overflowed
        
        '''
        memo = {}
        
        
        def _rec(i, j):
            #base case
            if i == 0 and j == 0:
                available = float(poured)
                in_glass = available if available <= 1 else 1
                excess = available - in_glass
                return in_glass, excess
            
            #retreival case
            if (i,j) in memo:
                return memo[(i,j)]
            
            #recursive case
            else:
                _, left = _rec(i - 1, j - 1) if j else (0, 0.0)
                _, right = _rec(i - 1, j) if j < i else (0, 0.0)
                available = left / 2.0 + right / 2.0
            in_glass = available if available <= 1 else 1
            excess = available - in_glass
            memo[(i,j)] = (in_glass,excess)
            return in_glass, excess
        
        return _rec(query_row, query_glass)[0]


#################################
# 271. Encode and Decode Strings
# 04MAR22
#################################
class Codec:
    def encode(self, strs: [str]) -> str:
        """Encodes a list of strings to a single string.
        """
        '''
        just use the 256'th ascii char as the delimeter in the string
        then split on that char
        '''
        delimiter = chr(256)
        encoded = delimiter.join(strs)
        return encoded
        

    def decode(self, s: str) -> [str]:
        """Decodes a single string to a list of strings.
        """
        delimiter = chr(256)
        return s.split(delimiter)

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.decode(codec.encode(strs))

class Codec:
    '''
    this is either a know or don't know, i'm not gonna whip it out of my ass
    it's based encoding used in http v1.1
    
    encode:
        each chunk is precended by it's size in bytes
        i.e its size is the delimiter
        iterate over the aray of chunks:
            for each chunk computes it's length and conver that length into a 4 bytes string
            append to encoded string
                4 bytes string with information about chunk size in byets
                chunk itself
        return encoded string
        
    decode: 
        read in b4 bytes, next one should be the string
    '''
    
    def len_to_str(self,x):
        x = len(x)
        #shift 4 bytes
        bytes = [chr(x >> (i * 8) & 0xff) for i in range(4)]
        #reverse and joing to get the bytes string for a chunk
        bytes.reverse()
        bytes_str = ''.join(bytes)
        return bytes_str
    
    def str_to_int(self, bytes_str):
        """
        Decodes bytes string to integer.
        """
        result = 0
        #256 bits is 8 bytes, 1 a byte == 8 bits
        for ch in bytes_str:
            result = result * 256 + ord(ch)
        return result
    
    def encode(self, strs: [str]) -> str:
        """Encodes a list of strings to a single string.
        """
        # encode here is a workaround to fix BE CodecDriver error
        return ''.join(self.len_to_str(x) + x.encode('utf-8') for x in strs)
        

    def decode(self, s: str) -> [str]:
        """Decodes a single string to a list of strings.
        """
        i = 0
        n = len(s)
        output = []
        while i < n:
            #compute length for first chunk
            length = self.str_to_int(s[i:i+4])
            #move up to its positsion
            i += 4
            #convert this chunk to string
            output.append(s[i:i+length])
            #advance to end of current chunk
            i += length
        return output
        


# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.decode(codec.encode(strs))
        
###########################
# 740. Delete and Earn
# 05MAR22
###########################
class Solution:
    def deleteAndEarn(self, nums: List[int]) -> int:
        '''
        we are given an int array nums and we want to maximize the number of points
        we can perform the following single operatio any number of times:
            picks nums[i] delete and earn nums[i]
            then delete every element that equlas nums[i] - 1 and nums[i] + 1
            
        we let dp(i) be the the max number of points i cant get if i delete nums[i]
        rather lets call it dp(num)
        if we deleted a num, we can get all occurecnes of the number
        so far we have dp(num) = counts[num]*num + previous calss
        but what is previous calles
        well if we have taken dp(num) the last number we possible would have taken would be dp(num-2)
        but we did take this num, we don't gain anything, but still need the answer from dp(num-1)
        
        dp(num) = max(gain + dp(num-2), + dp(num-1))
        base cases:
            if obvious that dp(0) = 0, because we get no points with the 0 num
            INTUITION ON BASE CASES:
                dp(num-2) would keep calling itself until we get to dp(0)
                but what about dp(1)?
                well, the only thing we stand to gain from dp(1) because 1*(number of 1 times) will always be that number
                
        before doing dp, we can convert the array to the number of points we can get by deleting the number
        rather counts[num] = number of times num shows up
        
        when we invoke, the dp, we start with the largest number in nums, and see the max for each
        '''
        points_obtained = Counter()
        max_number = 0
        for num in nums:
            points_obtained[num] += num
            max_number = max(max_number,num)
            
        memo = {}
        
        def dp(curr_num):
            if curr_num == 0:
                return 0
            if curr_num == 1:
                return points_obtained[1]
            if curr_num in memo:
                return memo[curr_num]
            ans = max(points_obtained[curr_num] + dp(curr_num-2), dp(curr_num-1))
            memo[curr_num] = ans
            return ans
        
        return dp(max_number)

class Solution:
    def deleteAndEarn(self, nums: List[int]) -> int:
        '''
        translating to dp
        we need states of up to max_number
        '''
        points_obtained = Counter()
        max_number = 0
        for num in nums:
            points_obtained[num] += num
            max_number = max(max_number,num)
            
        dp = [0]*(max_number + 1)
        dp[1] = points_obtained[1]
        
        
        for num in range(2,len(dp)):
            dp[num] = max(points_obtained[num] + dp[num-2], dp[num-1])
        
        return dp[max_number]

#saving space
class Solution:
    def deleteAndEarn(self, nums: List[int]) -> int:
        '''
        we can use up constance space just by saving the last two states
        then update
        '''
        points_obtained = Counter()
        max_number = 0
        for num in nums:
            points_obtained[num] += num
            max_number = max(max_number,num)
            
        dp_num_minus_two = 0
        dp_num_minus_one = points_obtained[1]
        curr_dp = 0
        
        
        for num in range(2,max_number+1):
            curr_dp = max(points_obtained[num] + dp_num_minus_two, dp_num_minus_one)
            #update
            dp_num_minus_two = dp_num_minus_one
            dp_num_minus_one = curr_dp
        
        return dp_num_minus_one

class Solution:
    def deleteAndEarn(self, nums: List[int]) -> int:
        '''
        instead of checking over each possible point up to max num
        we instead check only points in our counts mapp
        think of test cases like [1,2,3,1000]
        
        if we find that adjacent elements have a difference of 1, that means we can only take the points associated with one of them and we apply the normal recurrence
        howevery if they do not differ by 1, then we don't need to worry about deletions and just take the points
        
        '''
        points = defaultdict(int)
        # Precompute how many points we gain from taking an element
        for num in nums:
            points[num] += num
            
        elements = sorted(points.keys())
        two_back = 0
        one_back = points[elements[0]]
        
        for i in range(1, len(elements)):
            current_element = elements[i]
            if current_element == elements[i - 1] + 1:
                # The 2 elements are adjacent, cannot take both - apply normal recurrence
                two_back, one_back = one_back, max(one_back, two_back + points[current_element])
            else:
                # Otherwise, we don't need to worry about adjacent deletions
                two_back, one_back = one_back, one_back + points[current_element]

        return one_back

################################
# 280. Wiggle Sort
# 05MAR22
################################
class Solution:
    def wiggleSort(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        '''
        we want to reorder the array such that
        nums[0] <= nums[1] >= nums[2] <= nums[3]..
        any permutation with this characteristic suffices as an answer
        which just want a an array that goes up and down, but not strictly increasing or strictly decreasing
        or rather we want the difference to be alternating
        omg
        if i have an already sorted array
        [1,2,3,4,5,6]
        i could swap
        [2,1,3,4,5,6]
        [2,1,4,3,5,6]
        [2,1,4,3,6,5]
        sort and swap pairwise starting from the second element
        '''
        nums.sort()
        N = len(nums)
        for i in range(1,len(nums)-1,2):
            nums[i],nums[i+1] = nums[i+1], nums[i]
class Solution:
    def wiggleSort(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        '''
        we can traverse the array once and check whether or not it needs to be wiggled
        if it needs to ne wiggled we swap the current element with its next
        since the second element must be <= then the first, we start of by checking the less than requirement
        in order for the wiggle to work, we must define 1 starting case
        for example it cannot be nums[0] <= nums[1].... or nums[0] >= nums[1]
        also note, the array can be wiggle sorted
        
        here's another proof
        
        Suppose A[0: i-1] is wiggle sorted, when

        i is even:
            if A[i] <= A[i - 1], then A[0: i] is also wiggle sorted.
            if A[i] > A[i - 1], then we swap A[i] and A[i - 1], since A[0: i -1] is wiggle sorted, A[i - 1] >= A[i - 2], which means A[i] > A[i - 2]. After swapping, we have A[i - 2] <= A[i] >= A[i - 1], both A[i] and A[i - 1] refers to the elements before swapping. So, A[0: i] is wiggle sorted after we swap.
        i is odd, the proof is similar.

        '''
        N = len(nums)
        less = True
        for i in range(N-1):
            if less:
                if nums[i] > nums[i+1]:
                    nums[i],nums[i+1] = nums[i+1], nums[i]
            else:
                if nums[i] < nums[i+1]:
                    nums[i],nums[i+1] = nums[i+1], nums[i]
            less = not less
        
####################################################
# 1359. Count All Valid Pickup and Delivery Options
# 06MAR22
###################################################
#backtracking brute force
#https://leetcode.com/problems/count-all-valid-pickup-and-delivery-options/discuss/1069610/Java-Find-order-permutations-using-backtracking
class Solution:
    def countOrders(self, n: int) -> int:
        '''
        for n = 1
        (p1,d1)
        for n = 2
        (P1,P2,D1,D2), (P1,P2,D2,D1), (P1,D1,P2,D2), (P2,P1,D1,D2), (P2,P1,D2,D1) and (P2,D2,P1,D1
        
        rules:
            delivery(i) must always be after pickip(i)
            brute force would be to use backtracking
            we can pass over the orders and it is not picked up, pick up the order, mark and recurse
            if it is already picked up, but not delivered, then deliver it
            after delivering all unmark the delivers and pick up to make another combination
        '''
        count = [0]
        picked = [False]*n
        delivered = [False]*n
        paths = []
        
        def dfs(path,picked,delivered):
            p = sum(picked)
            d = sum(delivered)
            
            #ending
            if p == n and d == n:
                count[0] += 1
                count[0] %= 10**9 + 7
                paths.append(path[:])
                return
            
            for i in range(n):
                if picked[i]:
                    pass
                if delivered[i]:
                    pass
                
                #if not deliverd and picked
                if not delivered[i] and picked[i]:
                    path.append(f"D{i}")
                    delivered[i] = True
                    dfs(path,picked,delivered)
                    #backtrtack
                    path.pop()
                    delivered[i] = False
                    
                #not picked
                if not picked[i]:
                    path.append(f"P{i}")
                    picked[i] = True
                    dfs(path,picked,delivered)
                    path.pop()
                    picked[i] = False
                    
        dfs([],picked,delivered)
        print(paths)

#top down
class Solution:
    def countOrders(self, n: int) -> int:
        '''
        instead of generating each possible permutation and incrementing acount, perhaps we can use the counts
        from previous subproblems to generate the answer -> top down recursion
        
        intuion:
            if there are n unpicked orders, then we have n different options for orders that we can pick up at the current step, exmpla: 
            [p0,p1,p2] all are initially unpicked at step i, we for step i, we can choose from 3, so there are at least 3 ways (from this i'th step!)
            insteaf of picking up each one by one and recursinv, we could count the nunber of ways to pick (n-1)
            so intead of calling it n times we only call it at most one more additional time
            
            if we pick one order then we need to count the ways for the rest of the remaining orders
            remainin orders of just a smaller subproblem (0/1 knap sack really)
            
            lets so we have 'unpicked' number of orders that have not been pickedup and 'undelivered' number of orders to be delivered.
            If we want to pick one order than there are 'unpicked' different choide to pick at this step
            If we want to deliver one oder, then there are undelivred - unpicked differecnt choices 
            
            rather we can say:
            // If we want to pick one order then,
            waysToPick = unpicked * totalWays(unpicked - 1, undelivered)

            // If we want to deliver one order then,    
            waysToDeliver = (undelivered - unpicked) * totalWays(unpicked, undelivered - 1)
            
            then the num ways for this step would be:
                waysToPick + waysToDeliver mod 10*9 + 7
                
            base case, when unpcicked > undelieer, return 1 way,or we have delivered everything
        '''
        memo = {}
        mod = 10**9 + 7
        
        #dp(n,n) rerpesetns the number of ways to pickup and deliver n itesm
        #we we want dp(n-1,n) and dp(n,n-1)
        def dp(unpicked,undelivered):
            #we have picked up and delivered everything
            if unpicked == 0 and undelivered == 0:
                return 1
            #when we cannot pick or deliver if there is nothing to pick from
            #we cannot deliever of if wedon't have enough picked up
            if unpicked < 0 or undelivered < 0 or undelivered < unpicked: 
                return 0
            if (unpicked,undelivered) in memo:
                return memo[(unpicked,undelivered)]
            
            #number ways pick up order
            ans = unpicked*dp(unpicked-1,undelivered)
            ans %= mod
            
            #number of ways to get deliver picked order
            ans += (undelivered - unpicked)*dp(unpicked,undelivered-1)
            ans %= mod
            
            memo[(unpicked,undelivered)] = ans
            return ans

#bottom up
class Solution:
    def countOrders(self, n: int) -> int:
        '''
        we can just translate this to bottom up
        don't worry about the special cases code that in the loop
        '''
        mod = 10**9 + 7
        dp = [[0]*(n+1) for _ in range(n+1)]
        for unpicked in range(n+1):
            for undelivered in range(n+1):
                if (unpicked,undelivered) == (0,0):
                    dp[unpicked][undelivered] = 1
                if unpicked < 0 or undelivered < 0 or undelivered < unpicked: 
                    dp[unpicked][undelivered] = 0
                #number of ways to pick up order, but make sure for boundary check
                if unpicked - 1 >= 0:
                    dp[unpicked][undelivered] += unpicked*dp[unpicked-1][undelivered]
                dp[unpicked][undelivered] %= mod
                #number of ways to deliver, with boundary check
                if undelivered - 1 >= 0 and (undelivered - unpicked > 0):
                    dp[unpicked][undelivered] += (undelivered - unpicked)*dp[unpicked][undelivered -1]
                dp[unpicked][undelivered] %= mod
                    
        
        return dp[n][n]
                
#permutations
class Solution:
    def countOrders(self, n: int) -> int:
        '''
        for n pickups, we can do this n!
        now after placing n pick ups in any random order, how many ways can we place n deliveris
        so we have placed:
            P2 P4 P1 P3
            we now need to place D3, we only have on spot to place it it must come after P3
            P2 P4 P1 P3 D3
            
            now we want to place D1, there are three choides
            to place D4, there are 5 choices
            to place D2, there are 7 choides
            so the way to arrang N pickes up and delivers is: N! \prod_{i=1}^{N} (2*i- 1)
        '''
        MOD = 1_000_000_007
        ans = 1

        for i in range(1, n + 1):
            # Ways to arrange all pickups, 1*2*3*4*5*...*n
            ans = ans * i
            # Ways to arrange all deliveries, 1*3*5*...*(2n-1)
            ans = ans * (2 * i - 1)
            ans %= MOD
        
        return ans

##############################
# 07MAR22
# 21. Merge Two Sorted Lists
##############################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        '''
        two pointers into list1 and list2
        take the smaaller
        '''
        dummy = ListNode()
        curr = dummy
        
        l1_ptr = list1
        l2_ptr = list2
        
        while l1_ptr and l2_ptr:
            if l1_ptr.val <= l2_ptr.val:
                curr.next = l1_ptr
                l1_ptr = l1_ptr.next
            else:
                curr.next = l2_ptr
                l2_ptr = l2_ptr.next
            curr = curr.next
            
        #adding remaingin
        curr.next = l1_ptr if l1_ptr else l2_ptr
        
        return dummy.next

class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        '''
        we can do this recursively
        helper function to merge(l1,l2)
        base cases:
            if one list is empty, return the other
        
        recursive case:
            if l1.val < l2.val:
                just take l1.val and call on a smapller subproblem (l1.next,l2)
            same for the other
        '''
        def rec(l1,l2):
            if not l1:
                return l2
            if not l2:
                return l1
            elif l1.val < l2.val:
                l1.next = rec(l1.next,l2)
                return l1
            else:
                l2.next = rec(l1,l2.next)
                return l2
        
        return rec(list1, list2)


###########################
# 281. Zigzag Iterator
# 08MAR22
###########################
class ZigzagIterator:
    def __init__(self, v1: List[int], v2: List[int]):
        '''
        just alternate between 0 and 1, i can increment a poitner by one each time,
        then just mod 2 it
        more precsiely the pointer to vector moves cyclically
        and only pointer to an element in the vector increments
        '''
        self.vecs = [v1,v2]
        self.p_elem = 0
        self.p_vec = 0
        self.size = len(v1) + len(v2)
        self.count = 0

    def next(self) -> int:
        iter_num = 0 
        ans = None
        
        while iter_num < len(self.vecs):
            #if we can get a return value from this pointer
            curr_vec = self.vecs[self.p_vec]
            if self.p_elem < len(curr_vec):
                ans = curr_vec[self.p_elem]
                
            #move current iter_num
            iter_num += 1
            #make sure to adjust the p_vec
            self.p_vec = (self.p_vec + 1) % len(self.vecs)
            
            #increment element poitner once iterating all vectors, i.e we started from zero again
            if self.p_vec == 0:
                self.p_elem += 1
                
            #if we have ans to return, return it
            if ans is not None:
                self.count += 1
                return ans
        raise Exception 
        
class ZigzagIterator:
    def __init__(self, v1: List[int], v2: List[int]):
        '''
        the issue with the first approach is not the most effecient, when the input vectors are not equal in size
        examples:
            [1] and [1,2,3,4,5]
            we would keep cycling between 1 and 2, but we know we should just stay on 2
        
        we can use a queue to keep pointers in input vectors
        
        initially each input vector will have a correspinding pointer in q
        at each next() call, we pop out a pointer from the q,
        then we go into this pointed vector to retreive the element
            if the vector sill has elements let, we append another poitner pointed to this vectors at the end of the q
            if all the elemnt in th chose vector are outputted, we do not add this pointer to the end of the q
            we won't need to pass over vectors that have been exhausted
            
        for has next:
            as long as there are pointers in q to process
            
        However, the key point here is that we could simply use some index and integer to implement the role of pointer in the above idea.
        '''
        self.vecs = [v1,v2]
        self.q = deque()
        for i,v in enumerate(self.vecs):
            if len(v) > 0:
                self.q.append((i,0)) #ptrs and start element

    def next(self) -> int:
        if self.q:
            vec_index, elem_index = self.q.popleft()
            #get nextt
            next_elem_index = elem_index + 1
            #check if there are more
            if next_elem_index < len(self.vecs[vec_index]):
                #append the rest
                self.q.append((vec_index,next_elem_index))
            return self.vecs[vec_index][elem_index]
        
        #error
        raise Exception
        

    def hasNext(self) -> bool:
        return len(self.q) != 0

###############################################
# 82. Remove Duplicates from Sorted List II
# 09MAR22
###############################################
#close one! works at de-duplicating
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        i can use two pointers, curr and next
        and if next == curr keep advancing it
        '''
        curr = head
        
        while curr:
            next_node = curr.next
            while next_node and next_node.val == curr.val:
                next_node = next_node.next
            curr.next = next_node
            curr = curr.next
        
        return head
            
     
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        i can use two pointers, curr and next
        and if next == curr keep advancing it
        
        need to use dummy and maintin head and predecessor
        then delete duplicates 
        return dummy.next
        '''
        dummy = ListNode(-1)
        dummy.next = head
        pred = dummy
        curr = head
        
        while curr:
            #duplicates
            if curr.next and curr.val == curr.next.val:
                #move next head
                while curr.next and curr.val == curr.next.val:
                    curr = curr.next
                #delete all duplicates, head stops at the last duplicated valu
                pred.next = curr.next
            else:
                pred = pred.next
                
            #move head
            curr = curr.next
        
        return dummy.next


#################################
# 398. Random Pick Index
# 09MAR22
#################################
class Solution:

    def __init__(self, nums: List[int]):
        '''
        we want to pick a random index given a target
        '''
        self.mapp = defaultdict(list)
        for i,num in enumerate(nums):
            self.mapp[num].append(i)

    def pick(self, target: int) -> int:
        #return random index
        return random.choice(self.mapp[target])
        


# Your Solution object will be instantiated and called as such:
# obj = Solution(nums)
# param_1 = obj.pick(target)

#resrvoir sampling

class Solution:

    def __init__(self, nums: List[int]):
        '''
        we can use algorithm R, reservoir sampling
        for example, say we have n numbers to chose from, the probability of picking the current number of 1/n
        this implies we dont pic the other n+1 numbers with probability (1-1/n)
        rather we do not pick any number further from index (i+1)
        \prod_{i=1}^{n-1} \frac{i}{i+1}
        
        we can interpret this as:
            * picking the ith number from the list of i numbers
            * which means not picking the (i+1)th number from the list of (i+1) numbers
            * rather not picking the (nth) number from the list of (n) numbers, picking the rest (n-1)
        '''
        self.nums = nums
        

    def pick(self, target: int) -> int:
        N = len(self.nums)
        count = 0 #current count of target num
        idx = 0
        for i in range(N):
            #if nums is target, this index is a candidate
            if self.nums[i] == target:
                count += 1
                
                #pick or don't pick this current index
                if random.randint(1,count) == count:
                    idx = i
            
        return idx
            
        


# Your Solution object will be instantiated and called as such:
# obj = Solution(nums)
# param_1 = obj.pick(target)

class Solution:

    def __init__(self, nums: List[int]):
        '''
        we can use algorithm R, reservoir sampling
        for example, say we have n numbers to chose from, the probability of picking the current number of 1/n
        this implies we dont pic the other n+1 numbers with probability (1-1/n)
        rather we do not pick any number further from index (i+1)
        \prod_{i=1}^{n-1} \frac{i}{i+1}
        
        we can interpret this as:
            * picking the ith number from the list of i numbers
            * which means not picking the (i+1)th number from the list of (i+1) numbers
            * rather not picking the (nth) number from the list of (n) numbers, picking the rest (n-1)
        '''
        self.nums = nums
        

    def pick(self, target: int) -> int:
        N = len(self.nums)
        count = 0 #current count of target num
        idx = 0
        for i in range(N):
            #if nums is target, this index is a candidate
            if self.nums[i] == target:
                count += 1
                
                #pick or don't pick this current index, if greater than current count
                if random.random() < 1/count:
                    idx = i
            
        return idx

###################################
# 2. Add Two Numbers
# 10MAR22
###################################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        '''
        create dummy list
        and through lists by moving l1 and l2
        create carry
        '''
        dummy = ListNode(-1)
        carry = 0
        curr = dummy
        
        while l1 or l2:
            l1_val = l1.val if l1 else 0
            l2_val = l2.val if l2 else 0
            #get current answer
            curr_sum = l1_val + l2_val + carry
            #get node answer after modding
            node_ans = curr_sum % 10
            #update carry
            carry = curr_sum // 10
            #add nodes
            curr.next = ListNode(node_ans)
            curr  = curr.next
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next
        if carry:
            curr.next = ListNode(carry)
        return dummy.next

###############################
# 10MAR22
# 288. Unique Word Abbreviation
###############################
class ValidWordAbbr:
    '''
    we define abbreviation of a word as the concat of first letter + num letters betweeen first and last + last
    isUnique returns True if:
        there is no word in dict whose abbreviation is equal to word's abbreviation
        for any word in dict whose abbreviation == word's abbreivation that word and word are the same
    '''

    def __init__(self, dictionary: List[str]):
        '''
        i can use hashmap: it should be abbrev:word
        abbreivation: set(words)
        then just check
        '''
        self.mapp = defaultdict(set)
        for s in dictionary:
            val = s
            if len(s) > 2:
                s = s[0]+str(len(s)-2)+s[-1]
            self.mapp[s].add(val)

        

    def isUnique(self, word: str) -> bool:
        val = word 
        if len(word) > 2:
            word = word[0]+str(len(word)-2)+word[-1]
        # if word abbreviation not in the dictionary, or word itself in the dictionary (word itself may 
        # appear multiple times in the dictionary, so it's better using set instead of list)
        
        #left side, if abbrev not in mapp, then there is no word in dictionary whose abbrev == word's abbrev
        #right side, if the current word's abbrev == a dictionary word's abbrev, check if word in the set
        return word not in self.mapp or (len(self.mapp[word]) == 1 and val in self.mapp[word])

# Your ValidWordAbbr object will be instantiated and called as such:
# obj = ValidWordAbbr(dictionary)
# param_1 = obj.isUnique(word)
# obj = ValidWordAbbr(dictionary)
# param_1 = obj.isUnique(word)

###############################
# 61. Rotate List
# 11MAR22
################################
#messy buy it works
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        '''
        first pass is find the length of the the array
        we can only rotate at most N times, at which point N+1 becomes the original linked list
        find the actual shift be doing N % k
        hold current head as pointer
        advance another pointer N % k times
        '''
        if not head:
            return head
        
        #find the size:
        N = 0
        curr = head
        
        while curr:
            N += 1
            curr = curr.next
            
        #find actualy rotation size
        k = k % N
        
        #if it goes evenly, no switch at all
        if k == 0:
            return head
        
        #otherwise we need to find the head of the reconnecionted list
        #which would be at N - k
        #advance to new head
        new_head_count = N - k
        prev = None
        curr = head
        while new_head_count > 0:
            prev = curr
            curr = curr.next
            new_head_count -= 1
        
        #make prev.next none
        prev.next = None
        curr_temp = curr
        while curr_temp.next:
            curr_temp = curr_temp.next
        
        curr_temp.next = head
        return curr

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        '''
        an easier way would be to reconnect the linked list into a ring
        and break at the point where rotation happens
        
        if we have n nodes, the new head is at n-k and the tail is at n-k-1
        in the case where k >= n, we jus take to be k % n
        '''
        if not head:
            return None
        if not head.next:
            return head
        
          # close the linked list into the ring
        old_tail = head
        n = 1 #recall we presently have at least 1 node to start
        while old_tail.next:
            old_tail = old_tail.next
            n += 1
        old_tail.next = head
        
        # find new tail : (n - k % n - 1)th node
        # and new head : (n - k % n)th node
        new_tail = head
        for i in range(n - k % n-1):
            new_tail = new_tail.next
        new_head = new_tail.next
        
        # break the ring
        new_tail.next = None
        
        return new_head

#######################################
# 12MAR22
# 138. Copy List with Random Pointer
#####################################
#recursive, using up space
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        '''
        i can use a hashmap to make copies
        old node:copy
        just traverse and put into mapp
        we can use dfs to travers the linked list
        at each call put into mapp
        
        we dfs along the graph
        
        base cases:
            not node, return None
            already seen node, return its copy
            
        recursive case:
            make a copy
            put in mapp
            add its next and random pointers
            
        intuition on recursive function:
            the function returns a copy of the current node and also copy its next and random
        '''
        #mapp to store node:copey
        self.visited = {}
        
        def make_copy(node):
            if not node:
                return None
            #return its copy
            if node in self.visited:
                return self.visited[node]
            
            cloned_node = ListNode(node.val)
            #put into mapp
            self.visited[node] = cloned_node
            #adjust each recursively
            cloned_node.next = make_copy(node.next)
            cloned_node.random = make_copy(node.random)
            
            return cloned_node

#iterative
class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        '''
        we can also treay this iteratively
        traverrse and copy along
        if a next or randomo node already exsist in visited, retrieve
        ohterwise make copy
        
        we define a helper functiono that makes a copy of the corresponding node
        
        '''
        self.visited = {}
        
        if not head:
            return head
        
        curr = head
        copied = Node(curr.val)
        self.visited[curr] = copied
        
        while curr:
            copied.next = self.copy_node(curr.next)
            copied.random = self.copy_node(curr.random)
            
            copied = copied.next
            curr = curr.next
        
        return self.visited[head]
        
    def copy_node(self,node):
        if node:
            if node in self.visited:
                return self.visited[node]
            else:
                copied = Node(node.val)
                self.visited[node] = copied
                return self.visited[node]
        return None


#O(1) space
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        '''
        from the hint, we can avoid using extra space by putting copies right next to the ooriginal
        Old List: A --> B --> C --> D
        InterWeaved List: A --> A' --> B --> B' --> C --> C' --> D --> D'
        The interweaving is done using next pointers and we can make use of interweaved structure to get the correct reference nodes for random pointers.
        
        it's a rather tricky implementation to get correct
        
        algo: three passes
            1. traverse the original list and clone the nodes, placing cloonded copy next to original node
                clonded_node.next = originial_nodex.next
                original_node.next = cloned_node
            2. second pass, we conntect randoms
                clonded_node.random = original_node.random
                original_node.random = cloned_node
            3. last pass, we need to unweave
                next pointers need to be assgiend so that we only have the A', B' and C' connected
                
        '''
        if not head:
            return head

        # Creating a new weaved list of original and copied nodes.
        ptr = head
        while ptr:

            # Cloned node
            new_node = Node(ptr.val, None, None)

            # Inserting the cloned node just next to the original node.
            # If A->B->C is the original linked list,
            # Linked list after weaving cloned nodes would be A->A'->B->B'->C->C'
            new_node.next = ptr.next
            ptr.next = new_node
            ptr = new_node.next

        ptr = head

        # Now link the random pointers of the new nodes created.
        # Iterate the newly created list and use the original nodes random pointers,
        # to assign references to random pointers for cloned nodes.
        while ptr:
            ptr.next.random = ptr.random.next if ptr.random else None
            ptr = ptr.next.next

        # Unweave the linked list to get back the original linked list and the cloned list.
        # i.e. A->A'->B->B'->C->C' would be broken to A->B->C and A'->B'->C'
        ptr_old_list = head # A->B->C
        ptr_new_list = head.next # A'->B'->C'
        head_new = head.next
        while ptr_old_list:
            ptr_old_list.next = ptr_old_list.next.next
            ptr_new_list.next = ptr_new_list.next.next if ptr_new_list.next else None
            ptr_old_list = ptr_old_list.next
            ptr_new_list = ptr_new_list.next
        return head_new
            
###############################
# 71. Simplify Path
# 14MAR22
###############################
class Solution:
    def simplifyPath(self, path: str) -> str:
        '''
        important bits
        we need to used a stack and split on the correct delimiter '/'
        
        1. init stack
        2. split input string on '/' because anything in between '/' is a valid command/director
        3. traverse over the split string
        4. if we see a '.' or empty, we do nothing
        5. if we see a '..' we go up a level, and pop from stack
        6. othewsie simply add to the stack
        7. join using delimtier '/'
        '''
        stack = []
        
        for part in path.split('/'):
            #if we see a .. then we have to pop from the stack, i.e returnf rom directoy
            if part == '..':
                if stack:
                    stack.pop()
            #if we see . or is empty, it's just a read out
            elif part == '.' or not part:
                continue
            #must be a new directory access
            else:
                stack.append(part)
        
        return "/"+"/".join(stack)
        

###############################
# 291. Word Pattern II
# 14MAR22
################################
class Solution:
    def wordPatternMatch(self, pattern: str, s: str) -> bool:
        '''
        return true if s matches the pattern
        s matches pattern if there is somoe bijective mapping oof strings in pattern to a string in s
        
        we can solve this using backtracking, we just have to keep trying to use char in the pattern to match diffetn legnth of substrings in s
        we keep going until we through the whole string and pattern
        
        pattern = "abab", s = "redblueredblue"
        
        a matches r
        b matches e
        a matches d, whoops not a good match for a!
        b matched ed instead
        
        while recursing, if the pattenr char exsists in hashamp already, we just have tosee fi we can use to match the same lentgh of the string
        
        example: say so far we have
        a:red
        b:blue
        when we see a again, it better be trying to matche red
        so lets to swe str[i...i+3] matches a
        
        we use hash set to avoid duplicate matches
        
        if char a matches string red, then char b cannot be use to match red
        
        '''
        class Solution:
    def wordPatternMatch(self, pattern: str, s: str) -> bool:
        '''
        we need to use backtracking
        two pointers in each call (i,j)
        i points into pattern and j points into string s
        if i have moved all the way through pattern and s, we can make mapping and return tru
        '''
        mapp = {}
        seen = set()
        
        def dfs(i,j):
            #base cases moved through both, pattern mapping exits
            if i == len(s) and j == len(pattern):
                return True
            if i == len(s) or j == len(pattern):
                return False
            
            #get curr char from pattern
            curr_pattern = pattern[j]
            
            #if pattern char exists
            if curr_pattern in mapp:
                word_string = mapp[curr_pattern]
                
                #if this pattern doesn't match where we are so far
                if not s.startswith(word_string,i): #i.e no match
                    return False
                #keep going on to try matching, because we've aleady matched up to len(s)
                return dfs(i+len(word_string),j+1)
                
            #recursive case, check all ends, we move through our ends
            for k in range(i,len(s)):
                word_string = s[i:k+1]
                if word_string in seen:
                    continue
                ##put intp mapp
                mapp[curr_pattern] = word_string
                seen.add(word_string)
                #if i can still dfs
                if dfs(k+1,j+1):
                    return True
                #backtrack
                del mapp[curr_pattern]
                seen.remove(word_string)
            #reached the end
            return False
            
        return dfs(0,0)

class Solution:
    def wordPatternMatch(self, pattern: str, s: str) -> bool:
        '''
        we can still use bactracking but this time with two maps to get the correct bijection
        (pattern in p) -> (substring in s)
        (substring in s) -> (pattern in p)
        
        we need to push both i and j pointers in backtracking call to meet the true criterie
        '''
        pattern_to_substring = {}
        substring_to_pattern = {}
        def backtrack(i,j):
            #ending conditions
            if i == len(pattern) and j == len(s):
                return True
            #only one pointer has moved all the way
            if i == len(pattern) or j == len(s): 
                return False
            #get the current candidate pattern
            p = pattern[i]
            #check if we have currently mapped it
            if p in pattern_to_substring:
                #get it
                w = pattern_to_substring[p]
                #check if we can this current substring in s is the word or not
                if not s[j:j+len(w)] == w:
                    return False
                #if we don't return false, keep going, advance pointer in pattern 1, and pointer in s len of current substring
                return backtrack(i+1,j+len(w))
            #recursive case
            else:
                #check all ending points in s
                for k in range(j,len(s)):
                    w = s[j:k+1]
                    #if we've seen this already keep going
                    if w in substring_to_pattern:
                        continue
                    #map if
                    pattern_to_substring[p] = w
                    substring_to_pattern[w] = p
                    #backtrack
                    if backtrack(i+1,k+1):
                        return True
                    pattern_to_substring.pop(p)
                    substring_to_pattern.pop(w)
                
                return False
            
        if len(s) < len(pattern):
            return False
        
        return backtrack(0,0)

################################################
# 1249. Minimum Remove to Make Valid Parentheses
# 15MAR22
#################################################
class Solution:
    def minRemoveToMakeValid(self, s: str) -> str:
        '''
        we need to store the indices of the paranthese we want to remove
        recall the balance functino of parantheses
        balance = 0
        for char in s:
            if char '(':
                balance += 1
            if char ')':
                balance -= 1
            if balance < 0:
                return False
        return balance == 0
        
        we push on to a stack the index of each (
        when we see a ), we pop from the stack if there is a stack
        
        we are left with the indices we want to remove
        but we just can't leave un matched ) on the stack
        we need to keep a set of all unmatched indices == )
        in cases like )), these are all unmatched
        only when there is a stack we clear it with )
        '''
        stack = []
        unmatched_idxs = set()
        
        for i,c in enumerate(s):
            #an actual char
            if c not in '()':
                continue
            #openindg
            if c == '(':
                stack.append(i)
            #not opening but close, with empty stack, nothing to pop, we need to remove
            elif not stack:
                unmatched_idxs.add(i)
            #closing and we can match it
            else:
                stack.pop()
                
        #add unmatched in stack
        for i in stack:
            unmatched_idxs.add(i)
        
        ans = ""
        for i,c in enumerate(s):
            if i not in unmatched_idxs:
                ans += c
        
        return ans

class Solution:
    def minRemoveToMakeValid(self, s: str) -> str:
        '''
        from the previous apporach, we know that for all invalid ')' we know immediately that they are invalid 
        i.e the ones we are adding to the set
        it is the ( we don't know abut until the end (since they were left on the stack at the end)
        
        we do two passes still
        but in the first pass clear the unmatched )
        reverse the string
        then clear the unmathced ) again
        
        just need to look at the string in reverse. We do this by swapping the "(" and ")" for each other, and reversing the order of all characters in the string.

        
        '''
        s = self.delete_invalid_closing(s,'(',')')
        s = self.delete_invalid_closing(s[::-1],')','(')
        
        return s[::-1]
        
    def delete_invalid_closing(self,string,open_symbol,close_symbol):
        sb = ""
        balance = 0
        for c in string:
            if c == open_symbol:
                balance += 1
            if c == close_symbol:
                #don't need to include it
                if balance == 0:
                    continue
                balance -= 1
            #add to string
            sb += c
        
        return sb

#######################
# 294. Flip Game II
# 16MAR22
#######################
class Solution:
    def canWin(self, s: str) -> bool:
        '''
        we are given a current state
        in one move a play can only change '++' to '--'
        
        the game ends when a person can no longer make a move,
        
        we can use backtracking to figure out if we can win or not
        backtrack(state) returns whether or not a player can win if they switch a ++ to --
        
        say i have ++++
        
        i make it ++--
        opponenet makes it ++++
        i can just make it ++--
        
        so try making it +--+
        at this point i can win
        
        we try flipping all ++ in s, if we can flip check that the other person cannot win with the change starte
        
        
        '''
        
        memo = {}
        
        def backtrack(state):
            if state in memo:
                return memo[state]
            
            #try switch a ++ and see if we can win from here
            for i in range(len(state)-1):
                if state[i] == '+' and state[i+1] == "+":
                    if backtrack(state[:i]+"--"+state[i+2:]) == False:
                        memo[state] = True
                        return True
            memo[state] = False
            return False
        
        return backtrack(s)
        
#################################
# 946. Validate Stack Sequences
# 16MAR22
##################################
class Solution:
    def validateStackSequences(self, pushed: List[int], popped: List[int]) -> bool:
        '''
        we can simulate pushing and popping
        instead of using a deque, just reverse popped and pushed
        
        we keep pushing onto a an empty stack
        if we get through all elements, return true
        if we get to an invalid move return false
        '''
        stack = []
        
        #reverse
        pushed = pushed[::-1]
        popped = popped[::-1]
        
        while pushed:
            #get first push element and put onto a stack
            stack.append(pushed.pop())
            #now pop if we can
            while stack and stack[-1] == popped[-1]:
                stack.pop()
                popped.pop()

        
        return len(pushed) == 0 and len(popped) == 0

##########################
# 1057. Campus Bikes
# 16MAR22
###########################
#this one is actually tricky, you can't just compared bike to worker for all workers
#we want to pair them with the shortest ones
#close though
class Solution:
    def assignBikes(self, workers: List[List[int]], bikes: List[List[int]]) -> List[int]:
        '''
        assign each worker to its neareast bike
        we the manhattan distance as the distance from worker to bike
        
        ties:
            if there are multiple (worker_i, bike_j) pairs with the same shortest distance, chose pair with smallest worker index
            if there are again multiple ways to do that, choose the pair with the smalest bike index
            
        return array of n elements
        where each element i represents what bike the ith person will take
        
        for each bike, examine all workers and pair with with the closest one, if there is a tie, the worker with the smaller index takes it
        
        '''
        N = len(workers)
        M = len(bikes)
        
        ans = [0]*N
        found = set()
        
        for i,w in enumerate(workers):
            worker_to_bike_distances = []
            for j,b in enumerate(bikes):
                if j in found:
                    continue
                worker_to_bike_dist = abs(b[0] - w[0]) + abs(b[1] - w[1])
                #add to list as [index,dist]
                worker_to_bike_distances.append([j,worker_to_bike_dist])
            #find smallest without sorting
            smallest_dist = min([dist for worker,dist in worker_to_bike_dist])
            #scan again to find candidates
            candidates = []
            for worker,dist in worker_to_bike_dist:
                if dist == smallest_dist:
                    candidates.append(worker)
            #get closest
            closest = min(candidates)
            ans[closest] 

class Solution:
    def assignBikes(self, workers: List[List[int]], bikes: List[List[int]]) -> List[int]:
        '''
        we can't just do a pairwise worker to bike relationship, then take the smallest
        we need to examine all worker-bike pairs then choose the smallest for each one
        
        algo:
            1. Generate all (worker,bike pairs) and find its manhattan distance
            2. sort all the truples in ascending order
            3. pass over the triplets
                if worker has not been aissnged a bike, and bike is still available, then assing the bike to the woker and mark as unavailable for both
                if al workers have been assigned a bike, we are done!
        '''
        triplets = []
        for i,worker in enumerate(workers):
            for j,bike in enumerate(bikes):
                dist = abs(worker[0]-bike[0]) + abs(worker[1]-bike[1])
                triplets.append([dist,i,j])
        
        #sort ascending
        #note for sorting in python, we priortize first, then second, then third
        triplets.sort()
        
        ans = [None]*len(workers)
        count = 0 #we cant terminate if we have mapped all workers to bikes
        
        bikes_assigned = set()
        
        for dist,worker,bike in triplets:
            if not ans[worker] and bike not in bikes_assigned:
                #marker
                bikes_assigned.add(bike)
                ans[worker] = bike
                count += 1
                
                #early termination
                if count == len(workers):
                    return ans
        
        return ans

#bucket sort
class Solution:
    def assignBikes(self, workers: List[List[int]], bikes: List[List[int]]) -> List[int]:
        '''
        we can use bucket sort, 
        intuition notes:
            we can have at most 1000X1000 pairs
            but the maximum possible distanace would be 1998 (if bike is at (0,0) and person is at (999))
            make 1998 buckets!
            groups by distance, and then go in order
            we iteratre  over all possible distances in ascending order
        '''
        min_dist = float('inf') #start from here
        dist_pairs = collections.defaultdict(list)
        for i,worker in enumerate(workers):
            for j,bike in enumerate(bikes):
                dist = abs(worker[0]-bike[0]) + abs(worker[1]-bike[1])
                dist_pairs[dist].append([i,j])
                min_dist = min(min_dist,dist)
        
        #starting off
        curr = min_dist
        bike_taken = [False]*len(bikes)
        worker_ans = [-1]*len(workers)
        pair_count = 0
        
        #we need to keep going until all workers have been paired
        while pair_count < len(workers):
            #check pair at each dist
            for worker,bike in dist_pairs[curr]:
                if worker_ans[worker] == -1 and not bike_taken[bike]:
                    #if bothe worker and bike are free, assigne
                    bike_taken[bike] = True
                    worker_ans[worker] = bike
                    pair_count += 1
            
            curr += 1
        
        return worker_ans

################################################
# 298. Binary Tree Longest Consecutive Sequence
# 17MAR22
################################################
#TLE
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestConsecutive(self, root: Optional[TreeNode]) -> int:
        '''
        return the length of longest consective path
        path must be from parent to child node, and it must be strictly increasing
        dfs carrying the path length, pass path length along
        only incrment by one if it differs by 1
        update globally
        '''
        self.ans = 1
        
        def dfs(parent,child,path_length):
            if not child:
                return
            if parent:
                if child.val - parent.val == 1:
                    path_length += 1
                    self.ans = max(self.ans,path_length)
                    dfs(child,child.left,path_length)
                    dfs(child,child.right,path_length)

            dfs(child,child.left,1)
            dfs(child,child.right,1)

        dfs(None,root,1)
        return self.ans
        

#careful not to call the function too many times within it self
#top down
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestConsecutive(self, root: Optional[TreeNode]) -> int:
        '''
        return the length of longest consective path
        path must be from parent to child node, and it must be strictly increasing
        dfs carrying the path length, pass path length along
        only incrment by one if it differs by 1
        update globally
        '''
        self.ans = 1
        
        def dfs(parent,child,path_length):
            if not child:
                return
            if (parent != None) and (child.val == parent.val + 1):
                path_length += 1
            else:
                path_length = 1
            self.ans = max(self.ans,path_length)
            dfs(child,child.left,path_length)
            dfs(child,child.right,path_length)


        dfs(None,root,1)
        return self.ans
        
#bottom up
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestConsecutive(self, root: Optional[TreeNode]) -> int:
        '''
        no global
        '''
        
        def dfs(parent,child,path_length):
            #just return what we have so far
            if not child:
                return path_length
            if (parent != None) and (child.val == parent.val + 1):
                path_length += 1
            else:
                path_length = 1
            left = dfs(child,child.left,path_length)
            right = dfs(child,child.right,path_length)
            return max(path_length,max(left,right))


        return dfs(None,root,1)

#another way

        # Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def longestConsecutive(self, root: Optional[TreeNode]) -> int:
        '''
        bottom up
        '''
        self.ans = 0
        
        def dfs(node):
            if not node:
                return 0
            left = dfs(node.left) + 1
            right = dfs(node.right) + 1
            #check for left if left
            if node.left != None and node.val + 1 != node.left.val:
                left = 1
            if node.right != None and node.val + 1 != node.right.val:
                right = 1
            self.ans = max(self.ans, max(left,right))
            return max(left,right)
        
        dfs(root)
        return self.ans

##################################
# 316. Remove Duplicate Letters
# 18MAR22
###################################
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        '''
        this is a confusing problem
        a string is greater depens on the comparison between the frist unequal corresponding char in the two strings
        optimal soluiont will have the smallest chars as early as possible
        
        1. the left most letter in our solution will be the smallest letter such that the suffix from that letter contains every other, each solution must have one and only one copy of every letter
        if there are multiple smalelst letter, then we pick the lefmost one, because it gives us the most options
        
        2. as we iterate over our string, if char is > than char i+1 and another occurence of char i i in the strin later on, deleting char i will always lead to the optimal solution
        we choose to minmize the left most char first
        
        i.e we greedibly try to remove chars as early as possible, and picking the ebst letter at each step leads to the optimal solution
        
        we solve this recusrively
        
        We use idea number one from the intuition. In each iteration, we determine leftmost letter in our solution. This will be the smallest character such that its suffix contains at least one copy of every character in the string. We determine the rest our answer by recursively calling the function on the suffix we generate from the original string (leftmost letter is removed).
        '''
        
        #find pos - the idnex of the let most letter in our solution
        #create counter adn end interation ones the suffix doesnt have a unique char
        #pos will be the index of the samlelst char we ecnounter before the iteration ends
        def rec(s):
            counts = Counter(s)
            leftmost = 0
            for i in range(len(s)):
                if s[i] < s[leftmost]:
                    #new leftmost
                    leftmost = i
                counts[s[i]] -= 1
                #if this only has a single occurence we are done!
                if counts[s[i]] == 0:
                    break
            
            return s[leftmost] + rec(s[leftmost:].replace(s[leftmost],"")) if s else ''
        
        return rec(s)

class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        '''
        we can use a stack to store the solution we have built up so far
        we delete chars off the stack wehenver it is possible
        
        at each iteration we add teh current char to the solution if it hasn't already been used
        we try to remove as many chars as possible off the stack
        
        The conditions for deletion are:

        The character is greater than the current characters
        The character can be removed because it occurs later on

        '''
        stack = []
        
        seen = set()
        
        last_occur = {c:i for i,c in enumerate(s)}
        
        for i,c in enumerate(s):
            #we only ty to add c if it's not already in our solution
            #this is to maintain only one of each cahr
            if c not in seen:
                #if last letter, exists, is greater than current c, so we can make it smalled, its not the last occurence
                #for the last part, if it's the last occurence, then i need to include it
                while stack and c < stack[-1] and i < last_occur[stack[-1]]:
                    removed = stack.pop()
                    seen.remove(removed) #obie we would have seen this, since we add outside this loop
                
                seen.add(c)
                stack.append(c)
            
            
        return "".join(stack)

#############################
# 895. Maximum Frequency Stack
# 20MAR22
#############################
#heap solution
class FreqStack:
    '''
    constraints:
        note that there will be atlmost 10^9 elements
        and thtat push and pop will be colled at most 10^4 times
        we need to make this operation fast
    
    we can try use a max heap, where we push the (count,index,number)
    
    we keep three things
    heap, count mapp and counter of eleents
    
    when we push:
        increment count map
        incremant counter
        push(count,index,number)
        
    note: we need to push  count, and the size number to break ties
    '''

    def __init__(self):
        self.heap = []
        self.count_mapp = collections.defaultdict(int)
        self.counter = 0
        

    def push(self, val: int) -> None:
        self.count_mapp[val] += 1
        #push
        heapq.heappush(self.heap,(-self.count_mapp[val],-self.counter,val))
        self.counter += 1

    def pop(self) -> int:
        (count,size,number) = heapq.heappop(self.heap)
        #used up an elment
        self.count_mapp[number] -= 1
        return number
        


# Your FreqStack object will be instantiated and called as such:
# obj = FreqStack()

#stack of stacks
class FreqStack:
    '''
    we obvie care about the frequence of each element and the amx freq
    now we ask:
        among elements with the same frequency with same element, how do we know which element is the most recent
        
    we let grop be a map from frequecne to stack of eleemnts with that frequence
    '''
    def __init__(self):
        #to store counts of each 
        self.counts = Counter()
        #elemetns with that count
        self.group = defaultdict(list)
        #to query with max freq
        self.max_freq = 0
        
    def push(self, val: int) -> None:
        #before adding to group, retrieve count + 1 (after adding)
        f = self.counts[val] + 1
        #add it in
        self.counts[val] = f
        #update
        if f > self.max_freq:
            self.max_freq = f
        #add to groups
        self.group[f].append(val)

    def pop(self) -> int:
        #retrieve
        ans = self.group[self.max_freq].pop()
        #drecemtn the freq for ans
        self.counts[ans] -= 1
        #get read for the next pop
        if not self.group[self.max_freq]:
            self.max_freq -= 1
        
        return ans
        
################################################
# 1007. Minimum Domino Rotations For Equal Row
# 20MAR22
##################################################
class Solution:
    def minDominoRotations(self, tops: List[int], bottoms: List[int]) -> int:
        '''
        just check if i can flip a domino to the top or bottom for all nums 1-6
        do the same for the bottom
        then take the minimum
        '''
        N = len(tops)
        #check tops
        min_movs = float('inf')
        
        for dot in range(1,7):
            #hold current answers
            curr_top_ans = 0
            cant_do_top = False
            
            curr_bot_ans = 0
            cant_do_bot = False
            
            #check swapping to top
            for i in range(N):
                if tops[i] == dot:
                    continue
                if tops[i] != dot and bottoms[i] == dot:
                    curr_top_ans += 1
                else:
                    cant_do_top = True
                    break
                    
            #check bots
            for i in range(N):
                if bottoms[i] == dot:
                    continue
                if bottoms[i] != dot and tops[i] == dot:
                    curr_bot_ans += 1
                else:
                    cant_do_bot = True
                    break
                    
            if cant_do_top and cant_do_bot:
                continue
            else:
                min_movs = min(curr_top_ans,curr_bot_ans)
                
        
        return min_movs if min_movs != float('inf') else -1
                    

class Solution:
    def minDominoRotations(self, tops: List[int], bottoms: List[int]) -> int:
        '''
        we have three possibilites, note we do not need to check if we can make each number
        fore each bot row and top row
        we only need to check the first!
        
        intuition
        
        We need the whole row to match. Therefore, if one tile does not match, we cannot have a matching row. We could abitrarily pick any tile, but we pick the 0th tile since it always exists.

In other words, one side of the first tile MUST match at least one side of every single one of the other tiles, or we cannot have an equal row.

only need to check one domino for both top and bot
        
        algo:
            pick up the first element, check if we cna make top A[0] or bottom B[0] by flipping
        '''
        self.N = len(tops)
        rotations = self.check(tops,bottoms,tops[0])
        #if one could make all elelemtns in A or B equal to A[0]
        if rotations != -1 or tops[0] == bottoms[0]:
            return rotations
        else:
            return self.check(tops,bottoms,bottoms[0])
    
    def check(self,tops,bottoms,x):
        rotations_to_top = 0
        rotations_to_bot = 0
        
        for i in range(self.N):
            #how many rotatinos should be done to make it be tops[0]
            if tops[i] != x and bottoms[i] != x:
                return -1
            elif tops[i] != x:
                rotations_to_top += 1
            elif bottoms[i] != x:
                rotations_to_bot += 1
        
        return min(rotations_to_top,rotations_to_bot)
    
##############################
# 306. Additive Number
# 20MAR22
##############################
class Solution:
    def isAdditiveNumber(self, num: str) -> bool:
        '''
        this is a backtracking problem
        keep pointer in num, and if we can advance it through we are done
        then for all ends j in range(i+1,N) check that there num[j+1:] is the sum
        
        similar to #842 (SPlit Array into Fibonnci Sequences)
        '''
        self.ans= []
        
        def helper(index,nums):
            if index== len(nums) and len(self.ans)>= 3:
                return True    
            for i in range(index,len(nums)):
                if nums[index] == "0" and i != index : break
                n= int(nums[index:i +1])
                
                if len(self.ans) >=2  and self.ans[-2]+ self.ans[-1] < n: break
                
                if len(self.ans) <=1 or self.ans[-2]+ self.ans[-1] == n:
                    self.ans.append(n)
                    if helper(i+1,  nums):
                        return True
                    self.ans.pop()
            
            return False
    
        helper(0,num)
        return len(self.ans)>0

##############################
# 763. Partition Labels
# 21MAR22
##############################
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        '''
        we want to partition s into as many parts as possible so that each letter appears in at most one part
        
        naively:
            generate all possible partitions recursively
            for each part, check that each letter appears in at more one part
        
        which won't work, and that's really hard to code
        
        we need to treat this greedily like the capturing intervals problem
        
        figure out the rightmost index for ech char
        then use two pointers, left and rigth
        we keep advanving right as far as we can
        then when out current pointer == right most, this is the parition that we want
        '''
        
        rightmost = {char:i for i,char in enumerate(s)}
        left,right = 0,0
        sizes = []
        
        
        #keep pushing right as far as we can
        for i,char in enumerate(s):
            right = max(right,rightmost[char])
            
            #if we have reached our rightmost, this ends the current paritition
            if i == right:
                sizes.append(right - left + 1)
                #catch up left
                left = right + 1
                
        return sizes

class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        '''
        just another way
        '''
        L = len(s)
        
        rightmost = {s[i]:i for i in range(L)}
        
        i = 0
        sizes = []
        
        while i < L:
            end = rightmost[s[i]]
            j = i + 1
            #now try to extend j
            while j < end:
                #check candidate part
                if rightmost[s[j]] > end:
                    end = rightmost[s[j]]
                j += 1
            
            #gett the current part size
            sizes.append(end-i+1)
            i = end + 1
        
        return sizes

####################################
# 311. Sparse Matrix Multiplication
# 21MAR22
#####################################
class Solution:
    def multiply(self, mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
        '''
        recall matrix multiplcation is dot product of row and column
        we assume matrix multiplation is always valid
        this can still be done naively
        '''
        mat1_rows = len(mat1)
        mat1_cols = len(mat1[0])
        
        mat2_rows = len(mat2)
        mat2_cols = len(mat2[0])
        
        ans = [[0]*mat2_cols for _ in range(mat1_rows)]
        
        for i in range(mat1_rows):
            for j in range(mat2_cols):
                for k in range(mat1_cols):
                    ans[i][j] += mat1[i][k]*mat2[k][j]
        
        return ans

#we can optimze by seeinf if there is a zero in this row
class Solution:
    def multiply(self, mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
        '''
        recall matrix multiplcation is dot product of row and column
        we assume matrix multiplation is always valid
        this can still be done naively
        recall (m by k) times (k by n)
        
        ans (m by n) with inner dimension k
        '''
        mat1_rows = len(mat1)
        mat1_cols = len(mat1[0])
        
        mat2_rows = len(mat2)
        mat2_cols = len(mat2[0])
        
        ans = [[0]*mat2_cols for _ in range(mat1_rows)]
        
        for i in range(mat1_rows):
            #innder dimension
            for k in range(mat1_cols):
                if mat1[i][k]:
                    for j in range(mat2_cols):
                        ans[i][j] += mat1[i][k]*mat2[k][j]
        
        return ans

#compression
class Solution:
    def multiply(self, mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
        '''
        follow up, is that mat1 and mat2 are spar matrices, so we want to be efficient with storage
        we can stor the matrix as a list of lists 
        at each ith row
        we store values as pairs (value, column index)
        row i : [(value,j)....(value_n,j_n)] if there are values
        
        we first compress each matrix
        then carry out the multiplation, note, since we are only adding in non zero elements during compression
        we don't need to check for non zero elements during the multiplcation 
        so we can carry out the multiplacation normally
        
        '''
        def compress(mat):
            rows = len(mat)
            cols = len(mat[0])
            
            compressed = [[] for _ in range(rows)]
            
            for row in range(rows):
                for col in range(cols):
                    if mat[row][col]:
                        compressed[row].append([mat[row][col],col])
            
            return compressed
        
        m = len(mat1)
        k = len(mat1[0])
        n = len(mat2[0])
        
        # Store the non-zero values of each matrix.
        A = compress(mat1)
        B = compress(mat2)
        
        ans = [[0] * n for _ in range(m)]
        
        for mat1_row in range(m):
            for element1,mat1_col in A[mat1_row]:
                for element2,mat2_col in B[mat1_col]:
                    ans[mat1_row][mat2_col] += element1*element2
        
        return ans

##################################################
# 1663. Smallest String With A Given Numeric Value
# 22MAR22
###################################################
#repeat
class Solution:
    def getSmallestString(self, n: int, k: int) -> str:
        '''
        we can only use chars a to z, and they have values/scores in range [1,26]
        we want to return the lexicogpahilly smallest string of length n
        
        if we think greedily it makes sense to use the smallest chars first
        it's really just going to be some number of a's, middle chars, and some number of z's
        
        we can start off with all a's then just try to max it the last one if we can
        '''
        mapp = {}
        for i in range(26):
            mapp[i] = chr(ord('a')+i)
        
        #initially start with all a scores, or just zeros
        ans = [0]*n
        #we've at least use up n from k
        k -= n
        right = len(ans) - 1
        
        while k >= 0 and right >=0:
            #try using up all of k
            if k > 25:
                ans[right] = 25
                k -= 25
            else:
                #use what's left
                ans[right] = k
                k = 0
            right -= 1
        
        return "".join([mapp[num] for num in ans])
        
class Solution:
    def getSmallestString(self, n: int, k: int) -> str:
        '''
        going left to right, we need to keep track of the smallest k we can use
        and also how many spots are left
        going left to right, we always to to make it the smallest it can be by using a and reserve the last spot for z
        '''
        result = ""
        for i in range(n):
            #get remainng sports
            spots_left = n - i - 1
            #if current k is > the spots_left*26, use the larget you can
            if k > spots_left*26:
                score = k -  (spots_left*26)
                k -= score
                result += chr(ord('a') + score - 1)
            #if its less, well just use a
            else:
                result += 'a'
                k -= 1
        
        return result

##################################
# 313. Super Ugly Number
# 22MAR22
##################################
#close one....
class Solution:
    def nthSuperUglyNumber(self, n: int, primes: List[int]) -> int:
        '''
        we define a super ugly number whose prime factors are in the array of primes
        return the nth super ugly number
        
        for each prime p, to a list of n*p, or rather its multiples, but how far should i go?
        i can generat the super ugly numbers in order, since the primes p are in order
        
        first try generating candidates
        '''
        candidates = []
        N = len(primes)
        
        #i need to keep track of the current multiple that i'm on for each prime
        primes = [[p,1] for p in primes]
        start = 0 #this gets +1 % N
        
        while len(candidates) < n:
            curr_prime,curr_multiple = primes[start]
            next_prime,next_multiple = primes[(start+1) % N]
            while curr_prime*curr_multiple < next_prime*next_multiple:
                cand = curr_prime*curr_multiple
                candidates.append(cand)
                curr_multiple += 1
            #update curr multiple
            primes[start][1] = curr_multiple
            start = (start + 1) % N
            
        print(candidates)
            