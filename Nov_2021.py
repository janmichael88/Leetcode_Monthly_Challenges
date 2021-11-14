#####################################
# 01NOV21
# 130. Surrounded Regions
#####################################
#fuck yeah
#dfs solution
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        '''
        this is simlar to capturing islands
        note: surrounded regions should not be on the border
        dfs from all the O's on the borders and, since these cannot be captured, add  the zeros to the non capture group
        then re traverse the grid and if an O is not in the capture group change it
        '''
        rows = len(board)
        cols = len(board[0])
        
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        non_capture_zeros = set()
        
        def dfs(r,c):
            #add to non_capture
            non_capture_zeros.add((r,c))
            for dx,dy in dirrs:
                neigh_r = r + dx
                neigh_c = c + dy
                #bounds check
                if 0 <= neigh_r < rows and 0 <= neigh_c < cols:
                    #is an 'O'
                    if board[neigh_r][neigh_c] == 'O': 
                        #is not seen
                        if (neigh_r,neigh_c) not in non_capture_zeros:
                            dfs(neigh_r,neigh_c)
            return
                        
        #now dfs along borders
        #first row
        for c in range(cols):
            if board[0][c] == "O":
                dfs(0,c)
                
        #first col
        for r in range(rows):
            if board[r][0] == "O":
                dfs(r,0)
        
        #last row
        for c in range(cols):
            if board[rows-1][c] == 'O':
                dfs(rows-1,c)
        #last col
        for r in range(rows):
            if board[r][cols-1] == 'O':
                dfs(r,cols-1)
                
        #now pass board and make sure this O is a non capture
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == 'O':
                    if (r,c) not in non_capture_zeros:
                        board[r][c] = 'X'
                        
        return board
        print(non_capture_zeros)

#bfs
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        '''
        this is simlar to capturing islands
        note: surrounded regions should not be on the border
        dfs from all the O's on the borders and, since these cannot be captured, add  the zeros to the non capture group
        then re traverse the grid and if an O is not in the capture group change it
        '''
        rows = len(board)
        cols = len(board[0])
        
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        non_capture_zeros = set()
        
        def bfs(r,c):
            #add to non_capture
            non_capture_zeros.add((r,c))
            q = deque([(r,c)])
            while q:
                curr_r,curr_c = q.popleft()
                for dx,dy in dirrs:
                    neigh_r = curr_r + dx
                    neigh_c = curr_c + dy
                    #bounds check
                    if 0 <= neigh_r < rows and 0 <= neigh_c < cols:
                        #is an 'O'
                        if board[neigh_r][neigh_c] == 'O': 
                            #is not seen
                            if (neigh_r,neigh_c) not in non_capture_zeros:
                                q.append((neigh_r,neigh_c))
                                non_capture_zeros.add((neigh_r,neigh_c))
                        
        #now dfs along borders
        #first row
        for c in range(cols):
            if board[0][c] == "O":
                bfs(0,c)
                
        #first col
        for r in range(rows):
            if board[r][0] == "O":
                bfs(r,0)
        
        #last row
        for c in range(cols):
            if board[rows-1][c] == 'O':
                bfs(rows-1,c)
        #last col
        for r in range(rows):
            if board[r][cols-1] == 'O':
                bfs(r,cols-1)
                
        #now pass board and make sure this O is a non capture
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == 'O':
                    if (r,c) not in non_capture_zeros:
                        board[r][c] = 'X'
                        
        return board
        print(non_capture_zeros)

############################
# 02NOV21
# 980. Unique Paths III
############################
class Solution:
    def uniquePathsIII(self, grid: List[List[int]]) -> int:
        '''
        dfs to build all possible paths from target to source, making sure we only touch a square only once
        this is a backtracking problem
        
        first identify start and end points, and also count up empty spaces
        when we dfs pass in curr row and curr col as well as the number of empty spaces taken up
        if its the end and empty count matches, we have a path
        dfs on in bounds and empty and no obstacel cells
        then bactrack
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        start_r,start_c = 0,0
        end_r,end_c= 0,0
        
        empty = 0
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    start_r,start_c = i,j
                elif grid[i][j] == 2:
                    end_r,end_c = i,j
                elif grid[i][j] == 0:
                    empty += 1
        
        self.num_paths = 0
        #backtracking and dfs
        visited = set()
        def dfs(row,col,visited,walk): #walk counts up the number of empty squares we have hit
            #to end our recurion
            if row == end_r and col == end_c:
                #but also make sure we walked all squares including the end
                if walk == empty + 1:
                    self.num_paths += 1
                return
            #if we aren't here we can recurse
            #constraints, in bounds, not and obstalce and not visited
            if 0<= row < rows and 0 <= col <cols and grid[row][col] != -1 and (row,col) not in visited:
                #first add
                visited.add((row,col))
                #now dfs
                for i,j in [(0,1),(0,-1),(1,0),(-1,0)]:
                    dfs(row+i,col+j,visited,walk+1)
                #backtrack
                visited.remove((row,col))
            
        #invoke
        dfs(start_r,start_c,visited,0)
        return self.num_paths

################################
# 03NOV21
# 129. Sum Root to Leaf Numbers
################################
#recursive
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        '''
        i could generate all paths as a string,
        convert strings to int and add
        better way is to conert on the flow
        '''
        self.sum_paths = 0
        
        def dfs(node,curr_number):
            if node:
                curr_number = curr_number*10 + node.val
                if not node.left and not node.right:
                    self.sum_paths += curr_number
                
                dfs(node.left,curr_number)
                dfs(node.right,curr_number)
        
        dfs(root,0)
        return self.sum_paths

#iterative
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        '''
        i could generate all paths as a string,
        convert strings to int and add
        better way is to conert on the flow
        '''
        sum_paths = 0
        
        stack = [(root,0)]
        
        while stack:
            node, curr_number = stack.pop()
            if node:
                curr_number = curr_number*10 + node.val
                if not node.left and not node.right:
                    sum_paths += curr_number
            if node.left:
                stack.append((node.left,curr_number))
            if node.right:
                stack.append((node.right,curr_number))
            
        return sum_paths

##########################
# 04NOV21
# 404. Sum of Left Leaves
##########################
class Solution:
    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        '''
        its a left leaf only if i had gone left before
        just pass in gone left in call
        '''
        self.sum = 0
        
        def dfs(node,went_left):
            if not node:
                return
            if not node.left and not node.right:
                if went_left:
                    self.sum += node.val
                
                return
            
            dfs(node.left, True)
            dfs(node.right,False)
            
    
        dfs(root,False)
        return self.sum
            
#terative with stack
class Solution:
    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        '''
        can also so iterativley using stack
        '''
        ans = 0
        
        
        stack = [(root,False)]
        
        while stack:
            curr, went_left = stack.pop()
            
            if not curr:
                continue
            if not curr.left and not curr.right:
                if went_left:
                    ans += curr.val
                    
            if curr.left:
                stack.append((curr.left,True))
            
            if curr.right:
                stack.append((curr.right,False))

        return ans

class Solution:
    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        '''
        another recursive way but no global update
        return sum of problesm
        '''
        if not root:
            return root
        
        def dfs(node,went_left):
            if not node:
                return 0
            if not node.left and not node.right:
                if went_left:
                    return node.val
            
            return dfs(node.left,True) + dfs(node.right,False)
        
        return dfs(root, False)

#########################
# 441. Arranging Coins
# 05Nov21
#########################
#welp, it passes
class Solution:
    def arrangeCoins(self, n: int) -> int:
        '''
        i could just subtract each the number of coins for each level,
        when i can't thats' the level
        '''
        level = 1
        coins = 1
        
        n -= 1
        while n > coins:
            level += 1
            coins += 1
            n -= coins
        
        return level
            
#how about the binary search solution
class Solution:
    def arrangeCoins(self, n: int) -> int:
        '''
        i know that if i'm at the kth level, 
        i need k(k+1) / 2 coins
        the number of coins are in the range [1,2**31 -1]
        so we want to find the maximum k, such k(k+1) / 2 <= N
        we can solve this using binary serach sine the function k(k+1) /2 is increaing
        '''
        left = 1
        right = n
        
        while left <= right:
            level = (left + (right - left) // 2)
            coins = level*(level+1) // 2
            if coins == n:
                return level
            if coins > n:
                right  = level - 1
            else:
                left = level + 1
        
        return right
        

#O(1), just use math
class Solution:
    def arrangeCoins(self, n: int) -> int:
        return int(((2 * n + 0.25)**0.5 - 0.5))


#########################
# 260. Single Number III
# 06NOV21
########################
class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
        '''
        naive way is to just count
        '''
        counts = Counter(nums)
        ans = []
        for k,v in counts.items():
            if v == 1:
                ans.append(k)
        return ans


class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
        '''
        it's a bit mask problem
        bitmask ^= x, sets bith with the num
        just xor it
        second occurence of x clears it
        x  & (-x) isloates right most bit
        
        so lets create a bistmask: bitmaxk ^x, and since some elemnts are written two, the
        mask will net keep it
        '''
        #xor all nums to get bist maks, remember only two elements have muliplict 1
        bitmask = 0
        for num in nums:
            bitmask ^= num
            
        #now find the right most bit
        diff = bitmask & (-bitmask)
        
        first = 0
        for num in nums:
            if num & diff:
                first ^= num
        
        #get second one
        second = bitmask^first
        return [first, second]

########################
# 01NOV21
# 1231. Divide Chocolate
########################
class Solution:
    def maximizeSweetness(self, sweetness: List[int], k: int) -> int:
        '''
        turns out this happens to be a binary search problem
        there is a standard template for binary search problems like these
        
        instea of trying all possible k-1 cuts, try all possible minimum sweet vaulue
        first try to cut using the best possible min value, or 1 more than
        example,
            let x = sum(sweetness) + 1
            check if we can cut the bar into k+1 values having x - 1 as sweetness
            if it works, well we can go to x - 2, then x -3, then x - 4....
        
        now we have two new qustions, given a min sweetness, how to check if k+1 cutting exists
        is there a more efficient way to find the max workable value x
        
        how to check if cutting plan exists?
            go piece by piece until we get to x, then make a new piece
            two ending conditons:
                reach end of bar, and everyone has a piece with sweetness no less than x
                if we reached the ending position, not everyone has gotten a pice, so this x cannot be true
                if we go to the second, we know this x works, so we can go lower
        
        now, is there a better way to search fo the best optimal value
        intuion, if a cutting plan with a min value exists, then there also exists a cutting plan with min value of x -1
        i.e if 5 works, 4 is guaranteed to work
        
        so once we find a cutting plan with a minimum workable value of x there are two scenarios
            1. there exists a piece with sweentess of exactly x
            2. there does not exist piece with an exact sweetness of x, but a workable value exists larger than x
            
        DIVISION of workable and un workable values == BINARY SEARCh
        
        what is the search space?
            left is the min(sweetness), right is sum(sweetness) / (k+1), i.e the largest possible sweetness we can get for ourselves
            mid = (left + right + 1) / 2
            If we can successfully cut the chocolate into k + 1 pieces and every piece has a sweetness no less than mid, this means mid is a workable value and that there may exist a larger workable value. 
            
        NOTE? Why do we use mid = (left + right + 1) / 2 instead of mid = (left + right) . 2
        special case, it would looop foever
        
        algo;
            1. setup two boundaries left and right
            2. git middle
            3. check if we can gut into k+1 pieces with no sweetness less than mid, where mid is workable value
            4. if cutting the chocolate bar in this method results in everyone recieing a piece at elast mid, then left == mid, other wisse, right = mid - 1
            5. reapat, 
            6. return left or right as answer
            
        '''
        num_people = k + 1
        left = min(sweetness)
        right = sum(sweetness) // num_people
        
        while left < right:
            #get middle, curr sweetness is this peron's current bar
            #make sure we have at least k+1
            mid = (left + right + 1) //2
            cur_sweetness = 0
            people_with_chocolate = 0
            for s in sweetness:
                cur_sweetness += s
                
                #new cut?
                if cur_sweetness >= mid:
                    people_with_chocolate += 1
                    cur_sweetness = 0
            #can we do it
            if people_with_chocolate >= num_people:
                left = mid
            else:
                right = mid - 1
        
        return right

#########################
# 07NOV21
# 43. Multiply Strings
#########################
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        '''
        we are not allowed to convert inputs direcly
        so we can just use elementary math
        consider the the nums 123*456
        123*(6 + 50 + 400)
        (123*6) + 123(*50) + 123*(400)
        123*6 + 123*(5*10) + 123*4*100
        
        \sum (firstNumber*(jth digit ot second Number)*10^(index of j digit counting from end))
        
        we need to start from the last position for both strings when multiplying
        we can just reverse both nums
        
        for each digit in nums2, we multiply by num1 and get an inntermediate result
        call this currResult, which will be storing in a list
        
        algo:
            1. reverse both numbers
            2. for each digit in secondNumber:
                keep carry variable set to0
                init currResult array beginning with appropriate number of zeros accoroding tot he place of the secondNUmber diidgt
                for each digit in secondNUmber
                    multiplye secondNumbers diigt firstnumbers digit and add carry
                    take reminder of with 10 as arry
                    append lest digit to currresult
                    divide multiplication by 10 to get the new value for carry
                append remaining value for carry if any
                push the curr resutls into the results array
            3. compute the cum sum overall
            4. reverse and return ans
        '''
        #special cases
        if num1 == '0' or num2 == '0':
            return '0'
        
        #reverse numbers
        first_num = num1[::-1]
        second_num = num2[::-1]

        #for each digit in second number, multupky by first number digit, then store the multiplation result
        #reverse in res aray
        res = []
        for i,digit in enumerate(second_num):
            res.append(self.multiply_one(digit,i,first_num))
            
        #sum all
        ans = self.sum_results(res)
        
        # Reverse answer and join the digits to get the final answer.
        return ''.join(str(digit) for digit in reversed(ans))
            
        print(ans)
            
    def multiply_one(self,digit2,num_zeros,first_num):
        curr_res = [0]*num_zeros
        carry = 0
        for digit1 in first_num:
            mult = int(digit1)*int(digit2) + carry
            #get new carry
            carry = mult // 10
            #append last digit
            curr_res.append(mult % 10)
        
        #final carry
        if carry != 0:
            curr_res.append(carry)
        return curr_res
    
    def sum_results(self, results: List[List[int]]) -> List[int]:
        # Initialize answer as a number from results.
        answer = results.pop()

        # Add each result to answer one at a time.
        for result in results:
            new_answer = []
            carry = 0

            # Sum each digit from answer and result. Note: zip_longest is the
            # same as zip, except that it pads the shorter list with fillvalue.
            for digit1, digit2 in zip_longest(result, answer, fillvalue=0):
                # Add current digit from both numbers.
                curr_sum = digit1 + digit2 + carry
                # Set carry equal to the tens place digit of curr_sum.
                carry = curr_sum // 10
                # Append the ones place digit of curr_sum to the new answer.
                new_answer.append(curr_sum % 10)

            if carry != 0:
                new_answer.append(carry)

            # Update answer to new_answer which equals answer + result
            answer = new_answer

        return answer

#saving space
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        '''
        we can save space by first getting the results array hoding at least M + N digits
        the proof is ugly
        then same as before
        note,
        M+N >= (M*N)
        Algorithm

        Reverse both numbers.
        Initialize ans array with (N+M)(N+M) zeros.
        For each digit in secondNumber:
        Keep a carry variable, initially equal to 0.
        Initialize an array (currentResult) that begins with some zeros based on the place of the digit in secondNumber.
        For each digit of firstNumber:
        Multiply secondNumber's digit and firstNumber's digit and add previous carry to the multiplication.
        Take the remainder of multiplication with 10 to get the last digit.
        Append the last digit to currentResult array.
        Divide the multiplication by 10 to obtain the new value for carry.
        After iterating over each digit in the first number, if carry is not zero, append carry to the currentResult.
        Add currentResult to the ans.
        If the last digit in ans is zero, before reversing ans, we must pop the zero from ans. Otherwise, there would be a leading zero in the final answer.
        Reverse ans and return it.
        '''
        if num1 == "0" or num2 == "0": 
            return "0"
        
        # Reverse both numbers.
        first_number = num1[::-1]
        second_number = num2[::-1]
        
        # To store the multiplication result of each digit of secondNumber with firstNumber.
        N = len(first_number) + len(second_number)
        answer = [0] * N

        # Multiply each digit in second_number by the first_number
        # and add each result to answer
        for index, digit in enumerate(second_number):
            answer = self.addStrings(self.multiplyOneDigit(first_number, digit, index), answer)

        # Pop excess zero from the end of answer (if any).
        if answer[-1] == 0:
            answer.pop()

        # Ans is in the reversed order.
        # Reverse it to get the final answer.
        answer.reverse()
        return ''.join(str(digit) for digit in answer)
    
    def multiplyOneDigit(self, first_number: str, digit2: str, num_zeros: int):
        # Insert 0s at the beginning based on the current digit's place.
        currentResult = [0] * num_zeros
        carry = 0

        # Multiply firstNumber with the current digit of secondNumber.
        for digit1 in first_number:
            multiplication = int(digit1) * int(digit2) + carry
            # Set carry equal to the tens place digit of multiplication.
            carry = multiplication // 10
            # Append the ones place digit of multiplication to the current result.
            currentResult.append(multiplication % 10)

        if carry != 0:
            currentResult.append(carry)
        return currentResult
    
    def addStrings(self, result: list, answer: list) -> list:
        carry = 0
        i = 0
        new_answer = []
        for digit1, digit2 in zip_longest(result, answer, fillvalue=0):
            # Add current digits of both numbers.
            curr_sum = digit1 + digit2 + carry
            carry = curr_sum // 10
            # Append last digit of curr_sum to the answer.
            new_answer.append(curr_sum % 10)
            i += 1
        
        return new_answer

############################
# 96. Unique Binary Search Trees
# 08NOV21
############################
#recusive 
class Solution:
    def numTrees(self, n: int) -> int:
        '''
        this is just the catalan number
        \sum_{i}^{n} i(n-i)
        each element i can be a root, then we have 1,i-1 elements on right, and i+1,n on the left
        for the number of unique trees at i, i need the number of unique trees for i-1 and i+1
        then mulitply them
        '''
        memo = {}
        
        def rec(n):
            #base case
            if (n == 0) or (n==1):
                return 1
            #retrieve
            if n in memo:
                return memo[n]
            ans = 0
            for i in range(n):
                left = rec(n-i-1)
                right = rec(i)
                ans += left*right
            
            memo[n] = ans
            return ans
        
        return rec(n)

#iterative
class Solution:
    def numTrees(self, n: int) -> int:
        '''
        now just translate bottom up
        '''
        dp = [0]*(n+1)
        dp[0] = 1
        
        for node in range(1,n+1):
            #markt the left side bound
            for i in range(node):
                left = dp[node - 1 - i]
                right = dp[i]
                dp[node] += left*right
        
        return dp[-1]

#catlan closure number
class Solution:
    def numTrees(self, n: int) -> int:
        '''
        C_{0} = 1
        C{n+1} = (2(2n + 1) / (n+2))*C_{n}
        '''
        C = 1
        for i in range(0, n):
            C = C * 2*(2*i+1)/(i+2)
        return int(C)

###########################
# 1178. Number of Valid Words for Each Puzzle
# 09NOV21
###########################
#TLE
class Solution:
    def findNumOfValidWords(self, words: List[str], puzzles: List[str]) -> List[int]:
        '''
        a word in words is valid if:
            word containss the first letter of the pusszle (could be in puzzles)
            for each letter in word, that letter in in pusszle
        we want the number of valid words for each puzzle i
        brute force check maybe?
        #first pass puzzles and set each one
        #then check conditoins for eacch word
        '''
        N = len(words)
        M = len(puzzles)
        
        ans = [0]*(M)
        
        #set puzzle
        set_puzzles = [set(p) for p in puzzles]
        #set words
        set_words = [set(w) for w in words]
        
        for i in range(M):
            count = 0
            for word,chars in zip(words,set_words):
                if set(puzzles[i][0]).issubset(chars) and chars.issubset(set_puzzles[i]):
                    #print(word,puzzles[i])
                    count += 1
            
            ans[i] = count
        
        return ans


#hashing and frozen sets, would be faster using bit masking and Trie
class Solution:
    def findNumOfValidWords(self, words: List[str], puzzles: List[str]) -> List[int]:
        '''
        we can also use frozenses to answer this question
        part 1:
            for each w in words, calculate the set of words letters
            count the number of different sets
        part 2:
            for each p in puzzles, get all subsets using p as first
            for each set if subpuzzle, finds its occruence
            sum up and append answer
        '''
        #stored state of each word as frozenset with occurnce of 1
        count = collections.Counter(frozenset(w) for w in words)
        res = []
        for p in puzzles:
            subs = [p[0]]
            for c in p[1:]:
                subs += [s + c for s in subs]
            #print(subs)
            #this prings all combinations of puzzle start wit its first char
            curr_count = 0
            for s in subs:
                curr_count += count[frozenset(s)]
            
            #append to answer for each puzzle
            res.append(curr_count)
        
        return res

#hashing and bitmasks
class Solution:
    def findNumOfValidWords(self, words: List[str], puzzles: List[str]) -> List[int]:
        '''
        using bit masking, notice from the constraints that we are given more puzzzle than words
        but the words are longer than the puzzles
        brute force owule be to check each word
            and for eah word check first letter in pusszle
            and puzzle contains every letter in word
            O(len(words)*len(puzzles))
            we now know we must use hasing or Trie
        we need to count number of words matching a puzzle
        where the value stores the number of strings in the bin
        in python we can use native structure called frozenset as keys in dict, but for other languages
        we would need to write a custom hasing function
        BITMASK as the key
        remember target work should contain first letter in puzzle and cannt contain any letters that are not in puzzle
        challenge:
            how to iterate over all the subsets of a set, DFS
            use the mask trick to find all possible subsets for a given mask
            use subset = (subset - 1) & bitmask to ensure each subset only contains chars that exists in bit mask
        intution:
            iteralte over all subsets of letters contains in puzzle that also contain first letter of puzzle
            then for each subset, add number of words that match the subset to the count of valid words for curr_puzzle
        algo:
            1. build map
                for word in words
                    transform into bitmask fo chars
                    if bitmakse has not eebn sen before, store occurnece, it is has been seen before increment by 1
            2. count number of valid words for each puzzle
                for each puzzle
                    get bitmask
                    iterat over every possible submask contains the first letter in puzzle
                    a word is valid for a puzzle if bit makse matches one of the puzzles submasks
                    for eachsubmaks found, increment count by 1
        '''
        def bitmask(word):
            mask = 0
            for letter in word:
                mask |= 1 << (ord(letter) - ord('a'))
            return mask
        
        #get counts
        word_counts = Counter(bitmask(word) for word in words)
        
        result = []
        for puzzle in puzzles:
            #first letter
            first = 1 << (ord(puzzle[0]) - ord('a'))
            #get its count
            count = word_counts[first]
            #now use the remaing of the words
            mask = bitmask(puzzle[1:])
            #iterate over every possible subset of char
            submask = mask
            while submask:
                #increment count by the number of words that matach
                count += word_counts[submask | first]
                submask = (submask - 1) & mask
            result.append(count)
        return result

###################################
# 425. Word Squares
# 08NOV21
###################################
#hashtable
class Solution:
    def wordSquares(self, words: List[str]) -> List[List[str]]:
        '''
        i word square is an n x n matrix such that evey row and every col is a word in words
        we know that this matrix must by symmetrical
        if we know the upper right part, we could get its lower part
        
        intuion: backtracking
            construct the word square row by row top to bottom
            place a word in a row and see if we can add another
            keep going until we can
            backtrack once we can't
            take advantage of the symmextric property:
                example if i place the word ball
                second row word must start with a
                third row word must start with l....and so on
                
        we can use hashtable for fast lookup
        
        '''
        N = len(words[0])
        #build prefix hashtable
        prefixHash = {}
        for word in words:
            for i in range(1,len(word)+1):
                prefixHash.setdefault(word[:i],set()).add(word)
        
        #helper function to find word withs ppref
        def getWords(pref):
            if pref in prefixHash:
                return prefixHash[pref]
            else:
                return set([])
        
        results = []
        word_squares = []
        #define backtracking
        def backtrack(step,word_squares,results):
            if step == N:
                #build up candidates
                results.append(word_squares[:])
                return
            
            pref = "".join([word[step] for word in word_squares])
            for cand in getWords(pref):
                word_squares.append(cand)
                backtrack(step+1,word_squares,results)
                word_squares.pop()
        
        for word in words:
            word_squares = [word]
            backtrack(1,word_squares,results)
        return results

#trie
class Solution:
    def wordSquares(self, words: List[str]) -> List[List[str]]:
        '''
        we could laso use a trie
        when building the trie, store index to to work to mark completion of that word
        but more so, to show so far this this pref belongs to this wordd
        '''
        N = len(words[0])
        
        trie = {}
        
        results = []
        word_squares = []
        
        for i,word in enumerate(words):
            node = trie
            for char in word:
                if char in node:
                    node = node[char]
                else:
                    newNode = {}
                    newNode["#"] = []
                    node[char] = newNode
                    node = newNode
                #mark with index
                node['#'].append(i)
        
        #helper function find words with pref
        def getWords(pref):
            node = trie
            for char in pref:
                if char not in pref:
                    return []
                node = node[char]
            
            #find indes at this node
            return [words[i] for i in node['#']]
        
        def backtracking(step, word_squares, results):
            if step == N:
                results.append(word_squares[:])
                return

            prefix = ''.join([word[step] for word in word_squares])
            for candidate in getWords(prefix):
                word_squares.append(candidate)
                backtracking(step+1, word_squares, results)
                word_squares.pop()
                
        for word in words:
            word_squares = [word]
            backtracking(1,word_squares,results)
            word_squares.pop()
        
        return results

########################
# 10Nov21
# 122. Best Time to Buy and Sell Stock II
#########################
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        '''
        no cool down, we can buy immediatley after selling
        we can sell immediately after buying
        only add profit if there is a difference between i and i+1
        '''
        N = len(prices)
        profit = 0
        for i in range(N-1):
            if prices[i+1] - prices[i] > 0:
                profit += prices[i+1] - prices[i]
        
        return profit
        
#peak and valley approach
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        '''
        find first valley
        then find first peak
        '''
        n = len(prices)
        i = 0
        valley = prices[0]
        peak = prices[0]
        maxprof = 0
        while i < n - 1:
            #find valley
            while (i < n-1) and (prices[i] >= prices[i+1]):
                i += 1
            valley = prices[i]
            #find peak
            while (i < n-1) and (prices[i] <= prices[i+1]):
                i += 1
            peak = prices[i]
            maxprof += peak - valley
        
        return maxprof

######################################################
# 11Nov21
# 1413. Minimum Value to Get Positive Step by Step Sum
######################################################
class Solution:
    def minStartValue(self, nums: List[int]) -> int:
        '''
        we we want to find the minimum start value such that the the step by step sum is never less than 1
        [-3,2,-3,4,2]
        get initial differences
        [-1,-1,1,6]
        len(nums) is also very small, and the range for each int is [-100,100]
        step by steo sum is a rolling sum
        you could brute force start with 1 and check
        then incrementally add 1 to a start value until we cant
        '''
        min_start = 1
        while True:
            total = min_start
            is_valid = True
            
            for num in nums:
                total += num
                #check
                if total < 1:
                    is_valid = False
                    break
            
            if is_valid:
                return min_start
            else:
                min_start += 1

#binary search
class Solution:
    def minStartValue(self, nums: List[int]) -> int:
        '''
        side note:
            imagine the case [1,1,1,1....-m,-m,-m]
            its step by step sum is
            n = len(nums)
            (n//2) + (n//2 + 1)(-m)
            (n//2) + (n//2)*(-m) - m
            (n//2)(1 - m) - m
        we can use binary search to find min start value
        left = 1
        right should be larger enough to make sure each step by step sum is > 1
        we want the largest workable value times the length of nums
        100*len(nums)+1
        '''
        n = len(nums)
        m = 100
        
        left = 1
        right = n*m+1
        
        #we want to found lower bound for workable value, 
        while left < right:
            mid = left + (right - left) //2
            total = mid
            is_valid = True
            
            for num in nums:
                total += num
                #check
                if total < 1:
                    is_valid = False
                    break
            
            if is_valid:
                #if this value works, anything after mid should work, no need to look there
                right = mid
            else:
                # we need to keep up more
                left = mid + 1
        #return the lower bound after finising, we dont want to set the invariant to left <= right
        #otherwise left would have moved up and we'd be wrong
        return left 

#pref sum
class Solution:
    def minStartValue(self, nums: List[int]) -> int:
        '''
        if i get pref sums for each num
        i know that that sum so far must be greater than 1
        suppose we have the array [a,b,c,d]
        prefixes would be
        [0,a,a+b,a+b+c,a+b+c+d]
        key: the min start value is the value that make the minimum element equal exactly 1
        if at any point our prefsum goes below 1, we need to negate this and make it on1 more
        i.e for all pref in pref sum
        find its min, and set min = -x + 1
        minVal + startVal = 1
        startVal = 1 - minval
        so find the smallest pref sum
        as long as the smallest step by step sum is >= 1, every other step by step sum will be >= 1
        find the min start value , which is the value that makes min step by step > 1
        '''
        startValue = 0
        total = 0
        
        for num in nums:
            total += num
            startValue = min(startValue,total)
            
        # We have to change the minimum step-by-step total to 1, 
        # by increasing the startValue from 0 to -min_val + 1, 
        # which is just the minimum startValue we want.
        return -startValue + 1

#####################################
# 203. Remove Linked List Elements
# 12Nov21
#####################################
class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        '''
        loop through LL, and maintain curr and next pointers, also maitain curr and curr.next
        if node.val is val make .next connection
        #edge cases require dummy node to help out
        '''
        dummy = ListNode(-1)
        dummy.next = head
        
        prev = dummy
        curr = head
        
        while curr:
            if curr.val == val:
                prev.next = curr.next
            else:
                prev = curr
            curr = curr.next
        
        return dummy.next

######################
# 739. Daily Temperatures
# 13NOV21
######################
#brute force
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        '''
        we want to return an array dp where dp[i] is the num days we have to wait to get to wa 
        temp in the temps array that is wamemer
        brute  force would be to just  linear scan to the right for each temp in the array
        if im at temp[i] and i know the  temp[i+1] > temp
        and i have dp[i], then its  just one more
        
        i don't think  i can use dp on this one, the sub problem is not  repeatable
        and we want to return an array
        
        increasing  motonic stack?
        keep pushing on to stack  
        '''
        N = len(temperatures)
        ans  = [0]*N
        
        for i in range(N):
            for j in  range(i+1,N):
                if temperatures[j] > temperatures[i]:
                    ans[i] = j - i
                    break
        return ans
        

 #montonic stack
 class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        '''
        imagine the case we have that at the first day, the  temp that  is higher is on the last day
        then we would have to go through the whole array
        if we make sense of the fact that temps in desecnidng order can share  the  same answer day
        we can imporve time comp
        
        NOTE:
        montonic stacks are a good option when a problem ivoles comparing size of numeric eleemtns
        with their order being relevant
        
        for a day, there are two posssibiltes:
            1. curr day is not warmer than temp on top of stack, we push temp on to stack
            2. if curr day is warmer: it menas that curr day is the irst day with a warmer temp!
                when  we find a warmer temp, the number of  days is the diff between curr index  and index ono top of stack
         when we  can find a warmer temp, we  cant stop checking only after  one element
         we pop from the stack  until there is no longer a colder temp than current
        '''
        N = len(temperatures)
        ans  = [0]*N
        
        stack = []
        for curr_day,curr_temp in enumerate(temperatures):
            #if we have a stack and  the temp  curr_temp is  larger then what is at top
            #we  can  update
            while stack and temperatures[stack[-1]] < curr_temp:
                #update temp for  curr index  at top of stack
                prev_day = stack.pop()
                ans[prev_day] = curr_day - prev_day
            stack.append(curr_day)
        
        return ans
        









