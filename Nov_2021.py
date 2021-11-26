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

###############################
# 14NOV21
# 1286. Iterator for Combination
###############################
class CombinationIterator:

    def __init__(self, characters: str, combinationLength: int):
        '''
        number of combinations is N! / N
        i could just recursivley generate each one in order
        then use pointer to move
        '''
        self.combos = []
        N = len(characters)
        def rec(i,path):
            if len(path) == combinationLength:
                self.combos.append("".join(path[:]))
                return
            for j in range(i,N):
                path.append(characters[j])
                rec(j+1,path)
                path.pop()
        
        rec(0,[])
        self.num_combos = len(self.combos)
        self.curr_combo = 0

    def next(self) -> str:
        temp = self.curr_combo
        self.curr_combo += 1
        return self.combos[temp]

    def hasNext(self) -> bool:
        return self.curr_combo < self.num_combos

# Your CombinationIterator object will be instantiated and called as such:
# obj = CombinationIterator(characters, combinationLength)
# param_1 = obj.next()
# param_2 = obj.hasNext()

#precompute using algorithm L
class CombinationIterator:

    def __init__(self, characters: str, combinationLength: int):
        '''
        backtracking is risky because we can't control the flow
        we can use the bitmasking trick and subset generation to generate the next 
        combination in order
        the algo is affectionately called algorithm L, by Knuth
        if we have N elements, we have 2**n different masks
        only take bits that are set
        we take characters[i] if the n - 1 - ith bit is set
        trick: to test if ith bit in bitmask is set
        bitmask & (1 << i) != 0
        '''
        self.combs = []
        self.N = len(characters)
        self.k = combinationLength
        
        for bitmask in range(1 << self.N):
            if self.count_ones(bitmask) == self.k:
                #take chars
                curr = ""
                for j in range(self.N):
                    if bitmask & (1 << self.N - 1 - j):
                        curr += characters[j]
                
                self.combs.append(curr)
        #print(self.combs)
        #note this does in reverse
    
    def count_ones(self,num):
        count = 0
        while num > 0:
            if num & 1:
                count += 1
            num = num >> 1
        return count

    def next(self) -> str:
        return self.combs.pop()
        

    def hasNext(self) -> bool:
        return self.combs


#instead of precomputing, build one at a time
class CombinationIterator:

    def __init__(self, characters: str, combinationLength: int):
        '''
        start with highest bitmask 1 k times followed by 0 n-k times
        at each step generate a combinaton of the curr bistmask
        if n-1-jth bit is set, take it
        generate next mask, decrease until we hit k requirement
        
        '''
        self.n = len(characters)
        self.k = combinationLength
        self.chars = characters
        
        #generate first mask
        self.bitmask = (1 << self.n) - (1 << self.n -self.k)

    def next(self) -> str:
        #get curr combination
        curr = ""
        for j in range(self.n):
            if self.bitmask & (1 << self.n -j - 1):
                curr += self.chars[j]
        
        #get next mask
        self.bitmask -= 1
        while self.bitmask > 0 and self.count_ones(self.bitmask) != self.k:
            self.bitmask -= 1
        
        return curr
        

    def hasNext(self) -> bool:
        return self.bitmask > 0
    
    def count_ones(self,num):
        count = 0
        while num > 0:
            if num & 1:
                count += 1
            num = num >> 1
        return count

#################################
# 14NOV21
# 368. Largest Divisible Subset
################################
class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        '''
        basically we want the largest subset where each AN i,j pair can divide each other
        given a list of claues [E,F,G] sorted in increasing order, and this list already forms a divisible subset we can say two things:
            1. for any value that can be divided by the largest element in the divible subset, by adding the new value into the subset, one can form another subset for all h if h% G == 0 making [E,F,G,h]
            2. for all values that can divide the smallest element in the subset, by adding the new value into the subset, one cam form another divisible subset
            i.f for all d if E % d == 0, then [d,E,F,G]
        
        inutions, sort list increasingly to help enumerations
        claim 1:
            for an ordered list [X1,X2,...Xn] we claim that the largest divisible subset from this list is the largest subset amont all possible subsets ending with each number in the list
        define EDS function
        EDS(X_{i}), X_i is a subset of the list nums ending with X_i
        given list [2,4,7,8]
        EDS(4) = {2,4}
        EDS(2) = {2}
        
        lets call our target LDS
        LDS([X1,X2...XN]) = max(EDS(Xi) for all X_i)
        when we call EDS(8), we have alreayd called EDS for all elements less than 8
        to obtain EDS(8) we simply enumerate all elements before and their EDS
        
        '''
        nums.sort()
        subsets = {-1:set()}
        
        for num in nums:
            EDS = []
            for k in subsets:
                #if i can divide it
                if num % k == 0:
                    EDS.append(subsets[k])
            EDS = max(EDS,key=len)
            subsets[num] = EDS | {num}
        
        return list(max(subsets.values(),key = len))

#recursive
class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        '''
        if we define dp(i)
        this returns the largest subset in nums where the the subset ends in nums[i]
        we can further extend this by seeing if the max or min of dp(i) can be further divided
        a single number represents a divisible subset
        
        
        EDS(i) = max(EDS(j) for all nums[j] < nums[i])
        
        ours LDS for [X1,X2,X3...XN] = max(EDS(i) for i in range(1,n))
        [2,4,7,8]
        EDS(8) calls EDS(2), EDS(4),EDS(7)
        EDS(7) calls EDS(2) EDS(4)
        EDS(4) calls EDS(2)
        we store answers for each
        
        '''
        #corner case
        if len(nums) == 0:
            return []
        
        nums.sort()
        memo = {}
        
        def EDS(i):
            if i in memo:
                return memo[i]
            curr_num = nums[i]
            #trying to calc current maxSubset
            maxSubset = []
            #value of EDS(i) depends on prev elements
            for j in range(0,i):
                #corrloary 1
                if curr_num % nums[j] == 0:
                    #recuse
                    subset = EDS(j)
                    if len(maxSubset) < len(subset):
                        maxSubset = subset
            #extend the found max subset wut the curr_num
            maxSubset = maxSubset.copy()
            maxSubset.append(curr_num)
            memo[i] = maxSubset
            return maxSubset
        
        all_subsets = []
        for i in range(len(nums)):
            all_subsets.append(EDS(i))
        
        #print(all_subsets)
        #print(memo)
        
        return max(all_subsets,key=len)

#tabulation dp soltution
#similar to LIS
#https://leetcode.com/problems/largest-divisible-subset/discuss/684738/Python-Short-DP-with-O(n2)-explained-(update)
class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        '''
        we can treat this like LIS:
        say we have: nums = [4,5,8,12,16,20]
        
        sol[0] = [4], the biggest divisible subset has size 1.
        sol[1] = [5], because 5 % 4 != 0.
        sol[2] = [4,8], because 8 % 4 = 0.
        sol[3] = [4,12], because 12 % 4 = 0.
        sol[4] = [4,8,16], because 16 % 8 = 0 and 16 % 4 = 0 and we choose 8, because it has longer set.
        sol[5] = [4,20] (or [5,20] in fact, but it does not matter). We take [4,20], because it has the biggest length and when we see 5, we do not update it.
        Finally, answer is [4,8,16].
        '''
        if len(nums) == 0:
            return []
        
        nums.sort()
        
        dp = [[num] for num in nums] #each num is trivially a disivisble subset
        for i in range(len(nums)):
            #check everything up to i
            for j in range(i):
                if nums[i] % nums[j] == 0 and len(dp[i]) < len(dp[j]) + 1:
                    dp[i] = dp[j] + [nums[i]]
        
        print(dp)
        return max(dp,key = len)

##########################
# 15Nov21
# 1522. Diameter of N-Ary Tree
##########################
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children if children is not None else []
"""

class Solution:
    def diameter(self, root: 'Node') -> int:
        """
        :type root: 'Node'
        :rtype: int
        """
        '''
        from the binary tree problem just take the max of left and right and add 1
        but we need to do this for if childre
        but we want the two longest depths at level
        
        '''
        self.diameter = 0
        
        def dfs(node):
            if not node:
                return [0,0]
            depths = [0,0]
            if node.children:
                for child in node.children:
                    depths.append(dfs(child))
            
            self.diameter = max(self.diameter,sum(sorted(depths)[-2:]))
            return max(depths) + 1
        
        dfs(root)
        return self.diameter

#keeping track of top 2
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children if children is not None else []
"""

class Solution:
    def diameter(self, root: 'Node') -> int:
        """
        :type root: 'Node'
        :rtype: int
        """
        '''
        a couple of insights, first is that the longest path in a tree can only happend between two leaf nodes
        or between a lead node and the root
        second: each non leaf node acts as a brdige between the paths on its left and right
        combining them together gives a larger path
        for eaach node we want to find its two longest paths
        we could use height or depth for this one
        
        height:
            recall the height of a a node is defined as the length of the longest path downward from node to a leaf
            leaf nodes have height zero
            need combination of two longest paths
            if we are at a node (call it l with children m and n)
            height(node) = height(child m) + height(child n) + 2
        
        height(node) = max(height(child)) + 1 for all child in node.children
        we could keep all heightgs in an array and sort and take top 2
        or just ecord top 2
        '''
        self.diam = 0
        
        def height(node):
            if not node:
                return 0
            #find top two heights
            max_height_1 = 0
            max_height_2 = 0
            for child in node.children:
                parent_height = height(child) + 1
                if parent_height > max_height_1:
                    max_height_1, max_height_2 = parent_height, max_height_1
                elif parent_height > max_height_2:
                    max_height_2 = parent_height
            #diam update
            combined = max_height_1 + max_height_2
            self.diam = max(self.diam,combined)
            return max_height_1
        
        height(root)
        return self.diam

#############################
# 668. Kth Smallest Number in Multiplication Table
# 16NOV21
#############################
class Solution:
    def findKthNumber(self, m: int, n: int, k: int) -> int:
        '''
        we can use binary search to find the kth value in the array
        we define an enough function that is true only of there are k or more valuees in the table
        i.e it describes whether or not x is large enough to be the kth value
        how do we count whether are are at least k elements for each mid point
        if we are at the ith row, it looks like
        [i,2*i,3*i....n*i]
        the larggest possible k*i <= x 
        basically we are counting the number of elements at least x by using the number of elements in a row
        we inrease count by min(x//i,n)
        return count >= k (at leat k)
        '''
        def enough(x):
            count = 0
            for i in range(1,m+1):
                count += min(x // i, n)
            return count >= k
        
        lo,hi = 1,m*n
        #return the lower bound, in the case we cannot find exactly k
        #we don't want to cross over into hi, otherwise we don't get a lower bound
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if not enough(mid):
                #we need more element
                lo = mid + 1
            else:
                #we have more than enough elements so try smaller
                hi = mid
        
        return lo

##########################
# 62. Unique Paths
# 17NOV21
###########################
#recursion with memo
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        '''
        this is a classis dp problem
        dp(i,j) represents the number of ways to get to (i,j) from the start
        dp(i,j) = num ways from above + num ways from left (remember we can only go down and right)
        dp(i,j) = dp(i,j-1) + dp(i-1,j)
        base case, only on first row and col, there's only one way
        '''
        memo = {}
        
        def dp(i,j):
            if i == 1 or j == 1:
                return 1
            if (i,j) in memo:
                return memo[(i,j)]
            ans = dp(i,j-1) + dp(i-1,j)
            memo[(i,j)] = ans
            return ans

        return dp(m,n)

#iterative dp class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        '''
        iterative dp
        '''
        #dont forget to fill in bases cases
        dp = [[1]*n for _ in range(m)]
        
        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        
        return dp[-1][-1]

#math poly
from math import factorial
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        '''
        this is a classic combinatorial problem, there are h+v moves to from start to finish
        h = m- 1 horizontal moves and v = n-1 vertical moves
        # from start to destination, we need (m-1) ↓ moves and (n-1) → moves
        # Thus, the number of unique paths is the number of permutations of (m-1) ↓ and (n-1) →
        #
        # Number of unique paths = ( m-1 + n-1 ) ! / (m-1)! * (n-1)!
        
        
        '''
        return factorial( m+n-2 ) // ( factorial( m-1 ) * factorial( n-1 ) )    

###################################
# 238. Product of Array Except Self
# 17NOV21
###################################
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        '''
        just multiply all to get the product of everything in the first pas
        then pass through the array again dividing by the num
        the problem is with zeros, becaue that reduces product to 0
        keep track of indices that have zero
        '''
        all_prod = 1
        zero_idxs = set()
        
        for i in range(len(nums)):
            if nums[i] != 0:
                all_prod *= nums[i]
            else:
                zero_idxs.add(i)
        
        answer = []
        
        #in the case we don't have zeros
        if len(zero_idxs) == 0:

            for i in range(len(nums)):
                answer.append(all_prod // nums[i])
        
            return answer
        #we have zero
        else:
            for i in range(len(nums)):
                if i in zero_idxs:
                    answer.append(all_prod)
                else:
                    answer.append(0)
            return answer


class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        '''
        idea, if we are at the ith element, all we want is the product to its left side and the product to its right side
        then we just multiply these two to get the product
        we can generate in two pass the products to the left of each number
        and the products to the right of each number
        base case being 1 for the starting and ending elements
        '''
        N = len(nums)
        answer = [0]*N
        left = [0]*N
        right = [0]*N
        
        #for the first element, there is nothing, so its left product is 1
        left[0] = 1
        
        for i in range(1,N):
            left[i] = nums[i-1]*left[i-1]
        
        #get products to the right
        right[N-1] = 1
        for i in range(N-2,-1,-1):
            right[i] = nums[i+1]*right[i+1]
        
        #put into answer
        for i in range(N):
            answer[i] = left[i]*right[i]
        
        return(answer)

#using constant space
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        '''
        for constance space we can use the answer array to first start products to the left
        then on the second pass we find products to the right and put into answer
        '''
        N = len(nums)
        answer = [0]*N
        #for the first element, there is nothing, so its left product is 1
        answer[0] = 1
        
        for i in range(1,N):
            answer[i] = nums[i-1]*answer[i-1]
        
        #get products to the right
        #we need to hold the base case for products to the right
        right = 1
        for i in range(N-1,-1,-1):
            answer[i] = answer[i]*right
            right *= nums[i]
        
        
        return(answer)

################################################
# 17NOV21
# 448. Find All Numbers Disappeared in an Array
################################################
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        '''
        the numbers in the array can be [1,n]
        dumb way turn the numbers int a set
        '''
        n = len(nums)
        nums = set(nums)
        ans = []
        for num in range(1,n+1):
            if num not in nums:
                ans.append(num)
        
        return ans
        
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        '''
        since all the numbers are positive and are in beteween [1,n]
        we can mark the number as visited by going to its index and making it negative
        then pass the array once again and take indices whose num[at index] is > 0
        '''
        n = len(nums)
        
        for i in range(n):
            #finds where it should be in the array
            j = abs(nums[i]) - 1
            if nums[j] > 0:
                #negate it
                nums[j] *= -1
        
        #now check if a num is positive, then its index+1 was not seen
        ans = []
        for i in range(n):
            if nums[i] > 0:
                ans.append(i+1)
        return ans

#############################
# 461. Hamming Distance
# 18NOV21
#############################
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        '''
        just xor the two ti find where bits are different
        then count them up
        '''
        diff = x ^ y
        count = 0
        while diff:
            count += diff & 1
            diff = diff >> 1
        
        return count

#brian kernighan
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        '''
        recall the brian kernighan algo, to switch of last bit
        we can do evern faster than just shifiting over 1 bit at a time
        skip bits of zeros in between bits of 1
        i can turn off right most bit
        num & (num - 1) clears the right most bit
        '''
        diff = x ^ y
        distance = 0
        
        while diff:
            distance += 1
            diff = diff & (diff - 1)
        
        return distance

########################################
# 540. Single Element in a Sorted Array
# 20NOV21
#######################################
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        '''
        just count and return
        '''
        counts = Counter(nums)
        return [num for num,count in counts.items() if count == 1][0]
        
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        '''
        check every other element
        '''
        for i in range(0,len(nums)-1,2):
            if nums[i] != nums[i+1]:
                return nums[i]
        
        return nums[len(nums)-1]

#logN and constant space
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        '''
        recall each element is either present once or twice
        also notice that the starting array will alwasy be of odd length; 2*(some number of elemnte) + 1
        notice that if we have a subarray, the single element would appear only in the one that is odd length
        algo:
            1. keep dividing in half, and if we get to the point where reduced search space is 1, it must be that eleemtn
            2. at each iteration we need to determine the parity of the legnth of each sub array
            there are a few cases
                a. right side remains odd length, lo = mid + 2
                b. left side is odd length, hi = mid - 1
                c. mid's pair is to th left, and halves ar even, hi = mid - 2
                d. mid's pair if to the left, and havel are odd, lo = mid + 1
            
            return the lower bound
            note that this algo still works even if the array is not sorted
            invariant was the a side can always be odd or even lengthed
        notes on lo and hi bounds
        if you are returning from inside the loop, use left <= right
        if you are reducing the search space, use left < right and finally return a[left]
        
        if you discard mid for the next iteration (i.e. l = mid+1 or r = mid-1) then use while (l <= r).
        if you keep mid for the next iteration (i.e. l = mid or r = mid) then use while (l < r)
        '''
        lo = 0
        hi = len(nums) - 1
        
        while lo < hi:
            mid = lo + (hi - lo) // 2
            halves_even = (hi - mid) % 2 == 0
            if nums[mid+1] == nums[mid]:
                if halves_even:
                    lo = mid+ 2
                else:
                    hi = mid - 1
            elif nums[mid-1] == nums[mid]:
                if halves_even:
                    hi = mid - 2
                else:
                    lo = mid + 1
            else:
                return nums[mid]
        
        return nums[lo]

#binarys search for the right even index
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        '''
        it turns out that we can just binary search on even indices
        the single element if at the first even index not followed by a pair (from the first first linear seach idea)
        now in stead of doing a linear search to find this index, we use binary search to find this index
        after the single element, the pattern chnages to being odd indexes followed by their pair
        this means that the single element (an even index) and all other elements after it are even indices not followed by their pair
        therefoce, given any index in the array, we an easily determin whether the single elemnt is to the left or to the right
        
        algo:
            we can set lo and hi in the usual way
            we need to make sure our mid is even, so we can check its pair
            do this by decrementing 1 if odd
            then we check whether ot not the mid index is the same as the one after it
                if i is, we know that mid is not the single leement, and the singel element must be on the side afther this index
                if it is not, we know that the single element is at either mid or before mid
        '''
        lo = 0
        hi = len(nums) - 1
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if mid % 2 == 1:
                mid -= 1
            if nums[mid] == nums[mid + 1]:
                lo = mid + 2
            else:
                hi = mid
        return nums[lo]

########################
# 106. Construct Binary Tree from Inorder and Postorder Traversal
# 21NOV21
########################
#recursion but re scan in order array
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        '''
        in a post order traversal, the last element is the root
        for in order, if im at i, then anything from 0 to i-1 is on left substree 
        and anything from i+1 to N is ont the right subtree
        if i had pre order, then i would know what's root, left and right
        go in reverse of post order
        '''
        if not inorder:
            return None
        post_idx = [len(postorder) - 1]
        
        def helper(in_left,in_right):
            if in_left > in_right:
                return None
            #current root fomr post order
            root = TreeNode(postorder[post_idx[0]])
            #get ready for next call
            post_idx[0] -= 1
            if in_left == in_right:
                return root
            #find middle of in order using post_idx
            mid = inorder.index(root.val)
            #recursion right first then left
            root.right = helper(mid+1,in_right)
            root.left = helper(in_left,mid - 1)
            return root
        
        return helper(0,len(inorder)-1)

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        '''
        instead of having to re scan the inorder array, mapp the vals to their indices
        since the values will be uniqe
        '''
        if not inorder:
            return None
        
        mapp = {val:i for i,val in enumerate(inorder)}
        
        post_idx = [len(postorder) - 1]
        
        def helper(in_left,in_right):
            if in_left > in_right:
                return None
            #current root fomr post order
            root = TreeNode(postorder[post_idx[0]])
            #get ready for next call
            post_idx[0] -= 1
            if in_left == in_right:
                return root
            #find middle of in order using post_idx
            mid = mapp[root.val]
            #recursion right first then left
            root.right = helper(mid+1,in_right)
            root.left = helper(in_left,mid - 1)
            return root
        
        return helper(0,len(inorder)-1)
        
        print(mapp)

##################################
# 28. Implement strStr()
# 18NOV21
##################################
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        M = len(haystack)
        N = len(needle)
        
        for i in range(M-N+1):
            if haystack[i:i+N] == needle:
                return i
        return -1

#KMP search O(m+n)
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        '''
        we can use KMP to find substring match in O(n+m) time
        we first need to creat the longest prefix also suffix array 
        this answer, if we are at the ith index in the pattern then then there exsits a prefix of this length
        that is also a suffix
        
        then use the array to find the first occurrence of the needle in the haystack
        
        '''
        if len(needle) == 0:
            return 0
        
        #build longes prefix also suffix array
        lps = [0]*len(needle)
        i = 1
        j = 0
        
        while i < len(needle):
            if needle[i] == needle[j]:
                j += 1
                lps[i] = j
                i += 1
            else:
                if j != 0:
                    j = lps[j-1]
                else:
                    i += 1
        
        n = len(haystack)
        m = len(needle)
        i,j = 0,0
        while i < n:
            if haystack[i] == needle[j]:
                i += 1
                j += 1
                #if found solution
                if (j == m):
                    return i - m
            else:
                if j != 0:
                    j = lps[j-1]
                else:
                    i += 1
        
        return -1

###########################
# 450. Delete Node in a BST
# 21NOV21
###########################
#fuckkk
class Solution:
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        '''
        if the node we are at is a leaf, then just make that next pointer to none
        if the node has a left child, my the left the node
        if the node has a right children, make the right the curr node
        if left and right child, make right he node, and set node to the node that we deleted's left
        '''
        prev = None
        curr = root
        
        while curr:
            #on the left side
            if curr.val > key:
                prev = curr
                curr = curr.left
            #on the right
            elif curr.val < key:
                prev = curr
                curr = curr.right
            else:
                #must be the curr node
                #it's a leaf
                if not curr.left and not curr.right:
                    if prev.val > curr.val:
                        prev.left = None
                    else:
                        prev.right = None
                elif not curr.left

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        '''
        from a node, we can find its predecessor, the node in value just before the curr node
        from a node, we can find its successor, the node in value just after the curr node
        INORDER -> Left, Node, Right
        to find pred, go left once, then right as far as we can
        to find succ, go right once, then far lest as we can
        
        case:
            1 if its a leaf, delete it, node = null
            2. node has right, then repplace with succ, then recursively replace with succ
            3. node has a left, then its succ is somwhere in the upper tree, recursivley place with pred
            
                If key > root.val then delete the node to delete is in the right subtree root.right = deleteNode(root.right, key).

    If key < root.val then delete the node to delete is in the left subtree root.left = deleteNode(root.left, key).

    If key == root.val then the node to delete is right here. Let's do it :

    If the node is a leaf, the delete process is straightforward : root = null.

    If the node is not a leaf and has the right child, then replace the node value by a successor value root.val = successor.val, and then recursively delete the successor in the right subtree root.right = deleteNode(root.right, root.val).

    If the node is not a leaf and has only the left child, then replace the node value by a predecessor value root.val = predecessor.val, and then recursively delete the predecessor in the left subtree root.left = deleteNode(root.left, root.val).

    Return root.
        '''
        if not root:
            return None
        
        #delete from right subtree
        if key > root.val:
            root.right = self.deleteNode(root.right,key)
        #delete from left
        elif key < root.val:
            root.left = self.deleteNode(root.left,key)
            
        #we are there
        else:
            if not root.left and not root.right:
                root = None
            elif root.right:
                root.val = self.succ(root)
                root.right = self.deleteNode(root.right,root.val)
            else:
                root.val = self.pred(root)
                root.left = self.deleteNode(root.left,root.val)
        
        return root
    def succ(self,root):
        root = root.right
        while root.left:
            root = root.left
        return root.val
    
    def pred(self,root):
        root = root.left
        while root.right:
            root = root.right
        return root.val

######################
# 22NOV21
# 1429. First Unique Number
######################
#brute force in finding the unique element
class FirstUnique:

    def __init__(self, nums: List[int]):
        '''
        we have queue of integers, we can only add and not remove
        '''
        self.deq = deque(nums)
        

    def showFirstUnique(self) -> int:
        '''
        reads from left to right
        '''
        for num in self.deq:
            if self.deq.count(num) == 1:
                return num
        return -1
        

    def add(self, value: int) -> None:
        self.deq.append(value)
        


# Your FirstUnique object will be instantiated and called as such:
# obj = FirstUnique(nums)
# param_1 = obj.showFirstUnique()
# obj.add(value)

#q and hashmap
class FirstUnique:

    def __init__(self, nums: List[int]):
        '''
        we need a fast way of determing what is the first unique number in the deq
        otherwise when we call showFirstUnique we have to go through the whole deque
        intuion:
            we want to know if this number has occurred once or more than once
            instead of couting how many times a number occurred in the deque, we could instead keep a hashmap of numbers to booleans
           mapp of nums to booleans if we have not seen them, if true, update to false and do not add to q
        the problem now lies in whether we want to delete in the shorFirstUnique method or in the add method
        deleting after the add method, forces use to pass the array once more to carry out a deletion
        however, adding to the showFirstUnique method, we can save more time (amortized)
        '''
        self.queue = deque(nums)
        self.isUnique = {}
        #add using method
        for num in nums:
            self.add(num)

    def showFirstUnique(self) -> int:
        '''
        we need to start clearing the q of any non uniques before return
        because of the add invariant, any number in the que must be unique
        '''
        while self.queue and self.isUnique[self.queue[0]] == False:
            self.queue.popleft()
        #what should be remaining is the next unique
        if self.queue:
            return self.queue[0]
        return -1

    def add(self, value: int) -> None:
        #case 1: we need to ad the number
        if value not in self.isUnique:
            self.isUnique[value] = True
            self.queue.append(value)
        #case 2: its not unique, dont add
        else:
            self.isUnique[value] = False

#ordered dict, or linked hash set
from collections import OrderedDict
class FirstUnique:

    def __init__(self, nums: List[int]):
        '''
        O(1) time complexity will alwasy be better than amotrized o(1)
        to get O(1) for showFirst unique, we would need to have each removal happen with its corresponding add()
        not after some abritray call to showFirstUnique()
        we can remove and get next if we use LinkedList (gives us previous right away)
        there is very not so well known data structure known as a linked hahsset
        or ordereddcit in pythong
        same O(1) add, delete, update, etc operations, but also gives as the elements added in order
        we can use ordereddict as set, if we make its values None
        '''
        self.q = OrderedDict()
        self.isUnique = {}
        for num in nums:
            self.add(num)
        
    def showFirstUnique(self) -> int:
        if self.q:
            return next(iter(self.q))
        return -1

    def add(self, value: int) -> None:
        #case 1, not unique so add it
        if value not in self.isUnique:
            self.isUnique[value] = True
            self.q[value] = None
        #case 2, remove from q
        elif self.isUnique[value] == True:
            self.isUnique[value] = False
            self.q.pop(value)
        #case 3, nothing, since after the second time, it would have been removed

##############################
# 8. String to Integer (atoi)
# 23NOV21
##############################
class Solution:
    def myAtoi(self, s: str) -> int:
        '''
        keep advanving left pointer if its not an ord
        then keep decrementing right pointer if is not order
        * discard all leading whitespace
        * get sign of number
        * overflow
        * invalid input
        '''
        INT_MIN = -(2**31)
        INT_MAX = 2**31 - 1
        
        s = s.strip()
        
        N = len(s)
        sign = 1
        num = 0
        i = 0
        
        if N == 0:
            return 0
        
        
        #sign correction
        if s[0] == '-':
            sign *= -1
            i += 1
        elif s[0] == '+':
            i += 1
        
        #go through string
        while i < N and '0' <= s[i] <= '9':
            curr_digit = ord(s[i]) - ord('0')
            #abandon if we know its oveflown
            if num > INT_MAX // 10 or (num == INT_MAX // 10 and curr_digit >7):
                return INT_MAX if sign == 1 else INT_MIN
            num = num*10 + curr_digit
            i += 1
        
        return num*sign


################################################
# 23NOV21 
# 952. Largest Component Size by Common Factor
###############################################
##union find and check all factors by group
class DisjointSetUnion(object):
    
    def __init__(self,size):
        #start off with each node pointing to itself
        self.parent = [i for i in range(size+1)]
        #keep track of the size of each componeent
        self.size = [1]*(size+1)
        
    def find(self,x):
        #recursively search for x's parent
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self,x,y):
        px = self.find(x)
        py = self.find(y)
        
        #if two nodes are same set, take the left guy
        if px == py:
            return px
        #otherwise make the parent be the bigger set
        if self.size[px] > self.size[py]:
            px,py = py,px
        #add the smaller component to the larger one
        self.parent[px] = py
        self.size[py] += self.size[px]
        return py

class Solution:
    def largestComponentSize(self, nums: List[int]) -> int:
        '''
        we need to make the graph connecting nums if they have common factor greater than 1
        how to determine if two numbers share common factor > 1?
        solution is too long, just go over union find for connected components but first finding common factors
        in pseudo code this is what we want:
        group_count = {}
        for num in number_list:
            group_id = group(num)
            group_count[group_id] += 1
        return max(group_count.values())
        
        then this becomes a graph partition problem
        build groups by common factors o fht numbers, which can done in a single iteration for each number
        think of venn diagram
        
        algo:
            attribute each number to a serious of groups who's parent is a common factor
                iterate through num from 1 to sqrt(num)
                for each factor check is it is and group num and factor
                also perform opration on complement as well
            go through the groups and find the largest connected one
            
        '''
        dsu = DisjointSetUnion(max(nums))
        for num in nums:
            for factor in range(2,int(num**.5) + 1):
                #check if if is a factor
                if num % factor == 0:
                    #union the two
                    dsu.union(num,factor)
                    #also check is complement factor
                    dsu.union(num,num//factor)
        #count the size of group one by one
        max_size = 0
        group_count = defaultdict(int)
        for num in nums:
            group_id = dsu.find(num)
            group_count[group_id] += 1
            max_size = max(max_size, group_count[group_id])

        return max_size

#using seive of erastothenese
class DisjointSetUnion(object):
    
    def __init__(self,size):
        #start off with each node pointing to itself
        self.parent = [i for i in range(size+1)]
        #keep track of the size of each componeent
        self.size = [1]*(size+1)
        
    def find(self,x):
        #recursively search for x's parent
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self,x,y):
        px = self.find(x)
        py = self.find(y)
        
        #if two nodes are same set, take the left guy
        if px == py:
            return px
        #otherwise make the parent be the bigger set
        if self.size[px] > self.size[py]:
            px,py = py,px
        #add the smaller component to the larger one
        self.parent[px] = py
        self.size[py] += self.size[px]
        return py

class Solution:
    def largestComponentSize(self, nums: List[int]) -> int:
        '''
        we need to make the graph connecting nums if they have common factor greater than 1
        how to determine if two numbers share common factor > 1?
        solution is too long, just go over union find for connected components but first finding common factors
        in pseudo code this is what we want:
        group_count = {}
        for num in number_list:
            group_id = group(num)
            group_count[group_id] += 1
        return max(group_count.values())
        
        using sieve:    
        instead of checking all factors, some of which might be non essential for the grouping
        we could find the prime factorization of a number using the seuve method
        
        algo:
        decompose each number into its prime fators and apply union ops on the primes
            iterate through each number, decompose
            join all groups that possess these prime factors by applying union on adjacent pairs
            keep hash for mapping between each number and its any of prime factors
            later use table to find out which group that each number belongs to
        iterate through each number a second time to find out the groups
            since we build union find off primes, just find the prime_factor -> group_id
        
        '''
        
        dsu = DisjointSetUnion(max(nums))
        num_factor_map = {}
        
        for num in nums:
            prime_factors = list(set(self.primeDecompose(num)))
            # map a number to its first prime factor
            num_factor_map[num] = prime_factors[0]
            # merge all groups that contain the prime factors.
            for i in range(0, len(prime_factors)-1):
                dsu.union(prime_factors[i], prime_factors[i+1])
        
        max_size = 0
        group_count = defaultdict(int)
        for num in nums:
            group_id = dsu.find(num_factor_map[num])
            group_count[group_id] += 1
            max_size = max(max_size, group_count[group_id]) 
        
        return max_size

    
    def primeDecompose(self, num):
        factor = 2
        prime_factors = []
        while num >= factor*factor:
            if num % factor == 0:
                prime_factors.append(factor)
                num = num // factor
            else:
                factor += 1
        prime_factors.append(num)
        return prime_factors

#############################
# 24NOV21
# 986. Interval List Intersections
############################
class Solution:
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        '''
        if there is an intersection, we want the maxes of the lefts
        and the mins of the right
        '''
        ans = []
        i = j = 0
        
        while i < len(firstList) and j < len(secondList):
            #get the bounds for the interval
            lo = max(firstList[i][0],secondList[j][0])
            hi = min(firstList[i][1],secondList[j][1])
            
            if lo <= hi:
                ans.append([lo,hi])
            #move the smaller of the two
            if firstList[i][1] < secondList[j][1]:
                i += 1
            else:
                j += 1
        
        return ans

#######################
# 24NOV21
# 53. Maximum Subarray
#######################
#recursion
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        '''
        dp(i) represent the max sum ending at nums[i]
        if i know the previous dp(i-1), then its max owuld dp dp(i) = max(dp(i-1) + nums[i], nums[i])
        '''
        memo = {}
        N = len(nums)
        def dp(i):
            if i < 0:
                return 0
            if i in memo:
                return memo[i]
            ans = max(dp(i-1)+nums[i],nums[i])
            memo[i] = ans
            return ans
        
        dp(N-1)
        return max(memo.values())

#dp, kadanes algo
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        '''
        dp(i) represent the max sum ending at nums[i]
        if i know the previous dp(i-1), then its max owuld dp dp(i) = max(dp(i-1) + nums[i], nums[i])
        '''
        N = len(nums)
        dp = [0]*N
        dp[0] = nums[0]
        
        for i in range(1,N):
            dp[i] = max(nums[i],dp[i-1]+nums[i])
        return max(dp)

#constant space
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        '''
        dp(i) represent the max sum ending at nums[i]
        if i know the previous dp(i-1), then its max owuld dp dp(i) = max(dp(i-1) + nums[i], nums[i])
        '''
        N = len(nums)
        curr_max = highest_max = nums[0]
        
        for i in range(1,N):
            curr_max = max(nums[i], curr_max + nums[i])
            highest_max = max(highest_max, curr_max)
        return highest_max

##########################
# 24NOV21
# 219. Contains Duplicate II
##########################
#TLE
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        '''
        indicies i and j need to be at least k away from each other
        and nums[i] == nums[j]
        brute force would be to check all i and j k away then return true
        '''
        N = len(nums)
        for i in range(N):
            for j in range(max(i-k,0),i):
                if nums[i] == nums[j]:
                    return True
        return False

#sliding hash window if fixed size
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        '''
        i can keep a hash set of size k,
        keep adding nums[i] to it
        if im within k, and i see the nums again, i know it was between i and j
        once set size becomes more than k, remove the element i-k
        '''
        seen = set()
        N = len(nums)
        for i in range(N):
            if nums[i] in seen:
                return True
            seen.add(nums[i])
            if len(seen) > k:
                seen.remove(nums[i-k])
        
        return False
        
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        '''
        using hashmap
        '''
        seen = {}
        for i,v in enumerate(nums):
            if v in seen:
                if abs(i - seen[v]) <= k:
                    return True
            seen[v] = i
        
        return False

