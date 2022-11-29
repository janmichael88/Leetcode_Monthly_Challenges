##############################
# 1706. Where Will the Ball Fall
# 01NOV22
##############################
class Solution:
    def findBall(self, grid: List[List[int]]) -> List[int]:
        '''
        for each ball dropped from the top of the 0th row and the jth column traces its path
        to find the path we need to examine the current cell as well as the cell to its right
        
        say we are at cell (i,j)
        and grid[i][j] == 1, we need to check grid[i][j+1]
        if grid[i][j] == grid[i][j+1]:
            we can advance down a row in that direction
        
        we can use dfs to track the state of the ball
        keep state (i,j)
        if grid at i,j == 1, it must move right
        so move ball down a row and right 1
        
        if grid at (i,j) == -1, it must move left
        so move ball down  and left
        
        each time we have to check for base cases
        if we got to the last row, return the column
        
        if we go outside the walls or hit a v, the ball can't make it down, return -1
        '''
        rows = len(grid)
        cols = len(grid[0])
        def dfs(row,col):
            #got to the bottom
            if row == rows:
                return col
            
            #going right
            if grid[row][col] == 1:
                #if we can at least check right and go right
                if col < cols-1 and grid[row][col] == grid[row][col+1]:
                    return dfs(row+1,col+1)
                else:
                    return -1
            #going left
            elif grid[row][col] == -1:
                if col > 0 and grid[row][col] == grid[row][col-1]:
                    return dfs(row+1,col-1)
                else:
                    return -1

            
        ans = []
        for i in range(cols):
            ans.append(dfs(0,i))
        
        return ans
            
#also could have done dfs iteratively
class Solution:
    def findBall(self, grid: List[List[int]]) -> List[int]:
        '''
        for each ball dropped from the top of the 0th row and the jth column traces its path
        to find the path we need to examine the current cell as well as the cell to its right
        
        say we are at cell (i,j)
        and grid[i][j] == 1, we need to check grid[i][j+1]
        if grid[i][j] == grid[i][j+1]:
            we can advance down a row in that direction
        
        we can use dfs to track the state of the ball
        keep state (i,j)
        if grid at i,j == 1, it must move right
        so move ball down a row and right 1
        
        if grid at (i,j) == -1, it must move left
        so move ball down  and left
        
        each time we have to check for base cases
        if we got to the last row, return the column
        
        if we go outside the walls or hit a v, the ball can't make it down, return -1
        '''
        rows = len(grid)
        cols = len(grid[0])
        def find_col(row,col):
            while row < rows and col < cols:
                #got to the bottom
                if row == rows:
                    return col
                #going right
                if grid[row][col] == 1:
                    #if we can at least check right and go right
                    if col < cols-1 and grid[row][col] == grid[row][col+1]:
                        row += 1
                        col += 1
                    else:
                        return -1
                #going left
                elif grid[row][col] == -1:
                	#if we can at least check left
                    if col > 0 and grid[row][col] == grid[row][col-1]:
                        row += 1
                        col -= 1
                    else:
                        return -1
            return col if row == rows else -1

            
        ans = []
        for i in range(cols):
            ans.append(find_col(0,i))
        
        return ans

############################################################
# 1198. Find Smallest Common Element in All Rows (REVISTED)
# 02NOV22
###########################################################
class Solution:
    def smallestCommonElement(self, mat: List[List[int]]) -> int:
        '''
        for each element in the first row, binary search for it in the other rows
        
        '''
        to_search = mat[0]
        for num in to_search:
            check = 0
            for row in mat[1:]:
                #look for it
                idx = bisect_left(row,num)
                if (0 <= idx < len(row)) and num == row[idx]:
                    check += 1
            
            if check == len(mat) - 1:
                return num
        
        return -1


class Solution:
    def smallestCommonElement(self, mat: List[List[int]]) -> int:
        '''
        since the numbers in the rows are sorted in strictly increasing order, we can just count up the numbers
        then traverse the the numbers in order and return the one that has count at least n
        '''
        rows = len(mat)
        cols = len(mat[0])
        
        # know that largest number
        counts = [0]*(10**4 + 1)
        for i in range(rows):
            for j in range(cols):
                num = mat[i][j]
                counts[num] += 1
        
        for num in range(1, len(counts)):
            if counts[num] == rows:
                return num
        
        return -1

class Solution:
    def smallestCommonElement(self, mat: List[List[int]]) -> int:
        '''
        we can speed things up by going down a column
        '''
        rows = len(mat)
        cols = len(mat[0])
        
        # know that largest number
        counts = [0]*(10**4 + 1)
        for j in range(cols):
            for i in range(rows):
                num = mat[i][j]
                counts[num] += 1
                if counts[num] == rows:
                    return num
        
        return -1

#using merge k sorted lists
class Solution:
    def smallestCommonElement(self, mat: List[List[int]]) -> int:
        '''
        we can treat this like the merge k sorted problem
        for each we track the position of the current element starting from zero
        then we find the smallest element among all positions and advance the position for the row that we are on
        
        algo:
            1. init row positions, current max and counter (we are couting that have at least the maximum)
            2. for each row:
                increment the row position until the value is equal or greater than the current max
                    if we reach the end of a row, return -1
                if the value equals the current max, increase the counter
                    otherwise reset counter to 1 and update max
                
        '''
        rows = len(mat)
        cols = len(mat[0])
        
        
        #store poisitions of each row int an array
        positions = [0]*rows
        curr_max = 0
        count = 0
        
        while True:
            for i in range(rows):
                #keep advancing along a row
                curr_pos = positions[i]
                while curr_pos < cols and mat[i][curr_pos] < curr_max:
                    curr_pos += 1
                positions[i] = curr_pos
                
                #if we have reached the end of a row
                if positions[i] >= cols:
                    return -1
                
                if mat[i][positions[i]] != curr_max:
                    count = 1
                    curr_max = mat[i][positions[i]]
                
                else:
                    count += 1
                    if count == rows:
                        return curr_max

#abusing set notation
class Solution:
    def smallestCommonElement(self, mat: List[List[int]]) -> int:
        '''
        we can also use sets
        
        '''
        #turn this into iter object
        mat = iter(mat)
        temp = set(next(mat)) #grabs the first row and advance mat
        for row in mat:
            temp = temp & set(row)
        
        return min(temp,default=-1)




#################################
# 951. Flip Equivalent Binary Trees
# 02NOV22
#################################
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flipEquiv(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        '''
        if there is a flip in a a node of tree
        then it should be the case that at the flipped node
        (node1.left = node2.right) == (node1.right == node2.left)
        problem is we can so some number of flips
        
        if there is say N number of flips, then the permuations in a level should be the same
        rather perm of a level in tree 1 should be prem of that same level in tree 2
        actually nope this case doesn't hold
        
        
        4 5 6 7
        5 4 6 7
        5 4 7 6
        4 5 7 6
        6 7 4 5
        7 6 4 5
        7 6 5 4
        6 7 4 5
        
        gahh, straigh up tricky recursion
        
        rules:
            root1 and root2 have the same root value, then we only need to check if their children are equal
            1. if root1 or root2 is null, then they are equivlanet if  and only if they are both null
            2. else if root1 and root have different values, they ar enot equivalent
            3. else, lets check whether the children of root1 are equivalent to the children of root 2
            thtere are two ways to pair these chidlren
        '''
        def dp(p,q):
            #base case both are none, this is trieivally true
            if not p and not q:
                return True
            #if only one is empty, and the values don't match, not true
            if not p or not q or p.val != q.val:
                return False
            #check the options
            op1 = dp(p.left,q.left)
            op2 = dp(p.right,q.right)
            op3 = dp(p.left,q.right)
            op4 = dp(p.right,q.left)
            
            return (op1 and op2) or (op3 and op4)
        
        return dp(root1,root2)

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flipEquiv(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        '''
        there is actually a characteristic for binary trees where all the nodes have unique values
        flip each node such that the left child < right child
            call this the canonical representation
            all equivlant trees have eactly one
        
        preorder both trees, for empty nodes make value -1
        '''
        vals1 = []
        vals2 = []
        
        def dfs(node,vals):
            if node:
                vals.append(node.val)
                left = node.left.val if node.left else -1
                right = node.right.val if node.right else -1
            
                if left < right:
                    dfs(node.left,vals)
                    dfs(node.right,vals)
                else:
                    dfs(node.right,vals)
                    dfs(node.left,vals)
            else:
                vals.append(None)
                
        dfs(root1,vals1)
        dfs(root2,vals2)
        
        return vals1 == vals2

############################################################
# 2131. Longest Palindrome by Concatenating Two Letter Words
# 03NOV22
#############################################################
#almost
class Solution:
    def longestPalindrome(self, words: List[str]) -> int:
        '''
        brute force would be to examine all orderings of words
        for each ordering check if it is a palindrome and update the longest length found
        we need to intelligently build a palindrome from scratch
        
        if we start with 'ab' we need at least add 'ba'
        the number of times we can do this is the minimum number of 'ab' and 'ba'
        for words that are already palindromes
        
        ["ab","ty","yt","lc","cl","ab"]
        
        first make ty + yt
        then make lc cl
        
        while traversing the words keep track of counts
        if a word and it reverse are in there we can make a palindrome
        we can actually just keep track on the size of the currently made palindrom
        '''
        counts = {}
        current_size = 0
        
        for w in words:
            if w not in counts:
                counts[w] = 1
            else:
                counts[w] += 1
            
            if w[::-1] in counts:
                current_size += 4
            
            
        return current_size
            
class Solution:
    def longestPalindrome(self, words: List[str]) -> int:
        '''
        if we have a valid palindrome made then the last word is the reverse of the first
        second to last is reverse of the second and so on.
        
        center words, if there is one, must be of the same character
        if the word is not a palindrome we can make it one with its reverse, we take the min number of times
        
        for word the is already a palindrome, it ocurrs an odd number of times in the final string, if and only if it is a centrla word
        
        there are two special cases
        for words that are already palindromes, these must exist as the cetner words
            for an even number of such words, they can all be the pivot
            for an odd number, there can be that even number of times plus 1
            
        
        '''
        #count up the words
        counts = Counter(words)
        longest = 0
        has_center = False
        
        for word,count in counts.items():
            #if the word is a palindrome
            if word[0] == word[1]:
                #even occurence
                if count % 2 == 0:
                    #we can use them all, doesn't matter what, its jsut that we can
                    longest += count
                else:
                    longest += count - 1
                    #we can use this is center
                    has_center = True
                
            #otherwise its not
            elif word[0] < word[1]: #to consider each word once not twice
                #find the freverse
                longest += 2*min(count,counts[word[1]+word[0]])
        
        #if we have a centre increase by 1
        if has_center:
            longest += 1
        return longest*2


#another different way
'''
case 1: the word is not same as the reversed self, e.g. "ab" != "ba"
in this case, we need its reveresd string, i.e. ba to form "abba" as a palindrome

case 2: the word is same as the reversed self, e.g. "aa" == "aa"
case 2.1: if it is even, we could place it in the middle or on the side
e.g. [aa]abba[aa]
case 2.2: if the frequency of "aa" is odd, we could only place it in the middle
e.g. ab[aa]ba
since even + 1 = odd, we can put all even "aa" on the side, and put one in the middle
e.g. [aa]ab[aa]ba[aa]
'''
class Solution:
    def longestPalindrome(self, words: List[str]) -> int:
        ans = 0
        middle = 0
        
        counts = Counter(words)
        for word,count in counts.items():
            #if the words is not a palindrome
            if word[0] != word[1]:
                #we can only take the minimum of the count
                ans += min(counts[word],counts[word[::-1]])
            #if the word is a palindrome
            else:
                #we could add to both sides
                ans += count
                if count & 1:
                    #if odd, we need to compensate, by using once less count of the word
                    ans -= 1
                    middle = 1
        
        #correct for middle
        ans += middle
        return 2*ans


'''
class Solution {
public:
    //reverse function
        string reversed(string s){
            string t = s;
            reverse(t.begin(),t.end());
            return t;
        };
    int longestPalindrome(vector<string>& words) {
        
        int ans = 0;
        int middle = 0;
        
        //get the word count, just be lazy and use unordered map
        unordered_map<string,int> counts;
        
        for (auto word:words){
            counts[word]++;
            //std::cout << counts[word] << '\n';
        }
        
        //go through the mapp
        for (auto& [word,count]:counts){
            string reversed_word = reversed(word);
            //std::cout << reversed_word << '\n';
            
            if (word != reversed_word) {
                if (counts.count(reversed_word)){
                    ans += min(counts[word],counts[reversed_word]);
                }
            }
            
            else{
                ans += count;
                
                if (count & 1){
                    ans -= 1;
                    middle = 1;
                }
            }
        }
        
        ans += middle;
        
        return 2*ans;
    }
};
'''

class Solution:
    def longestPalindrome(self, words: List[str]) -> int:
        '''
        we first get the counts of all the words
        then for words the are already palindromes, we know can stitch these together
        the problem is the there might be some palindromes that are left unpaired, in which case, this must be the center
        counts % 2
        
        so really we are building in increments of 4
        and if there is an unapired two length palindrome, it must be the center
        '''
        res = 0
        unpaired_centers = 0
        
        counts = Counter(words)
        
        for word,count in counts.items():
            if word[0] == word[1]:
                unpaired_centers += count % 2
                #imagine cases for input like ['xx','ll','gg']
                res += 4*(count // 2)
            else:
                #this is important so that we don't compare ab and ba seperately
                if word[0] < word[1]:
                    res += 4*min(count,counts[word[1]+word[0]])
        

        if unpaired_centers > 0:
            res += 2
        return res

class Solution:
    def longestPalindrome(self, words: List[str]) -> int:
        '''
        we can use a hashamp to store all possible 2 lenght strings (there will be 26*26 of them)
        
        then just use it as that
        '''
        counts = Counter(words)
        answer = 0
        has_unpaired_center = False
        
        for i in range(26):
            a = chr(ord('a') + i)
            if counts[a+a] % 2 == 0:
                answer += counts[a+a]
            else:
                answer += counts[a+a]
                answer -= 1
                has_unpaired_center = True
            
            #check all other pairs from i+1 to 26
            for j in range(i+1,26):
                b = chr(ord('a') + j)
                answer += 2*min(counts[a+b],counts[b+a])
        
        if has_unpaired_center:
            answer += 1
        return answer*2
    
###################################
# 273. Integer to English Words
# 07NOV22
###################################
#recursive solution
class Solution:
    
    def __init__(self):
        #store list of english word numbers, and index into them
        self.lessThan20 = ["","One","Two","Three","Four","Five","Six","Seven","Eight","Nine","Ten","Eleven","Twelve","Thirteen","Fourteen","Fifteen","Sixteen","Seventeen","Eighteen","Nineteen"]
        self.tens = ["","Ten","Twenty","Thirty","Forty","Fifty","Sixty","Seventy","Eighty","Ninety"]
        self.thousands = ["","Thousand","Million","Billion"]

    def helper(self,num):
        #we can use a reucrsive function and pass the reduced parts into the function
        #allowable limit for this function is 2000
        #in the out function we reduce by the largest divisible unit, in this case it is a thousand
        if num == 0:
            return ""
        elif num < 20:
            return self.lessThan20[num] + " "
        elif num < 100:
            return self.tens[num//10] + " " + self.helper(num % 10)
        else:
            return self.lessThan20[num//100] + " Hundred " + self.helper(num % 100)
        
    def numberToWords(self, num: int) -> str:
        #store list of english word numbers, and index into them
        #return self.helper(96)
        if num == 0:
            return "Zero"
        
        res = ""
        #start from thousands to bullions
        for i in range(len(self.thousands)):
            
            #if the curr group cannot be divided exactly by 1000, tranlate it
            #i.e call the function on this part and current suffix
            if num % 1000 != 0:
                res = self.helper(num % 1000) + self.thousands[i] + " " + res
            num //= 1000
        
        return res.strip()
###############################
# 1544. Make The String Great
# 08NOV22
################################
#i can't belive this got accepted
class Solution:
    def makeGood(self, s: str) -> str:
        '''
        we define a good string as a string that does not have two adjacent characets where
        s[i] is lower case and s[i+1] is the same letter but in upper case
        
        
        '''
        #function to check == and opposite in case
        def helper(p,q):
            #fist check same letter
            if p.lower() == q.lower():
                if (p.isupper() and q.islower()) or (p.islower() and q.isupper()):
                    return True
            return False
        
        def isNotGood(s):
            N = len(s)
            for i in range(1,N):
                first = s[i-1]
                second = s[i]
                if helper(first,second):
                    return True
            
            return False
        
        s = list(s)
        while isNotGood(s):
            N = len(s)
            i = 1
            while i < N:
                first = s[i-1]
                second = s[i]
                if helper(first,second):
                    del s[i]
                    del s[i-1]
                    N -= 2
                i += 1
        
        return "".join(s)
                
class Solution:
    def makeGood(self, s: str) -> str:
        '''
        to see if adjacent characters are invalid pairs just check that abs(ascii(first - second)) == 32
        '''
        while len(s) > 1:
            isNotGood = False
            for i in range(len(s)-1):
                first = s[i]
                second = s[i+1]
                
                if abs(ord(first) - ord(second)) == 32:
                    #delete and rebuild
                    s = s[:i] + s[i+2:]
                    isNotGood = True
                    break
            
            if isNotGood == False:
                break
        
        return s

#recursion
class Solution:
    def makeGood(self, s: str) -> str:
        #we can use recursion
        def rec(s):
            for i in range(len(s)-1):
                first = s[i]
                second = s[i+1]
                if abs(ord(first) - ord(second)) == 32:
                    #delete and rebuild
                    return rec(s[:i] + s[i+2:])
            #base case is here in the case we get through the whole thing just return the string
            return s

        
        return rec(s)
                
class Solution:
    def makeGood(self, s: str) -> str:
        '''
        typical stack solution
            if element we are are about to add and last eleemtn on stack make an invalid pair, pop it
        '''
        stack = []
        for ch in s:
            if stack and abs(ord(stack[-1]) - ord(ch)) == 32:
                stack.pop()
            else:
                stack.append(ch)
        
        return "".join(stack)

####################################
# 339. Nested List Weight Sum (REVISITED)
# 08NOV22
####################################
# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """
#class NestedInteger:
#    def __init__(self, value=None):
#        """
#        If value is not specified, initializes an empty list.
#        Otherwise initializes a single integer equal to value.
#        """
#
#    def isInteger(self):
#        """
#        @return True if this NestedInteger holds a single integer, rather than a nested list.
#        :rtype bool
#        """
#
#    def add(self, elem):
#        """
#        Set this NestedInteger to hold a nested list and adds a nested integer elem to it.
#        :rtype void
#        """
#
#    def setInteger(self, value):
#        """
#        Set this NestedInteger to hold a single integer equal to value.
#        :rtype void
#        """
#
#    def getInteger(self):
#        """
#        @return the single integer that this NestedInteger holds, if it holds a single integer
#        Return None if this NestedInteger holds a nested list
#        :rtype int
#        """
#
#    def getList(self):
#        """
#        @return the nested list that this NestedInteger holds, if it holds a nested list
#        Return None if this NestedInteger holds a single integer
#        :rtype List[NestedInteger]
#        """

class Solution:
    def depthSum(self, nestedList: List[NestedInteger]) -> int:
        '''
        just use dfs on each item to return the depth by multiplied by its sum
        '''
        def dfs(nl,depth):
            #base case if its an integer
            if nl.isInteger():
                return depth*nl.getInteger()
            else:
                ans = 0
                for child in nl.getList():
                    ans += dfs(child,depth+1)
                return ans
        
        ans = 0
        for nl in nestedList:
            ans += dfs(nl,1)
        return ans

###################################
# 493. Reverse Pairs
# 08NOV22
####################################
#using merger sort
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        '''
        we can just copy merge sort and count the reverse pairs during the merge part
        '''
        self.inverse_pairs = 0
        print(self.merge(nums))
        return self.inverse_pairs
    
    def merge_count(self,left_list,right_list):
        #modification to count inversions
        #don't actuall do the merge, just count, when we need to return, use the bultin sort method
        left_ptr = right_ptr = 0
        while left_ptr < len(left_list) and right_ptr < len(right_list):
            #check for no inversions
            if left_list[left_ptr] <= 2*right_list[right_ptr]:
                left_ptr += 1
            #check for inversion
            else:
                self.inverse_pairs += len(left_list) - left_ptr
                right_ptr += 1
        
        return sorted(left_list+right_list)
    
    def merge(self,nums):
        if len(nums) <= 1:
            return nums
        pivot = len(nums) // 2
        #recurse
        left = self.merge(nums[:pivot])
        right = self.merge(nums[pivot:])
        return self.merge_count(left,right)
                

#another way just for fun
class Soltuion:
     def reversePairs(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) <= 1:
            return 0
        count = [0]

        def merge(nums):
            if len(nums) <= 1: return nums
            
            left, right = merge(nums[:len(nums)//2]), merge(nums[len(nums)//2:])
            l = r = 0
            
            while l < len(left) and r < len(right):
                if left[l] <= 2 * right[r]:
                    l += 1
                else:
                    count[0] += len(left) - l
                    r += 1
            return sorted(left+right)

        merge(nums)
        return count[0]


#bst
class Node:
    def __init__(self, val):
        self.val = val
        self.ge_cnt = 0
        self.left = None
        self.right = None

class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        root = None
        ans = 0
        cp_nums = nums.copy()
        cp_nums.sort()
        #make bst using sorted nums first
        root = self.bst(cp_nums, 0, len(nums) - 1)
        
        for num in nums:
            #in this tree we want to find the value the (num,root) such that there exists an inversion
            #so we look into the bst fro 2*num+1, since it must be strictly increasing (open interval)
            #we we search we get the counts, which stores the inversions for each number
            ans += self.search(root, 2 * num + 1)
            #we need to update the counts for this number
            root = self.update(root, num)
        
        return ans

    def search(self, root, val):
        #find node and count inverse pairs
        if not root:
            return 0
        
        if root.val < val:
            ans = self.search(root.right, val)
        elif root.val > val:
            ans = self.search(root.left, val) + root.ge_cnt
        else:
            ans = root.ge_cnt
        
        return ans
    
    def update(self, root, val):
        if root.val == val:
            root.ge_cnt += 1
        elif root.val < val:
            root.ge_cnt += 1
            self.update(root.right, val)
        else:
            self.update(root.left, val)
        return root
    
    def bst(self, nums, l, r):
        if l > r:
            return None
        m = l + (r - l) // 2
        root = Node(nums[m])
        root.left = self.bst(nums, l, m - 1)
        root.right = self.bst(nums, m + 1, r)
        return root
        


######################################
# 901. Online Stock Span (REVISTED)
# 09NOV22
######################################
#brute force, count the streaks
class StockSpanner:

    def __init__(self):
        '''
        span is max consecutive days for which the stock price was <= than the current price starting from today and going backward
        i could brute force and just count the streaks
        break once we can't can't continue a streak
        
        '''
        self.stocks = []
        

    def next(self, price: int) -> int:
        self.stocks.append(price)
        N = len(self.stocks)
        max_span = 0
        curr_span = 0
        
        curr_price = self.stocks[N-1]
        
        for i in range(N-1,-1,-1):
            if self.stocks[i] <= curr_price:
                curr_span += 1
            else:
                max_span = max(max_span,curr_span)
                break
        
        return max(max_span,curr_span)
            


# Your StockSpanner object will be instantiated and called as such:
# obj = StockSpanner()
# param_1 = obj.next(price)


#montonic stack solution
class StockSpanner:
    def __init__(self):
        self.stack = []

    def next(self, price: int) -> int:
        ans = 1
        while self.stack and self.stack[-1][0] <= price:
            ans += self.stack.pop()[1]
        
        self.stack.append([price, ans])
        return ans

# Your StockSpanner object will be instantiated and called as such:
# obj = StockSpanner()
# param_1 = obj.next(price)

#################################
# 1021. Remove Outermost Parentheses
# 10NOV22
#################################
class Solution:
    def removeOuterParentheses(self, s: str) -> str:
        '''
        we already know that s is a valid parantheses
        what if i store the balance so far for each expression
        
        a primitive decomposition is whenever we hit a balance of zero
        find the primitive decompositions and clear the out most
        '''
        bal = 0
        curr_primitive = []
        primitives = []
        for ch in s:
            if ch == '(':
                bal += 1
            else:
                bal -= 1
            curr_primitive.append(ch)
            if bal == 0:
                primitives.append(curr_primitive)
                curr_primitive = []
        
        #for each primitive, clear the outer most
        ans = []
        for prim in primitives:
            stripped = prim[1:-1]
            ans.append("".join(stripped))
        
        return "".join(ans)
            
class Solution:
    def removeOuterParentheses(self, s: str) -> str:
        '''
        we already know that s is a valid parantheses
        what if i store the balance so far for each expression
        
        a primitive decomposition is whenever we hit a balance of zero
        find the primitive decompositions and clear the out most
        '''
        bal = 0
        curr_primitive = ""
        primitives = ""
        for ch in s:
            if ch == '(':
                bal += 1
            else:
                bal -= 1
            curr_primitive += ch
            if bal == 0:
                primitives += curr_primitive[1:-1]
                curr_primitive = ""
        
        return primitives

#########################
# 731. My Calendar II
# 12NOV22
#########################
class MyCalendarTwo:
    '''
    we maintain a list of bookinds AND a list of double bookings
    when a booking conflicts with andything in the double booking, this cannot be a valid booking
    othewise, part that overlap the calendar will be a double boking
    
    
    '''
    def __init__(self):
        self.calendar = []
        self.double_bookings = []
        

    def book(self, start: int, end: int) -> bool:
        #first check for a violation in the double bookings array
        for s,e in self.double_bookings:
            #over lap with a double bookings?
            if start < e and end > s:
                return False
        
        #update double bookings by going through the whole calendar to find a double booking
        for s,e in self.calendar:
            #single overlap with single event?
            if start < e and end > s:
                #update double bookings:
                self.double_bookings.append([max(start,s),min(end,e)])
            
        #otherwise add to calendaar
        self.calendar.append([start,end])
        return True
        
class MyCalendarTwo:
    '''
    boundary count
    for a new booking event on the closed and open interval [start,end)
    we increment a count for every delta in between start and end
    if the sum is 3 or more, than that time is triple bookied
    
            - We will consider 'start' time as +1 and 'end' time as -1
            - If we currently only have 'start' and 'end' time
                - The sum between them will equal to 0, which will balance out
        - Now, if we add an overlap between the 'start/end' time we will have the following
            - s0, s1, e0, e1
        - Then the sum will be
            - 1   2   1  0
        - Since, there is an overlap, we can see that our highest sum is equal to 2
    - We can continue this approach to 3 or more overlaps
        - Example:
            - s0, s1, s2, e0, e1, e3
            - 1   2   3   2   1   0
        - In this case, our sum has reached 3 and we have found our triple booking
        
    we don't have a tree set available in python, so we can mimic that with bisect method and insort
    '''
    def __init__(self):
        self.calendar = []

    def book(self, start, end):
        bisect.insort(self.calendar, (start, 1))
        bisect.insort(self.calendar, (end, -1))
        #print(self.calendar)
        
        bookings = 0
        for time, freq in self.calendar:
            bookings += freq
            if bookings == 3:
                self.calendar.pop(bisect.bisect_left(self.calendar, (start, 1)))
                self.calendar.pop(bisect.bisect_left(self.calendar, (end, -1)))
                return False
        
        return True

class MyCalendarTwo:

    def __init__(self):
        self.mapp = defaultdict()

    def book(self, start: int, end: int) -> bool:
        self.mapp[start] = self.mapp.get(start,0) + 1
        self.mapp[end] = self.mapp.get(end,0) - 1
        
        bookings = 0
        for time,freq in sorted(self.mapp.items()):
            bookings += freq
            if bookings == 3:
                self.mapp[start] = self.mapp.get(start,0) - 1
                self.mapp[end] = self.mapp.get(end,0) + 1
                return False
        
        return True

##################################################
# 947. Most Stones Removed with Same Row or Column
#  14NOV22
##################################################
class Solution:
    def removeStones(self, stones: List[List[int]]) -> int:
        '''
        we can treat this is a graph problem
        stones are connected if they share a row or column with another stone
        so anye stone with a non zero degree can be removed because it means there exist at least one stone with the same row or column
         
        we just need to find the best order for the stone removal
         
        the graph that we build could have a number of connected components
        we need to process each each component seperately
        for a grouped componnet, we can remove all of but one stone (think going backwards)
         	 	 
        DFS
        we start from any nodes in a compoenent and keep removing stones by visting in topolical order
        except the one we start with
         	 	 
        algo
        1. make the adjacent list (bidirectional)
        2. make seen hash set
        3. dfs each stone and increment count
        4. return length stones
        '''
        adj_list = defaultdict(list)
        N = len(stones)
        for i in range(N):
            #we dont need duplicates
        	for j in range(i+1,N):
                #make sure they sure an edge
        	 	if stones[i][0] == stones[j][0] or stones[i][1] == stones[j][1]:
        	 	    adj_list[i].append(j)
        	 	 	adj_list[j].append(i)

        #we dfs on a connected component
        #we can retun the size of this copmonent, and return 1 less

        def dfs_helper(node,seen):
            self.size = 0
            def dfs(node,seen):
                seen.add(node)
                self.size += 1
                for neigh in adj_list[node]:
                    if neigh not in seen:
                    dfs(neigh,seen)
            dfs(node,seen)
            return self.size - 1
        	 	 
        #dfs on each node
        ans = 0
        seen = set()
        for i in range(N):
            if i not in seen:
        	   ans += dfs_helper(i,seen)

        return ans

#len(stones) - num components
class Solution:
    def removeStones(self, stones: List[List[int]]) -> int:
        '''
        instead of counting the number of nodes in a connected component
        we can just countt up the number of connected components
        the largest number of stones that can be removed is len(stones) - connected compoenents
        why? for each componenet, we will be left with one stone, the other stones are to be deletd
        '''
        adj_list = defaultdict(list)
        N = len(stones)
        
        for i in range(N):
            for j in range(i+1,N):
                if stones[i][0] == stones[j][0] or stones[i][1] == stones[j][1]:
                    adj_list[i].append(j)
                    adj_list[j].append(i)
        
        def dfs(node,seen):
            seen.add(node)
            for neigh in adj_list[node]:
                if neigh not in seen:
                    dfs(neigh,seen)
                    
        num_comps = 0
        seen = set()
        for i in range(N):
            if i not in seen:
                dfs(i,seen)
                num_comps += 1
        
        return N - num_comps

class UnionFind:
    def __init__(self,size):
        self.groups = [i for i in range(size)]
        self.size = [1 for i in range(size)]
        
    def find(self,x):
        #using path compression
        if x == self.groups[x]:
            return x
        self.groups[x] = self.find(self.groups[x])
        return self.groups[x]
        
    def union(self,x,y):
        #combine stones, returns 1 if they were not connected
        x = self.find(x)
        y = self.find(y)
        
        #if they were already connected
        if x == y:
            return 0
        
        #updates
        if self.size[x] > self.size[y]:
            self.size[x] += self.size[y]
            self.groups[y] = x
        else:
            self.size[y] += self.size[x]
            self.groups[x] = y
        
        #we combined so there must be connection
        return 1

class Solution:
    def removeStones(self, stones: List[List[int]]) -> int:
        '''
        we can use union find,
        union on all pairs of stones
        the inital number of connected components is the number of stones
        every time we perform a union on two stones, we decrement the number of componenets by 1
        then same as the first part
        '''
        N = len(stones)
        uf = UnionFind(N)
        
        comp_size = N
        for i in range(N):
            for j in range(i+1,N):
                if stones[i][0] == stones[j][0] or stones[i][1] == stones[j][1]:
                    comp_size -= uf.union(i,j)
        
        return N - comp_size


#optimized DFS
#adjlist between row to col and col to row
class Solution:
    def removeStones(self, stones: List[List[int]]) -> int:
        '''
        instead of checking all pairs for generate the adj_list
        just check all rows and all cols
        
        insteaf of making adj list from stone to stone
        we link a row to a column, but we need a way to differentiate between the same row and same col
            OFFSET with + 10001
        
        we connect all stones with the same x coordiante and all stones with the same y cooridante
        each row and each col is a vertex
        
        inution:
            a stone at row x, touches stones at col y
            a stone at col y, touches a stone at row x
        '''
        #could also have used array
        adj_list = defaultdict(list)
        N = len(stones)
        K = 10001
        
        for i,stone in enumerate(stones):
            row,col = stone
            col += K
            adj_list[row].append(col)
            adj_list[col].append(row)
        
        def dfs(node,seen,adj_list):
            seen.add(node)
            for neigh in adj_list[node]:
                if neigh not in seen:
                    dfs(neigh,seen,adj_list)
                    
        num_comps = 0
        seen = set()
        for i in range(2*K + 1):
            if i not in seen and len(adj_list[i]) > 0:
                num_comps += 1
                dfs(i,seen,adj_list)
        
        return N - num_comps

#another solution
class Solution:
    def removeStones(self, stones: List[List[int]]) -> int:
        '''
        first group by grouping together rows and cols 
        rows map to stone indices in same row
        same goes for cols
        
        make connections
        '''
        
        N = len(stones)
        x2n, y2n, adj = defaultdict(list), defaultdict(list), defaultdict(list)
        visited = set()
        
        # group/connect nodes to x-coordinate and y-coordinate [x=0,nodes=1,3,4,5]
        for n,(x,y) in enumerate(stones):
            x2n[x].append(n), y2n[y].append(n)

        # build adj_list and connect grouped/connected nodes to the first node in a group
        # adj list is stone index being connected to other stone indices
        for n in chain(x2n.values(), y2n.values()):
            for i in range(1, len(n)):
                adj[n[0]].append(n[i]), adj[n[i]].append(n[0])
                
        
        def bfs(node):
            if node in visited: return 0
            queue = deque([node])
            visited.add(node)
            while queue:
                node = queue.popleft()
                for nei in adj[node]:
                    if nei not in visited:
                        visited.add(nei)
                        queue.append(nei)
            return 1
        
        def dfs(node):
            if node in visited: return 0
            visited.add(node)
            for nei in adj[node]:
                if nei not in visited:
                    dfs(nei)
            return 1
        
        return N - sum(dfs(i) for i in range(N))

################################
# 503. Next Greater Element II
# 15NOV22
################################
#cheeky O(N^2)
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        '''
        brute force, i can concat nums with nums
        then for each i up to len(nums) check i + len(nums)
        '''
        N = len(nums)
        nums = nums + nums
        
        ans = []
        for i in range(N):
            next_greater = -1
            for j in range(i+1,i+N):
                if nums[j] > nums[i]:
                    next_greater = nums[j]
                    break
            ans.append(next_greater)
        
        return ans

class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        '''
        brute force, i can concat nums with nums
        then for each i up to len(nums) check i + len(nums)
        
        we don't need to conct arrays, 
        when indexing j 
        do j % N
        '''
        N = len(nums)
        
        ans = []
        for i in range(N):
            next_greater = -1
            for j in range(i+1,i+N):
                if nums[j % N] > nums[i]:
                    next_greater = nums[j % N]
                    break
            ans.append(next_greater)
        
        return ans

#############################
# 223. Rectangle Area (REVISTED)
# 17OCT22
#############################
class Solution:
    def computeArea(self, ax1: int, ay1: int, ax2: int, ay2: int, bx1: int, by1: int, bx2: int, by2: int) -> int:
        '''
        area is defined as:
            area1 + area2 - intersection
            is there is an intersection
            line sweep?
            
        how to find intersection?
        if there is an overlap along x direction:
        xOverlap=min(ax2,bx2)−max(ax1,bx1) should be positive
        
        if there is an overlap in the y direction
        yOverlap=min(ay2,by2)−max(ay1,by1)
        
        the area of the overlap is:
        overlap = xOverlap*yOverlap
        
        if ther eis no overlap, then both of these would be negtaive
            
        very similar to intessection over union
        '''
        area1 = (ax2 - ax1)*(ay2 - ay1)
        area2 = (bx2 - bx1)*(by2 - by1)
        
        #find overlap in x direction
        xOverlap = min(ax2,bx2) - max(ax1,bx1)
    
        #find overlap in y direction
        yOverlap = min(ay2,by2) - max(ay1,by1)
        
        #if there is ovalap calculate it
        overlap = 0
        if xOverlap > 0 and yOverlap > 0:
            overlap += xOverlap*yOverlap
        
        return area1 + area2 - overlap

####################################
# 552. Student Attendance Record II
# 17NOV22
###################################
#nice try...
class Solution:
    def checkRecord(self, n: int) -> int:
        '''
        student can get an award if:
            1. count of absent < 2
            2. longest consectuive L cannot be more than 3
            
        return the number of possible ways a student can get an award for size n
        base case, i n == 1:
            there are three ways A,L,P
        
        if i knew the number of ways for some number i, call that dp(i)
        
        build up from i = 0 to n
        keeping track of A's used so far, and, first and second L
        
        throw in a memo too
        '''
        memo = {}
        mod = 10**9 + 7
        
        def dp(i,count_A,first_L,second_L):
            if i == n:
                return 1
            if (i,count_A,first_L,second_L) in memo:
                return memo[(i,count_A,first_L,second_L)]
            
            ways = 0
            for record in ['A','L','P']:
                #P case is easy
                if record == 'P': 
                    if first_L and second_L:
                        #break it
                        ways += dp(i+1,count_A,False,False)
                    else:
                        ways += dp(i+1,count_A,first_L,second_L)
                
                #the A case
                if record == 'A':
                    if count_A < 2:
                        if first_L and second_L:
                            ways += dp(i+1,count_A +1,False,False)
                        else:
                            ways += dp(i+1,count_A +1, first_L,second_L)
                #L case
                if record == 'L':
                    if not first_L and not second_L:
                        ways += dp(i+1,count_A,True,False)
                    if first_L and not second_L:
                        ways += dp(i+1,count_A,True,True)
            ways %= mod
            memo[(i,count_A,first_L,second_L)] = ways
            return ways % mod
        
        return dp(0,0,False,False)

class Solution:
    def checkRecord(self, n: int) -> int:
        '''
        lets first examine having a rewardable string of length n using only L and P
        call this number f(n)
        if we go to f(n-1) we can have two a string ending with L and a string ending with P
        a string ending with P is always rewardable, so to all the valid strings at f(n-1), we can get n-1 more, just by adding P
        (i.e) increase by factor of f(n-1) to f(n)
        
        now for the string ending with L,
        the rewardability is based upon looking at strings for f(n-3), becaise i cannot add a LL
        
        for f(n-3), we coudl have
                LPL
                PLL
                PPL
        but not
            P LLL
        BUT, very important BUT
            since we've considered only rewardable strings of length n-3, for the last string to be rewardable at lenght n-3
            and unrewardable at length n-1, it must have been preceded by a P before LL
            
        so, accounting for the first string again, all the rewardable strings of length n-1, except the string of length n-4
        followed by PLL, can contribute to a rewardable string of length n
        
        so we want to include all strings f(n-1) not including strings at f(n-4)
        
        so far
        f(n) = (including f(n-1) ending with P) + (inclduing f(n-1) ending with L not including f(n-4))
        f(n) = f(n-1) + f(n-1) - f(n-4)
        f(n) = 2*f(n-1) - f(n-4)
        
        NOTES on the recurrences part for f(n-1)
            we are saying we having rewardable strings at f(n-1) but by doing so, we incur a violation, i.e we are over counting
            so we need to adjust this overcount by decrementing by f(n-4)
            
        we store all valuies f(n) for i up n
        
        now what about inlcuding A
            only 1 A is allowed
            
        two cases
        1. No A is present, in thie case, the number of rewardblse strings is the same as f(n)
        2. A single A is present
            A can be present anywhere in a string of length N
            [...]A[....]
            [string[i-1:]]+A[i]+[string[n-i]]
            so for an A at position i we have an additional
                f(i-1)*f*(n-1)
        
        '''
        #functino f(n) for string ending in LP
        memo = {}
        mod = 10**9 + 7
        def fn(n):
            #if there are no days using LP only, we can get a rewardable string
            if n == 0:
                return 1
            #using only L,P, lenght 1 two ways
            if n == 1:
                return 2
            #lenght 2, 4 ways obvie
            if n == 2:
                return 4
            #length 3, well for length 2, we alrady had 4 ways, but only 3 of those 4 can be made rewardeable by adding L or P
            #so 3 + 4 = 7
            if n == 3:
                return 7
            #memo
            if n in memo:
                return memo[n]
            #recurse
            ans = (2*fn(n-1) - fn(n-4)) % mod #the mod here fucks everything upppp
            memo[n] = ans
            return ans
        
        #first find all rewardable strinsg using only LP for all i from 1 to N
        all_LP_rewardable_strings = [0]*(n+1)
        all_LP_rewardable_strings[0] = 1
        
        for i in range(1,n+1):
            all_LP_rewardable_strings[i] = fn(i)
        
        #to include A, we can place it anways in a rewardable string made up only LP
        #first we include anwer for all_LP_rewardable_strings
        #then we need to increment this by using A in all positions for string lengths i to N
        ans = all_LP_rewardable_strings[-1]
        for i in range(1,n+1):
            ans += all_LP_rewardable_strings[i-1]*all_LP_rewardable_strings[n-i]
            ans %= mod
        
        return ans % mod

#bottom up
class Solution:
    def checkRecord(self, n: int) -> int:
        '''
        converting top down to bottom up
        '''
        LP_rewardable = [0]*(n+1)
        mod = 10**9 + 7
        
        LP_rewardable[0] = 1
        
        for i in range(1,n+1):
            if i == 1:
                LP_rewardable[i] = 2
            if i == 2:
                LP_rewardable[i] = 4
            if i == 3:
                LP_rewardable[i] = 7
            else:
                LP_rewardable[i] = (2*LP_rewardable[i-1] - LP_rewardable[i-4]) % mod
        
        ans = LP_rewardable[-1]
        for i in range(1,n+1):
            ans += LP_rewardable[i-1]*LP_rewardable[n-i]
            ans %= mod
        
        return ans % mod

#############################
# 224. Basic Calculator (REVISTED)
# 21NOV22
##############################
class Solution:
    def calculate(self, s: str) -> int:
        '''
        without string reversal, we just take the minus sign and switch, i.e minus becomes + negative
        which makes the expression obey associtvity
        so we don't need to reverse
            the problem is that expressions could be deeply ensted
            (A-(B-C))
            we need to evaluate the exrpession on the go
            i.e, we don't need to keep adding back on the stack after a completed expression
            
        algo:
            1. iterate exrpession one char at at ime
            2. careful when reading digits and no difits
                this is char is read as a difit we need to form the operand by muluptuing by 10 *duh
            3. whenever we encounter an operatore such + or - we first evluate the epxression to the left and then save this sign\
                +.i,) makr the end of an operand
            4. if char is opening parenth,we just puch the result calulcated on to the stack and start a new fresh expression
            5. if char is closing
                first calculate the exrpession to the left
        '''
        stack = []
        operand = 0
        res = 0 # For the on-going result
        sign = 1 # 1 means positive, -1 means negative  

        for ch in s:
            if ch.isdigit():

                # Forming operand, since it could be more than one digit
                operand = (operand * 10) + int(ch)

            elif ch == '+':

                # Evaluate the expression to the left,
                # with result, sign, operand
                res += sign * operand

                # Save the recently encountered '+' sign
                sign = 1

                # Reset operand
                operand = 0

            elif ch == '-':

                res += sign * operand
                sign = -1
                operand = 0

            elif ch == '(':

                # Push the result and sign on to the stack, for later
                # We push the result first, then sign
                stack.append(res)
                stack.append(sign)

                # Reset operand and result, as if new evaluation begins for the new sub-expression
                sign = 1
                res = 0

            elif ch == ')':

                # Evaluate the expression to the left
                # with result, sign and operand
                res += sign * operand

                # ')' marks end of expression within a set of parenthesis
                # Its result is multiplied with sign on top of stack
                # as stack.pop() is the sign before the parenthesis
                res *= stack.pop() # stack pop 1, sign

                # Then add to the next operand on the top.
                # as stack.pop() is the result calculated before this parenthesis
                # (operand on stack) + (sign on stack * (result from parenthesis))
                res += stack.pop() # stack pop 2, operand

                # Reset the operand
                operand = 0

        return res + sign * operand   

##############################
# 1926. Nearest Exit from Entrance in Maze
# 21NOV22
##############################
#close one
class Solution:
    def nearestExit(self, maze: List[List[str]], entrance: List[int]) -> int:
        '''
        bfs obvie to find the nearext exit
        exit is defind as the an empty cell at the border
        '''
        rows = len(maze)
        cols = len(maze[0])
        dirrs = [(1,0),(-1,0),(0,-1),(0,1)]
        
        q = deque([(entrance[0],entrance[1],0)]) #store as (i,j) steps to entrance
        global_seen = set([])
        global_seen.add((entrance[0],entrance[1]))
        
        while q:
            curr_x,curr_y,steps = q.popleft()
            #at the border, and it must have been a '.'
            if curr_x == 0 or curr_x == rows or curr_y == 0 or curr_y == cols:
                return steps
            for dx,dy in dirrs:
                neigh_x = curr_x + dx
                neigh_y = curr_y + dy
                #bounds check
                if 0 <= neigh_x < rows and 0 <= neigh_y <cols:
                    #empy space
                    if maze[neigh_x][neigh_y] == '.':
                        #if i have yet to see it
                        if (neigh_x,neigh_y) not in global_seen:
                            global_seen.add((neigh_x,neigh_y))
                            q.append((neigh_x,neigh_y,steps+1))
        
        return -1
                            

#think about mutating board next time
class Solution:
    def nearestExit(self, maze: List[List[str]], entrance: List[int]) -> int:
        rows, cols = len(maze), len(maze[0])
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
        
        # Mark the entrance as visited since its not a exit.
        start_row, start_col = entrance
        maze[start_row][start_col] = "+"
        
        # Start BFS from the entrance, and use a queue `queue` to store all 
        # the cells to be visited.
        queue = collections.deque()
        queue.append([start_row, start_col, 0])
        
        while queue:
            curr_row, curr_col, curr_distance = queue.popleft()
            
            # For the current cell, check its four neighbor cells.
            for d in dirs:
                next_row = curr_row + d[0]
                next_col = curr_col + d[1]
                
                # If there exists an unvisited empty neighbor:
                if 0 <= next_row < rows and 0 <= next_col < cols \
                    and maze[next_row][next_col] == ".":
                    
                    # If this empty cell is an exit, return distance + 1.
                    if 0 == next_row or next_row == rows - 1 or 0 == next_col or next_col == cols - 1:
                        return curr_distance + 1
                    
                    # Otherwise, add this cell to 'queue' and mark it as visited.
                    maze[next_row][next_col] = "+"
                    queue.append([next_row, next_col, curr_distance + 1])
            
        # If we finish iterating without finding an exit, return -1.
        return -1

###############################
# 279. Perfect Squares (REVISTED)
# 22NOV22
################################
#yes!!
#sometimes it passes and sometimes it does not!
class Solution:
    def numSquares(self, n: int) -> int:
        '''
        i can generate perfect squares up to n
        then use dp to minimize
        '''
        #create list for squares up to n
        squares = set([1])
        curr_square = 1
        while curr_square*curr_square < n:
            curr_square += 1
            squares.add(curr_square*curr_square)
        
        
        
        memo = {}
        def dp(n):
            #bottome cases, the number if a square already!
            if n in squares:
                return 1
            if (n) in memo:
                return memo[(n)]
            ans = float('inf')
            for sq in squares:
                if n - sq >= 0:
                    ans = min(ans, dp(n-sq) + 1)
            
            memo[(n)] = ans
            return ans
        
        return dp(n)

#bfs review
class Solution:
    def numSquares(self, n: int) -> int:
        # list of square numbers that are less than `n`
        square_nums = [i * i for i in range(1, int(n**0.5)+1)]
    
        count = 0
        q = deque([n])
        while q:
            count += 1
            N = len(q)
            # construct the queue for the next level
            for _ in range(N) :
                curr_n = q.popleft()
                for square_num in square_nums:    
                    if curr_n == square_num:
                        return count  # find the node!
                    elif curr_n < square_num:
                        break
                    else:
                        q.append(curr_n - square_num)
        return level
                

###############################
# 907. Sum of Subarray Minimums
# 26NOV22
###############################
#brute force
class Solution:
    def sumSubarrayMins(self, arr: List[int]) -> int:
        '''
        turns out not to be dp at all but rather a monotoinc stack
        try out brute force first,
        we don't need to all subarrays, just all mins between any given i,j
        '''
        N = len(arr)
        ans = 0
        
        for i in range(N):
            curr_min = arr[i]
            for j in range(i,N):
                curr_min = min(curr_min,arr[j])
                ans += curr_min
                
        
        return ans

class Solution:
    def sumSubarrayMins(self, arr: List[int]) -> int:
        '''
        generating ranges is the most expensive part of the algorithm 
        focus on the current element in the range instead of the range
        for this element to be a min, every element in this range must be greater than it, otherwise we would have a new min
        
        intution:
            once we have a smallest element in a given range (i,j)
            we can determing the number of subarrays in this range that contain this element
            becaue the eleemnt is smalles in the range, it will also be the smallest in all subarrays
            the count of subarray multiplied by the smallest element in this current range will give us the contribution to the sum
            
        
        so we figure out the count of subarrays with this smallest elemeent
        
        example:  [0, 3, 4, 5, 2, 3, 4, 1, 4]
        question, find number of subarrays with 2 as the samllest interger
        2 is smallest in the range [3, 4, 5, 2, 3, 4] or (1,6)
        in the given range (1,6), find the count of subarrays which contain 2
        
        Each subarray is a continuous series of elements that contains 2 from the given range. So to count them, we can count every subarray that starts before 2 or at 2, and ends after 2 or at 2.
        
        break current subarray into left: less than 2, at 2 and greather than 2
        to find the count we just do:
            (index of current min - index of start of left)*(index end of right - index of current min)
            
        we can get the total count by multiplying two numbers - the count of elements before (and including) 22 and the count of elements after (and including) 2.
        
        So, if we know the count of subarrays where each element is the smallest, we can deduce the amount each element will contribute to the final summation. 
        
        Now the only remaining part of the puzzle is how to get the range in which each element is the smallest
        
        . For this, we find the nearest element on the left, which is less than itself. Then, find the closest element on the right, which is less than itself. If ii and jj are the indices of these elements on the left and right, then [i + 1, j - 1][i+1,j−1] indices create our range
        
        we can use montonic stack to calculate the previous smallest element and th next smallet element (very similary to LC 84)
        
        we only care about montonic increasing stack
        
        review:
            keep pushing elements on to stack so lon as they are increasing
            
        how does this help:
            As a new item gets added to the stack, older items are removed from the top if they are bigger. In other words, the items that are getting popped must be greater than or equal to the incoming element. o, every time an item is popped, we get to know about its next smaller item.
            
        If the stack becomes empty at the time of removal of an item, it indicates that the outgoing item is the smallest item seen so far.
            
        Also note that once the process is complete, the stack contains a series of items sorted in increasing order. These are also the items that have no smaller items after themselves. And their previous smaller items are stored right below them in the stack.
        
        edge cases:
             we should make sure that we don't count the contribution by an element twice. This is possible in the cases such as [2, 2, 2].
             
             while finding boundary elements for a range, we look for elemenets that are strictly less than the current element to on the left
             to decide the right boundary, we look for elemeents which are less than or equal to the current eleemnt
             
             
        algo:
        Declare a monotonically increasing stack stackstack, and a variable to hold the summation of minimums sumOfMinimumssumOfMinimums.

        Create a monotonically increasing stack. Iterate index ii from 00 to nn (inclusive) where nn is the length of the given array arrarr. While in practice, in a 00-indexed array, index values extend until n - 1n−1 only, we use value nn to indicate that we have reached the end of the array, and everything left in the stack can then be removed.

        Do the following for each index ii in the array arrarr -

        If the stack isn't empty, pop all the items from the top until ii has reached nn or the item at the top of the stack, stackTop <= arr[i]stackTop<=arr[i].

        According to our constraint, the stack can contain only the increasing items. So, before we push the current index ii into the stack, we need to ensure that stackTopstackTop is smaller than the current item, arr[i]arr[i]. So, the items bigger than or equal to the current item, are removed from the stack top.

        Please note that we must be careful about duplicate elements in the array. So, while considering the next smaller items, we also allow equal elements. When it comes to previous smaller items, though, we keep them strictly smaller (not equal).

        For each item midmid, popped from the stack, we get the range in which it is minimum. The range is defined by all the items between the previous smaller item and the next smaller item.

        The next smaller item's index is ii. The previous smaller item's index comes from the current top of stack. If the stack is empty, we consider it -1−1.

        Calculate the contribution of the element as -

        contribution = arr[mid] * (i - mid) * (mid - previousSmallerIndex)contribution=arr[mid]∗(i−mid)∗(mid−previousSmallerIndex)

        When ii reaches nn, we would have pushed all the array elements into the stack. Some of them would have been removed as well. The remaining items are the ones that have no smaller items after them. So we can consider the array's length as the nextSmallerIndexnextSmallerIndex for them. At the same time, the previousSmallerpreviousSmaller index would be the item below them in the stack.

        We can use the same logic as explained in the previous approach to calculate their contribution.

        This contributioncontribution gets added to the running total of minimums. sumOfMinimums += contributionsumOfMinimums+=contribution

        Because all the bigger items have already been removed from the stack, we can now push the index ii into the stack.

        Return the running total sumOfMinimumssumOfMinimums as the final answer (because this number could be huge, return the mod with the given number

            
        '''
        mod = 10**9 + 7
        stack = []
        sum_of_mins = 0
        
        for i in range(len(arr)+ 1): #the case where we have to to the end of the array in which case we have all increaing
            #when i reach the length of the array
            #indicates that all the elements have been processed and the rmeaining elements on the stack need to be popped out
            
            while stack and (i == len(arr) or arr[stack[-1]] >= arr[i]):
                
                 # Notice the sign ">=", This ensures that no contribution
                # is counted twice. right_boundary takes equal or smaller 
                # elements into account while left_boundary takes only the
                # strictly smaller elements into account
                
                #do this every time, since the array is montonic increasing, we have found a place where nums[i] can be a minimum
                mid = stack.pop()
                left = -1 if not stack else stack[-1]
                right = i
                
                #contribution to min sum
                count = (mid - left)*(right - mid)
                sum_of_mins += count*arr[mid]
            
            stack.append(i)
            
        return sum_of_mins % mod

#################################
# 446. Arithmetic Slices II - Subsequence (REVISTED)
# 28NOV22
#################################
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        '''
        brute force would use dfs to generate all paths
        
        to determine an arithmetic sequence we need the first or last number in the sequecnes, and the common differece d
        if the elemenet we want to add has common difference d, we can extend it
        
        one transition that may work is
        
        dp(i,d) = number of arithmetic subsequences ending with A[i] and common diffeerence d
        
        note:
            so we could say the number is sum of dp(i,d) for i in range(N) for d in all possible D
        
        further we can say:
            for all j < i: dp(i,nums[i]-nums[j]) += dp(j,nums[i]-nums[j])
            
            but if all dp(i,d) are set to 0, how can we form a new arithmetic subsequence is there no existing ones before
            
        we define weak arithmetic subsequences as subseqs that consist of at least two elements and if the difference between any to is the same which have two properties:
            1. for any pair (i,j) such that i != j nums[i] and nums[j] can always form a weal arithemtic subseq
            2. if we can append a new element to a weak arithmetic subseq and keep it arithemtic, then this new subsequ must be arithemtic
            
        now we can say:
            dp(i,d) denotes the numebr of weak arithemtic subseqs ending with nums[i] and diffetn d
            then we can include + 1
            
            for all j < i: dp(i,nums[i]-nums[j]) += dp(j,nums[i]-nums[j]) + 1
            
            the 1 appears because we have made a new one
            
        now the number of all weak arithemtic subseqs is the sum of all dp(i,d), but how canw e get the number of arithmetic subseqs that are not weak?
        
            we can directly compute the number of weak arithemtic subseqs, which is just nC2:
                n*(n-1) / 2
                
            
            
        '''
        memo = {}
        N = len(nums)
        #use recurrences
        def dp(i,d):
            if i == N:
                return 0
            if (i,d) in memo:
                return memo[(i,d)]
            count = 0
            for j in range(i):
                diff = nums[i] - nums[j]
                count += dp(j,diff) + 1
            
            memo[(i,d)] = count
            return count
        
        #find all possible arithmetic inlcuding weak
        all_counts = 0
        for i in range(N):
            all_counts += dp(i,nums[i])

        #remove duplicates from weak
        return all_counts - (N*(N-1)//2)

#another recursive approach
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        '''
        Consider an element to be part of an Arithmetic Progression with
diffence d. See if the next term(s) exists in the array, get their
indexes and recursively call for them increasing the count by 1.
Recursive function will have 3 parameters- index,difference and count.
Ignore the dp memoization and understand the recursion for better intuition.
        '''
        #storing the indexes of the elements in dictionary a with values as a key
        #to check for elements using indieces
        N = len(nums)
        counter = defaultdict(list)
        for i,num in enumerate(nums):
            counter[num].append(i)
                
        print(counter)
        dp={}
        def rec(i,d,c):
            if (i,d,c) in dp:
                return dp[(i,d,c)]
            
            total=0
            if c>=3:
                total+=1
            if nums[i]+d in counter:
                for j in counter[nums[i]+d]:
                    if j>i:
                        total+=rec(j,d,c+1)
            dp[(i,d,c)]=total
            return total
        ans=0
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                #we alwayrs try to extend the current weak arithmetic sequence
                ans+=rec(j,nums[j]-nums[i],2)
        return ans

############################################
# 2225. Find Players With Zero or One Losses
# 28NOV22
############################################
class Solution:
    def findWinners(self, matches: List[List[int]]) -> List[List[int]]:
        '''
        we are given a matches history where there are len(matches) matches
        and each matches[i] indicates that player matches[i][0] won and player matches[i][1] lost
        
        we want the indices ot players that have not lost any matches (i.e they won all their matches)
        we want idncies of players that have lost exactly one match
        
        only include players that have played at least one match
        '''
        #keep count of winnders and losers
        winners = Counter([win for win,loss in matches])
        losers = Counter([loss for win,loss in matches])
        
        
        #losers is easy, just traverse losers and found counts == 1
        loser_idxs = [index for index,count in losers.items() if count == 1]
        
        winner_idxs = []
        for w in winners:
            if w not in losers:
                winner_idxs.append(w)
        
        #sort
        loser_idxs.sort()
        winner_idxs.sort()
        
        return [winner_idxs,loser_idxs]

#one pass three hash sets
class Solution:
    def findWinners(self, matches: List[List[int]]) -> List[List[int]]:
        '''
        we can do this in one pass
        just have three buckets
            1. zero loss, hold player id with only zero losses
            2. one loss, hold player id with ONLY one loss
            3. more loss, holds player id's with more than one less
        '''
        zero_loss = set()
        one_loss = set()
        more_losses = set()
        
        for winner, loser in matches:
            # Add winner
            if (winner not in one_loss) and (winner not in more_losses):
                zero_loss.add(winner)
            # Add or move loser.
            if loser in zero_loss:
                zero_loss.remove(loser)
                one_loss.add(loser)
            elif loser in one_loss:
                one_loss.remove(loser)
                more_losses.add(loser)
            elif loser in more_losses:
                continue
            else:
                one_loss.add(loser)          
            
        return [sorted(list(zero_loss)), sorted(list(one_loss))]


class Solution:
    def findWinners(self, matches: List[List[int]]) -> List[List[int]]:
        '''
        keep seen set for all player ids, and count up the losse
        then second pass look through losses
        '''
        seen = set()
        count_losses = Counter()
        
        zero_loss = []
        one_loss = []
        
        for w,l in matches:
            seen.add(w)
            seen.add(l)
            count_losses[l] += 1
        
        for player in seen:
            count = count_losses[player]
            if count == 0:
                zero_loss.append(player)
            elif count == 1:
                one_loss.append(player)
        
        return [sorted(zero_loss),sorted(one_loss)]

#counting sort
class Solution:
    def findWinners(self, matches: List[List[int]]) -> List[List[int]]:
        '''
        we can us bucket sort, since we know upper and lower limits of a player id
        this stems from the fact that we only need to count losses
        a win is a zero loss
        
        for count array we initalize each player index to -1
        -1 means player i has not played yet
        0 means player i has played and hase no loss
        1 means player i has played and has 1 loss
        >1 means player 1 has played an has more than one loss
        
        
        '''
        loss_count = [-1]*10001
        
        for w,l in matches:
            if loss_count[w] == -1:
                loss_count[w] = 0
            if loss_count[l] == -1:
                loss_count[l] = 1
            else:
                loss_count[l] += 1
        
        ans = [[],[]]
        
        for i in range(10001):
            if loss_count[i] == 0:
                ans[0].append(i)
            if loss_count[i] == 1:
                ans[1].append(i)
        
        return ans

