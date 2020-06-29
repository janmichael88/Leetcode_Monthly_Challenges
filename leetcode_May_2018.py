class Solution(object):
    def findComplement(self, num):
        """
        :type num: int
        :rtype: int
        """
        #first convert to binary
        def DecimalToBinary(num): 
	        if num > 0:
		        return(str(DecimalToBinary(num//2))+str(num%2))
        
        #get the string
        a = DecimalToBinary(num).split("None")[1]
        complement = 0
        for i in range(0,len(a)):
            complement += ((1 - int(a[i]))*2**(len(a)-i-1))
        return complement
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def isCousins(self, root, x, y):
        """
        :type root: TreeNode
        :type x: int
        :type y: int
        :rtype: bool
        """
        def isSibling(root,x,y):
            #base case
            if root is None:
                return 0
            return ((root.left == x and root.right ==y) or 
            (root.left == x and root.right == y)or
            isSibling(root.left, x, y) or
            isSibling(root.right, x, y)) 
        
        #define recursive function to find level of Node 'ptr'
        def level(root, ptr, lev): 
      
            # Base Case  
            if root is None : 
                return 0 
            if root == ptr:  
                return lev 

            # Return level if Node is present in left subtree 
            l = level(root.left, ptr, lev+1) 
            if l != 0: 
                return l
            # Else search in right subtree 
            else:
                return level(root.right, ptr, lev+1) 
        
        #def is cousins
        def isCousin(root, x,y):
            #the two nodes should be at the same level but not from the same parent
            if ((level(root,x,1) == level(root, y, 1)) and 
            not (isSibling(root, x, y))):
                return 1
            else:
                return 0
        return isCousin(root,x,y)


class Solution(object):
    def checkStraightLine(self, coordinates):
        """
        :type coordinates: List[List[int]]
        :rtype: bool
        """
        if len(coordinates) == 2:
            return 1
        first_point = coordinates[0]
        second_point = coordinates[1]
        v_slope = None
        slope = None
        to_return = 1
        #if vertical slope
        if first_point[0] == second_point[0]:
            v_slope = first_point[0]
        else:
            slope = (second_point[1] - first_point[1]) / (second_point[0]-first_point[0])
        #y - y_1 = m(x-x_1)
        for i in range(0,len(coordinates)):
            #regular slope case
            if slope is not None:
                if coordinates[i][1] - first_point[1] != slope*(coordinates[i][0]-first_point[0]):
                    to_return = 0
                    break
                else:
                    continue
            if v_slope is not None:
                for i in range(0,len(coordinates)):
                    if coordinates[i][0] != v_slope:
                        to_return = 0
                        break
                    else:
                        continue
        return to_return


'''
1.......16, where 16 is perfect
the mid is 16+1 / 2 =8.5
~72.25

72.25 > num, not in the right side
adjust righto over  on the left


1...8.5
mid is now 9.5 /2...4.75

'''

class Solution(object):
    def isPerfectSquare(self, num):
        """
        :type num: int
        :rtype: bool
        """
        if num < 0:
            return False
        if num == 0:
            return True
        
        #binary search across all numbers
        left = 1
        right = num
        
        while left <= right:
            mid = (left+right) / 2
            midsqrd  = mid*mid
            if num == midsqrd:
                return True
            else:
                if midsqrd > num:
                    right = mid -1
                else:
                    left = mid + 1
        return False


class Solution(object):
    def findJudge(self, N, trust):
        """
        :type N: int
        :type trust: List[List[int]]
        :rtype: int
        """
        import numpy as np
        judge = -1
        '''
        1: 3,4
        2: 3,4
        4: 3
        
        so its three
        
        1: 2
        so its 2
        
        1:3
        2:3
        so its three
        
        1:3
        2:3
        3:1
        
        no judge
        
        1:2
        2:3
        
        no judge
        '''
        if N == 1:
            judge = 1
        #create the peple list and trust counts
        count = np.zeros(N+1)
        for t in trust:
            count[t[0]] -= 1
            count[t[1]] += 1
        for i in range(1,N+1):
            if count[i] == N - 1:
                judge = i
        return judge

class Solution(object):
    def floodFill(self, image, sr, sc, newColor):
        """
        :type image: List[List[int]]
        :type sr: int
        :type sc: int
        :type newColor: int
        :rtype: List[List[int]]
        """
        #ge dimnesions
        R,C = len(image), len(image[0])
        #get the starting color
        color = image[sr][sc]
        if color == newColor:
            return image
        def dfs(r,c):
            if image[r][c] == color:
                image[r][c] = newColor
                if r >= 1: dfs(r-1, c)
                if r+1 < R: dfs(r+1, c)
                if c >= 1: dfs(r, c-1)
                if c+1 < C: dfs(r, c+1)
        dfs(sr,sc)
        return image

'''

[ 1,1,2,3,3]
get the middle
check middle+1
not the same as middle
its on the left side
[1,1,2]
check middle - 1
they;re the same

[1,1,2,2,3]
'''

class Solution(object):
    def singleNonDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums=nums
        low = 0
        high = len(nums) - 1
        
        def search(nums, low, high):
            #base cases
            if low > high:
                return None
            if low == high:
                return nums[low]

            #get the middle point
            mid = low + (high - low) / 2

            #if mid is even, and the element next to mid the same, then
            if mid % 2 == 0:
                if nums[mid] == nums[mid+1]:
                    return search(nums=nums, low=mid+2, high = high)
                else:
                    return search(nums=nums, low=low, high=mid)
            #for odd
            else:
                if nums[mid] == nums[mid-1]:
                    return search(nums=nums,low=mid+1,high=high)
                else:
                    return search(nums=nums, low=low, high=mid-1)
        return search(nums=nums,low=low,high=high)

#remove k digits


'4664564' k =5 want 44
'464564', k=4
'44564', k = 3
'4564', k = 2
'464', k = 1
'44', k = 0

'612113', k = 3 want 113
'12113', k =  2
'1113' k = 1
'113'

'''
function that takes in a string of numbers
compares first with second
if first > second
drop first
return number
if first < second
drop second
return number
recurse(number)
keep going until length of returned string is equalt to length  of starting string minus k

'''
class Solution(object):
    def removeKdigits(self, num, k):
        """
        :type num: str
        :type k: int
        :rtype: str
        """
        def reduce(num):
            if int(num[0]) > int(num[1]):
                return reduce(num[1:])
            else:
                return reduce(num[0]+num[2:])
class Solution(object):
    def removeKdigits(self, num, k):
        """
        :type num: str
        :type k: int
        :rtype: str
        """
        if len(num) <= k: 
            return '0'
        def reduce(num):
            if int(num[0]) > int(num[1]):
                return num[1:]
            else:
                return num[0]+num[2:]
        to_return = num
        while k>0:
            to_return = reduce(to_return)
            k -= 1
        return to_return.lstrip('0')

#the stack version
class Solution:
    def removeKdigits(self, num: str, k: int) -> str:
        # edge case
        if len(num) <= k: return '0'
        
        stack = []
        for i, digit in enumerate(num):
            while k > 0 and stack and int(stack[-1]) > int(digit):
                # remove the digit
                stack.pop()
                k -= 1
            stack.append(digit)
        
        while k > 0:
            stack.pop()
            k -= 1
            
        ans = ''.join(stack).lstrip('0')
        return ans if ans else '0'


## Trie problem
#nested dictionary
#last dict if last key with "#" as value marking the end of three
class Trie(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        #create root, which is just the dictionary
        self.root = dict()

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: None
        """
        #get the structure
        p = self.root
        for char in word:
            #if char isn't in the dict
            if char not in p:
                #add char to dict with another dict
                p[char] = dict()
            #get new root
            p = p[char]
        p["#"] = True
    
    def find(self,prefix):
        p = self.root
        for char in prefix:
            if char not in p:
                return None
            p = p[char]
        return p
        
                
    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        node = self.find(word)
        return node is not None and "#" in node
        

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        return self.find(prefix) is not None


#kadanes algo, naive
#set max sum to zero
max_sum = None
test = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
for i in range(0,len(test)):
	sub_array = test[i]
	if max_sum is None:
		max_sum = sub_array
	#set first
	for j in range(i+1,len(test)):
		#get sum of array
		sub_array = sum(test[i:j])
		if sub_array > max_sum:
			max_sum = sub_array

print(max_sum)

test = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
def max_subarry(numbers):
	best_sum = float('-inf')
	current_sum = 0
	for x in numbers:
		current_sum = max(x,x+current_sum)
		best_sum = max(best_sum, current_sum)
	return best_sum

print(max_subarry(test))
class Solution(object):
    def maxSubarraySumCircular(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        def max_subarray(A):
            best_sum = float('-inf')
            current_sum = 0
            for x in A:
                current_sum = max(x,x+current_sum)
                best_sum = max(best_sum, current_sum)
            return best_sum
        
        #calculate max ciruclar sub array
        def max_circular(A):
            #just for regular use of kadane's
            max_kadane_reg = max_subarray(A)
            
            #get sum for non contributing elements, which involves inverting each sign and re
            #running kadanes
            max_wrap = 0
            for i in range(0,len(A)):
                max_wrap = A[i]
                A[i] = -A[i]
                
            #add to original sum using Kadane's
            #this is why we flipped each element
            max_wrap = max_wrap + max_subarray(A)
            
            #return the greater of the two
            if max_wrap > max_kadane_reg:
                return max_wrap
            else:
                return max_kadane_reg
        return max_circular(A)

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return head
        #start add first odd and even and continute down
        odd, even = head, head.next
        evenHead = even
        while even and even.next:
            odd.next = even.next
            odd = odd.next
            even.next = odd.next
            even = even.next
        odd.next = evenHead
        return head
class Solution(object):
    def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """
        #scan s with window of lenght p
        #for each window see if all elements are in p,
        #if they are add to starting index
        starting_idxs = []
        for i in range(0, len(s)-len(p)+1):
            #get the window
            window = s[i:len(p)+i]
            #p list to remove
            p_list = list(p)
            for j in range(0,len(window)):
                if window[j] in p_list:
                    p_list.remove(window[j])
            if len(p_list) == 0:
                starting_idxs.append(i)
        return starting_idxs


#note this only get 34/36 on leet code

class Solution(object):
    def checkInclusion(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        #scan the array s1 with a window of length s2
        #for each window see if each char from s1 matches, note scan each time
        #counter increment for each pair
        #return True if counter is the length of s1
        overall_matches = 0
        for i in range(0, len(s2)-len(s1)+1):
            window = s2[i:len(s1)+i]
            s1_list = list(s1)
            for j in range(0,len(window)):
                if window[j] in s1_list:
                    s1_list.remove(window[j])
                if len(s1_list) == 0:
                    overall_matches += 1
                    break 
        if overall_matches > 0:
            return True
        else:
            return False

###not this only gets 73/103

class Solution(object):
    def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """
        #use two count arrays
        #the first count arrya stores frequenies of chars in pattern
        #the second stores freqs oc hars in current window of text
        #time complexity comparing two count arrays is O(1)
        #first store counts of frequencies of patter in frost count array, called P
        #also store counts of frequencies of charecters in frist window of text
        #now run a loop from M to N-1
        #if the two count arrays are identical, we found an occurrence
        #increment that count fo character text in TW
        #decrement counf of first character
        #the last window is not checked by above loop
        
        #create function that returns truee of arr and arr2 are the same, otherwise false
        MAX = 256
        def compare(arr1, arr2):
            for i in range(MAX):
                if arr1[i] != arr2[i]:
                    return False
            return True
        
        #function to seach fro all permutations
        def search(pattern, text):
            M = len(pattern)
            N = len(text)
            
            starting_idxs = []
            
            #count P stores count of all characters of pattern
            #count TW stores count of current window text
            countP = [0]*MAX
            countTW = [0]*MAX
            
            for i in range(M):
                countP[ord(pattern[i])] += 1
                countTW[ord(text[i])] += 1
            #traverse through remaining characters of pattern
            for i in range(M,N):
                #compare counts of current window text with counts of pattern
                if compare(countP,countTW):
                    #add to found index
                    starting_idxs.append(i-M)
                #add current character to current window
                countTW[ord(text[i])] += 1
                #removed the first character of previous window
                countTW[ord(text[i-M])] -= 1
            #check the last window
            if compare(countP,countTW):
                starting_idxs.append(N-M)
            return starting_idxs
        return search(pattern=p, text=s)

from collections import Counter
from collections import defaultdict

class Solution(object):
    def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """
        p_counter = Counter(p)
        window_counter = Counter(s[0:len(p)-1])
        starting_idxs = []
        
        #set left and right,
        #left is the index number
        left,right = [0,len(p)]
        
        while right <= len(s):
            window_counter[s[right-1]] += 1
            if p_counter == window_counter:
                starting_idxs.append(left)
                
            window_counter[s[left]] -= 1
            if window_counter[s[left]] == 0:
                del window_counter[s[left]]
            
            left,right = left+1, right+1
            
        return starting_idxs


#Permutation in a string
from collections import Counter
from collections import defaultdict

class Solution(object):
    def checkInclusion(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        #same as the previous problem, just terminated the loop after it found an occurence
        pattern_counter = Counter(s1)
        window_counter = Counter(s2[0:len(s1)-1])
        
        left,right = 0,len(s1)
        
        while right <= len(s2):
            window_counter[s2[right-1]] +=1
            if pattern_counter == window_counter:
                return True
            
            window_counter[s2[left]] -= 1
            if window_counter[s2[left]] == 0:
                del window_counter[s2[left]]
                
            left,right = left+1, right+1
        return False


'''
empty stack first

'''

class StockSpanner(object):
    
    '''
    [100, 80, 60, 70, 60, 75, 85]
    (85,)
    (75,4) 4 days < 75
    (60,1) 1 day < 60
    (70,2) 2 days < 70
    (60,1) 1 day < 60
    (80,1) 1 day < 80
    (100,1)
    '''

    def __init__(self):
        '''
        array will look like
        [
        (70,2)
        (60,1) reset here
        (70,2)
        (100,1)
        ]
        '''
        self.stack = []
        

    def next(self, price):
        """
        :type price: int
        :rtype: int
        """
        stack = self.stack
        span = 1
        while stack and stack[-1][0] <= price:
            span +=stack[-1][1]
            stack.pop()
        stack.append([price,span])
        return span
        

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        #print an in order traversal
        def traverse(root, k, kth):
            if root.left != None:
                left = traverse(root.left,k,kth)
                if kth[0] == k:
                    return left
            #gets called from smalles to biggest, when it hits here return the val
            kth[0] += 1
            if kth[0] == k:
                return root.val
            #print(root.val)
            if root.right != None:
                right = traverse(root.right, k,kth)
                if kth[0] == k:
                    return right
            return -1
        return traverse(root, k,kth=[0])


class Solution(object):
    def countSquares(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: int
        """
        #left neighbor
        #top neighbor
        #diagonal neighbor
        
        #init counter
        count = 0
        
        #edge case for top row, i.e no top neighbor edge case
        for col in range(0, len(matrix[0])):
            if matrix[0][col] == 1:
                count += 1
        #no left neighbor edge case, adjust 1 to avoid double counting
        for row in range(1,len(matrix)):
            if matrix[row][0] == 1:
                count += 1
        #go into the rest of the our matrix
        for row in range(1, len(matrix)):
            for col in range(1,len(matrix[0])):
                #computation for when new element is only 1
                if matrix[row][col] != 0:
                    #want the minimum of left, top, and diagonal
                    num_squares  = 1 + min(matrix[row-1][col], matrix[row][col-1],
                                  matrix[row-1][col-1])
                    count +=num_squares
                    #update matrix
                    matrix[row][col] = num_squares
        return count

class Solution(object):
    def frequencySort(self, s):
        """
        :type s: str
        :rtype: str
        """
        #create counter 
        #sort the counts using quicksort
        #for each sorted element get the chacter and print it
        
        counter = dict()
        for char in s:
            if char not in counter.keys():
                counter[char] = 1
            else:
                count = counter[char]
                counter[char] = count + 1
        #sort the dictionary
        counter = sorted(counter.items(), key = lambda x: x[1])
        #traverse counter in reverse printing each char by count
        res = ""
        for i in range(1,len(counter)+1):
            res += counter[-i][0]*counter[-i][1]
            
        return res

class Solution(object):
    def intervalIntersection(self, A, B):
        """
        :type A: List[List[int]]
        :type B: List[List[int]]
        :rtype: List[List[int]]
        """
        '''
        res = []
        [0,2] and [1,5]
        1 > 0, grab 1
        2 < 5, grab 2
        make [1,2] and to res
        set max of two to 5
        stil on [1,5] compare next after [0,2]
        [1,5] and [5,10]
        max([1,5]) in [5,10]
        store 5
        '''
        res = []
        i,j = 0,0
        
        while (i < len(A)) and (j < len(B)):
            #check if A[i] intersects B[j]
            #set low to be the start of the intersection
            #set high to be the end of the intersection
            lo = max(A[i][0], B[j][0])
            #go in to the second of the elements
            hi = min(A[i][1], B[j][1])
            
            #add the interval to the results
            if lo <= hi:
                res.append([lo,hi])
            #remove the interval with smaller endpoints
            if A[i][1] < B[j][1]:
                i += 1
            else:
                j += 1
        return res

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def bstFromPreorder(self, preorder):
        """
        :type preorder: List[int]
        :rtype: TreeNode
        """
        if len(preorder) < 1:
            return null
        return create(preorder, [0],float('inf'))
    
def create(preorder, pointer, constraint):
    node = TreeNode(preorder[pointer[0]])
    pointer[0] += 1
    if (pointer[0] < len(preorder)) and (preorder[pointer[0]] < node.val):
        #create the node
        node.left = create(preorder,pointer,node.val)
    #updates the pointer as we recurse
    #now the right side
    if (pointer[0] < len(preorder)) and (preorder[pointer[0]] < constraint):
        #create the right node
        node.right = create(preorder,pointer,constraint)
    return node


'''
[2,5,1,2,5]

[10,5,2,1,5,2]

'''

from collections import Counter

def findAnagrams(s, p):
    """
    :type s: str
    :type p: str
    :rtype: List[int]
    """
    p_counter = Counter(p)
    window_counter = Counter(s[0:len(p)-1])
    starting_idxs = []
    
    #set left and right,
    #left is the index number
    left,right = [0,len(p)]
    
    while right <= len(s):
        window_counter[s[right-1]] += 1
        if p_counter == window_counter:
            starting_idxs.append(left)
            
        window_counter[s[left]] -= 1
        if window_counter[s[left]] == 0:
            del window_counter[s[left]]
        
        left,right = left+1, right+1
        
    return starting_idxs


string1 = 'aaabbabbabababa'
string2 =  'ab'

print(findAnagrams(s=string1, p=string2))


'''
1 2 4 4 6 2

1 2 2 4 2 5
'''

class Solution(object):
    def maxUncrossedLines(self, A, B):
        """
        :type A: List[int]
        :type B: List[int]
        :rtype: int
        """
        '''
        A
    B   0 0 0 0 0
        0 1 1
        0
        0
        
        '''
        N = len(A)
        M = len(B)
        
        #create dynamic array
        dp = [[0 for n in range(N+1)] for m in range(M+1)]
        #pass over first row and first column
        
        for i in range(1,M+1):
            for j in range(1,N+1):
                #if there is an occurence increment one along with the top left
                if A[j-1] == B[i-1]:
                    dp[i][j] = 1 + dp[i-1][j-1]
                #no occurence, take the max of the left or the top    
                else:
                    dp[i][j] = max(dp[i][j-1],dp[i-1][j])
        #the total is just the element in the lower right hand corner            
        return(dp[M][N])

class Solution(object):
    def findMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        #do n squared first to get a handle
        #store max len
        max_length = 0
        for start in range(0,len(nums)):
            #keep track of zeros and ones
            zeros,ones = 0,0
            for end in range(start,len(nums)):
                if nums[end] == 0:
                    zeros += 1
                else:
                    ones += 1
	            if zeros == ones:
	                max_length = max(max_length, (end - start) + 1)
                
        return max_length

class Solution(object):
    def findMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        #scan the array
        #count: increment by 1 when 1 occurs, and decrement when 0 occurs
        #at zero equal amount at index
        #occurences of same count indicate equal zeros ond ones in subarray
        #unique counts between -len(nums) and len(nums)
        #unique_count = empty array of length 2*len(n) + 1
        #make entry into unique count when occurence exists
        #if same count occured, determine length of subarray lying between the indices
        #of the form count:index
        count_map = dict()
        count_map[0] = -1
        
        max_length = 0
        count = 0
        for i in range(0,len(nums)):
            if nums[i] == 0:
                count -= 1
            else:
                count += 1
            #add new count to hash map    
            if count in count_map.keys():
                #update max_length
                max_length = max(max_length,i-count_map[count])
            #if not dump the new count in the mao
            else:
                count_map[count] = i
        return max_length

'''
 N = 4, dislikes = [[1,2],[1,3],[2,4]]
 dict of dislikes? 
1 : 2, 3
2 : 1,2
3 : 1
4 : 2



'''

class Solution(object):
    def possibleBipartition(self, N, dislikes):
        """
        :type N: int
        :type dislikes: List[List[int]]
        :rtype: bool
        """
        #create adjaceny list using dictionary, default dict here
        edges = collections.defaultdict(list)
        for i in range(0, len(dislikes)):
            #grab the two members
            a = dislikes[i][0]
            b = dislikes[i][1]
            edges[a].append(b)
            edges[b].append(a)
        #create labels dict, to label each node pair as having an edge
        #ensure that every node was successfully labeled
        #if label[i] = 0, node has not been labeled
        labels = dict()
        for i in range(1,N+1):
            if ((len(edges[i]) > 0) and (labels[i] == 0) and (isBipartite(edges,i,labels,1) != True)):
                return False
            else:
                return True
            
    ##helper function isBipartite, which is jsut DFS in graph traversal
    def isBipartite(graph, node,labels,label ):
        #if already labeled but not the label we want
        if (labels[node] != 0) and (labels[node] != label):
            return False
        if (labels[node] == label):
            return True
        #process the edges
        for edge in edges[node]:
            if isBipartite(graph, edge,labels,-label):
                return False
        return True
    
 class Solution(object):
    def possibleBipartition(self, N, dislikes):
        """
        :type N: int
        :type dislikes: List[List[int]]
        :rtype: bool
        """
        #define the adjaceny matrix
        adj = defaultdict(list)
        for a,b in dislikes:
            adj[a].append(b)
            adj[b].append(a)
        #create dict to keep track of node markers
        node_markers = {i:None for i in range(N+1)}
        
        #define dfs traversal
        def dfs_traverse(node,label):
            #if it hasn't been marked
            if not node_markers[node]:
                node_markers[node] = label
            else:
                return node_markers[node] == label
            #do dfs in the adjaceny list
            for neighbor in adj[node]:
                if not dfs_traverse(neighbor,2 if label == 1 else 1):
                    return False
            return True
        
        for n in range(1, N+1):
            if not node_markers[n] and not dfs_traverse(n,1):
                return False
        
        return True


'''
0: [0 0 0 0] [0]
1: [0 0 0 1] [0,1]
2: [0 0 1 0] [0,1,1]
3: [0 0 1 1] [0,1,1,2]
4: [0 1 0 0] [0,1,1,2,1]
5: [0 1 0 1] [0,1,1,2,1,2]
6: [0 1 1 0] [0,1,1,2,1,2,2]
7: [0 1 1 1] [0,1,1,2,1,2,2,3]
8: [1 0 0 0]
'''
class Solution(object):
    def countBits(self, num):
        """
        :type num: int
        :rtype: List[int]
        """
        output = [0]
        
        while (len(output) <= num):
            output.extend([i+1 for i in output])
        return output[:num+1]

import collections
class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        #create dictionary showing courses and their pre-reqs
        catalog = collections.defaultdict(list)
        #create dictionary showing the number of prereqs per class
        prereqs = {i:0 for i in range(numCourses)}
        for edge in prerequisites:
            #a: [b...] means to take a you ned b
            a = edge[0]
            b = edge[1]
            #dump into catalog
            catalog[b].append(a)
            #increment the preqs at a accordlingly
            prereqs[a] += 1
       
        #what are our available courses
        available = collections.deque([])
        for i in range(0, numCourses):
            if prereqs[i] == 0:
                available.append(i)
                
        #go through the catalog simulating taking courses
        while (len(available) > 0):
            course = available.popleft()
            numCourses -= 1
            for nextt in catalog[course]:
                #prereqs that this course satisfies
                prereqs[nextt] -= 1
                if prereqs[nextt] == 0:
                    #add to availabile list
                    available.append(nextt)
                    
        return numCourses == 0

import math
class Solution(object):
    def kClosest(self, points, K):
        """
        :type points: List[List[int]]
        :type K: int
        :rtype: List[List[int]]
        """
        def calc_distance(point):
            return math.sqrt(point[0]**2 + point[1]**2)
        
        sorted_points = sorted(points, key =lambda x: calc_distance(x))
        return sorted_points[:K]

class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        M = len(word1)
        N = len(word2)
        #word1 across the row, word2 down the column
        dp = [[0 for i in range(M+1)] for j in range(N+1)]
        #populate edge cases first
        #first row
        for j in range(0,M+1):
            dp[0][j] = j
        #do column
        for i in range(0,N+1):
            dp[i][0] = i
        #traverse through dp array, taking the minimum of top, left, diag left, and increment 1
        #if chars are the same, pull min and do not icrement
        for i in range(1,N+1):
            for j in range(1,M+1):
                #the do nothing case
                if word1[j-1] == word2[i-1]:
                    dp[i][j] = min(dp[i][j-1],dp[i-1][j], dp[i-1][j-1])
                else:
                    dp[i][j] = min(dp[i][j-1],dp[i-1][j], dp[i-1][j-1]) + 1 
        return dp[N][M]


