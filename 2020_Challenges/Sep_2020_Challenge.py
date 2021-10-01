 #########################################
 #Largest Time for Given Digits 01/09/2020
 #########################################
#NICE TRY ONCE AGAIN
class Solution(object):
    def largestTimeFromDigits(self, A):
        """
        :type A: List[int]
        :rtype: str
        """
        '''
        rules, to make it max time 2 is in the front 9 can be either at the end or the second
        first max out the first two spots, which is just 2 + max()
        what if i build it by digit adding zeros
        [1 2 3 4]
        max of 2000 but 
        [2 0 0 0]
        max of 300 but not 400
        [2 3 0 0]
        max of 5
        [2 3 4 0]
        max of 9
        [2 3 4 1]
        '''
        results = [0]*len(A)
        
        maxes = [2000,400,50,9]
        multipliers = [1000,100,10,1]
        pointer = 0
        while pointer < 4:
            #make the current digit
            digit_to_add = float('-inf')
            for i in range(0,len(A)):
                if (A[i]*multipliers[pointer] > digit_to_add) and (A[i]*multipliers[pointer]<= maxes[pointer]):
                    digit_to_add = A[i]
            #remove the digit
            A.remove(A[i])
            results[pointer] = digit_to_add
            pointer += 1
        print results

from itertools import permutations
class Solution(object):
    def largestTimeFromDigits(self, A):
        """
        :type A: List[int]
        :rtype: str
        """
        '''
        warm up before backtracking and recursion,generate all permutations first and check if it is va valid time, output a max
        if it is valid time format, update the max
        '''
        out = ""
        for p in permutations(A):
            if p[0]*10 + p[1] <= 23 and p[2] <= 5: #valid number
                out = max(out, str(p[0])+str(p[1])+":"+str(p[2])+str(p[3]))
        return out

class Solution(object):
    def largestTimeFromDigits(self, A):
        """
        :type A: List[int]
        :rtype: str
        """
        '''
        not using itertools, but backtracking
        based on divide and conquer,swapping, and backtracking
        start with the whole array A[0:n],prefix the first one 
        then look for A[i:n] permuations
        https://leetcode.com/problems/largest-time-for-given-digits/solution/
        function is called permutate(array,start)
        base case is when the start gets to the length of the array, which means there is nothing left to permute, just add the current array
        if we still have some elements that can be permuted
        i.e start < len(array) we backtrack trying out all possible permutations for the posfixes
        when generating a permutation, check if valid time and if greater than the max
        
        '''
        #set max time
        self.max_time = -1
        
        def create_time(permutation):
            self.max_time
            
            h,i,j,k = permutation
            hour = h*10 + i
            minute = j*10 + k
            if hour < 24 and minute < 60:
                #time update
                self.max_time = max(self.max_time, hour*60 + minute) #storing as seconds
        
        def swap(array,i,j):
            #during the permute call, we need to swap teh start element with each of the reamaining elements not in start
            if i != j:
                array[i],array[j] = array[j],array[i]
        
        def permute(array,start):
            #base case, checking the permutation
            if start == len(array):
                create_time(array)
                return
            
            #permute for each element
            for idx in range(start,len(array)):
                #swap current element with starting
                swap(array,idx,start)
                #permuate on the next
                permute(array,start+1)
                #swap again
                swap(array,idx,start)
        
        #invoke
        permute(A,0)
        if self.max_time == -1:
            return ""
        else:
            return "{:02d}:{:02d}".format(self.max_time // 60, self.max_time % 60)

##############################
#Contains Duplicate 09/02/2020
##############################

#sooooo close 30/ 41
import heapq
class Solution(object):
    def containsNearbyAlmostDuplicate(self, nums, k, t):
        """
        :type nums: List[int]
        :type k: int
        :type t: int
        :rtype: bool
        """
        '''
        k difference means that the index j will alwasy be after i, and canonly be at most k away
        what if we use a window of size k?
        then i would check each window for the constraints, but this becomes n squared,\
        which brings us to hint 2, i can use the previous state for the nex state
        i can use a heap for pushing
        example
        [1,5,9,7,2,4,6,8] k = 3, t = 3
        first window
        [1,5,9,7] sort
        [1,5,7,9] compare
        abs(1-5) > t, since its greater than t every never in the windo after that will have differnce greater than t, so move on to the next window
        next window
        [5,7,9,2]
        calling a sort every time costs nlogn, is there a structure that can take a sorted structe and putt it back in place
        in log n time! yues, BST, or in this case sortlest
        '''
        output = False
        N = len(nums)
        for i in range(0,N-k):
            current_window = nums[i:k+i+1]
            current_window.sort()
            for j in range(1,len(current_window)):
                if abs(current_window[0] - current_window[j]) <= t:
                    output = True
                    break
                elif abs(current_window[0] - current_window[j]) > t:
                    break
        return output

class Solution:
    def containsNearbyAlmostDuplicate(self, nums: List[int], k: int, t: int) -> bool:
        if t == 0 and len(nums) == len(set(nums)):
            return False
        for i in range(len(nums)):
            for j in range(i + 1, i + k + 1):
                if j >= len(nums):
                    break
                if abs(nums[i] - nums[j]) <= t:
                    return True
        return False

from sortedcontainers import SortedList
class Solution(object):
    def containsNearbyAlmostDuplicate(self, nums, k, t):
        """
        :type nums: List[int]
        :type k: int
        :type t: int
        :rtype: bool
        """
        '''
        we need a structure that orders a list, and when adding in a new element it can be inserted in log k time,
        so we need to use some for of A BST SortedList
        since this is now sorted, we can do binary searchs for left and right boundaries
        https://leetcode.com/problems/contains-duplicate-iii/discuss/825267/Python3-summarizing-Contain-Duplicates-I-II-III
        '''
        #edge case
        if k<0 or t <0:
            return False
        S = SortedList()
        for i,n in enumerate(nums):
            #if we have gone beyonf the first range of K,need to remove the beginning elements
            if i > k:
                S.remove(nums[i-k-1])
            #binary search for left boundary
            lo,hi = 0, len(S)-1
            left = lo
            while lo < hi:
                mid = (lo+hi) // 2
                if S[mid] < nums[i] - t:
                    lo = mid+1
                else:
                    hi = mid
            left = lo
            #binary search for right boundary
            lo,hi = 0,len(S)-1
            right = lo
            while lo < hi:
                mid = (lo+hi) // 2
                if S[mid] > nums[i] + t:
                    hi = mid
                else:
                    lo = mid + 1
            right = lo
            
            #if left and right are not the same, then we have found anumber in k within +- t
            if left !=right:
                return True
            S.add(n)
        return False
                    
####################################
#Repeated Substring Pattern 09/03/20      
####################################

#nice try again!
#58/120
class Solution(object):
    def repeatedSubstringPattern(self, s):
        """
        :type s: str
        :rtype: bool
        """
        '''
        two pointer hash, or set
        well, since the string will only be up to 10000 long, n squared seems ok
        create substrings, and check if sliding windoiw that substring
        '''
        for i in range(0,len(s)//2):
            curr_substring = s[:i+1]
            window_length = len(curr_substring)
            #now check windows
            num_times = 0
            for j in range(window_length,len(s)-window_length+1):
                if s[j:j+window_length] == curr_substring:
                    num_times += 1
            
            #now check
            if len(s) % window_length == 0:
                if num_times == (len(s) // window_length) -1:
                    return True
        return False
######################################
import re
class Solution(object):
    def repeatedSubstringPattern(self, s):
        """
        :type s: str
        :rtype: bool
        """
        '''
        https://leetcode.com/problems/repeated-substring-pattern/solution/
        using regex
        patten is r'^(.*
        1*$
        ^ matches the beginnig of the string
        (.+) greedy, matches everything, and does not stop at the frist one
        \1* cases the resulting RE to match 1 or more reps of the proceeding RE
        &matches end of string
        '''
        pattern = re.compile(r'^(.+?)\1+$')
        return pattern.match(s)     
###########################################
import re
class Solution(object):
    def repeatedSubstringPattern(self, s):
        """
        :type s: str
        :rtype: bool
        """
        '''
        https://leetcode.com/problems/repeated-substring-pattern/solution/
        concatenation
        if s is 'abcabc'
        s concated is abcabcabcabc
        drop the ends
        bcabcabcab
        abcabc is in there
        
        '''
        return s in (s+s)[1:-1]

############################
#From Time
class Solution(object):
    def repeatedSubstringPattern(self, s):
        """
        :type s: str
        :rtype: bool
        """
        '''
        create patterns up to length N//2 and check if that pattern matches in in window of length pattern created
        '''
        N = len(s)
        
        def pattern_match(n):
            #n is a length
            for i in range(0,N-n,n):
                if s[i:i+n] != s[i+n:i+2*n]:
                    return False
            return True
        
        #invoke starting from 1
        for j in range(1,N//2+1):
            if pattern_match(j):
                return True
        return False



#############################
# Partition Labels 09/04/2020
#############################
#drawing a blank here....

class Solution(object):
    def partitionLabels(self, S):
        """
        :type S: str
        :rtype: List[int]
        """
        '''
        im noticing something here...
        ababcbaca|
                  defegde|
                          hijhklij
        maybe try counting the items first?
        aaa bbb ccc wont work because you're breaking up the string
        you just need to partition
        first count up the char counts
        build the first parition exhausting the first char, end this parition only if the next char is exhausted, it there is a remaining next element expand up to that element
        then parition
        keep going until i reach the end
        use hashmap to mark the last chars position
        
        '''
        #count up chars
        count = dict()
        for s in S:
            if s not in count:
                count[s] = 1
            else:
                count[s] += 1
        last_char = dict()
        
        #get the last chars index number
        i = len(S) -1
        unique_chars = set(count.keys())
        while unique_chars:
            if S[i] in unique_chars:
                last_char[S[i]] = i
                unique_chars.remove(S[i])
            i -= 1
       
        #build paritions
        splitter = 0
        results = [] #marked with split point
        while splitter < len(S):


from collections import defaultdict
class Solution(object):
    def partitionLabels(self, S):
        """
        :type S: str
        :rtype: List[int]
        """
        '''
        we need to do this greedily,starting with the first characetr, we need to make a partition that goes up to the last occruence of the characer, but not only that, we need to make sure that each of the chars in parition do not go into another partition
        '''
        char_pos = defaultdict(list)
        for i,char in enumerate(S):
            char_pos[char].append(i)
            
        results = []
        splitter = 0
        
        #find the last occurence of each letter and move the spitter up to that point
        last_occur = max(char_pos[S[0]])
        while splitter < len(S):
            #get the last occurence
            last_occur = max(last_occur, max(char_pos[S[splitter]]))
            #if the splitter has got to the last point
            if splitter == last_occur: #we've made it to the end of the window
                #results update
                results.append(splitter+1-sum(results))
            splitter += 1
        return results


###############################
# Read N Characters Given Read4
###############################
#so close!
class Solution(object):
    def read(self, buf, n):
        """
        :type buf: Destination buffer (List[str])
        :type n: Number of characters to read (int)
        :rtype: The number of actual characters read (int)
        """
        first_call = read4(buf)
        if n < first_call:
            return n
        output = first_call
        while output <= n:
            output += read4(buf)
        return output

class Solution(object):
    def read(self, buf, n):
        """
        :type buf: Destination buffer (List[str])
        :type n: Number of characters to read (int)
        :rtype: The number of actual characters read (int)
        """
        '''
        https://leetcode.com/problems/read-n-characters-given-read4/solution/
        these are a litte bit of notes on theory
        standard approach is to use the internal buffer of 4 chars
        init the number of copier chars to 0 and number of reach hcars to 4
        its convineint to init the readchars to 4
        #init an interal buffer of 4
        while the number of copies chars is less than N and there are still chars in the file
        read from file inot interla buffer
        copy the cars from internal buuffer into main buffer one by one
        increase copiedChars after each char
        '''
        copied_chars = 0
        read_chars = 4
        #internal buffer
        buf4 = ['']*4


        while copied_chars < n and read_chars == 4:
            read_chars = read4(buf4)
            
            for i in range(read_chars):
                #return statement
                if copied_chars == n:
                    return copied_chars
                #recall buff is currently empty, read4 writes int buf
                buf[copied_chars] = buf4[i]
                copied_chars += 1
        
        return copied_chars

####################################################
# All Elements in Two Binary Search Trees 09/05/2020
####################################################

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def getAllElements(self, root1, root2):
        """
        :type root1: TreeNode
        :type root2: TreeNode
        :rtype: List[int]
        """
        '''
        in order traverse both trees gets nodes list
        traverse both lists creating a new one
        '''
        def traverse(root,nodes=[]):
            if not root:
                return
            traverse(root.left,nodes)
            nodes.append(root.val)
            traverse(root.right,nodes)
            return nodes
        if not root1 and not root2:
            return []
        elif root1 and not root2:
            return traverse(root1)
        elif not root1 and root2:
            return traverse(root2)
        else:
            #both are valid
            nodes1 = traverse(root1)
            nodes2 = traverse(root2)
            return sorted(nodes1)

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def getAllElements(self, root1, root2):
        """
        :type root1: TreeNode
        :type root2: TreeNode
        :rtype: List[int]
        """
        '''
        in order traverse both trees gets nodes list
        traverse both lists creating a new one
        '''
        output1,output2 = [],[]
        def traverse(root,nodes):
            if not root:
                return
            traverse(root.left,nodes)
            nodes.append(root.val)
            traverse(root.right,nodes)
        if not root1 and not root2:
            return []
        elif root1 and not root2:
            traverse(root1,output1)
            return output1
        elif not root1 and root2:
            traverse(root2,output2)
            return output2
        else:
            #both are valid
            traverse(root1,output1)
            traverse(root2,output2)
            #merge sorted lists
            i,j,N1,N2=0,0, len(output1),len(output2)
            results = []
            while i < N1 and j < N2:
                if output1[i] < output2[j]:
                    results.append(output1[i])
                    i += 1
                else:
                    results.append(output2[j])
                    j += 1
            return results + output1[i:] + output2[j:]


#########################
#Image Overlap 09/06/2020
#########################
class Solution(object):
    def largestOverlap(self, A, B):
        """
        :type A: List[List[int]]
        :type B: List[List[int]]
        :rtype: int
        """
        '''
        translate A, use zero padding to keep dims
        traverse A and B, increment count by 1 when A[i][j] == B[i][j]
        how do you translate? O(N!) brute force
        i need to start from B and match A....
        what if i placed B on to A, then moved A around?
        this is going to be a long one, maybe an all day thing? but we are going to go over the leetcode official solutions
        #first shift and count
        a simple idea is that one coould come up with all possible overlapping zones by shifting for all combindationas up,downleft right
        https://leetcode.com/problems/image-overlap/solution/
        images a grid x,and y, where the x,y positions indicates the amounts of shifts in each direction
        also not the a shift in xy of an image, is the same effect as -x,-y from the other
        define a shift and count function, that shifts a matrix M in reference to an image R, and count the overlapping ones
        the range for the x,y shfits is going to be between (0, N-1)
        invoke matrix twice RM MR, update max overlap
        https://leetcode.com/problems/image-overlap/solution/
        '''
        dim = len(A)
        
        def shift_count(x_shift,y_shift, M,R):
            #xy shifts
            #M is the moved matrix and R is the ref matrix
            #recall (1,1) has the same effect as (-1,-1)
            #so i don't need the negative x and y directions, just swap the matrices
            count = 0
            for r_row, m_row in enumerate(range(y_shift, dim)):
                for r_col, m_col in enumerate(range(x_shift, dim)):
                    if M[m_row][m_col] == 1 and M[m_row][m_col] == R[r_row][r_col]:
                        count += 1
            return count
        
        #invoke twice
        max_overlaps = 0
        for y_shift in range(0,dim):
            for x_shift in range(0,dim):
                max_overlaps = max(max_overlaps, shift_count(x_shift,y_shift,A,B))
                max_overlaps = max(max_overlaps, shift_count(x_shift,y_shift,B,A))
        return max_overlaps

from collections import defaultdict
class Solution(object):
    def largestOverlap(self, A, B):
        """
        :type A: List[List[int]]
        :type B: List[List[int]]
        :rtype: int
        """
        '''
        https://leetcode.com/problems/image-overlap/solution/
        Linear Transofrmation
        the drawback of the count shift approach is that we ould scan through those zones filled with zeros over and over, despite the zones being of interest
        filter out cells with ones in both matrics, and apply linear transformation to align the cells
        if we have vector position (0,0) and apply vector (1,1),which is up right, we get (1,1)
        for both matrices, if we hace no zero entries at P_a = X_a,Y_a and P_b =X_b,Y_b, we can allign these clles by using transofrmation vector V_ab = (X_b - X_a, Y_b- Y_a) so that  P_a + V_ab = P_b
        use V_ab as a key to group all the non zeros cells alignments between the two matrices.
        each group represents an overlapping zone
        first filters out non zero cells in each matrix repsectiel
        next do a cartesian product of the non zeros cells, for each pair of products, get the corresponding linear transformation vector
        count the number of pairs that have the same transformation vector, which is alsoe the number ofones in the overlappign sonze
        '''
        #store linear transofrmations dict counter
        lt_dict = defaultdict(int)
        A_ones = []
        B_ones = []
        x_dim,y_dim = len(A[0]),len(A)
        for i in range(y_dim):
            for j in range(x_dim):
                if (A[i][j]==1):
                    A_ones.append((i,j))
                if (B[i][j] ==1):
                    B_ones.append((i,j))
        
        max_overlaps = 0
        for x_a,y_a in A_ones:
            for x_b,y_b in B_ones:
                #compute LT
                lt = (x_b - x_a,y_b - y_a)
                #add to dict
                lt_dict[lt] += 1
                #update max_overlaps
                max_overlaps = max(max_overlaps, lt_dict[lt])
        return max_overlaps

class Solution(object):
    def largestOverlap(self, A, B):
        """
        :type A: List[List[int]]
        :type B: List[List[int]]
        :rtype: int
        """
        '''
        brute force variant from time
        def shifter counter, check shift is in bounds and new shift is equal to 1 and B
        '''
        dim = len(A)
        def shifter(x_shift,y_shift):
            overlaps = 0
            for x in range(dim):
                for y in range(dim):
                    #in bounds and 1 and euqal
                    if 0 <= x +x_shift< dim and 0 <= y +y_shift < dim and A[x+x_shift][y+y_shift] == 1 and B[x][y] == 1:
                        overlaps += 1
            return overlaps
        
        max_overlaps = 0
        for i in range(-dim,dim):
            for j in range(-dim,dim):
                max_overlaps = max(max_overlaps,shifter(i,j))
        return max_overlaps
        

import numpy as np
class Solution(object):
    def largestOverlap(self, A, B):
        """
        :type A: List[List[int]]
        :type B: List[List[int]]
        :rtype: int
        """
        '''
        we can just take the convolution of the two matrices,
        enumerate all shifted matrices B and convolve with A
        remember to pad B with zeros
        dim is 0 to 2n -1
        
        '''
        A = np.array(A)
        B = np.array(B)
        
        dim = len(A)
        #pad be 0 to 2n-1
        B_padded = np.pad(B,dim-1,mode='constant',constant_values=(0,0))
        
        max_overlaps = 0
        for x_shift in range(dim*2-1):
            for y_shift in range(dim*2-1):
                #get the kernel
                kernel = B_padded[x_shift:x_shift+dim, y_shift:y_shift+dim]
                # convolution between A and kernel
                non_zeros = np.sum(A * kernel)
                max_overlaps = max(max_overlaps, non_zeros)
        return max_overlaps


#########################
#Word Pattern 09/07/2020
#########################
#so close! 34/37
class Solution(object):
    def wordPattern(self, pattern, str):
        """
        :type pattern: str
        :type str: str
        :rtype: bool
        """
        '''
        just map to a dictionary
        or the length of a set on pattern does not match the length of set(str)
        '''
        str = str.split(" ")
        if len(set(pattern)) == len(set(str)) and len(str) == len(pattern) :
            return True
        else:
            return False

class Solution(object):
    def wordPattern(self, pattern, str):
        """
        :type pattern: str
        :type str: str
        :rtype: bool
        """
        '''
        just map to a dictionary
        or the length of a set on pattern does not match the length of set(str)
        '''
        
        str_split = str.split(" ")
        if len(set(pattern)) == len(set(str_split)) and len(str_split) == len(pattern) :
            #make the mapper
            mapper  = {k:v for (k,v) in zip(set(str_split),set(pattern))}
            #recreate and see if it matches
            for i in range(0,len(str_split)):
                if mapper[str_split[i]] != pattern[i]:
                    return False
            return True
        else:
            return False

#wohooooo
class Solution(object):
    def wordPattern(self, pattern, str):
        """
        :type pattern: str
        :type str: str
        :rtype: bool
        """
        '''
        just map to a dictionary
        or the length of a set on pattern does not match the length of set(str)
        '''
        
        str_split = str.split(" ")
        if len(set(pattern)) == len(set(str_split)) == len(set(zip(str_split,pattern))) and len(str_split) == len(pattern) :
            return True
        else:
            return False


class Solution(object):
    def wordPattern(self, pattern, str):
        """
        :type pattern: str
        :type str: str
        :rtype: bool
        """
        '''
        https://leetcode.com/problems/word-pattern/solution/
        if thinking naively, use a single hash map mapping char to word, as you scan update the pairs in the hash map. 
        if you see a char already in the map, check whether the current word matches teh char it not, return false. 
        however this does not work for all cases
        the idea is to have two hash maps char to word and word to char
        if char is not in the word mapping check whether that word is also in the char mapping
        if the word is already in the word to char mapping, return false
        else update both
        if char is in the char to word mapping check whether to current word matches the word in the word to chat mappng if not return Fasle
        
        '''
        map_char = {}
        map_word = {}
        
        words = str.split(" ")
        if len(words) != len(pattern):
            return False
        
        for char,word in zip(pattern,words):
            if char not in map_char:
                if word in map_word:
                    return False
                else:
                    map_char[char] = word
                    map_word[word] = char
            else:
                if map_char[char] != word:
                    return False
        return True


##############################################
#Sum of Root to Leaf Binary Numbers 09/08/2020
##############################################
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def sumRootToLeaf(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        '''
        traverse each root leaf path,
        dump into list
        convert to base 10, and add all of them
        '''
        #recursively call update outside vars
        def find_paths(root,current_path,all_paths):
            if not root:
                return
            if not root.left and not root.right:
                all_paths.append(current_path)
                current_path = []
            current_path.append(root.val)
            find_paths(root.left,current_path,all_paths)
            find_paths(root.left,current_path,all_paths)
            return all_paths
        print find_paths(root,[],[])


### YES!!!!
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def sumRootToLeaf(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        '''
        traverse each root leaf path,
        dump into list
        convert to base 10, and add all of them
        '''
        all_paths = []
        def find_paths(root,current_path):
            if not root:
                return
            if not root.left and not root.right:
                all_paths.append(current_path + [root.val])
            else:
                find_paths(root.left, current_path + [root.val])
                find_paths(root.right, current_path + [root.val])
        find_paths(root,[])
        #travers all paths
        result = 0
        for p in all_paths:
            number = 0
            for i in range(0,len(p)):
                number += p[i]*2**(len(p)-i-1)
            result += number
        return result
            

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def sumRootToLeaf(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.total = 0
        
        def pre_order(root,binary):
            if not root:
                return
            pre_order(root.left,binary+str(root.val))
            pre_order(root.right,binary+str(root.val))
            
            if not root.left and not root.right:
                self.total += int(binary+str(root.val),2)
                
        pre_order(root,"")
        return self.total

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def sumRootToLeaf(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        
        def pre_order(root,binary):
            if not root:
                return 0
            
            if not root.left and not root.right:
                return int(binary+str(root.val),2)
            else:
                return pre_order(root.left,binary+str(root.val)) + pre_order(root.right,binary+str(root.val))
                
        return pre_order(root,"")

###################################
#Compare Version Numbers 09/09/2020
####################################
class Solution(object):
    def compareVersion(self, version1, version2):
        """
        :type version1: str
        :type version2: str
        :rtype: int
        """
        '''
        split on periods for both strings
        compare each split
        '''
        v1 = version1.split(".")
        v1 = [int(i) for i in v1]
        v2 = version2.split(".")
        v2 = [int(i) for i in v2]
        
        #need to check lenths
        if len(v1) > len(v2):
            #append zeros to the smaller length up to the largest length
            max_len = len(v1)
            for i in range(max_len-len(v2)):
                v2.append(0)
        #now the other way
        if len(v1) < len(v2):
            #append zeros to the smaller length up to the largest length
            max_len = len(v2)
            for i in range(max_len-len(v1)):
                v1.append(0) 
        #now compare using two pointers, since they are now the same length i can just for looop it
        for i in range(0,len(v1)):
            if v1[i] > v2[i]:
                return 1
            elif v1[i] < v2[i]:
                return -1
        return 0

class Solution(object):
    def compareVersion(self, version1, version2):
        """
        :type version1: str
        :type version2: str
        :rtype: int
        """
        '''
        split on periods for both strings
        compare each split
        '''
        v1 = version1.split(".")
        v2 = version2.split(".")
        N1 = len(v1)
        N2 = len(v2)
        
        for i in range(max(N1,N2)):
            num1 = 0 if i >= N1 else int(v1[i])
            num2 = 0 if i >= N2 else int(v2[i])
            
            if num1 > num2:
                return 1
            elif num1 < num2:
                return -1
        return 0


############################
#Bulls and Cows 10/09/2020
############################
class Solution(object):
    def getHint(self, secret, guess):
        """
        :type secret: str
        :type guess: str
        :rtype: str
        """
        '''
        1807
        7810
        1A3B
        
        1123
        0111
        
        1A
        
        find bulls create A
        find cows create B
        '''
        secret = list(secret)
        guess = list(guess)
        bulls = 0
        char_matches = []
        for i in range(0,len(secret)):
            if secret[i] == guess[i]:
                bulls += 1
                #record matching indices
                char_matches.append(secret[i])
        #from the matches pop them off both secret and guess
        for char in char_matches:
            secret.remove(char)
            guess.remove(char)
            
        cows = 0
        for g in guess:
            if g in secret:
                cows += 1
                secret.remove(g)
        
        return str(bulls)+'A'+str(cows)+'B'

#from tim
from collections import Counter
class Solution(object):
    def getHint(self, secret, guess):
        """
        :type secret: str
        :type guess: str
        :rtype: str
        """
        bulls,cows = 0,0
        s = list(secret)
        g = list(guess)
        
        i,j = 0,0
        while i < len(secret):
            if s[j] == g[j]:
                bulls += 1
                s.pop(j)
                g.pop(j)
            else:
                j += 1
            i += 1
            
        count = Counter(s)
        for foo in g:
            if foo in count and count[foo] > 0:
                cows += 1
                count[foo] -= 1
        
        return "{}A{}B".format(bulls,cows)


##########################
#Maximum Product Subarray
#########################
#TLE exceed
class Solution(object):
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        n squared solution just search all possible subarrays
        '''
        def product(array):
            res = 1
            for i in range(0,len(array)):
                res *= array[i]
            return res
        
        if len(nums) == 1:
            return nums[0]
        N = len(nums)
        max_product = float('-inf')
        for i in range(0,N):
            for j in range(i,N):
                max_product = max(max_product,product(nums[i:j+1]))
                
        return max_product


class Solution(object):
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        if a sub array is all positive, the max product is just all of them
        if a sub array has an even number of negative, the maxproduct is all of them
        mark the indices of pos and negative afte the first scan
        https://leetcode.com/problems/maximum-product-subarray/solution/
        this is a natural extension of Kadanes algo
        we want to keep builiding up combo chains, zeros restart the product, and negatives can negate or increase (odd or even)
        approach,
        while going through nums keep track of the max product up to that number max_so_far and min_so_far
        update max_so_far by:
        current_number in the case that the product has been bad
        product of last_max_so_far and current number this value will be picked if accumualted product has been steadily increasing
        product of last min so far, this value will be picked if the current number is a negative and the combo chain has been disrupted by a single negative number
        '''
        if len(nums) == 0:
            return 0
        
        #initilize max and min
        max_so_far = nums[0]
        min_so_far = nums[0]
        absolute_max = max_so_far #we update this
        for i in range(1,len(nums)):
            current = nums[i]
            #get the local_max
            local_max = max(current,max(max_so_far*current,min_so_far*current))
            #get the local_min
            local_min = min(max_so_far*current,min_so_far*current)
            #update min_so_far
            min_so_far = min(current,local_min)
            #update the max_so_far
            max_so_far = local_max
            #update absolute max
            absolute_max = max(absolute_max,max_so_far)
        return absolute_max

class Solution(object):
    def maxProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        using two dp arrays
        dp_max storing the max product up to i
        dp_min storing the min product up to i
        we needboth because a really small negative number multiplied by another negative can be huge
        traver nums updating both dp
        if nums[i] > 0, the the bigged number can be dp_max[i-1]*nums[i]
        likewise the smallest number if the minimum
        if nums[i] <= 0, then the biggest numbers is the max(dp_min[i-1],nums[i])
        and the smallest is the mi of dp[i-1]*nums[i] or nums[i]
        '''
        N = len(nums)
        if N < 1:
            return 0
        
        dp_max = [0]*N
        dp_min = [0]*N
        
        #initlize
        dp_max[0] = dp_min[0] = nums[0]
        for i in range(1,N):
            #for the positive case
            if nums[i] > 0:
                dp_max[i] = max(dp_max[i-1]*nums[i],nums[i])
                dp_min[i] = min(dp_min[i-1]*nums[i],nums[i])
            #in the negative or zero case
            else:
                dp_max[i] = max(dp_min[i-1]*nums[i],nums[i])
                dp_min[i] = min(dp_max[i-1]*nums[i],nums[i])
        
        return max(dp_max)
        

######################
#Combindation Sum III 09/12/20
######################
#i couldn't get it.. wannnhh!!
#you almost got it!

class Solution(object):
    def combinationSum3(self, k, n):
        """
        :type k: int
        :type n: int
        :rtype: List[List[int]]
        """
        '''
        there is a maximum number that can be generated using 1 to 9
        which is 45, i know that n cannot be greater than 45
        
        '''
        candidates = list(range(1,10))
        results = []
        
        
        def recurse(comb,k,n):
            #base case when to add to our results
            if len(comb) == k:
                results.append(comb)
            
            #recurse for all candidates, making sure not to include duplicates
            for c in candiates:
                #get the first diffeences
                first_diff = n - c
                #now this is just two sum
                new_candidates = candidates.remove(c)
                mapping = {}
                for n in new_candidates:
                    mapping[n] = first_diff - n
                #now look
                for k,v in mapping:
                    if c + k + v = n:
                        comb.append((c,k,v))

class Solution(object):
    def combinationSum3(self, k, n):
        """
        :type k: int
        :type n: int
        :rtype: List[List[int]]
        """
        '''
        there is a maximum number that can be generated using 1 to 9
        which is 45, i know that n cannot be greater than 45
        you almost had it
        https://leetcode.com/problems/combination-sum-iii/solution/
        this is just a back tracking problem
        starting with the candidates, pick 1 at a time, on the n + 1 one, remove the previous candidate
        chose the next one so that n - results[:n-1] = n
        for example 
        k = 3, n = 9
        [1,0,0]
        [1,2,0]
        9 - 2 -1 = 6
        [1,2,6]
        backtrack
        [1,0,0]
        try 3
        [1,3,0]
        9 - 3 - 1 = 5
        [1,3,5]
        backtrack
        to avoid duplicates in a combination, and redundancy in the results, traverse the candidates in order
        
        
        '''
        results = []
        
        
        def back_track(remaining, comb,next_start):
            #base case when to add to our results
            if remaining == 0 and len(comb) == k:
                #when we have found matches andr reduced n to zero
                results.append(list(comb))
                return
            elif remaining < 0 or len(comb) == k:
                #we have gone negative but our comb is equal to k
                return
            
            #recurse for all candidates, making sure not to include duplicates
            for i in range(next_start,9):
                comb.append(i+1)
                back_track(remaining - i - 1,comb,i+1)
                #backtrack to the current choice
                comb.pop()
        
        back_track(n,[],0)
        return results


class Solution(object):
    def combinationSum3(self, k, n):
        """
        :type k: int
        :type n: int
        :rtype: List[List[int]]
        """
        
        results = []
        
        def dfs(comb,num,sum_so_far):
            '''
            comb is the current combination
            num is any of the numbers 1 to 9
            sum_so_far is the sum of our comb list
            '''
            if len(comb) == k:
                if sum_so_far == n:
                    results.append(comb)
                return
            
            #recurse
            for i in range(num,9+1):
                if sum_so_far + i > n:
                    break
                dfs(comb+[i],i+1,sum_so_far+i)
        
        dfs([],1,0)
        return results
        
#############################
#Insert Interval 09/13/2020
###############################
class Solution(object):
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[List[int]]
        :type newInterval: List[int]
        :rtype: List[List[int]]
        """
        '''
        ive done this problem before from LC blind curated 75, lets review it!
        say we have intervals like
        ---- --- -- ---- - - -----
              --------
        this is an overlapping interval
        keep the first segment, get the start part of the second
        on this second segment compare end to the newInterval's start
        assign new start, now keep comparing the start and ends to the new interval
        merge
        '''
        #edge case
        if not intervals:
            return [newInterval]
        
        N = len(intervals)
        newStart,newEnd = newInterval
        output = []
        i = 0
        
        #first add original intervals up to the new start
        while i < N and intervals[i][0] <= newStart:
            output.append(intervals[i])
            i += 1
        #second, append or update or merge using the new interval
        if not output:
            #if nothing was addeddin and in the new one
            output.append([newStart,newEnd])
        elif output[-1][1] < newStart: #this is a start of a merge
            output.append([newStart,newEnd])
        else:
            output[-1][1] = max(output[-1][1],newEnd)
        
        #third, merge the rest of the intervals if necessary
        while i < N:
            newStart,newEnd = intervals[i]
            if output[-1][1] < newStart:
                output.append([newStart,newEnd])
            else:
                output[-1][1] = max(output[-1][1],newEnd)
            i += 1
        return output

class Solution(object):
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[List[int]]
        :type newInterval: List[int]
        :rtype: List[List[int]]
        """
        '''
        create new pairs of intervals as you traverse down the intervals
        when a new interval is present adjust
        maybe use two pointers, and while loop until both hit the last one
        at any one moment you need to examine three intervals
        
        '''
        '''
        --- ----- --- ---- -----
            -------
        '''
        #three stagee
        #edge case empty intervals
        if not intervals:
            return [newInterval]
        
        N = len(intervals)
        i = 0

        output = []
        
        #add intervals whose end is less than the newIntervals start
        while i < N and intervals[i][1] < newInterval[0]:
            output.append(intervals[i])
            i += 1
        
        #merge intervals that share bounds if newINtervals
        missing_interval = newInterval
        while i < N and intervals[i][0] <= newInterval[1]:
            missing_interval[0] = min(missing_interval[0],intervals[i][0])
            missing_interval[1] = max(missing_interval[1],intervals[i][1])
            i += 1
        output.append(missing_interval)
        
        #we are done merging, add in the remaining intervals
        while i < N:
            output.append(intervals[i])
            i += 1
        return output

class Solution(object):
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[List[int]]
        :type newInterval: List[int]
        :rtype: List[List[int]]
        """
        '''
        ive done this problem before from LC blind curated 75, lets review it!
        say we have intervals like
        ---- --- -- ---- - - -----
              --------
        this is an overlapping interval
        keep the first segment, get the start part of the second
        on this second segment compare end to the newInterval's start
        assign new start, now keep comparing the start and ends to the new interval
        merge
        https://www.youtube.com/watch?v=SdxgI6zFuWY&ab_channel=TimothyHChang
        '''
        #keep appending intervals that whose start is less than the newInterval's start
        output = []
        middle = 0
        for start,end in intervals:
            if start < newInterval[0]:
                output.append([start,end])
                middle += 1
            else:
                break #probably done better in a while loop
        #instert merge new up to mid
        if not output and output[-1][1] < newInterval[0]:
            output.append(newInterval)
        else:
            #update
            output[-1][1] = max(output[-1][1],newInterval[1])
            
        #merge the rest
        for start,end in intervals[middle:]:
            if output[-1][1] < start:
                output.append([start,end])
            else:
                output[-1][1] = max(output[-1][1],end)
        return output

##########################
#Max Robber 09/14/2020
##########################
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        this is a classic DP problem
        define f(k) as the largest amount you ran rob from the first k housts
        A_i is the amount of mooney at the ith house
        
        examine at n = 1
        f(1) = A_1
        f(2) = max(A_1,A_2)
        for the third hoous, you have two optioonns:
            rob the third and add its amount
            do not roob the third house, and stick with the current max
        we can define a recusrive relatinship
        f(k) = max(f(k-2) + A_k,f(k-1))
        the recursive solution gets a TLE
        '''
        if not nums:
            return 0
        def rec_max(k):
            if k == 0:
                return nums[0]
            elif k == 1:
                return max(nums[1],nums[0])
            return max(rec_max(k-2)+nums[k],rec_max(k-1))
        
        return rec_max(len(nums)-1)
        

class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        this is a classic DP problem
        define f(k) as the largest amount you ran rob from the first k housts
        A_i is the amount of mooney at the ith house
        
        examine at n = 1
        f(1) = A_1
        f(2) = max(A_1,A_2)
        for the third hoous, you have two optioonns:
            rob the third and add its amount
            do not roob the third house, and stick with the current max
        we can define a recusrive relatinship
        f(k) = max(f(k-2) + A_k,f(k-1))
        the recursive solution gets a TLE
        we chose the base case as f(-1) = f(0) = 0
        '''
        prevMax = 0
        currMax = 0
        for num in nums:
            local_max = currMax
            #updates
            currMax = max(prevMax + num,currMax)
            prevMax = local_max
        return currMax
        
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        max of current house and robbed n-2 hous
        so we had the array
        3   10  9   1   10  1   9   2
        3   10  12  12  22
        '''
        if not nums:
            return 0
        elif len(nums) < 3:
            return max(nums)
        
        amounts = [0]*len(nums)
        
        #init 0 and 1 places
        amounts[0] = nums[0]
        amounts[1] = max(nums[1],nums[0])
        for i in range(2,len(nums)):
            amounts[i] = max(amounts[i-2] + nums[i],amounts[i-1])
        return amounts[-1]
    

################################
#Length of Last Word 09/15/2020
#################################
class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        words = s.split(" ")
        lengths = [len(w) for w in words]
        i = len(lengths) - 1
        while i > 0:
            if lengths[i] != 0:
                return lengths[i]
            i -= 1
            
        return lengths[i]

class Solution(object):
    def lengthOfLastWord(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        single loop
        start counting at the frist occurence of ' ' backwards
        '''
        i = len(s)
        length = 0
        
        while i > 0:
            i -= 1
            if s[i] != ' ':
                length += 1
            elif length > 0:
                return length
        return length


############################
#Inroder Successor of BST II
############################
"""
# Definition for a Node.
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None
"""

class Solution(object):
    def inorderSuccessor(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        '''
        in order successor is the the smallest node.val greater than tha node
        i first need to find the node.val = node
        things we know in a binary tree:
        all nodes to the left of a node are greather than that node
        all nodes to the right of node are less
        more clearly defined, a predeessor is to the left, but the greatest on the left
        a succ is on the right, but the smallest
        there are two possible scenarios here,, a nods a right child ant is succ is womhere in the tree
        to find the succ go right once, and keep going left until you cant
        node has not right, then its succ is womhere up!
        go until the node that is left child of its parent, the answer is the parent
        
        1.if the node has a right child, its succ is somwhere in lower tree, right one, then left as many times as you can until you get a node
        2. node has no right child, and its succ is somehwere up, ascend until the node is a left child of its parent!
        '''
        #the succ is in the lower right
        if node.right:
            node = node.right
            while node.left:
                node = node.left
            return node
        
        #the succ is somewhere up, ascend until node is the left child of its parent
        while node.parent and node == node.parent.right:
            node = node.parent
        return node.parent


########################################
#Maximum XOR of Two Numbers in an ARRAY
########################################
class Solution(object):
    def findMaximumXOR(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        n squared solution is to check two way pairs
        gets TLE
        '''
        maxx = 0
        for i in range(len(nums)):
            for j in range(len(nums)):
                if i != j:
                    maxx = max(maxx, nums[i]^nums[j])
        return maxx


class Solution(object):
    def findMaximumXOR(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        well this one was just way too hard
        https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/discuss/849128/Python-O(32n)-solution-explained
        we want to ask are there two numbers that can be XORED to get the lnogest binary rep
        we can use two sum:
            put all the prefixes of length on to set and then try to find two numbers in this set stuch that their XOR starts with 1
            imagein we have
            1...... find two that give me that
            now find
            11.....
            we again iterate through all numbers and find two tih XOR starting with 11
            it cna happen on hat next stpe we did not find an XOR with 111....
            we don't update in this case
        1. iterate starting from the frist binary rep of the number on the right
        2. for each travered digit update our binary mmaskt 10000,11000,11100......
        3. create set of all possible starts of numbers using num & mask
            on the first iteration it will be the first digit
        4. apply two sum algo: if we gounf two numbers with XOR starting with start, then we are good!
        '''
        result, mask = 0,0
        for i in range(31,-1,-1):
            #create mask 
            #31 0b0
            #30 0b10000000000000000000000000000000
            #29 0b11000000000000000000000000000000
            #28 0b11100000000000000000000000000000
            #27 0b11110000000000000000000000000000
            mask |= 1 << i
            
            #now seek all nums that can be foud using this mask
            found = set([num & mask for num in nums])
            
            start  = result | 1 << i
            for pref in found:
                if start^pref in found:
                    result = start
                    break
        return result

nums = [3,10,5,25,2,8]
ans, mask = 0,0
for i in range(31,-1,-1):
    mask = mask | 1 << i
    found = set([num & mask for num in nums])
    start = ans | 1 << i
    for pref in found:
        if start^pref in found:
            ans = start
        print(mask,bin(mask),start,found,i,start^pref)
    #print(mask,bin(mask),found,start)

#########################
#Robot Bounded in Circle
#########################
#103 /110! so close
class Solution(object):
    def isRobotBounded(self, instructions):
        """
        :type instructions: str
        :rtype: bool
        """
        '''
        the hints say find the final position vector, if final position vector == initial
        true
        cycle detection
        keep going until i hit the same initival vector
        return true
        current on +vf[0]
        call L +vf[1]
        call L -vf[0]
        call L -vf[1]
        
        for right
        '''
        instructions_2 = instructions*4
        v_f = [0,0]
        partial = 0 #need to flip between 0 and 1
        direction = 1 #need to flip between 1 and -1
        count = {'L':0,'R':0}
        for inst in instructions_2:
            if inst == 'G':
                v_f[partial] += direction*1
            elif inst == 'L':
                if count[inst] % 2 != 0:
                    direction *= -1
                #update
                partial = 1 - partial
                count[inst] += 1
            
            elif inst == 'R':
                #update
                if count[inst] % 2 == 0:
                    direction *= -1
                partial = 1 - partial
                count[inst] += 1

        if v_f == [0,0]:
            return True
        else:
            return False

class Solution(object):
    def isRobotBounded(self, instructions):
        """
        :type instructions: str
        :rtype: bool
        """
        #initali direction
        direction = (0,1)
        start = [0,0]
        
        for i in instructions:
            if i == 'G':
                start[0] += direction[0]
                start[1] += direction[1]
            elif i == 'L':
                #swap
                direction = (-direction[1],direction[0])
            elif i == 'R':
                direction = (direction[1],-direction[0])
        return start == [0,0] or direction != (0,1)


class Solution(object):
    def isRobotBounded(self, instructions):
        """
        :type instructions: str
        :rtype: bool
        """
        '''
        https://www.youtube.com/watch?v=t6nUzD41G5U&ab_channel=TimothyHChang
        '''
        #initliaze start
        x,y = 0,0
        #directions oriented like up,right,down,left
        #we start at the direction
        d = 0
        #so right is +1, and left is -1, used the modular operator
        directions = [(0,1),(1,0),(0,-1),(-1,0)]
        for i in instructions:
            if i == 'G':
                x += directions[d%4][0]
                y += directions[d%4][1]
            elif i == 'R':
                d +=1
            elif i == 'L':
                d -= 1
                
        #are we at the same point or still facing the same direction
        return [x,y] == [0,0] or d%4 != 0


##################################
#Best Time to Buy and Sell Stock
##################################
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        '''
        THIS IS THE CLASSIC DP problem, not ii...i should finish this series of problems
        this is just finding the maximum difference
        '''
        if not prices:
            return 0
        max_diff = 0
        curr_min = prices[0]
        curr_max = prices[0]
        for i in range(1,len(prices)):
            if prices[i] < curr_min:
                curr_min = prices[i]
                curr_max = 0
            if prices[i] > curr_max:
                curr_max = prices[i]
                
            max_diff = max(max_diff, curr_max - curr_min)
        return max(max_diff,0)

class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        '''
        THIS IS THE CLASSIC DP problem, not ii...i should finish this series of problems
        this is just finding the maximum difference
        '''
        #the dum way
        curr_min = prices[0]
        max_profit = 0
        for i in range(1,len(prices)):
            max_profit = max(max_profit,prices[i]-curr_min)
            curr_min = min(prices[i],curr_min)
        return max_profit

#############################
#Sequential Digits 09/19/2020
##############################
#i cant figure this out, but i got a recursion to work from scratch!
class Solution(object):
    def sequentialDigits(self, low, high):
        """
        :type low: int
        :type high: int
        :rtype: List[int]
        """
        '''
        RECURSION!!!
        really we can only use the digits 1 to 9
        '''
        candidates = list(range(1,10))
        
        results = []
        #update outside
        #we are adding sequentially
        def rec_build(start,digit,low,high):
            if int(digit) > high:
                return
            elif low < int(digit) < high:
                results.append(digit)
                
            #create the digit one by one sequentially
            for i in range(start,9):
                digit += str(candidates[i])
                rec_build(i+1,digit,low,high)
        
        rec_build(1,'1',low,high)
        return results


class Solution(object):
    def sequentialDigits(self, low, high):
        """
        :type low: int
        :type high: int
        :rtype: List[int]
        """
        '''
        recursive backtracking, 
        being inclusive 1 to 9
        digits the next gigit that is about to apeend the integet
        cur is the current value
        
        '''
        results = []
        
        def recursive_build(current, digit_to_add):
            #add to results
            if digit_to_add <= 10 and low <= current <= high:
                results.append(current)
            elif current < low:
                pass
            else:
                return
            current = current*10 + digit_to_add #generating sequential digits
            #backtrack
            recursive_build(current,digit_to_add+1)
        for i in range(1,10):
            recursive_build(0,i)
        return sorted(results)

#sliding window approach
class Solution(object):
    def sequentialDigits(self, low, high):
        """
        :type low: int
        :type high: int
        :rtype: List[int]
        """
        '''
        we notice that a sequential digit is just alsing window from to 1 to 9
        exmple 
        low = 100
        high = 1000
        use window lengths from 3 to 4
        123456789
        123
         234
          345
           456
            567
             789
        1234 BUT greater than high so stop!
        '''
        digits ='123456789'
        n = 10
        results = []
        
        for length in range(len(str(low)),len(str(high))+1):
            for i in range(n-length):
                num = int(digits[i:i+length])
                if low <= num <= high:
                    results.append(num)
                    
        return results

#another way recursively! from tim
class Solution(object):
    def sequentialDigits(self, low, high):
        """
        :type low: int
        :type high: int
        :rtype: List[int]
        """
        '''
        https://www.youtube.com/watch?v=GwPfcriGlB8&t=5s&ab_channel=TimothyHChang
        recursive solution from time
        build up the digit one by one
        '''
        results = []
        
        def rec_build(start,current,results):
            if current % 10 == 0:
                return
            if low <= current <= high: 
                results.append(current)
            rec_build(start+1,current*10+start+1,results)
        for i in range(1,10):
            rec_build(i,i,results)
        results.sort()
        return results


############################
#Unique Paths III 09/20/2020
#############################
class Solution(object):
    def uniquePathsIII(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        '''
        1 represents the starting square.  There is exactly one starting square.
        2 represents the ending square.  There is exactly one ending square.
        0 represents empty squares we can walk over.
        -1 represents obstacles that we cannot walk over.
        area of thee grid cannot be more than 20
        we need to find a path that touches all zeros only once, DFS, because we need to look at all paths
        we can think of this as a grid taversesal, and immediatlye think of backtracking
        recall backtracking is a general algorithm for finding all or some solutions to some proble with constraints
        '''
        rows = len(grid)
        cols = len(grid[0])
        
        #first find the start and end and number of obstacles, 
        #store number of empty squares, to make sure we touch all of them, instead of keeping track of paths
        #visited set
        start_r,start_c = 0,0
        end_r,end_c = 0,0
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


#########################
#Carpooling 09/21/2020
##########################
#cant do it greedily!
from collections import deque
class Solution(object):
    def carPooling(self, trips, capacity):
        """
        :type trips: List[List[int]]
        :type capacity: int
        :rtype: bool
        """
        '''
        i would need to termineate the process when capcity goes negative and return false
        sort on starting location then process,
        pull all the starts and stops to begin with?
        pick a person up, decrement the capacity
        incremant the current location by 1, each increment, check if somone needs to be dropped off
        pop off that dropped person, increase the capacity by that number
        
        break conditions: when cap goes negative
        we are checking after each increment
        use q, when done, return True
        '''
        #first sort on start time
        trips = sorted(trips,key=lambda x:[1])
        current = 0
        
        
        while trips:
            #check cap
            if capacity < 0:
                return False
            #now search the q if a start is at this block
            for passenger in trips:
                #to pick up
                if passenger[1] == place:
                    #decrement cap
                    capacity -= passenger[0]
                    place += 1
                #to drop off
                if passenger[2] == place:
                    capacity += passenger[0]
                    trips.remove(passenger)
                    place += 1
                place += 1

        return True

class Solution(object):
    def carPooling(self, trips, capacity):
        """
        :type trips: List[List[int]]
        :type capacity: int
        :rtype: bool
        """
        '''
        don't get upset if you couldn't get it, its one of those problems where you need to see an example of the strategy first
        similar to the meeint rooms II problems
        
        the ides is to go through from the start to end and check if capacity has been exceeded
        to know the cap, we need the number of passengers at each time stamp
        save the number of passengerts at each time, then sort on the timestamp, traverse one least  time to get the actual capacity
        
        1. initlize a list to store the number of passengers and the corresponding time stamp then sort it
        2. iterate fomr the start timestamp and check if the actual cap meets the condition
        '''
        timestamps = []
        for trip in trips:
            timestamps.append([trip[1],trip[0]]) #[start, passengers], +passenggers
            timestamps.append([trip[2],-trip[0]]) #[end, - paasengers] -passengers
            
        #sort
        timestamps.sort()
        
        used_capacity = 0 #compare against current cap
        for time,passenger_change in timestamps:
            used_capacity += passenger_change
            if used_capacity > capacity:
                return False
        return True

class Solution(object):
    def carPooling(self, trips, capacity):
        """
        :type trips: List[List[int]]
        :type capacity: int
        :rtype: bool
        """
        '''
        bucket sort
        notice the constraints the the trips[i][1] < trips[i][2] <= 1000
        we can use bucket sort
        the idea is to create N buckets(1001) #for each pickup/drop location
        in each bucket, update the nmber of passesngers
        then traverse one last time seeing if it meets the constraint
        '''
        timestamps = [0]*1001
        for trip in trips:
            timestamps[trip[1]] += trip[0]
            timestamps[trip[2]] -= trip[0]
            
        used_cap = 0
        for pass_change in timestamps:
            used_cap += pass_change
            if used_cap > capacity:
                return False
        return True

class Solution(object):
    def carPooling(self, trips, capacity):
        """
        :type trips: List[List[int]]
        :type capacity: int
        :rtype: bool
        """
        '''
        https://www.youtube.com/watch?v=bQu0jrhjBMQ&pp=wgIECgIIAQ%3D%3D&ab_channel=TimothyHChang
        walk through from TIME
        at each time, store whether it is a pickup or drop off and the number of passengers
        pass this once, and if used cap is greater than cap, return false
        '''
        times = []
        for passengers,start,end in trips:
            times.append((start,passengers,1)) #time, amount, pockup
            times.append((end,passengers,0))
            
        #sort on the start, and by pickup, remember we want to dropoff first
        times.sort(key = lambda x: (x[0],x[2]))
        
        used_cap = 0
        
        for time,passengers,pickup in times:
            if pickup:
                used_cap += passengers
            else:
                used_cap -= passengers
            if used_cap > capacity:
                return False
        return True


class Solution(object):
    def carPooling(self, trips, capacity):
        """
        :type trips: List[List[int]]
        :type capacity: int
        :rtype: bool
        """
        '''
        i could use buckets for the sort and calc the number the change at each time
        '''
        #get the max end
        M = max([end for _,_,end in trips])
        times = [0]*(M+2) #adjust for 1 less the beginning and 1 more the end
        
        for people,start,end in trips:
            times[start+1] += people
            times[end+1] -= people
        
        used_cap = 0
        for change_pass in times:
            used_cap += change_pass
            if used_cap > capacity:
                return False
        return True
            

##############################
#Majority Element II 09/22/20
##############################
from collections import Counter
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        '''
        for an array of nums length n, the number of majority elements is just 3(n/3), which
        floor is just integer division
        naive way is to just count
        '''
        count = Counter(nums)
        results = []
        for k,v in count.items():
            if v > len(nums) // 3:
                results.append(k)
        return results

from collections import Counter
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        '''
        for an array of nums length n, the maximum number of majority elements is n // ((n // 3) + 1)
        it can can have elements in range(1,n // ((n // 3) + 1))
        notice the pattern if the constraint is n/2, then at at most 1
        n//3 at most 2
        n//4 at most 3 and so on
        this is called Boyer-Moore Voting
        
        to gain an inution examine fiding majority with threshold n //2
        the idea is to keep two variables, one holding a potential candidate for majority and a counter to keep track of whether to swap a potential candidate or not, two vars because there can be at most two!
        while scanning the array, the counter is incremented you an encounter an elemnt which is eactly the same as the ptenital cnadidated but decremented otherwise
        when the counter gets to zero, the next element becomes the new candidate
        but pass one more to ensure teh cand candidate is n // 3
        since we are asked to find  n//3 majority elements we need four vars, two holding candidates, and two holding correspoding counters
        
        algorithm
        1. if the current element is equal to one of the potential candidates, the count for that candidate is increased while leaing the other candidate as is
        2. if the counter reaches zero for any candidate, replace the candidate with the next element if the next element is not equal to the other candidate as well
        3. both counters are decremented only when the current eleemtn is different from both candidates
        
        '''
        if not nums:
            return []
        
        #first pass
        candidate1, candidate2 = None,None
        count1,count2 = 0,0
        for num in nums:
            if candidate1 == num:
                count1 += 1
            elif candidate2 == num:
                count2 += 1
            #new candidate updates
            elif count1 == 0:
                candidate1 = num
                count1 += 1
            elif count2 == 0:
                candidate2 = num
                count2 += 1
            else:
                count1 -= 1
                count2 -= 1
                
        #second pass
        result = []
        for c in [candidate1,candidate2]:
            #check that it is greater than n//3
            if nums.count(c) > len(nums) // 3:
                result.append(c)
        return result

######################
#Gas Station 09/23/2020
######################
#gave up :(
class Solution(object):
    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """
        '''
        i need to travel around in a circuit, i could just simulate for each gas station
        if i end up at the place i started with zero gas, i made a cycle
        repeat for all gas stations
        it i exhaust all of them without gettin back to zero return a negative one
        '''
        N = len(gas)
        
        index_to_return = -1
        
        def simulate(index):
            #init gas
            current_gas = gas[index]
            while current_gas > 0:
                index += 1
                index %= N
                if current_gas == gas[index]:
                    index_to_return = index
                    break
                current_gas -= cost[index-1]
                current_gas += gas[index]
        
        #invoke on each
        for i in range(0,N):
            simulate(i)

#brute force solution
class Solution(object):
    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """
        '''
        inution 
        the first idead is to check every single gas station
        perform the road trip and check how much gas we have in tank at each station
        the means it N squared do that first
        '''
        N = len(gas)
        for i in range(0,N):
            tank = 0
            for j in range(0,N):
                tank += gas[(i+j) % N]
                tank -= cost[(i+j) % N]
                if tank < 0:
                    break
                if j == N -1:
                    return i
        return -1


class Solution(object):
    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """
        '''
        inution 
        the first idead is to check every single gas station
        perform the road trip and check how much gas we have in tank at each station
        the means it N squared do that first
        
        there are two things,
        its impossible to perform the roadtrup if sum(gas) < sum(cost) duh!
        one could compute the total amount of gas in the tank 
        total_tank = sum(gas) - sum(cost) during the round trip and then return -1 of tank <0
        its impossible to start as a station if gas[i] - cost[i] < 0
        second fast could be generalize, introduce current tank
        if current tank is less than 0, it means we could not start at this startion
        algo:
        init total_tank and curr_tank to 0 and start with 0
        pass all stations
        update total_tank and curent_tank at each step by adding gas[i] and subtracting cost[i]
        if current_tank < 0 at i + 1, make i + 1 the new starting point and reset curr_tank to 0
        why this works?
        imagine the case where total_tank >= 0 and I can cycle starting at N_s
        is i can cycle i should be able to from N_s to end, but what about 0 to N_s
        '''
        N = len(gas)
        
        total_tank,current_tank = 0,0
        starting_station = 0
        for i in range(0,N):
            total_tank += gas[i] - cost[i]
            current_tank += gas[i] - cost[i]
            #if we couldnt get there
            if current_tank < 0:
                starting_station = i + 1
                current_tank = 0
        return starting_station if total_tank >= 0 else - 1


class Solution(object):
    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """
        '''
        another way, we would have to be able to across at least twice
        at each point check if the tank is negative and increase the maxlength
        return 2*N - maxlen
        [1,2,3,4,5][1,2,3,4,5]
        [3,4,5,1,2][3,4,5,1,2]
        -2-2-2 3 6 4 2 0 3 6
         0 0 0 1 2 4 4 5 6 7
        '''
        N = len(gas)
        maxlength = 0
        tank = 0 #starting with an empty tank
        for i in range(0,N*2):
            tank += gas[i % N] - cost[i % N]
            if tank >= 0:
                maxlength += 1
            else:
                maxlength = 0
            tank = max(tank,0)
        return 2*N - maxlength if maxlength >= N else - 1 #we need to have been able to go around at least once!
        


###############################
#Find the difference 09/24/2020
###############################
#32/54
from collections import Counter
class Solution(object):
    def findTheDifference(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        '''
        the length of t will always be 1 greater than the lenght of s!
        get the char counts of s
        and see if the char counts match the first window length in t
        if they match, the last letter in t is the answer, if they dont, compare counts
        '''
        N = len(s)
        count = Counter(s)
        for i in range(0,len(t)-N):
            temp = Counter(t[i:N])
            if temp == count:
                return t[-1]
            else:
                #compare temp and count, there should be one missing in temp
                for k,v in temp.items():
                    if count[k] != v:
                        return k

class Solution(object):
    def findTheDifference(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        '''
        sort and compare characters, N log N + n
        '''
         # Sort both the strings
        sorted_s = sorted(s)
        sorted_t = sorted(t)

        # Character by character comparison
        i = 0
        while i < len(s):
            if sorted_s[i] != sorted_t[i]:
                return sorted_t[i]
            i += 1

        return sorted_t[-1]


from collections import Counter
class Solution(object):
    def findTheDifference(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        '''
        one pass hash with extra pass
        get char counts of s
        traverse chars of t
        if char not in count s or count s at t is 0, we found it
        if t is in count s, decrement by one to remove a false match
        why? use the chars all up and the last one remaining is the extra char count!
        '''
        count_s = Counter(s)
        for char in t:
            if char not in count_s or count_s[char] == 0:
                return char
            else:
                count_s[char] -= 1
        return t[-1]


from collections import Counter
class Solution(object):
    def findTheDifference(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        count_s = Counter(s)
        count_t = Counter(t)
        
        #now compare
        for char,num in count_t.items():
            if char not in count_s or num != count_s[char]:
                return char


########################
#largest Number 09/25/20
########################
#wrong!!!

class Solution(object):
    def largestNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        '''
        this is just a recusrvie
        left build all premutations of the number and replace it with the max
        '''
        self.max_so_far = 0
        
        def rec_build(build,remaining):
            if not remaining:
                self.max_so_far = max(self.max_so_far,int(build))
                return
                
            for num in remaining:
                rec_build(build+str(num),remaining.remove(num))
                
        rec_build("",nums)
        return self.max_so_far

class Solution(object):
    def largestNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        '''
        this is just a recusrvie
        left build all premutations of the number and replace it with the max
        recursivel build the digit
        '''
        
        self.number = 0
        self.length = sum([len(str(foo)) for foo in nums])
        
        def rec_build(build,nums):
            if not nums:
                return
            if len(build) == self.length:
                self.number = max(self.number,int(build))
                
            for i in range(0,len(nums)):
                rec_build(build+str(nums[i]),nums.remove(nums[i]))
                
                
        rec_build("",nums)


class Solution(object):
    def largestNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        '''
        [3,30,34,5,9]
        just make the largest digit you can by sorting
        9534330
        if i just sorted
        but if we just sorted the numbers lexographically in reverse we would get
        ['9', '5', '34', '30', '3']
        
        but it whold really be 330 not 303
        
        then just use a new compator to breka the two 303 vs 330, 330 wins!
        define a compartor function and use bubble sort
        '''
        def largest(x,y):
            if x+y > y+x:
                return 0
            elif x+y < y+x:
                return 1#swap here
            else:
                return 0
        
        ##bubble sort
        for i in range(len(nums)-1,-1,-1):
            for j in range(i):
                if largest(str(nums[j]),str(nums[j+1])):
                    nums[j],nums[j+1] = nums[j+1],nums[j]
        return str(int("".join(map(str,nums))))


class Solution(object):
    def largestNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        '''
        [3,30,34,5,9]
        just make the largest digit you can by sorting
        9534330
        if i just sorted
        but if we just sorted the numbers lexographically in reverse we would get
        ['9', '5', '34', '30', '3']
        
        but it whold really be 330 not 303
        
        then just use a new compator to breka the two 303 vs 330, 330 wins!
        define a compartor function and use bubble sort
        '''
        def largest(x,y):
            if x+y > y+x:
                return -1
            elif x+y < y+x:
                return 1#swap here
            else:
                return 0
            

        
        return str(int("".join(sorted(map(str,nums),key = cmp_to_key(lambda x,y: largest(x,y))))))

##########################
#Teeemo Attacking 09/26/20
##########################

class Solution(object):
    def findPoisonedDuration(self, timeSeries, duration):
        """
        :type timeSeries: List[int]
        :type duration: int
        :rtype: int
        """
        '''
        [1,4,6,10] 3
        3 6  8  11
        for loop across the array with a window of two,
        add to time poinsied comparing time poison stay and differencec
        '''
        if not timeSeries:
            return 0
        time = 0
        N = len(timeSeries)
        for i in range(0,N-1):
            if timeSeries[i+1] - timeSeries[i] >= duration:
                time += duration
            elif timeSeries[i+1] - timeSeries[i] < duration:
                time += timeSeries[i+1] - timeSeries[i]
        #last one
        time += duration

        return time

class Solution(object):
    def findPoisonedDuration(self, timeSeries, duration):
        """
        :type timeSeries: List[int]
        :type duration: int
        :rtype: int
        """
        '''
        [1,4,6,10] 3
        3 6  8  11
        for loop across the array with a window of two,
        add to time poinsied comparing time poison stay and differencec
        '''
        if not timeSeries:
            return 0
        time = 0
        N = len(timeSeries)
        for i in range(0,N-1):
            time += min(timeSeries[i+1] - timeSeries[i], duration)
        #last one
        time += duration

        return time



############################################
#Insert into a Sorted Circular Linked List
############################################
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val=None, next=None):
        self.val = val
        self.next = next
"""

class Solution(object):
    def insert(self, head, insertVal):
        """
        :type head: Node
        :type insertVal: int
        :rtype: Node
        """
        '''
        https://leetcode.com/problems/insert-into-a-sorted-circular-linked-list/solution/
        usually linked lists problems involve two pointers
        traverse the linked list using prev and curr pointers, when we found a suitable place to under the new val, it should go in between the prev and curr nodes
        terminatino condition is when prev == head
        during the loop at each step, check if the current place bounded by the two pointers is the right place, if not move forward for both pointers
        there are three cases
        1. value of new node between minimual and maximal nodes
        2. the value is beyond the max or less than the min of the linked list (max add to to mil prev to head)
            first find tail by descending, where prev.val > curr.val
            so find the head and tail nodes
        3. uniform values, just add the node anywhere, usually right after the start
            in thise case we would end up looping through the list and getting back to the starting point
        4. aaaaand the empty linked list
        '''
        #empty linked list case
        if head == None: 
            newNode = Node(insertVal,None)
            newNode.next = newNode
            return newNode
        
        prev,curr = head,head.next
        toInsert = False
        
        while True:
            #case 1, finding where to insert
            if prev.val <= insertVal <= curr.val:
                toInsert = True
            #case2, at the ends
            elif prev.val > curr.val:
                #check
                if insertVal >= prev.val or insertVal <= curr.val:
                    toInsert= True
            #adding
            if toInsert:
                prev.next = Node(insertVal,curr)
                return head
            #updating
            prev,curr = curr,curr.next
            
            #loop detection
            if prev == head:
                break
        #unable to add anywhere
        prev.next = Node(insertVal,curr)
        return head

###########################
#Evaluate Division 09/27/20
###########################
#nice try, but you almost had it :(
from collections import defaultdict,deque
class Solution(object):
    def calcEquation(self, equations, values, queries):
        """
        :type equations: List[List[str]]
        :type values: List[float]
        :type queries: List[List[str]]
        :rtype: List[float]
        """
        '''
        just a system of equations,solve for each query
        i could solve for each variable,
        i could generate all 'node1','node2pairs' into the dictionary
        if i have (a/b) and (b/c) then (a/c) is just (a/b)(b/c)
        just forward multiply for in order, and invert forward multiply the reverse the ger the reverse
        '''
        
        #all nodes
        nodes = []
        for foo in equations:
            nodes += foo
        nodes = set(nodes)
        
        #create adjacency list
        adj_list = defaultdict(list)
        for start,end in equations:
            adj_list[start].append(end)
            adj_list[end].append(start)
        
        #create distance dict
        distances = defaultdict()
        for i in range(0,len(values)):
            distances[tuple(equations[i])] = values[i]
            distances[tuple(equations[i])[::-1]] = 1 / values[i]
            
        results = []
        for start,end in queries:
            if start not in nodes or end not in nodes:
                results.append(-1.0)
            else:
                #find the total edge by multiplying along the path, BFS starts here, i need to be able to make a path from start to end
                visited, q = set(), deque()
                visited.add(start)
                q.append(start)
                edge = 1
                
                while q:
                    s = q.popleft()
                    visited.add(s)
                    for n in adj_list[s]:
                        if n not in visited:
                            if n == end:
                                edge *= distances[(s,n)]
                                break
                            visited.add(n)
                        q.extend(adj_list[n])
                results.append(edge)


##BFS almost the same
from collections import defaultdict,deque
class Solution(object):
    def calcEquation(self, equations, values, queries):
        """
        :type equations: List[List[str]]
        :type values: List[float]
        :type queries: List[List[str]]
        :rtype: List[float]
        """
        '''
        just a system of equations,solve for each query
        i could solve for each variable,
        i could generate all 'node1','node2pairs' into the dictionary
        if i have (a/b) and (b/c) then (a/c) is just (a/b)(b/c)
        just forward multiply for in order, and invert forward multiply the reverse the ger the reverse
        '''
        #build dictionary, key is a node, and value is dictionary with connection nodes and edge legnth
        relations = defaultdict(dict)
        for (start,end),val in zip(equations,values):
            relations[start][end] = val
            relations[end][start] = 1/val
        
        results = []
        for start,end in queries:
            visited  = set()
            q = deque([(start,1)]) #the end node and the cost to get there
            res = -1
            while q:
                node,cost = q.popleft() #start off
                visited.add(node)
                if end in relations[node]: #check if i can get to the start from the end
                    res = cost*relations[node][end]
                    break #done here
                #otherwise,find the neighbors with their costs
                neighbors_costs = []
                for neighbor, n_cost in relations[node].items():
                    #not visited yet
                    if neighbor not in visited:
                        neighbors_costs.append((neighbor,cost*n_cost))
                #add to the q
                q.extend(neighbors_costs)
            results.append(res)
                
                    
        return results


#DFS from my man Tim!
from collections import defaultdict,deque
class Solution(object):
    def calcEquation(self, equations, values, queries):
        """
        :type equations: List[List[str]]
        :type values: List[float]
        :type queries: List[List[str]]
        :rtype: List[float]
        """
        '''
        just a system of equations,solve for each query
        i could solve for each variable,
        i could generate all 'node1','node2pairs' into the dictionary
        if i have (a/b) and (b/c) then (a/c) is just (a/b)(b/c)
        just forward multiply for in order, and invert forward multiply the reverse the ger the reverse
        '''
        #build dictionary, key is a node, and value is dictionary with connection nodes and edge legnth
        graphs = defaultdict(dict)
        N = len(equations)
        for i in range(0,N):
            graphs[equations[i][0]][equations[i][1]] = values[i]
            graphs[equations[i][1]][equations[i][0]] = 1/values[i]
        
        def dfs(start,end,visited):
            if start not in graphs or end not in graphs:
                return -1
            #check first if we have gotten to the end
            if end in graphs[start]:
                return graphs[start][end]
            
            #have we seen this node? or not able to find the ned
            for i in graphs[start]:
                if i not in visited:
                    visited.add(i)
                    #DFS!!!!
                    temp = dfs(i,end,visited)
                    if temp == -1:
                        continue
                    else:
                        return temp*graphs[start][i]
            return -1
        output = []
        for a,b in queries:
            output.append(dfs(a,b,set()))
        return output


######################################
#Subarray Product Less than K 09/28/20
######################################
class Solution(object):
    def numSubarrayProductLessThanK(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        '''
        n squared solution is to just check all contiguous subarrays
        do this first
        '''
        def multiply(array):
            if len(array) == 1:
                return array[0]
            product = array[0]
            for i in range(1,len(array)):
                product *= array[i]
            return product
                
        count = 0   
        for i in range(0,len(nums)):
            for j in range(i,len(nums)):
                if multiply(nums[i:j+1]) < k:
                    count += 1
        return count

#prefix array and binary search
class Solution(object):
    def numSubarrayProductLessThanK(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        '''
        recall the log(a*b*c*d) is just log(a) + log(b) + log(c) + log(d)
        we can write:
            log \prod_{i} x_{i} = \sum_{i} log(x_{i})
            
        now we can treat this a max sum subarray problesm for logs
        now an element becomes log(x), we can take the prefix sums
        prefix[i+1] = nums[0] + nums[1]...nums[i]
        now we are left with the problem of finiding for each i, the largest j so that nums[i]...+nums[j] = prefix[j] - prefix[i] < l
        now since prefex is increasing montonically, we can solve it with binary search
        '''
        #edge case
        if k == 0:
            return 0
        
        k = math.log(k)
        #get the prefx sums array but using the logs, now we can just sum
        prefix = [0]
        for num in nums:
            prefix.append(prefix[-1] + math.log(num))
        #now prefix is increasing, we want to find the smallest j (which is the index to pref)
        #so that nums[i] + ....nums[j] = prefix[j] - prefix[i] < k  
        #instead of doing another pass in n time, we can do binary search!
        
        answer = 0
        for i in range(0,len(prefix)):
            lo = i + 1
            hi = len(prefix) 
            while lo < hi:
                mid = lo + (hi - lo) // 2
                if prefix[mid] < prefix[i] + k - 1e-9: #i can still make a larger sub array
                    lo = mid +1
                else:
                    hi = mid
            answer += lo -i - 1
        return answer

class Solution(object):
    def numSubarrayProductLessThanK(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        '''
        10  5  2  6
        1   2  3  4   10 total subarrays
        10  50 100 600
        but once you get to 2, divide by the first pointer to see if we can get it to less than k
        decrease the number of subarrays by the moved pointer
        we need total subarrays at each number
        product so far
        and the smallest i
        '''
        N = len(nums)
        product = 1
        mini = 0
        subarrays = 0
        total = 0
        
        for i in range(N):
            #product so far,add to sub array
            product *= nums[i]
            subarrays += 1
            
            #check/move pointer,while loop to increment mini and decrese subarrays
            while product >= k and mini <= i:
                product /= nums[mini]
                mini += 1
                subarrays -= 1 #decrement here because product is to much
            #check product less than k
            #add subarrays to total
            if product < k:
                total += subarrays
        return total


#######################
#Word Break II 09/29/20
#######################
def rec_check(string,words):
            #use up string
            if len(string) == 0:
                return True
            for i in range(len(words)):
                if words[i] in string:
                    rec_check(string[len(words[i]):],words)
            return False
        rec_check(s,wordDict)

# 29/43
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        '''
        i cant figure this out recursively, just try it iteratively
        we use a pointer to mark our positin in worDict, then check if the word is in s
        is there is a match, shorten s, and move the pointer back to zero
        '''
        i = 0
        N = len(wordDict)
        
        while i < N:
            if wordDict[i] in s:
                s = s[len(wordDict[i]):]
                i = 0
            i += 1
            
        for w in wordDict:
            if w in s:
                s = s[len(w):]
        
        if len(s) == 0:
            return True
        else:
            return False

#recursive naive
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        '''
        naive recursive approach, this is pretty much what i was thinking about
        '''
        

        def dfs(s,wordDict,start):
            if start == len(s):
                return True
            
            for end in range(start+1,len(s)+1):
                if s[start:end] in wordDict and dfs(s,wordDict,end):
                    return True
            return False
        
        return dfs(s,set(wordDict),0)

#recursive naive with memoization
class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        '''
        recursive approach with memoization, this is pretty much what i was thinking about
        '''

        def dfs(s,wordDict,start,memo):
            #memo is just boolean array of length s, we need to mark whether or not we invoke our recursive call, othewise we would be checking forever
            if start == len(s):
                return True
            if memo[start] != None:
                return memo[start]
            
            for end in range(start+1,len(s)+1):
                #print s[start:end]
                if s[start:end] in wordDict and dfs(s,wordDict,end,memo):
                    memo[start] = True
                    return memo[start]
            memo[start] = False
            return memo[start]
        
        return dfs(s,set(wordDict),0,[None for _ in range(len(s))])


class Solution(object):
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        '''
        the idea is the s can be divided into subproblems s1 and s2
        if s1 and s2 solve then s is solved
        catsanddog  
        catsand         dog satisfied
        cats    and     dog satified
        
        the dp arry can be started with n+1, with two pointers i and j
        i refers to the length of the candidate substring (s')
        j refers to the index paritioning (s')
            index s' by s'[0:j] and s'[j+1:i]
        to fill the dp array, init dp[0] as True, since the null string is always in the  dict
        we consider substrings of all possible lenghts starting from the beg by makign use of index i
        for every such substring, partiion the string into two furthr substrings s1' and s2' in all possible ways using j
        note that i refers to the ending index of s2'
        to fill the dp[i] entry we check if dp[j] contains try (i.e whether or no s1' meets criteria)
        if so we further check if s2' is in the diction, if both meer we make dp[i true]
        dp solution
        'abc'   {a,bc,c}
            ''  a   ab  abc
        ''  T   F   F    F
        a       T   F    F
        b
        c
        '''
        d = set(wordDict)
        dp = [False for _ in range(len(s)+1)]
        dp[0] = True
        #we need to examine every prefix, recall 1 more than the length
        for i in range(1,len(s)+1):
            for j in range(0,i):
                #print s[j:i]
                if dp[j] and s[j:i] in d:
                    dp[i] = True
                    break
                    
        return dp[len(s)]


################################
#First Missing Positive 09/30/20
#################################
class Solution(object):
    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        7,8,9,11,12
        well the min of this array is 7
        and the max is 12
        the minimum positive array is just min(range(1,min)) nut in nums
        '''
        if not nums:
            return 1
        largest = max(nums)
        possible_numbers = set(range(1,largest+2))
        possible_numbers =  set(possible_numbers)
        
        #traverse nums removing each,then return the min of the set
        for num in nums:
            if num in possible_numbers:
                possible_numbers.remove(num)
        return min(possible_numbers)


class Solution(object):
    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        7,8,9,11,12
        well the min of this array is 7
        and the max is 12
        the minimum positive array is just min(range(1,min)) nut in nums
        if i find the min, i know that i can get a smaller min positive, which is just one
        but that min can't be in nums
        im so stupid, start at 1 and keep adding until i can't find anymore
        '''

        if not nums:
            return 1
        nums = [num for num in nums if num > 0]
        nums = set(nums)
        if not nums:
            return 1
        
        candidate = 1
        while candidate <= max(nums) + 1:
            if candidate not in nums:
                return candidate
            candidate += 1
        return candidate


class Solution(object):
    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        use another array to coun the number of times we see
        we know the min is going to be in the range of 1 to N +1
        '''
        N = len(nums)
        nums.append(0)
        
        array = [0 for _ in range(N)]
        
        for i in range(N):
            if 0 < nums[i] <= N:
                array[nums[i]-1] += 1
        
        for i in range(N):
            if array[i] == 0:
                return i+1
        

class Solution(object):
    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        0   2   2   4   0   1   0   1   3
        1   1   1   1   1   2
        
        '''
        N = len(nums)
        nums.append(0)
        
        #get rid of numbers outside our range
        for i in range(N):
            if nums[i] < 0 or nums[i] >N:
                nums[i] = 0
        
        temp = nums[0]
        for i in range(N):
            if nums[i] > 0:
                nums[nums[i]%N -1] += N #marking up numbers in range
        if nums[0] == temp:
            return 1
        for i in range(N):
            if nums[i] // N == 0:
                return i + 1
        return N