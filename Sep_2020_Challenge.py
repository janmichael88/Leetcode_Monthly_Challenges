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

foo = []
print(foo.pop(0))
print(foo)