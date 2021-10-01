############################
#Palindrome Linked List
############################
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        '''
        naive way would be to dump contents into container and check using two pointer method for alindrom
        '''
        arr = []
        while head:
            arr.append(head.val)
            head = head.next
        N = len(arr)
        left,right = 0,N-1
        while left < right:
            if arr[left] != arr[right]:
                return False
            left += 1
            right -= 1
        return True

#recursive approach
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        '''
        recursive
        
        '''
        #not part of the probloem, just printing elements of linked list in reverse
        def print_reverse(node):
            if node is not None:
                print_reverse(node.next)
                print node.val
        #print_reverse(head)
        
        #now for the main part of the problem
        self.front = head
        def rec_check(node):
            #still a node
            if node is not None:
                if not rec_check(node.next):
                    return False
                if self.front.val != node.val:
                    return False
                self.front = self.front.next
            return True
        
        return rec_check(head)

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        '''
        O(1) space, reverse second half in place and check
        algo:
            * find two halves
            * reverse second half
            * compare two halves
        #finding mid point inovlves slow and fast pointer method
        '''
        #makes helper for marking starts of first half ands econd helft
        def end_first_half(head):
            fast = head
            slow = head
            while fast.next and fast.next.next:
                slow = slow.next
                fast = fast.next.next
            return slow
        
        #helper to reverse in place, could also do recursively
        def reverse(head):
            prev = None
            cur = head
            while cur is not None:
                temp = cur.next
                cur.next = prev
                prev = cur
                cur = temp
            return prev
        
        if not head:
            return True
        #find the end of first haldf and reverse seocnd half
        first_half_end = end_first_half(head)
        second_half_start = reverse(first_half_end.next)
        
        #check whehter palin
        result = True
        first_pos = head
        second_pos = second_half_start
        while result and second_pos:
            if first_pos.val != second_pos.val:
                result = False
            first_pos = first_pos.next
            second_pos = second_pos.next
        
        #restore
        first_half_end.next = reverse(second_half_start)
        return result
        

#####################
#Larget Unique Number
#####################
class Solution(object):
    def largestUniqueNumber(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        '''
        naive way would be to count and sort by the counts
        and then see what count is count
        '''
        counts = Counter(A)
        
        ones = []
        for k,v in sorted(counts.items()):
            if v == 1:
                ones.append(k)
        if len(ones) == 0:
            return -1
        return ones[-1]

class Solution(object):
    def largestUniqueNumber(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        '''
        naive way would be to count and sort by the counts
        and then see what count is count
        max on the fly
        '''
        counts = Counter(A)
        
        result = -1
        for k,v in counts.items():
            if v == 1:
                result = max(result,k)
        
        return result

#################################
# Ones and Zeroes
##################################
#good article
#brute force TLE
class Solution(object):
    def findMaxForm(self, strs, m, n):
        """
        :type strs: List[str]
        :type m: int
        :type n: int
        :rtype: int
        """
        '''
        lets walk through all of the solutions, brute, force examine all all subsets
        for number of items N, the number of subsets is 2**N, take care to think about this for a bit
        '''
        #subset generation
        maxLen = 0
        for i in range(0,1<<len(strs)):
            zeros = 0
            ones = 0
            size = 0
            for j in range(len(strs)):
                if ((i & (1 << j)) != 0): #new subset,interseting way of generating all subsets itratively
                    counts = Counter(strs[j])
                    zeros += counts['0']
                    ones += counts['1']
                    #we can break out of this loop as soon as we exceed m and n
                    if zeros > m or ones > n:
                        break
                    size += 1
            
            if zeros <= m and ones <= n:
                maxLen = max(maxLen,size)
        
        return maxLen

#classic knapsack problem, but memo is lsow in python
class Solution(object):
    def findMaxForm(self, strs, m, n):
        """
        :type strs: List[str]
        :type m: int
        :type n: int
        :rtype: int
        """
        '''
        recursion no memo
        in the first apporach we examined generating subsets iteratively, however its better we use subset generation iteratveily
        we def a new rec functino cal(strs,i,one,zeros)
        the func takes a list of strinsg and give the size of the largest subset with oens adn zeros, conseirng the strings lying after the ith on in srts
        now in call of calc
            1. we include the current string in the subset being considered, but if we include the currente string, 
                we'll need to decut the number of 0\s and 1 in the current string from the total
                so we make a function call of the form
                calc(strs,i+1,zeros-zeros_curr, ones-ones_curr)
                we also increment the total number of strings sofar by 1
                we store the results obtains fromt his call in taken
            2. not including the current subset,
                in this case we need not update the count of ones andz zeros
                thus, the new call upon recursion is of the form
                calc(strs,i+1,zeros,ones)
            the larger value of taken not taken represent the required result to be returned in the functino call
            
            
        '''
        #adding memo
        memo = {} #tuple,answer to function, stores the max number of subsets possible consindering ithinex afterwakrds provided there only j 0s and k 1s
        #dict memo gets TLE, what is i generate the 3d array and get answer using index look up
        #memo =  [[[0]*(len(strs)) for _ in range(m)] for _ in range(n)]
        def calc(strs,idx,zeros,ones):
            #base case
            if idx == len(strs):
                return 0
            #retrieve
            if (idx,zeros,ones) in memo:
                return memo[(idx,zeros,ones)]
            counts = Counter(strs[idx])
            taken = -1
            if zeros - counts['0'] >= 0 and ones - counts['1'] >= 0:
                #take it
                taken = calc(strs,idx+1,zeros-counts['0'],ones-counts['1']) + 1
            #we dont take it 
            not_taken = calc(strs,idx+1,zeros,ones)
            #put into memory
            memo[(idx,zeros,ones)] = max(taken,not_taken)
            return memo[(idx,zeros,ones)]
        
        return calc(strs,0,m,n)

#even with a differnt structure for memo it still TLES
class Solution(object):
    def findMaxForm(self, strs, m, n):
        """
        :type strs: List[str]
        :type m: int
        :type n: int
        :rtype: int
        """
        '''
        recursion no memo
        in the first apporach we examined generating subsets iteratively, however its better we use subset generation iteratveily
        we def a new rec functino cal(strs,i,one,zeros)
        the func takes a list of strinsg and give the size of the largest subset with oens adn zeros, conseirng the strings lying after the ith on in srts
        now in call of calc
            1. we include the current string in the subset being considered, but if we include the currente string, 
                we'll need to decut the number of 0\s and 1 in the current string from the total
                so we make a function call of the form
                calc(strs,i+1,zeros-zeros_curr, ones-ones_curr)
                we also increment the total number of strings sofar by 1
                we store the results obtains fromt his call in taken
            2. not including the current subset,
                in this case we need not update the count of ones andz zeros
                thus, the new call upon recursion is of the form
                calc(strs,i+1,zeros,ones)
            the larger value of taken not taken represent the required result to be returned in the functino call
            
            
        '''
        #adding memo
        #memo = {} #tuple,answer to function, stores the max number of subsets possible consindering ithinex afterwakrds provided there only j 0s and k 1s
        #dict memo gets TLE, what is i generate the 3d array and get answer using index look up
        memo =  [[[0]*(1000) for _ in range(102)] for _ in range(102)]
        def calc(strs,idx,zeros,ones):
            #base case
            if idx == len(strs):
                return 0
            #retrieve
            if memo[idx][zeros][ones] != 0:
                return memo[idx][zeros][ones]
            counts = Counter(strs[idx])
            taken = -1
            if zeros - counts['0'] >= 0 and ones - counts['1'] >= 0:
                #take it
                taken = calc(strs,idx+1,zeros-counts['0'],ones-counts['1']) + 1
            #we dont take it 
            not_taken = calc(strs,idx+1,zeros,ones)
            #put into memory
            memo[idx][zeros][ones] = max(taken,not_taken)
            return memo[idx][zeros][ones]
        
        return calc(strs,0,m,n)
#dp
class Solution(object):
    def findMaxForm(self, strs, m, n):
        """
        :type strs: List[str]
        :type m: int
        :type n: int
        :rtype: int
        """
        '''
        we can use dp array, 2d, and dp[i][j] anwers the question?
        if i zeros and j ones, what is the maximum number of strings i can chose such that there are at least i zeros and j ones
        so how do we actaully fill up the dp array,
        we traverse the list of strings one by one
        at some point s_k conssting of x zeros and y ones, now how do we choose?
        choosing to add this string the the subset we wany by using previous strings traversed so far
        this is because for entries dp[i][j] with i < x or j < y, there will not be a suffcient numbers of 1's and 0's to accomodate teh current string in any subset
        as we go across the dp array, if we can take the current string and be above m zeros and n ones, we take it
        dp[i][j] = max(1+dp[i-zeros_curr_string][j-ones_curr_string],dp[i][j])
        '''
        dp = [[ 0 ] * (n+1) for _ in range(m+1)]
        for s in strs:
            zeros, ones = s.count("0"), s.count("1")
            for i in range(m, zeros - 1, -1):
                for j in range(n, ones - 1, -1):
                    # dp[i][j] indicates it has i zeros and j ones, can this string be formed with those ?
                    dp[i][j] = max( 1 + dp[i - zeros][j- ones], dp[i][j] )
            # print(dp)
        return dp[-1][-1]

#just another recusrive way od oing this
class Solution(object):
    def findMaxForm(self, strs, m, n):
        """
        :type strs: List[str]
        :type m: int
        :type n: int
        :rtype: int
        """
        memo = {}
        def calc(strs,idx,zeros,ones):
            #base case, end of array no options left
            if idx == len(strs):
                return 0
            if zeros == 0 and ones == 0:
                return 0
            if (idx,zeros,ones) in memo and memo[(idx,zeros,ones)] != 0:
                return memo[(idx,zeros,ones)]
            #current one strs
            counts = Counter(strs[idx])
            #keep recursing if we have enought
            if counts["0"] > zeros or counts["1"] > n:
                return calc(strs,idx+1,zeros,ones)
            #take vs not take
            take = 1 + calc(strs,idx+1,zeros - counts["0"],ones-counts["1"])
            not_take = calc(strs,idx+1,zeros,ones)
            memo[(idx,zeros,ones)] =  max(take,not_take)
            return memo[(idx,zeros,ones)]
        
        return calc(strs,0,m,n)

#another dp way
class Solution(object):
    def findMaxForm(self, strs, m, n):
        """
        :type strs: List[str]
        :type m: int
        :type n: int
        :rtype: int
        """
        dp = [[0]*(n+1) for _ in range(m+1)]
        for s in strs:
            zeros = s.count('0')
            ones = s.count('1')
            #backwards, we want to see if we can fit it in the big, if we started forwards, well we woeuld nee to backtrack a lot more
            #at least starting from the max, we are already constrained
            for i in range(m,zeros-1,-1):
                for j in range(n,ones-1,-1):
                    dp[i][j] = max(1+dp[i-zeros][j-ones],dp[i][j])
        return dp[m][n]
        
##########################
# Longest Valid Parentheses
##########################
#close one
class Solution(object):
    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        at each point in s, check the balance, could use array
        then go through the array finding the longest length
        ")()())"
        [-1,1,-1,1,-1,-1]
        convert the string into an array of 1's and -1's
        then use the array to find the longest conigusous subarray who's sum is zero
        brute force can be solve in n squared
        '''
        if not s:
            return 0
        arr = []
        for char in s:
            if char == '(':
                arr.append(1)
            else:
                arr.append(-1)
        maxlen = 0 
        N = len(arr)
        for i in range(N):
            for j in range(i,N):
                sub = arr[i:j+1]
                if sum(sub) == 0:
                    maxlen = max(maxlen,j-i+1)
        return maxlen

#brute force TLE
class Solution(object):
    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        check each window for valid paran, and update
        '''
        def isvalid(string):
            stack = []
            for char in string:
                if char == '(':
                    stack.append(char)
                elif char == ')':
                    if stack and stack[-1] == '(':
                        stack.pop()
                    else:
                        return False
                else:
                    return False
            return len(stack) == 0
        maxlen = 0
        N = len(s)
        for i in range(N):
            for j in range(i+2,N+1,2): #check in multipels of 2
                if isvalid(s[i:j]) == True:
                    print s[i:j]
                    maxlen = max(maxlen, j-i)
        return maxlen

class Solution(object):
    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        we can use dp to solve this problem
        dp array of size N,and each dp[i] answers the question, what is the longest valid paraenthis up to and including this index
        init element values to all zeros impies that each char enidng with (
        will always contains '0' 
        we fill in the dp as follows
            1. if s[i] == ')' and s[i-1] == '(',  then we have at least ()
                and so dp[i] = dp[i-2] + 2
                we do so becasue the enidng () portinos is a valid substring 
            2. if s[i] == ')' and s[i-1] == ')', the we have ))
            if s[i-dp[i-1]-1] == '(',
            then dp[i] = dp[i-1] + dp[i- dp[i-1]-2] + 2
            why?
            well if the second to last ) was already part of a valid substring, then for the last ) to be valid, there must be a corresponding starting (, which lies before the valid subtring of thesecond least
            so if the char before the valid sub_s happens to be (, we update dp[i] as an addition 2 plus the one one before it form the valid sub_s
            its something like this but i can't get it ot to work
        '''
        N = len(s)
        dp = [0]*N
        maxlen = 0
        for i in range(1,N):
            if s[i] == ')':
                #case 1, zero before and current pair is ()
                if s[i-1] == '(':
                    if i >= 2:
                        dp[i] = dp[i-2] + 2
                    else:
                        dp[i] = 2
                #case 2: variant of )), check previous valid subs and see if last ) is part of valid
                elif (i-dp[i-1] > 0) and (s[i-dp[i-1]-1] == '('):
                    if (i - dp[i - 1]) >= 2:
                        dp[i] = dp[i-1] + dp[i-dp[i-1]-2] + 2
                    else:
                        dp[i] = 2
                maxlen = max(maxlen,dp[i])
        
        return maxlen

#using stack
class Solution(object):
    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        using stack purely
        instead of checking every possible string and checking its validait, we use stack 
        and while scanning the givens tring, check if it is valid so far
        and also check the longest
        start by pushing onto -1 on to the stac
        algo;
            1. for every (, we push its index on to the stack
            2. for every ), we pop the topmost ands subtract the current elments index fromt he top element of the stack
            this give the length of the currently encountered valid string of parantehses
            if while popping element, the stack becomes empty, we push the current element elemtns index on to the stack
            in this way, we keep calcuating the lengths of the valid substrings, and return the maxlength
        '''
        maxlen = 0
        stack = [-1]
        for i in range(len(s)):
            if s[i] == "(":
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
                else:
                    maxlen = max(maxlen, i - stack[-1])
        return maxlen

class Solution(object):
    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        without using extra space
        we need to make use of two counters, call them left and right
        first, we start traversing the sttring from the left towards the right
        and for every '(' encountered we increment the left counter and for every ) encountered we increment the right counter
        whenever left == right, we calc the lenght of the current valind string and keep strack of the max length of the substring found so far
        if right > left, then we reset left and right to 0
        '''
        maxlen = 0
        l,r = 0,0
        #left to right scane
        for i in range(len(s)):
            if s[i] == '(':
                l += 1
            else:
                r += 1
            if l == r:
                maxlen = max(maxlen,2*r)
            elif r >= l:
                left = right = 0
        l,r = 0,0
        #right to left
        for i in range(len(s)-1,-1,-1):
            if s[i] == '(':
                l += 1
            else:
                r += 1
            if l == r:
                maxlen = max(maxlen, 2*l)
            elif l >= r:
                l = r = 0
            
        return maxlen

################################
#Design Circular Queue
##################################
class MyCircularQueue(object):
    '''
    i've done this proble before, this is a really great review problem!
    '''

    def __init__(self, k):
        """
        :type k: int
        """
        self.q = [0]*k
        self.head = 0
        self.tail = 0

    def enQueue(self, value):
        """
        :type value: int
        :rtype: bool
        """
        if self.isFull():
            return False
        #use modular
        self.q[self.tail % len(self.q)] = value
        self.tail += 1
        return True
        

    def deQueue(self):
        """
        :rtype: bool
        """
        if self.isEmpty():
            return False
        self.head += 1
        return True
        

    def Front(self):
        """
        :rtype: int
        """
        if self.isEmpty():
            return -1
        else:
            return self.q[self.head % len(self.q)]
        

    def Rear(self):
        """
        :rtype: int
        """
        if self.isEmpty():
            return -1
        else:
            return self.q[(self.tail-1) % len(self.q)] #one less, for zero indexing
        

    def isEmpty(self):
        """
        :rtype: bool
        """
        #both pointers catch up
        return self.head == self.tail
        

    def isFull(self):
        """
        :rtype: bool
        """
        return self.tail - self.head == len(self.q)
        


# Your MyCircularQueue object will be instantiated and called as such:
# obj = MyCircularQueue(k)
# param_1 = obj.enQueue(value)
# param_2 = obj.deQueue()
# param_3 = obj.Front()
# param_4 = obj.Rear()
# param_5 = obj.isEmpty()
# param_6 = obj.isFull()
class MyCircularQueue(object):
    '''
    official LC solution
    circular q's, also called a ring buffer
    recall the formulat for head and tail pointers
    tail = (head + len(q) -1) mod len(q)
    '''

    def __init__(self, k):
        """
        :type k: int
        """
        self.q = [0]*k
        self.head = 0 
        self.tail = 0
        self.k = k
        self.count = 0
        

    def enQueue(self, value):
        """
        :type value: int
        :rtype: bool
        """
        if self.isFull():
            return False
        self.q[self.tail] = value
        self.count += 1
        self.tail = (self.tail + 1) % self.k
        return True
        

    def deQueue(self):
        """
        :rtype: bool
        """
        if self.isEmpty():
            return False
        self.count -= 1
        self.head = (self.head + 1) % self.k
        return True
        

    def Front(self):
        """
        :rtype: int
        """
        if self.isEmpty():
            return -1 
        return self.q[self.head]
        

    def Rear(self):
        """
        :rtype: int
        """
        if self.isEmpty():
            return -1
        return self.q[self.tail - 1]

    def isEmpty(self):
        """
        :rtype: bool
        """
        return self.count == 0
        

    def isFull(self):
        """
        :rtype: bool
        """
        return self.count == self.k

#########################
#Global and Local Inversions
##########################
#TLE
class Solution(object):
    def isIdealPermutation(self, A):
        """
        :type A: List[int]
        :rtype: bool
        """
        '''
        brute force is count up local and global and compare
        
        '''
        count_global = 0
        for i in range(len(A)):
            for j in range(i+1,len(A)):
                if A[i] > A[j]:
                    count_global += 1
        count_local = 0
        for i in range(len(A)-1):
            if A[i] > A[i+1]:
                count_local += 1
        return count_local == count_global

#TLE abbreviated
class Solution(object):
    def isIdealPermutation(self, A):
        """
        :type A: List[int]
        :rtype: bool
        """
        '''
        brute force is count up local and global and compare
        what do i know?
        A will be a permuataion of [0,1,2...len(A)-1]
        
        for a strictly decreasing sequence, the number of local inversions i just N
        and the number of global inversions is N(N-1) - N
        stricly increasing, yeilds both to be zero
        
        we can degenerate this problem:
        a local inversion is also a global inversion
        so we only need to check if the permutaion has any non-lcaol inversions, i.e just count up global inversions
        inversions = local + global
        if local == global
        then inversions - local = global
        i.e (A[i] > A[j]) j - i > 1, 
        
        '''
        for i in range(len(A)):
            for j in range(i+2,len(A)):
                #if there is an inversion at all
                if A[i] > A[j]:
                    return False
        return True

#find the min after i + 2
class Solution(object):
    def isIdealPermutation(self, A):
        """
        :type A: List[int]
        :rtype: bool
        """
        '''
        we can degenerate this problem:
        a local inversion is also a global inversion
        so we only need to check if the permutaion has any non-lcaol inversions, i.e just count up global inversions
        inversions = local + global
        if local == global
        then inversions - local = global
        i.e (A[i] > A[j]) j - i > 1, 
        
        instead of repeatedly checking for an (i,j) pair such that j >= i + 2 and such that A[i] > A[j]
        this is the same thing as checking A[i] > min(A[i+2:])
        if we knowe these minimums, i.e min(A[0:]), min(A[1:]), min(A[2:]) we could make the check quickly
        algo:
            iterate thorough A from right to left, remembering the min value we've seen
            if we remembered the minimum: floor = min(A[i:]) and A[i-2] > floor
            then we should return False
            since the search is exhaustive, if we don't find a min, we return True
        
    
        '''
        N = len(A)
        floor = N #the largest element
        for i in range(N-1,-1,-1):
            floor = min(floor, A[i])
            if i >= 2 and A[i-2] > floor:
                return False
        return True

#best explanatino so far!
class Solution(object):
    def isIdealPermutation(self, A):
        """
        :type A: List[int]
        :rtype: bool
        """
        '''
        we can degenerate this problem:
        a local inversion is also a global inversion
        so we only need to check if the permutaion has any non-lcaol inversions, i.e just count up global inversions
        inversions = local + global
        if local == global
        then inversions - local = global
        i.e (A[i] > A[j]) j - i > 1, 
        
        KEY: if there is an inversion, and is that inversion is non local, i.e global, we return FALSE
        
        
        https://leetcode.com/problems/global-and-local-inversions/discuss/150991/Logical-Thinking-with-Clear-Code
    
        '''
        for i in range(len(A)):
            for j in range(i+1,len(A)):
                #if there is an inversion
                if A[j] < A[i] and  j - i > 1:
                    return False
        return True

class Solution(object):
    def isIdealPermutation(self, A):
        """
        :type A: List[int]
        :rtype: bool
        """
        '''
        we can degenerate this problem:
        a local inversion is also a global inversion
        so we only need to check if the permutaion has any non-lcaol inversions, i.e just count up global inversions
        inversions = local + global
        if local == global
        then inversions - local = global
        i.e (A[i] > A[j]) j - i > 1, 
        
        KEY: if there is an inversion, and is that inversion is non local, i.e global, we return FALSE
        
        https://leetcode.com/problems/global-and-local-inversions/discuss/150991/Logical-Thinking-with-Clear-Code
        
        we can also think of this abstractly, local inversion are just swaps between two adajcent elements
        if while traversing we find that abs(i - A[i]) > 1, we know this inversion cannot be local and a global must exsist in the array, we return False, pretty much saying all inversions must be local not global
        
    
        '''
        for i in range(len(A)):
            if abs(i - A[i]) > 1:
                return False
        return True

#########################################
# Minimum Operations to Make Array Equal
########################################
class Solution(object):
    def minOperations(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        arr is an array of length n, where arr[i] = (2 * i) + 1 and where i is in the range [0,n]
        we have two operations:
            1. select two distinct indictes (i,j), where 0 <= i and j < n, and arr[i] -= 1 and arr[j] += 1
        the goal is to make all of the elements equal
        return the min number of operations it takes
        guarnteed that array is able to be reduced
        target = sum(array) /n
        target always equals size
        [1,3,5,7]
        find elements less than target
        odd or even case
        '''
        arr = [0]*n
        for i in range(n):
            arr[i] = 2*i + 1
        target = sum(arr) / n
        ops = 0
        for i in range(n):
            if arr[i] > target:
                break
            ops += target - arr[i]
        return ops

class Solution(object):
    def minOperations(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        improvemtns, intead of allocating the array do it on the fly
        first we need the sum of the first n odd numbers
        \sum_{i=0}^{n-1} (2i + 1)
        2 \sum_{i=0}^{n-1} i + \sum_{i=0}^{n-1}
        2 \frac{(n-1)n}{n} + n 
        n^2
        and so the target is: \frac{n^2}{n}
        n
        '''
        count = 0
        for i in range(n):
            number = 2*i + 1
            if number > n:
                break
            count += n - number
        return count

class Solution(object):
    def minOperations(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        we can even go further by obtain an expressino for the sum without summation notation
        keeping in mind that we need to watch cases for off and even
        if n is our target
        S = (n-1) + (n-3) + (n-5) +.....(1 or so)
        EVEN CASE:
            n = 2k
        \sum_{i=0}^{n/2 -1} (2i + 1)
        2\sum_{i=0}^{n/2 -1} i + \sum_{i=0}^{n/2 -1}
        2 \frac{n/2(n/2 - 1)}{2} + \frac{n/2} 
        \frac{n^2/4}
        ODD CASE:
            same as even but the upper bound on the summation is \frac{n-1}{2}
        \sum_[i=0]^{i = \frac{n-1}{2}} 2i
        
        '''
        return n**2 // 4 if n % 2 == 0 else (n**2 -1) // 4

#########################
# Determine if String Halves Are Alike
#########################
class Solution(object):
    def halvesAreAlike(self, s):
        """
        :type s: str
        :rtype: bool
        """
        '''
        just count up the total number of vowels in both halves
        '''
        def count_vowels(string):
            count = 0
            N = len(string)
            vowels = set(['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'])
            for i in range(N):
                if string[i] in vowels:
                    count += 1
            return count
        
        N = len(s)
        return count_vowels(s[:N/2]) == count_vowels(s[N/2:])

#####################################
#  Letter Combinations of a Phone Number
##################################
class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        '''
        this is just a permutataionproblem
        so this is a recursive back tracking solutiono
        could also do it iterateively
        first make mapping, systematically do this
        try to solve this recsurively
        '''
        if not digits:
            return []
        #cant mape mapp this way, but everything else works
        mapp = defaultdict(list)
        for i in range(2,10):
            for j in range(3):
                mapp[str(i)].append(chr(ord('a')+(i-2)*3+j))
        #nine is not multiple of 3
        #edge cases
        mapp['7'].append
        mapp['9'].append('z')
        
        self.results = []
        
        def rec(idx,digits,build):
            if len(build) == len(digits):
                self.results.append(build)
                return
            chars =  mapp[digits[idx]]
            for ch in chars:
                rec(idx+1,digits,build+ch)
        rec(0,digits,"")
        return self.results

class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        '''
        this is just a permutataionproblem
        so this is a recursive back tracking solutiono
        could also do it iterateively
        first make mapping, systematically do this
        try to solve this recsurively
        '''
        if not digits:
            return []
        #cant mape mapp this way, but everything else works
        mapp = defaultdict(list)
        numbers  = [str(foo) for foo in range(2,10)]
        chars = ['abc','def','ghi','jkl','mno','pqrs','tuv','wxyz']
        for num,chs in zip(numbers,chars):
            for ch, in chs:
                mapp[num].append(ch)
        
        self.results = []
        
        def rec(idx,digits,build):
            if len(build) == len(digits):
                self.results.append(build)
                return
            chars =  mapp[digits[idx]]
            for ch in chars:
                rec(idx+1,digits,build+ch)
        rec(0,digits,"")
        return self.results

#immeditalty return without global results
class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        '''
        this is just a permutataionproblem
        so this is a recursive back tracking solutiono
        could also do it iterateively
        first make mapping, systematically do this
        try to solve this recsurively
        '''
        if not digits:
            return []
        #cant mape mapp this way, but everything else works
        mapp = defaultdict(list)
        numbers  = [str(foo) for foo in range(2,10)]
        chars = ['abc','def','ghi','jkl','mno','pqrs','tuv','wxyz']
        for num,chs in zip(numbers,chars):
            for ch, in chs:
                mapp[num].append(ch)
        

        
        def rec(idx,digits,build):
            results = []
            if len(build) == len(digits):
                results.append(build)
                return results
            chars =  mapp[digits[idx]]
            for ch in chars:
                results += rec(idx+1,digits,build+ch)
            return results
        return rec(0,digits,"")

class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        '''
        official LC, writeup, lets implement using backtracking and pop
        '''
        if not digits: 
            return 0
        
        letters = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", 
                   "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
        
        results = []
        
        def backtrack(idx,path):
            if len(path) == len(digits):
                results.append("".join(path)) #path is a list
                return #this is also a backtrack
            possible_letters = letters[digits[idx]]
            for ch in possible_letters:
                path.append(ch)
                backtrack(idx+1,path)
                path.pop()
        
        backtrack(0,[])
        return results

#iteratrive solution


############################
# Inorder Successor in BST
############################
#works but not very inefficient

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def inorderSuccessor(self, root, p):
        """
        :type root: TreeNode
        :type p: TreeNode
        :rtype: TreeNode
        """
        '''
        good review problem
        a nodes inorder successor is smallest of the largest nodes greater
        so the first of the succesors
        thanfully all nodes are unique
        #brute foce
        well brute force would be to get the inorder nodes into an array,
        scan the array, find the p node, return the next one
        '''
        in_order_nodes = []
        def in_order(node):
            if not node:
                return
            in_order(node.left)
            in_order_nodes.append(node)
            in_order(node.right)
        in_order(root)
        for i in range(len(in_order_nodes)):
            if in_order_nodes[i].val == p.val and i + 1 < len(in_order_nodes):
                return in_order_nodes[i+1]
        return None

#recursive in orderon the fly
class Solution(object):
    def inorderSuccessor(self, root, p):
        """
        :type root: TreeNode
        :type p: TreeNode
        :rtype: TreeNode
        """
        self.last = None
        self.inOrderSucc = None

        def inorder(node):
            if not node or self.inOrderSucc is not None:
                return
            inorder(node.left)
            if self.last == p:
                self.inOrderSucc = node
            self.last = node
            inorder(node.right)
        
        inorder(root)
        return self.inOrderSucc

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def inorderSuccessor(self, root, p):
        """
        :type root: TreeNode
        :type p: TreeNode
        :rtype: TreeNode
        """
        '''
        in order succ is simeplt the next node in the in order traversal of a tree
        if a node has a right, its inorder succ is node.right
        if not node.right, its just the parent
        if there is a node.right, go left as far as you can, then that's it
        if there isn't a right, then we came from above
        approach 1, any bianry tree, not necesarily a BST
        thre cases
        case 1: node has a right: go right first, then go left as far as we can
        case 2: node has no right,we need to perform in order traversal on three on keep track locally of prev node
        if at any point the pred pev == node given to us, the curr node should be in order succ
        algo:
            1. define two vars, prev and inorderSuccnode, prev used only for case 2, inorderSuccnode is the answer
            2. first start by idneityinf which of the two cases we are in
                if right child:
                    assing right child to a node called left most, and keep going left
                if not right child:
                    . For this, we define another function called inorderCase2 and we will pass it a node and the node p.
                    perform in order and recruse left
                    upon returning from recuison we check if the class varaible previous == node p
            return inorderSuccNode
        '''
        #global vars to help with cases
        self.prev = None
        self.inOrderSuccNode = None
        
        #case 2 helpoerm no right
        def case_2(node,p): #this is just inorder tarversl on node matching p but no right
            if not node:
                return
            case_2(node.left,p)
            #check if prev is in order pred of node
            if self.prev == p and not self.inOrderSuccNode:
                self.inOrderSuccNode = node
                return
            #keep updating previous for further recursions
            self.prev = node
            case_2(node.right,p)
        
        #case 1, simply need to find the leftmodet node in the subtree roots at p.right
        if p.right:
            leftmost = p.right
            while leftmost.left:
                leftmost = leftmost.left
            #answer
            self.inOrderSuccNode = leftmost
        else:
            case_2(root,p)

#most optimal
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def inorderSuccessor(self, root, p):
        """
        :type root: TreeNode
        :type p: TreeNode
        :rtype: TreeNode
        """
        '''
        here was can use properties of BST
        from node all its left descendants are less
        from that same node, all its right descendatrs are greter
        algo:
            1. start traversal with roote node and contnue until our currnet node reaches a null values
            2. at each steap we compare the value of node p
            3. if p.val >= node.val, we can discard left
            4. if p.val < node.val, this implies that succ must lie in left subtree
            and that the current node is a potential candidate for an inorder succ,
            so we update our local vars 
        '''
        succ = None
        while root:
            #not if left
            if p.val >= root.val:
                root = root.right
            #in left and current node might be succ
            else:
                succ = root
                root = root.left
        return succ
'''
what if there were dupes? well apporach 1 owuld generalize nicelt
and if p were from root, just compare memory locations to p and node
not sure about approach 2 for dupes
'''

####################################
#Verifying an Alien Dictionary
####################################
#welp, im a cheater lol
class Solution(object):
    def isAlienSorted(self, words, order):
        """
        :type words: List[str]
        :type order: str
        :rtype: bool
        """
        '''
        the ask is whether or not the words are sorted loxgraphically base on the alphabet
        if there is tie, the shorter word gets precendcne
        i could just define a comparator and sort the words then see if sorted words == words
        
        '''
        #defining comparator, sort then check
        sorted_words = sorted(words, key = lambda word: [order.index(c) for c in word])
        for i in range(len(words)):
            if words[i] != sorted_words[i]:
                return False
        return True

#good problem

class Solution(object):
    def isAlienSorted(self, words, order):
        """
        :type words: List[str]
        :type order: str
        :rtype: bool
        """
        '''
        im so dumb...for every word, make sure that all its words to the right are lexograpchially larger
        we can every do better by comparing adjacent words only
        if all pairs of adjacent words are sorted, then they are lexographically sorted
        to compare two adjacent words, we want to find the first letter that is different
        if words[i] has lex smllaer letter then we can exit fromt eh tieratiorn
        if words[i+1] is greater return False
        
        we also need to check boundary conditions
        if we cannot find a mis match, and len(word[i]) < len(word[i+1]) we're good, else break
        algo:
            1. inint order mapp using alphabet
            2 pass over words
                if words[i+1] ends before words[i] and no differnt letter then we found false
                if we find the first diffetn letter and the two words are in correct roder, go on to th next
                if we find the first diff letter and the two words are in the wrong orde, retunr flase
        '''
        mapp = {}
        for i,char in enumerate(order):
            mapp[char] = i
        
        N = len(words)
        for i in range(N-1):
            #compare the left and right
            for j in range(len(words[i])):
                #alwasy check if the word lengths don't match as we sort
                if j >= len(words[i+1]):
                    return False
                if words[i][j] != words[i+1][j]:
                    if mapp[words[i][j]] > mapp[words[i+1][j]]:
                        return False
                    break #they must be sorted
        return True

#abusing python builtins
class Solution(object):
    def isAlienSorted(self, words, order):
        """
        :type words: List[str]
        :type order: str
        :rtype: bool
        """
        mapp = {ch:i for i,ch in enumerate(order)}
        
        #convert to numbers
        words_converted = []
        for w in words:
            temp = []
            for ch in w:
                temp.append(mapp[ch])
            words_converted.append(temp)
        for i in range(len(words_converted)-1):
            if words_converted[i] > words_converted[i+1]:
                return False
        return True
        
##############################
#Longest Increasing Path in a Matrix
###############################
#FAILS
class Solution(object):
    def longestIncreasingPath(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: int
        """
        '''
        dfs
        well start by making the dfs functions add find the path stored in a container
        add container path to global var container
        find max from that and, return
        '''
        rows = len(matrix)
        cols = len(matrix[0])
        
        dirs = [(1,0),(-1,0),(0,1),(0,1)]
        
        def dfs(i,j,path):
            #bounds,path var is list
            if i < 0 or i >= rows or j < 0 or j >= cols:
                all_paths.append(path)
                return
            path.append((i,j))
            visited.add((i,j))
            for dx,dy in dirs:
                new_x,new_y = i + dx, j + dy
                if 0 <= new_x < rows and 0 <= new_y < cols and (new_x,new_y) not in visited:
                    if matrix[new_x][new_y] > matrix[i][j]:
                        dfs(new_x,new_y,path)
                    else:
                        return

        all_paths = []
        for i in range(rows):
            for j in range(cols):
                visited = set()
                dfs(i,j,[])
        print all_paths

#TLE, i think the solutino compiler is fucked up
class Solution(object):
    def longestIncreasingPath(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: int
        """
        '''
        TLE, using DFS
        '''
        rows = len(matrix)
        cols = len(matrix[0])
        
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        result = 0
        
        def dfs(i,j):
            path_length = 1
            for dx,dy in dirs:
                new_x,new_y = i + dx, j + dy
                if 0 <= new_x < rows and 0 <= new_y < cols and matrix[new_x][new_y] > matrix[i][j]:
                    path_length = max(path_length,dfs(new_x,new_y))
            return 1+path_length
        for i in range(rows):
            for j in range(cols):
                result = max(result,dfs(i,j))
        return result

#STILL TLE!!!
class Solution(object):
    def longestIncreasingPath(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: int
        """
        '''
        TLE, using DFS
        '''
        rows = len(matrix)
        cols = len(matrix[0])
        
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        visited = {}
        ans = 0
        def dfs(i, j):
            path_length = 0
            for dx,dy in dirs:
                new_row,new_col = i + dx,j+dy
                if 0 <= new_row <= len(matrix) - 1 and 0 <= new_col <= len(matrix[0]) - 1 and matrix[i][j] < matrix[new_row][new_col]:
                    if (new_row, new_col) in visited:
                        path_length = max(path_length, visited[(new_row, new_col)])
                    else:
                        path_length = max(path_length, dfs(new_row, new_col))

            visited[(row, col)] = path_length + 1
            return path_length + 1
        
        for row in range(len(matrix)):
            for col in range(len(matrix[0])):
                if (row, col) not in visited:
                    dfs(row, col)
        
        return max(visited.values())

#FINALLY
class Solution(object):
    def longestIncreasingPath(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: int
        """
        rows = len(matrix)
        cols = len(matrix[0])
        
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        cache = [[0]*cols for _ in range(rows)]
        
        result = 0
        
        def dfs(i,j):
            if cache[i][j] != 0:
                return cache[i][j]
            for dx,dy in dirs:
                new_x,new_y = i + dx, j + dy
                if 0 <= new_x < rows and 0 <= new_y < cols and matrix[new_x][new_y] > matrix[i][j]:
                    cache[i][j] = max(cache[i][j],dfs(new_x,new_y))
            cache[i][j] += 1
            return cache[i][j]
        for i in range(rows):
            for j in range(cols):
                result = max(result,dfs(i,j))
        return result

############################
#Deepest Leaves Sum
#############################
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def deepestLeavesSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        '''
        bfs level by dumping sum into hash
        return the largest sum of the deepeste level after traversing hash once more
        '''
        levels_sum = defaultdict(list)
        max_level = 0
        q = deque([(root,0)])
        
        while q:
            N = len(q)
            for i in range(N):
                node,level = q.popleft()
                if node:
                    levels_sum[level].append(node.val)
                    max_level = max(max_level,level)
                if node.left:
                    q.append((node.left,level+1))
                if node.right:
                    q.append((node.right,level+1))
        return sum(levels_sum[max_level])

class Solution(object):
    def deepestLeavesSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        '''
        we can optimize the iterative bfs by keeping nodes in the q that are at the last level
        then reurn the sum of the q
        initally q will be for the current levle, but can be in place of next
        make new q for curr and swap/reassign
        could also just reset the sum at the current deepest depth
        inrement when it get to that depth

        '''
        next_level = deque([root])
        
        while next_level:
            curr_level = next_level
            next_level = deque()
            
            for node in curr_level:
                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)
                
        return sum([node.val for node in curr_level])

#iterative dfs
class Solution(object):
    def deepestLeavesSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        '''
        iterative using stack
        algo:
            * push root into stack, 
            * while there is a stack
                pop form tack
                if leaf
                    update deepestleaves deepestes sum
                push right and left
        '''
        deepest_sum = 0
        depth = 0
        stack = [(root,0)] # element is of type (node,level)
        
        while stack:
            node, curr_depth = stack.pop()
            #if leaf
            if node.left is None and node.right is None:
                #check depth
                if depth < curr_depth:
                    deepest_sum = node.val
                    depth = curr_depth
                #if equal
                elif depth == curr_depth:
                    deepest_sum += node.val
            else:
                if node.left:
                    stack.append((node.left, curr_depth +1))
                if node.right:
                    stack.append((node.right,curr_depth + 1))
        return deepest_sum

class Solution(object):
    def deepestLeavesSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        '''
        pure recurison
        treat this two different problems
        find max depth
        then traverse the tree again, and if we descend into a depth == max_depth, increment
        '''
        def max_depth(node):
            if not node:
                return 0
            return max(max_depth(node.left),max_depth(node.right))+1
        
        def deepest_sum(node,d):
            if not node:
                return
            if d == depth:
                self.ans += node.val
            deepest_sum(node.left,d+1)
            deepest_sum(node.right,d+1)
            
        self.ans = 0
        depth = max_depth(root)
        deepest_sum(root,1)
        return self.ans

############################
#Beautiful Arrangement II
############################
class Solution(object):
    def constructArray(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[int]
        """
        '''
        make an array using [1,n], there needs to n different positive intergers, in any order
        so a permutation if size n, using [1,n]
        and there needs to be only k disttinc intergers where k is defined as integers
        [abs(nums[i]-nums[i+1]) for i in range(len(nums)-1)] where nums is the array
        if k is 1, then we just want a strictly increasing or decreasing sequence
        if i swap two element, what happens
        [1,2,3,4,5]
        [5,2,3,4,1] k = 3
        well if im buliding i can just recursively build it up, and if i can't abandon
        aye yai yai, just go over the solution from LC
        ''' 
        #BRUTE FORCE, generate all permutations and check
        def check_unique(arr):
            count = set()
            for i in range(1,len(arr)):
                count.add(abs(arr[i-1]-arr[i]))
            return len(count)
        for cand in itertools.permutations(range(1,n+1)):
            if check_unique(cand) == k:
                return cand

class Solution(object):
    def constructArray(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[int]
        """
        '''
        look at the constraint
        k can never be n, so the range of k is [1,n-1] and the range of n is [1,n]
        if k = n-1
        then we have [1,n,2,n-1,3,n-2....], the differences alternate between 1-n and n-2
        for this sequence
        for k = 1, you just return the strictly increaing or decreasing sequences
        this leads to the idea that for any k, we utilize both regimes
        [
        1-n,
        n-2,
        1-n,
        n-2,
        1-n,
        ]
        example, n = 5, k = 4
        [1,5,2,4,3]
        4,3,2,1
        also when k = 1, a valid construciton is the the decresing/increasing one
        we start with [1....n-k-1] first sot hat n is effectively k+1 and then finish k = n-1
        we can make k-1 unique differecnes up to n
        by doing k-1,k-2,k-3.....
        
        for examples n= 6, k = 3
        we start with [1,2], then we get [1,4,2,3] but we add to to all of them to get
        [1,2,3,6,4,5]
        algo:
            start with the first n-k-1 integers
            then starting with the k+1 integers, which can be written:
                [n-k,n-k+1,n-k+2,n-k+3,n-k+4....n] or in the range [n-k,k]
                the diffes for the elements indexed in range [n-k,n]
                [
                -2k-1,
                -2k-1,
                -2k-1
                ]
                now all the differences in this regime are 
            we start alternatingly from the head and tail of this range and append to our results?
            why?
            look at the explanation above, every even is chose from the head, and odd from the tail
        '''
        ans = list(range(1,n-k))
        #we are going to add to this
        for i in range(k+1):
            #even, we take from the head
            if i % 2 == 0:
                ans.append(n-k+i//2)
            else:
                #take add from the trail, but one less from n
                ans.append(n-i//2)
        return ans

class Solution(object):
    def constructArray(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[int]
        """
        '''
        this one is a good writeup, but i only would have seen it if i wrote down a few examples
        and then spottted a pattern
        https://leetcode.com/problems/beautiful-arrangement-ii/discuss/1154683/Short-and-Simple-Solution-or-Multiple-Approaches-Explained-with-Examples-!
        solution 1:
            1. start the permutation off with 1
            2. chose the next element such that the abs diff is k-1
            3. then chose another one such taht the differnce is k-2, or k-1 fromt he previous element
        algo:
            maintain two pointers, l = 1 and, r = k+1
            assing 1st idnex as l and incrementl, then assign second index as r and decremnt r
            assign thrid idnex as l and icnrment, and son
        '''
        ans = [0]*n #init the array palces
        i = 0
        left = 1
        right = k+1
        while i < k:
            ans[i] = left
            i += 1
            left += 1
            ans[i] = right
            i += 1
            right -= 1
        #if k is even
        if left == right:
            ans[i] = right #even case take from right then fill values for k+1 to n
            i += 1
        while i < n: #the last elements consecutive differences will be
            ans[i] = i + 1
            i += 1
        
        return ans

#second solution, just another way
class Solution(object):
    def constructArray(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[int]
        """
        '''
        this one is a good writeup, but i only would have seen it if i wrote down a few examples
        and then spottted a pattern
        https://leetcode.com/problems/beautiful-arrangement-ii/discuss/1154683/Short-and-Simple-Solution-or-Multiple-Approaches-Explained-with-Examples-!
        solution 2:, same as solution 1
            but this time well the sequnce this way
            [1,n,2,n-1,3,n-2....] 
            convince yourself this gives k unique differences
            but this only needs to be done for the frist k elements, the rest should be strincly increasing/decreassing with conseuctiver differeences of 1
        algo:
            1. same as above, but we init left and right to be 1 and n
            2. we fill the values the same way for the first k, however we need to check for two cases for the next k+1 to n values
                if k is odd, assign from left, incrment left
                if k is even assign from right, decrement right
        '''
        ans = [0]*n #init the array palces
        i = 0
        left = 1
        right = n
        while i < k-1: #elements should be filled in like [1,n,2,n-1,3,n-2....]
            ans[i] = left
            i += 1
            left += 1
            ans[i] = right
            i += 1
            right -= 1
        while i < n: 
            if k & 1 == 1:
                ans[i] = left
                left += 1
                i += 1
            else:
                ans[i] = right
                right -= 1
                i += 1

        return ans

class Solution(object):
    def constructArray(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[int]
        """
        '''
        just an additional way
        say for example we ahd 
        n = 5 k = 4
        1,5,2,4,3
        n = 5, k = 3
        1 2 5 3 4
        stricly icnreasing up to n-k, then zig zag from n-k to n
        algo:
            start with the first 1 to n-k digits
            then add the nedt digit whose differnce is i less/more than the previous element
        '''
        output = [i for i in range(1,n-k+1)]
        
        dr = 1
        diff = k
        for i in range(k):
            output.append(output[-1]+(diff*dr))
            diff -= 1
            dr *= -1
        return output

#################################
#Flatten Nested List Iterator
#################################
#using dfs to flatten list first then move through it
# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """
#class NestedInteger(object):
#    def isInteger(self):
#        """
#        @return True if this NestedInteger holds a single integer, rather than a nested list.
#        :rtype bool
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

class NestedIterator(object):

    def __init__(self, nestedList):
        """
        Initialize your data structure here.
        :type nestedList: List[NestedInteger]
        """
        '''
        flatten the nestedList right from the start
        '''
        self.flattened = []
        def dfs(nested):
            for nest in nested:
                if nest.isInteger():
                    self.flattened.append(nest.getInteger())
                else:
                    dfs(nest.getList())
        dfs(nestedList)
        self.ptr = -1 #instead of giving reference to curr pointer

    def next(self):
        """
        :rtype: int
        """
        self.ptr += 1
        return self.flattened[self.ptr]
            
        

    def hasNext(self):
        """
        :rtype: bool
        """
        return self.ptr + 1 < len(self.flattened)
        

# Your NestedIterator object will be instantiated and called as such:
# i, v = NestedIterator(nestedList), []
# while i.hasNext(): v.append(i.next())


#using single stack
class NestedIterator(object):

    def __init__(self, nestedList):
        """
        Initialize your data structure here.
        :type nestedList: List[NestedInteger]
        """
        '''
        instead of using recursion right in the constructor to flatten it
        we can control it by doing dfs iteratively
        we push all the elements of the nested list in reversr order on to the stack
        and if its not an integer at the stop, get the list and add back to stack in reverse
        instead of building this out in the constructor, makea new method in the class
        '''
        
        self.stack = list(reversed(nestedList))
        

    def next(self):
        """
        :rtype: int
        """
        #make top int
        self.make_top_int()
        return self.stack.pop().getInteger()
        

    def hasNext(self):
        """
        :rtype: bool
        """
        #again ensure top is int
        self.make_top_int()
        return len(self.stack) > 0
    
    def make_top_int(self):
        #whiel stack conatins nested on top 
        while self.stack and self.stack[-1].isInteger() == False:
            topList = self.stack.pop().getList()
            self.stack.extend(reversed(topList))
            
#using two stacks -  single stack with two element list
class NestedIterator(object):

    def __init__(self, nestedList):
        """
        Initialize your data structure here.
        :type nestedList: List[NestedInteger]
        """
        '''
        reversing the lists to put them on to stack is expensive
        instead of pushing every sub list (or single element eventually) we can in stead assocaite an index pointer with each sublist
        this keeps track of how far along the sublist we are
        adding a new sublit to the stack now becomes O(1) operation instead of O(length sublist)
        main take away from diagram
        push unpack
        push unpack
        until i cant pack
        pop
        pop
        until i have to pack
        we implement this approach in python on stack using tuples (index,nestedList element)
        
        '''
        self.stack = [[nestedList,0]] #list of lists, cannot change tuples
    def make_top_int(self):
        while self.stack:
            curr_list = self.stack[-1][0]
            curr_index = self.stack[-1][1]
            #if top is list, pop it and its index, we've moved through the list
            if len(curr_list) == curr_index:
                self.stack.pop()
                continue
            #otherwise its an int and we don't do anytihing
            if curr_list[curr_index].isInteger():
                break
            #otherwise it mist be a list
            #we need to incrmenet the idnex on the previous list and add the new list
            new_list = curr_list[curr_index].getList()
            self.stack[-1][1] += 1
            self.stack.append([new_list,0])
            
                
    def next(self):
        """
        :rtype: int
        """
        #make top of stack an int, and move the poitners
        self.make_top_int()
        curr_list = self.stack[-1][0]
        curr_index = self.stack[-1][1]
        self.stack[-1][1] += 1
        return curr_list[curr_index].getInteger()

    def hasNext(self):
        """
        :rtype: bool
        """
        self.make_top_int()
        return len(self.stack) > 0

#using generator
'''
background in generators, same as iteators but they are paused
used yeild instead of return
once the generator is expire, there is a StopIteation Error
'''
def range_gen(a,b):
    curr = a
    while curr <= b:
        yield(curr)
        curr += 1

foo = range_gen(1,5)
print(next(foo))
print(next(foo))
print(next(foo))
print(next(foo))
print(next(foo))
#print(next(foo)) #this is stop iteration
#we can also loop them
for num in range_gen(1,6):
    print(num)

class NestedIterator(object):

    def __init__(self, nestedList):
        """
        Initialize your data structure here.
        :type nestedList: List[NestedInteger]
        """
        '''
        we can use a generator, recall from the first apporach we recursively flattened
        this prompted the constructor to make everything into a list
        we can use a generator recursively, which will pause the recursion
        instead of yeild use yeild from
        we also need to add in a peeked value,
        why? the only way to know if there IS a next value is to take it out of generator
        and since generators cannot go back, we have to cache what we peeked!
        '''
        self.gen = self.int_gen(nestedList)
        self.peeked = None
    
    def int_gen(self,nested_list):
        #make generator from nested list
        for nest in nested_list:
            if nest.isInteger() == True:
                yield nest.getInteger()
            else:
                yield from self.int_gen(nest.getList()) #yeilf from using generator recursively
        

    def next(self):
        """
        :rtype: int
        """
        #check there are integers left, and if so, then this will also put one into peak
        if self.hasNext() == False:
            return None
        output = self.peeked
        self.peeked = None
        return output
        

    def hasNext(self):
        """
        :rtype: bool
        """
        if self.peeked is not None:
            return True
        #try to go get another value from the gen
        try:
            self.peeked = next(self.gen)
            return True
        except:
            return False #the generator is finished
        

# Your NestedIterator object will be instantiated and called as such:
# i, v = NestedIterator(nestedList), []
# while i.hasNext(): v.append(i.next())

#######################
#Partition List
#######################
class Solution(object):
    def partition(self, head, x):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        '''
        find nodes less than x, in the order
        make linked list from thi less elements
        add in the node x
        add the rest of the elements
        '''
        if not head:
            return None
        less_x = []
        greater_x = []
        temp = head
        while temp:
            if temp.val < x:
                less_x.append(temp.val)
            else:
                greater_x.append(temp.val)
            temp = temp.next
        #concat
        concat = less_x + greater_x
        dummy = ListNode()
        temp = dummy
        for num in concat[:-1]:
            temp.val = num
            temp.next = ListNode()
            temp = temp.next
        temp.val = concat[-1]
        return dummy
#two pointer technique
#takeaway, init two vars, but also need move through pointer
#reconnect the heads
class Solution(object):
    def partition(self, head, x):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        '''
        well it looks like we can use a two pointer approach with O(1) space
        essentiall we will end up with two partioned linked lists
        the lesser linkedlist will have all nodes less than x, the other half icludes x and everything after
        algo
        1. initlize before and after vars, these are just dummy heads
        2. traverse the linked list
        3. if less than x, add to before, else add to after
        4. before.next = after.next
        5. return it
        '''
        #we need our move through pointers when we reference
        before = before_head = ListNode(0)
        after = after_head = ListNode(0)
        #move before and after
        while head:
            if head.val < x:
                before.next = head
                before = before.next
            else:
                after.next = head
                after = after.next
            head = head.next
        
        #no we need to recombinde
        #fixe the last node in after
        after.next = None
        #connect
        before.next = after_head.next
        return before_head.next

######################
#Fibonacci Number
######################
class Solution(object):
    def fib(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        bleagh good review problem
        F(n) = F(n-1) + F(n-2)
        recusrive memo
        '''
        memo = {}
        
        def fib(n):
            if n == 0:
                return 0
            if n == 1:
                return 1
            if n in memo:
                return memo[n]
            result = fib(n-1) + fib(n-2)
            memo[n] = result
            return result
        return fib(n)

#bottom up dp O(1)
class Solution(object):
    def fib(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        bottom up dp 0(1)
        '''
        if n <= 1:
            return n
        if n == 2:
            return 1
        #now starting from the third
        current = 0
        num1 = 1
        num2 = 1
        for i in range(3,n+1):
            current = num1 + num2
            num2 = num1
            num1 = current
        return current

class Solution(object):
    def fib(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        after solving the recurrence
        the golden ration is : \frac{1+\sqrt{5}}{2}
        F_{n} = golden_ration^{N+1} / \sqrt{5}
        '''
        golden_ratio = (1+ 5**.5) / 2
        return int((golden_ratio**n + 1) / (5**0.5))

#########################################
#Minimum Swaps to Group All 1's Together
#########################################
#brute force sliding windows gives me TLE
#need to use anohter approach
class Solution(object):
    def minSwaps(self, data):
        """
        :type data: List[int]
        :rtype: int
        """
        '''
        intuion:
        say for example we are given the array
        [0,0,1,1,0,1]
        well we can swap once and get
        [0,0,0,1,1,1]
        we know that if we group the ones together, we have a subarray if size equal to number of ones!
        we can use a sliding window of the size numbers of ones, and for each window record the difference between zeros and ones, and minimizne the answer on the fly, 
        this would be the brute force sliding window approach
        
        '''
        ones = sum(data)
        answer = len(data) - ones
        for i in range(len(data)-ones+1): #check every window
            window = data[i:i+ones]
            curr_ones = sum(window)
            answer = min(answer,ones - curr_ones)
        return answer

#two pointer sliding windows
class Solution(object):
    def minSwaps(self, data):
        """
        :type data: List[int]
        :rtype: int
        """
        '''
        instead of for looping across every window in the array
        we can use a two pointer sliding window
        key: we need to find the maximum number of 1's in the window so that we can make 
        the smallest number of swaps to achieve the goal
        algo:
            * use two pointers left and right to maintain a sliding window
            * check every window in the array and we would calculate the number 1;s currentl seens
            * store the largest number of ones we've sesn
            * but also maintain the lenght of size ones
            * since the values in the array are only zero and one, well we can just actuall incrment by their elemenmt values
        '''
        ones = sum(data)
        count_ones = 0
        max_ones = 0
        left,right = 0,0
        
        #two pointers
        while right < len(data):
            #updating count
            count_ones += data[right]
            right += 1
            #maintain length
            if right - left > ones:
                count_ones -= data[left]
                left += 1
            
            max_ones = max(max_ones,count_ones)
        return ones- max_ones
        
class Solution(object):
    def minSwaps(self, data):
        """
        :type data: List[int]
        :rtype: int
        """
        '''
        we also could use a deque
        '''
        ones = sum(data)
        count_ones = 0
        max_ones = 0 
        array = deque()
        
        for i in range(len(data)):
            array.append(data[i])
            count_ones += data[i]
            
            if len(array) > ones:
                count_ones -= array.popleft() #remove the earliast eleemnt 
            max_ones = max(max_ones,count_ones)
        return ones - max_ones

##############################################
#Remove All Adjacent Duplicates in String II
#############################################
#YASSSS
class Solution(object):
    def removeDuplicates(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: str
        """
        
        stack = []
        for char in s:
            #increment to matchign char at top of stack
            if stack and stack[-1][0] == char:
                stack[-1][1] += 1
            #othewise just load in a new char
            else:
                stack.append([char,1])
            
            if stack[-1][1] == k:
                stack.pop()
        
        
        output = ""
        for char,count in stack:
            #we need to add the chars wit their final counts
            output += char*count
        return output

############################################
# Number of Submatrices That Sum to Target
###########################################
#well i have no idea...
class Solution(object):
    def numSubmatrixSumTarget(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: int
        """
        '''
        hard problem, dont worry about it too much
        we can degenerate this problem intot he subarry sum equals k
        one of the best solutions for this problem is to use a hashmap with key as the prefize sum
        prefix is just cumsum up to i for i in rang len(array)
        for i in range(len(array)): #for the 1 d case
            \sum_{i=0}^{i}
        we can use a prefix sum for the 2d cae
        let P be m by n
        P_{mn} = \sum{i=0}^{m} \sum_{j=0}^{n} x_{ij}, and so prefix sum is O(R X C)
        reduce 2d problem to 1d ;
            fix two rows, r1 and r2
            using 2d prefix sum, we can get the sum of each prefmize matrix"
                curr_sum = ps[r2][col] - ps[r1-1][col]
            the sum itselft could be considered as a 1d prefix sum, because when rows are fixed, there is just one paramater to play with
        takeaway:
            use 2d prefix sum to reduce the problems to lots of smaller 1d problems
            use 1d prefix sum to solve tehse 1 d problems
        algo:
            init count to 0
            compute rows and cols
            compure 2d prefix sum for 1 more cols and 1 more rows
            iterate from r1 (1 to r) 
                from r2 (r1 to r)
                in this double loop, the left and right row bondaries are fixed
                now we can treat this as the 1d cae
                init hashmap: number of matrices which use [r1,r2]
                iterate over columns from 1 to c+ 1:
                    compute curr 1d prefix sum using prev computed 2d sum
                    the number of times the sum occures defines the number of matrices which use r1....r2 rows and sum to target
                    increment count by that much
                    add the current 1d prefix sum into the hashamp
            return count
        
        '''
        #compute prefix sum for 2d case
        def prefix_2d(matrix):
            rows = len(matrix)
            cols = len(matrix[0])
            prefix = [[0*cols] for _ in range(rows)]
            for i in range(1,rows+1):
                for j in range(1,cols+1):
                    prefix[i][j] = prefix[i-1][j] + prefix[i][j-1] - prefix[i-1][j-1] + matrix[i-1][j-1]
            return prefix #idk how i'm gonna rememebr this
        
        rows, cols = len(matrix),len(matrix[0])
        
        #compute 2d prefix sum
        ps = [[0]*(cols+1) for _ in range(rows+1)]
        for i in range(1,rows+1):
            for j in range(1,cols+1):
                ps[i][j] = ps[i-1][j] + ps[i][j-1] - ps[i-1][j-1] + matrix[i-1][j-1]
        
        count = 0
        #reduce to 1d case, review this problem tonight!
        #fix two rows r1 and r2
        #compute 1d prefix sum for all matrices using r1..r2
        for r1 in range(1,rows+1):
            for r2 in range(r1,rows+1):
                h = defaultdict(int)
                h[0] = 1
                for col in range(1,cols+1):
                    #1d prefix sum
                    curr_sum = ps[r2][col] - ps[r1-1][col]
                    #add subarrays which sum up to curr_sum - target
                    count += h[curr_sum - target]
                    #save current prefix sum
                    h[curr_sum] += 1
        return count

class Solution(object):
    def numSubmatrixSumTarget(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: int
        """
        '''
        https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/discuss/1162767/JS-Python-Java-C%2B%2B-or-Short-Prefix-Sum-Solution-w-Explanation
        just another way of doing it
        takeaway from prefix sum:
            the sum of a subarray between indices i and j == the sum of the subarray from 0 to j minis the sum of the subarray form 0 to i-1
        intuition:
            find number of subarrays with the taget sum by using hashmap
        
        algo:
            rather than iteratively checking if sum[0,j] - sum[0,i-1] == T for every pair of i,j values, we can flit it around and say:
            sum[0,j] - T = sum[0,i-1] and since every earlier sum value has been stored in res, we can simply lookup and add or increment if found
        notes on prefix sum:
            we can build the prefix sum directly into the matrix
        '''
        rows, cols = len(matrix[0]), len(matrix) #notice the swap here
        count = 0
        
        #build prefix sum in place
        for r in matrix: #all rows
            for i in range(1,len(r)):
                r[i] = r[i-1] + r[i]
                
        #SSEK
        for start in range(rows):
            for end in range(start,rows):
                #start end are bounds for upper and lower parts of matrix
                lookup = defaultdict(int)
                cum_sum = 0
                lookup[0] = 1 #do SSEK on each submatrix
                #now go across columns
                for row in matrix:
                    #get current cum_sum
                    cum_sum += row[end] - (row[start-1] if start else 0) #falling out if index
                    #check if we already have a sum
                    if cum_sum - target in lookup:
                        count += lookup[cum_sum-target]
                    #update
                    lookup[cum_sum] += 1
        return count

##########################
#Subarray Sum Equals K
##########################
#TLE OBVIE O(N^3)
class Solution(object):
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        '''
        brute froce, examine every possible subarray
        dont use sum function, just + the next element
        whenever the sum == k incrment count
        '''
        count = 0
        for start in range(len(nums)):
            for end in range(start+1,len(nums)+1):
                sub_array_sum = 0
                for i in range(start,end):
                    sub_array_sum += nums[i]
                if sub_array_sum == k:
                    count += 1
        return count

#O(N^2)
class Solution(object):
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        '''
        O(N^2) time, using cum sum array
        instead of determing sum of all subarrays for a every new subaraay consider, we can make use of the prefix sum
        then in order to calculate the sum of elements lying between two indices, we can substract the cum sum corresponding to the two indicies to obtain the sum directly
        algo:
            init prefix sum array
            to determine the sum of elements for new subarray nums[i:j]
            we can us sum[j+1]- sum[i] ??? hmmmmm
        
        '''
        N = len(nums)
        count = 0
        sums = [0]*(N+1) #prefix array
        for i in range(1,N+1):
            sums[i] = sums[i-1] + nums[i-1] #offsett, could have also just made it from nums arrayh
        #find my starts and ends
        for start in range(N):
            for end in range(start+1,N+1):
                #the difference between the end and start into sums is the sum of new candidate subarray
                if sums[end] - sums[start] == k:
                    count += 1
        return count

#O(N) using haspmap and compelemnt count increment
class Solution(object):
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        '''
        we can use hashmap
        intuition:
            if the cumsum (represented by sums[i]) up to indices is the same, them sum of the elments lying in between those indices must be zero
            extending the same though, if the cum sum up to two indices say i and j is at a difference of k, them sum lying between those indices i and j is k
        
        algo:
            hashmap stores: (sum_i : num of ocurrences)
            we keep traversing the array and keep finding the cum sum
            every time we encounter a new sum we make a new entry and if the same sum occurs again, we increment the count corresponding sum in the hash
            further, for every sum encountered, we also determing the number of teims the sum sum - k has occurred alreasdy, since it will determing the number of times a subarray sum with sum k has has occurred up to the index
            we increment the count by the same amount, think about two sum
        '''
        count = 0
        SUM = 0
        mapp = defaultdict(int)
        mapp[0] = 1 #sum of zero initially, 1 occurrence
        for i in range(len(nums)):
            SUM += nums[i]
            if SUM - k in mapp:
                count += mapp[SUM-k]
            mapp[SUM] += 1
        return count
        
#naive, but really the only way i could get it
class Solution(object):
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        count = 0
        for start in range(len(nums)):
            summ = 0
            for end in range(start,len(nums)):
                summ += nums[end]
                if summ == k:
                    count += 1
        return count
        

#####################################
#Remove Nth Node From End of List
####################################
#meh it works
class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        '''
        dump elements in array, delete nth from end, rebuild
        '''
        temp = []
        curr = head
        while curr:
            temp.append(curr.val)
            curr = curr.next
        del temp[-n]
        #rebuild
        dummy = ListNode(0)
        curr = dummy
        for num in temp:
            curr.next = ListNode(num)
            curr = curr.next
        return dummy.nextr.next.next
        return head

#two pass
class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        '''
        two pass
        '''
        #get the length
        #watch the edge cases, we are starting from zero for both nodes
        dummy = ListNode(0)
        dummy.next = head #embedding!
        N = 0
        temp = head
        while temp:
            N += 1
            temp = temp.next
            
        #offset 1
        N -= n #this part is really clever, from the end
        temp = dummy
        #adavance
        while N > 0: #keep decrementing
            N -= 1
            temp = temp.next
        
        #reconnect and pass
        temp.next = temp.next.next
        return dummy.next

#one pass
class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        '''
        we can optimze to one pass using two pointers
        first pointer advances n+1 steps from the beginning
        while second pointer starts from the beginning of the list
        now both pointers are exactly separated n nodes part
        we can maintain this gap by advancing both pointers toegether until the first pointer arrives past the last node.
        this must mean the second pointer will be pointing to the nth node (from the last)
        we relinkg the next pointer of the node referenced by the second pointer to point to the node's next next node
        '''
        dummy = ListNode(0)
        dummy.next = head
        first = dummy
        second = dummy
        
        #advance first n away
        for i in range(n+1):
            first = first.next
        #move both
        while first:
            first = first.next
            second = second.next
        
        second.next = second.next.next
        return dummy.next

###########################
#Combination Sum IV
###########################
class Solution(object):
    def combinationSum4(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        '''
        recusrive backtracking
        but before recursing again and advancing through the nums array, check is i can keep using then current number
        '''
        memo  = {}
        #yassssss
        
        def rec(idx,target):
            count = 0
            #careful not to keep going after target diminishes
            if target <= 0:
                if target == 0:
                    #valid path
                    return 1
                return 0
            if (idx,target) in memo:
                return memo[(idx,target)]
            for i in range(idx,len(nums)):
                candidate = nums[i]
                if target - candidate > 0:
                    #stay on the index
                    count += rec(idx,target-candidate)
                else:
                    #move up the index
                    count += rec(idx+1,target-candidate)
            memo[(idx,target)] = count
            return count
        return rec(0,target)

#top down, recursive with memo
class Solution(object):
    def combinationSum4(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        '''
        just some notes from the official LC writeup
        really asking for permutations
        or any array of multiplicty taken from nums == target
        takeaway:
            everytime we take a num reduce the target
        expression:
            we need to count the number of combinations taht make target
            combs(target) = \intersection combs(target - nums[i]) if target >= nums[i]
        if the numbers sorted, we could terminate in the loop to prune
        '''
        memo = {}
        
        def combs(remain):
            if remain == 0:
                return 1
            if remain in memo:
                return memo[remain]
            
            count = 0
            for num in nums:
                #can still us the numer
                if remain - num >=0:
                    count += combs(remain-num)
            memo[remain] = count
            return count
        
        return combs(target)

class Solution(object):
    def combinationSum4(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        '''
        we can also come up with the bottom up DP solution
        the analgous cache memo in top down recusrive is out dp array
        each cell in the dp array answers the question:
            how many combs for this target sum are there?
            if when we take a num from nums to decrement target we are still above zero, we add to this cell the comb_sum - num count
        
        '''
        dp = [0]*(target+1)
        dp[0] = 1
        
        #for each sum up to target
        for comb_sum in range(target+1):
            #for this current sum,keep taking nums and reduct target
            for num in nums:
                if comb_sum - num >= 0:
                    dp[comb_sum] += dp[comb_sum-num]
        return dp[target]

############################
#N-ary Tree Preorder Traversal
#############################
#i cant belive this shit workd
class Solution(object):
    def preorder(self, root):
        """
        :type root: Node
        :rtype: List[int]
        """
        '''
        lets do this recursively first
        recall preorder is root, left, right
        '''
        self.results = []
        def dfs(node):
            if not node:
                return
            self.results.append(node.val)
            if node.children:
                for child in node.children:
                    dfs(child)
        dfs(root)
        return self.results

#embedding path into function
class Solution(object):
    def preorder(self, root):
        """
        :type root: Node
        :rtype: List[int]
        """
        '''
        lets do this recursively first
        recall preorder is root, left, right
        '''
        def dfs(node):
            path = []
            if not node:
                return
            path.append(node.val)
            if node.children:
                for child in node.children:
                    path += dfs(child)
            return path
        return dfs(root)

#iteratively
class Solution(object):
    def preorder(self, root):
        """
        :type root: Node
        :rtype: List[int]
        """
        '''
        iteratively, well just simulate the stack call
        '''
        if not root:
            return []
        results = []
        stack = [root]
        while stack:
            curr = stack.pop()
            if curr:
                results.append(curr.val)
            if curr.children:
                for child in reversed(curr.children):
                    #i only figured it out after seeing the pattern
                    #need to add children in reverse
                    stack.append(child)
        return results

########################
#Triangle
########################
#yasssss!
#i got to TLE
class Solution(object):
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        '''
        find min path fro top to bottom, min path is the smallest sum of the numbers in the path
        recursive backtracking
        dfs
        from a position 
        just build all possbile paths, and to global bucket, then find min, and return
        
        '''
        rows = len(triangle)
        #only one row,edge cae
        if rows == 1:
            return min(triangle[0])
        self.paths = []
        def dfs(row,col,path):
            #if i've gotten to final row,add our path
            if row == rows:
                self.paths.append(path)
                return
            dfs(row+1,col+1,path+[triangle[row][col]])
            dfs(row+1,col,path+[triangle[row][col]])

        
        dfs(0,0,[])
        #find the min of the results
        result = float('inf')
        for p in self.paths:
            result = min(sum(p),result)
        return result

class Solution(object):
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        '''
        recursive official solution from LC
        we define a helper function rec(i,j) that returns the minium path sum formt he current (i,j)
        down to the base
        we only recurse when there is still a row to go down into
        written succintly:
            return triangle[i][j] + min(rec(i+1,j+1),rec(i+1,j))
        then we can just cache along the long
        '''
        memo = {}
        
        def dfs(i,j):
            if (i,j) in memo:
                return memo[(i,j)]
            result = triangle[i][j]
            #we keep recusring as long we can go down into a row
            if i < len(triangle)-1:
                result += min(dfs(i+1,j),dfs(i+1,j+1))
            #cache
            memo[(i,j)] = result
            return result
        
        return dfs(0,0)

#now lets go over some of the dp solutions
class Solution(object):
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        '''
        if we want the min path sum
        then we we would want the minimum from left aboveus or right above us
        then we would add this min the current cell we are at
        this is the main idea
        we can do this in place, and just bring down the min sum level by level
        notes on in place algos:
            1. if the algo needs to run in a multi thread environment (i.e parallel), then overwriting the space may not be a good idea
            2. if there is a single thread, the algo has exlcusvive access to the array while running, the array might need to be resued later by another thread
        since this a trianlge, we need to be careful of how we index
        cases:
            1. if row == col, to only
            2. if col == 0, onle one above (row-1,col)
            3. if col == row, one above but to elft (row-1,col-1)
            4. other case have both (row-1,col-1) or (row-1,col)
        we can collapse case 4, by checkt checking all cells in the current column
        '''
        for row in range(1,len(triangle)):
            #starting from second row actually
            for col in range(row+1): #col will always be 1 more than row
                smallest_above = float('inf')
                #find all 'available' cells above
                if col == 0:
                    smallest_above = triangle[row-1][col]
                elif col == row:
                    smallest_above = triangle[row-1][col-1]
                else:
                    smallest_above = min(triangle[row-1][col],triangle[row-1][col-1])
                #update in palce
                triangle[row][col] += smallest_above
        #return the min in the final row
        return min(triangle[-1])

#O(N) space
class Solution(object):
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        '''
        using aux space,
        if we wanted we could have just made a deep copy of the triangle
        and impute taht copy, that way we would have access to the oringal still
        even better, we just update rows for every time we are trying to find the minimum
        O(N) space
        '''
        
        prev_row = triangle[0]
        for row in range(1,len(triangle)):
            #starting from second row actually
            curr_row = []
            for col in range(row+1): #col will always be 1 more than row
                smallest_above = float('inf')
                #find all 'available' cells above
                if col == 0:
                    smallest_above = prev_row[col]
                elif col == row:
                    smallest_above = prev_row[col-1]
                else:
                    smallest_above = min(prev_row[col],prev_row[col-1])
                #update in palce
                curr_row.append(triangle[row][col] +smallest_above)
            prev_row = curr_row
        #return the min in the final row
        return min(prev_row)

#starting from the bottom of the pyramid
class Solution(object):
    def minimumTotal(self, triangle):
        """
        :type triangle: List[List[int]]
        :rtype: int
        """
        '''
        we can also start from the bottom
        say we have
          1
         2 3
        4 5 6
        
        we can start from the bottom row
        4 5 6
         2 3 
          1
          
        then for each cell, we have acess to to 2 from above, 
        we take the minimum of those two in place and incrment
        
        4 5 6
         6 8
          7
        '''
        #coyuld have also used aux space, NOT in place
        N = len(triangle)
        for row in range(N-2,-1,-1):
            for col in range(row+1):
                triangle[row][col] += min(triangle[row+1][col+1],triangle[row+1][col])
        return triangle[0][0]

class Solution(object):
    def minimumTotal(self, triangle):
        N = len(triangle)
        below_row, above_row = triangle[N-2], triangle[N-1]
        for row in range(N-2,-1,-1):
            for col in range(row+1):
                below_row[col] += min(above_row[col+1],above_row[col])
            above_row = below_row
            below_row = triangl

############################
#Brick Wall
##############################
#idk how to handle the edge cases
class Solution(object):
    def leastBricks(self, wall):
        """
        :type wall: List[List[int]]
        :rtype: int
        """
        '''
        if i want to find the least number of bricks, i need the line that has the most number of edges
        but it could be the case where the most number of edges may not have the least number of bricks?
        is this true??? i dont think so, taking up an edge means one less space for bricks
        N = len(wall), which is the number of rows
        rows = bricks + edges
        maximizing edges minimizes bricks
        another thing sum(ith row) same for all i in N
        if i find the edges just subtract from N
        cumsum array marks the edges
        get cumsum array, 
        hash:
            count up edges at cumsum
            find max, deduct from N
            actually you want second highest, highest marks end
        fucking edge cases!
        '''
        #edge case, if all same and one brike, return 
        N = len(wall)
        def helper(arr):
            N = len(arr)
            dp = [0]*N
            dp[0] = arr[0]
            for i in range(1,N):
                dp[i] = dp[i-1] + arr[i]
            return dp
        cumsums = []
        for row in wall:
            cumsums.append(helper(row))
        counts = defaultdict(int)
        for row in cumsums:
            for num in row:
                counts[num] += 1
        #edge case
        if len(counts) == 1:
            return N

        max_edges = 0
        for k,v in counts.items():
            if k != N:
                max_edges = max(max_edges,v)
        return N - max_edges
            return N - max_edges
        else:
            return max_edges

#YASSSSSS
class Solution(object):
    def leastBricks(self, wall):
        """
        :type wall: List[List[int]]
        :rtype: int
        """
        '''
        if i want to find the least number of bricks, i need the line that has the most number of edges
        but it could be the case where the most number of edges may not have the least number of bricks?
        is this true??? i dont think so, taking up an edge means one less space for bricks
        N = len(wall), which is the number of rows
        rows = bricks + edges
        maximizing edges minimizes bricks
        another thing sum(ith row) same for all i in N
        if i find the edges just subtract from N
        cumsum array marks the edges
        get cumsum array, 
        hash:
            count up edges at cumsum
            find max, deduct from N
            actually you want second highest, highest marks end
        fucking edge cases!
        well hold on, i think you had the right idea, just build up the hash on the fly
        don't include the last cumsum though
        '''
        mapp = defaultdict(int)
        for row in wall:
            cum_sum = 0
            for i in range(len(row)-1): #don't iclude the last brick
                cum_sum += row[i]
                mapp[cum_sum] += 1
        result = len(wall)
        for k,v in mapp.items():
            result = min(result,len(wall)-v)
        return result

#update on the fly
class Solution(object):
    def leastBricks(self, wall):
        """
        :type wall: List[List[int]]
        :rtype: int
        """
        mapp = defaultdict(int)
        result = len(wall)
        for row in wall:
            cum_sum = 0
            for i in range(len(row)-1): #don't iclude the last brick
                cum_sum += row[i]
                mapp[cum_sum] += 1
                #update on the fly
                result = min(result, len(wall)-mapp[cum_sum])
        
        return result

#brute force, a few ways
class Solution(object):
    def leastBricks(self, wall):
        """
        :type wall: List[List[int]]
        :rtype: int
        """
        '''
        brute force TLE
        we traverse a unit brick starting from the first row
        we can keep track of what brick we are one by using another array to mark the bth brick in each row
        we also maintain a count variable to count the number of times we cross the brick
        for every rwo considered during the colummn by column traversal (incremented one at time) we check if we've hit
        this is done by updating the bricks widht in the array"
        IMPORTANT: make sure we can overwrite the input array, ask interviewer
        
        '''
        #get widht of wall
        width = 0
        for b in wall[0]:
            width += b
        
        #get height, positions arrary,and keep track of fewetet corssings
        height = len(wall)
        pos = [0]*height
        fewest_intersections = height #starting off we'd assume the first line is crossing all bricks
        for col in range(width -1):
            count = 0
            for i in range(height):
                curr_row = wall[i]
                #use up the current brick by cutting it
                curr_row[pos[i]] -= 1
                if curr_row[pos[i]] == 0:
                    #move up to the new brick
                    pos[i] += 1
                #otherwise we have a crossing
                else:
                    count += 1
            #after the first virutal brick of length 1, we update
            fewest_intersections = min(fewest_intersections,count)
            
        return fewest_intersections

###########################################
#Missing Number In Arithmetic Progression
##########################################
class Solution(object):
    def missingNumber(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        '''
        the big give away is that the first or last value was not removed
        we can just use the sum of an arithmetic sequence
        find the sum, and divide by the number of intervals n-1
        sum = n(a1 + aN) / 2
        subtract the sum(arr) from this sum
        '''
        expected_sum = (len(arr)+1)*(arr[0] + arr[-1]) / 2
        return expected_sum - sum(arr)

class Solution(object):
    def missingNumber(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        '''
        official LC solution
        we know the first and last were not removed,
        we can get there different and divide by the number of values
        idx = initial
        idx1 = initial + 1*diff
        idx2 = initial + 2*diff
        idxn = initital + n*diff
        
        algo:
            find the expected diffeence
            loop through the array and check if the elelent is expected or not
            if ist not return
            for starters assume expected be the first element
            this is a good trick, keep this in mind
        '''
        N = len(arr)
        diff = (arr[-1] - arr[0]) / N
        missing = arr[0]
        for num in arr:
            if num != missing:
                return missing
            missing += diff
        return missing

#binary search
class Solution(object):
    def missingNumber(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        '''
        in this approach we can use binary seach,
        we know that consective difference must be at least the expected difference
        when using binary search we can check if the mid_idx*difference is arr[0] + that
        if it is, we know all elements from lo are ok
        if not we check the right
        NOW: the lo idx pointer will be the element just before the missing?
        why? is this the case because it's supposed to be mid! 
        and it not!
        we return the index at the lowe + lo*difference
        '''
        N = len(arr)
        diff = (arr[-1] - arr[0]) / N
        lo = 0
        hi = N-1
        
        while lo < hi:
            mid = lo + (hi-lo) // 2
            #we check if mid matches, if it does, the left side could not contain missing
            if arr[mid] == arr[0] + mid*diff:
                lo = mid + 1
            else:
                hi = mid
        return arr[0]+diff*lo

#######################
#Count Binary Substrings
#######################
class Solution(object):
    def countBinarySubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        two pointer
        '''
        count = 0 
        for i in range(len(s)):
            #we need to scan for two cases 10 or 01
            l,r = i,i+1
            while l >=0  and r < len(s) and s[l] == '0' and s[r] == '1':
                count += 1
                l -= 1
                r += 1
            l,r = i,i+1
            while l >=0  and r < len(s) and s[l] == '1' and s[r] == '0':
                count += 1
                l -= 1
                r += 1
        return count

#could have also used a q
class Solution(object):
    def countBinarySubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        q = deque()
        ans = 0
        for char in s:
            #where we need to clear a bunch of ones and zeros
            #kind like capturing parantheses
            while q and q[-1] == c and q[0] != c:
                q.pop()
            if q and q[-1] != c:
                q.pop()
                ans += 1
            q.appendleft()
        return ans 

#grouping the elements and taking the min
class Solution(object):
    def countBinarySubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        we can group by character and push the counts of groups into an array
        this is pretty cool
        s = "110001111000000", then groups = [2, 3, 4, 6]
        for every binary string of
        '0'*k + 1*'k' or '1'*k + '0'*k, the middle of thes tring must occur between the groups
        now once we have the groups, we can take the minimum of each consective grouping
        when we take the min of those two, we want to sum them up
        we start of the array groups with 1, meaning 1 group present
        if the two consecutive elements are different, we start a new group
        '''
        groups = [1]
        for i in range(1,len(s)):
            if s[i-1] != s[i]:
                groups.append(1)
            else:
                groups[-1] += 1
                
        #we can only make substring with consective zeros and ones
        #but we take min count of the grouping
        count = 0
        for i in range(1,len(groups)):
            count += min(groups[i],groups[i-1])
        return count

#one pass
class Solution(object):
    def countBinarySubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        instead of building out the whole groups array, we notice that its similar to DP
        where we only care about the curr and prev grou
        then we can just update
        '''
        count = 0
        curr_group = 1
        prev_group = 0 
        #what does this mean? initally we have 1 group, but no prev group
        for i in range(1,len(s)):
            if s[i-1] != s[i]:
                count += min(prev_group,curr_group)
                prev_group = curr_group
                curr_group = 1
            else:
                #same group
                curr_group += 1
        
        #before returning, we need check the final group
        return count + min(curr_group,prev_group)

###############################
#Critical Connections in a Network
###############################
#brute force TLE
#be happy you got this one
class Solution(object):
    def criticalConnections(self, n, connections):
        """
        :type n: int
        :type connections: List[List[int]]
        :rtype: List[List[int]]
        """
        '''
        hard problem
        we are essentially looking for a bridge connection
        if we we remove this connection the graph becomes unconnected
        remember the edges are undirected
        deletion of bridge increases the number of conenctred nodes
        tarjan's algorithm is a standard way of finding the articulation of points and bridges in a grpah
        but really, you could just ust dfs to solve this problem
        https://www.youtube.com/watch?v=RYaakWv5m6o&ab_channel=TechRevisions
        brute force: drop all edges and see if we get anymore disconnected components
        then dfs to see if i can visit all nodes
        '''
        #brute force
        #dfs on the graph after removing and edge
        #if after removing the edge, i cannot
        adj_list = defaultdict(list)
        for a,b in connections:
            adj_list[a].append(b)
            adj_list[b].append(a) #need to add cmoplements because it is undireced
        
        #define dfs, traverse the graph and return the visited set
        def dfs(node):
            seen.add(node)
            for neigh in adj_list[node]:
                if neigh not in seen:
                    dfs(neigh)
        #i dont know if dfsing from each node would give me disconenctiend graph
        #well in voke on each n, freeze the set, add to it and call it inital start
        initial_state = set()
        for i in range(n):
            seen = set()
            dfs(i)
            temp = []
            for v in seen:
                temp.append(v)
            temp = frozenset(temp)
            initial_state.add(temp)
            
        bridges = set()
        
        #now simulate dropping an edge
        for i in range(len(connections)):
            dropped = copy.deepcopy(connections)
            del dropped[i]
            #build_adj for dropped edge
            adj_list = defaultdict(list)
            for a,b in dropped:
                adj_list[a].append(b)
                adj_list[b].append(a)
                #but now i have to dfs for all the nodes in this curr adjlist
            curr_state = set()
            for j in range(n):
                seen = set()
                dfs(j)
                temp = []
                for v in seen:
                    temp.append(v)
                temp = frozenset(temp)
                curr_state.add(temp)
            if curr_state != initial_state:
                v1,v2 = connections[i]
                bridges.add((v1,v2))
        #convert back to se
        return bridges

class Solution(object):
    def criticalConnections(self, n, connections):
        """
        :type n: int
        :type connections: List[List[int]]
        :rtype: List[List[int]]
        """
        '''
        LC writeu:
        tarjan's is an effienct articulation and bridge graph finding algorithm 
        that find's them in O(nodes + edges) time
        key:
            if an edge is part of cycle, than any edge in that cyle cannot be a critical componeent, i.e an edge is a criticle connection, if and only if it is not in a cycle, think on this. why?
            if there are mulitple ways, then our edge would not be part of the ccyle
        approach: dfs for cycle detection
        new concept: rank
            rank is similar to the concept of discovery times in tarjans
            we consider each node to be a root, temporariliy
            the rank of the root noe is always -, if not visited, keep rank as None
            using concept of rank is very similar to keeping set of visited nodes
            at each step of our traversal, we maintain the rank of the nodes we've come across until now in the current path
            if at any point, we come acorss a neigh that has a lower rank than then current node, we know then that the nieghbor must have already been visited
            SO! if we started along a path with rank 0 from the root, and aure at a dnow ith rank m and we discover a node that alreayd has rank assigned to ie 0 <=n <m, then that implies a cycle
        importance: 
            we can detect a cycle simply by checking if the rank has already been assigned to some neighbor or not, so when we detect a cycle, discard that edge!
            BUT what about ancestral edges?
            WELL, what we need is the minimum rank that a cycle inldues. we need our dfs helper to return this information so taht previous callers can use it to identify if an edge has to be discarded or not
            after discarding an edge, the returned min rank (from all neighboring nodes) is less than current, carry it back to the previous caller
        we know that only the current level knows of the presence of a cycle
        to make upper levels of recusion aware of this cycle, and to help discard edges, we return the min rank that our traversk fins
        during a step of recusion from u to b, if DFS return something <= rank of u, then u knows its neighbotr v is part of the cycle spanning back to u or some other node higher up in the recursion tree
        algo:
            1. dfs function will take in node and rank
            2. build adj list
            3. we need graph and conndict
            4. need array to rank our nodes
            6 dfs function: 
                * check if node has rank assigned 
                * else assing rankd of this node
                * iterate over neighbords, and recuse on each of them, returing value and doing two things
                    1. is rank <= discoer rank, edge is part of cycle and can be discared
                    2. record min rank
                return min rank
        
        '''
        rank = {}
        graph = defaultdict(list)
        conn_dict = {}
        
        #build rank array
        for i in range(n):
            rank[i] = None
        #build graph
        for a,b in connections:
            graph[a].append(b)
            graph[b].append(a) #graph is undirected
            
            #counting number of connections low rank to high rank initally
            conn_dict[(min(a,b),max(a,b))] = 1
        
        def dfs(node,discovery_rank):
            #if node has been already visited
            if rank[node]:
                return rank[node]
            #otherwise update the current rank for the node, i.e jsut marking
            rank[node] = discovery_rank
            #when we recurse we need to make sure we go up a rank
            min_rank = discovery_rank + 1 #could also just max it out
            #examin neighs
            for neigh in graph[node]:
                #skip parente
                if rank[neigh] and rank[neigh] == discovery_rank - 1:
                    continue
                #recurse on niegh
                rec_rank = dfs(neigh, discovery_rank+1)
                #1. check if edge needs to be discarded
                if rec_rank <= discovery_rank:
                    del conn_dict[(min(node,neigh),max(node,neigh))]
                #2. carry the min
                min_rank = min(min_rank,rec_rank)
            return min_rank
        result = []
        dfs(0,0)
        for a,b in conn_dict:
            result.append([a,b])
        return result

class Solution(object):
    def criticalConnections(self, n, connections):
        """
        :type n: int
        :type connections: List[List[int]]
        :rtype: List[List[int]]
        """
        '''
        https://leetcode.com/problems/critical-connections-in-a-network/discuss/1174196/JS-Python-Java-C%2B%2B-or-Tarjan's-Algorithm-Solution-w-Explanation
        notes:
        the algo is referred to as Tarjan's bridge finding algo, which combines recursion with union find
        takeaways:
            we dfs on our graph for each node, keeping track of the earliest node we can get circle back to: if from the curr node we see a node of lower rank, than the edge from that current node to the current far node must be a bridge
        need:
            disocvery time array
            lowest future node array
            time discovered, just += 1 for each invocation
        dfs helper"
            for each newly visited node set value for both discover time and lowest rank node before time is increment
            recurse on the neighbors of the current node if unvisited, 
            then if in of the next posible nodes is of a lower rank, we have a loop, and edge should be removed
            if the rank of the future node from this current node is higher, there is no looped conenction and th edge between curr and nex is  a bridge
        '''
        edges  = defaultdict(list)
        for a,b in connections:
            edges[a].append(b)
            edges[b].append(a)
        disc_times = [0]*n
        low_ranks = [0]*n
        times = [1] #instead of keeping global variable
        bridges = []
        
        def dfs(curr,prev):
            #makr curret node
            disc_times[curr] = times[0]
            low_ranks[curr] = times[0]
            #now update time
            times[0] += 1
            #neighbor check
            for neigh in edges[curr]:
                #if not markedt yet, dfs, but also update low ranks
                if not disc_times[neigh]:
                    dfs(neigh,curr)
                    low_ranks[curr] = min(low_ranks[curr],low_ranks[neigh])
                elif neigh != prev:
                    low_ranks[curr] = min(low_ranks[curr], disc_times[neigh])
                if low_ranks[neigh] > disc_times[curr]:
                    #bridge
                    bridges.append([curr,neigh])
        dfs(0,-1)
        return bridges

##################
#Rotate Image
##################
class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        '''
        well if i were to allocate extra space, read bottom up, and left to right of the original matrix into a new matrix, and then impute back in to the original matrix
        do this first
        allocation extra space
        '''
        rows = len(matrix)
        cols = len(matrix[0])
        
        new_mat = [[] for _ in range(rows)]
        new_mat_ptr = 0
        for c in range(cols):
            for r in range(rows):
                #lets be clever is use index offsetting :)
                candidate = matrix[rows-r-1][c]
                new_mat[new_mat_ptr].append(candidate)
            new_mat_ptr += 1
        #impute
        for i in range(rows):
            for j in range(cols):
                matrix[i][j] = new_mat[i][j]

#rotate in place
class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        '''
        we can't just swap, otherwise the matrix gets overwrite incorrectly
        we rotate in groups of fair
        algo:
            at at one time we reference four positions in the array
            1. we cache the bottom left, call it temp
            2. the bottom left becomes the bottom right
            3. the bottom right becomes the upper right
            4. the upper right becomes the upper left
            5. the upper left becomes what we cached
            the indexing is fiddly, but really we just shrink the window frame every time
            we only need to through N // 2, where N is th enumber of columns
            note, in this pass, iff N is odd, the center row and cols are manipualted last
            this is just one of those problems where you might have to draw ane example out and hard code the rules
        '''
        N = len(matrix[0])
        for i in range(N//2 + N%2):
            for j in range(N//2):
                tmp = matrix[N-1-j][i] #bottom left
                matrix[N-1-j][i] = matrix[N-1-i][N-j-1] #bottome right becomes bottom left
                matrix[N-1-i][N-j-1] = matrix[j][N-1-i] #bottom right becomes upper right
                matrix[j][N-1-i] = matrix[i][j] #upper right becomes upper left
                matrix[i][j] = tmp #upper left becomes what cached

#tranpose and reflect
class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        '''
        the cheeky way to transpose the matrix and reflect
        tranpose reflect!
        '''
        #transpose
        rows = len(matrix)
        cols = len(matrix[0])
        
        #tranpose, 
        for i in range(rows):
            for j in range(i,cols):
                matrix[j][i], matrix[i][j] = matrix[i][j], matrix[j][i]
        #reflect
        for i in range(rows):
            #now think of reverse string in place
            for j in range(cols//2):
                matrix[i][j], matrix[i][-j - 1] = matrix[i][-j - 1], matrix[i][j]

##########################
#Furthest Building You Can Reach
##########################
#general solution
class Solution(object):
    def furthestBuilding(self, heights, bricks, ladders):
        """
        :type heights: List[int]
        :type bricks: int
        :type ladders: int
        :rtype: int
        """
        '''
        think about the two questinos that can be asked.
        how far we can get? vs can we get to the end?
        we dont know which climbs we need to cover because we dont know the final building we canr ached
        if our initial idea was to use ladders on the largest jumps every time, well we dont know what the largest jump would be becasue we haven't seen it yet
        the idea would be to use ladders first, and then reallocate a ladder with the right amount of bricks when we can
        we can then furnisha new ladder, at the cost of using more bricks
        KEY:  If we're out of ladders, we'll replace the most wasteful ladder allocation with bricks
        reteive the smallest climb, MIN heap!
        '''
        N = len(heights)
        ladders_used = []
        for i in range(N-1):
            climb = heights[i+1] - heights[i]
            if climb <= 0:
                continue #no need for a climb
            #use up a ladder
            if ladders > 0:
                ladders -= 1
                heappush(ladders_used,climb)
            #what if we have no ladders, use brick
            else:
                if ladders_used:
                    smallest_ladder = ladders_used[0]
                    if climb < smallest_ladder: #forced to use bricks
                        bricks -= climb
                    else: #we can reallocate
                        smallest_ladder = heappop(ladders_used)
                        heappush(ladders_used,climb)
                        bricks -= smallest_ladder
                #watch for the empty heap case, we just use the current climb 
                #and deduct from bricks
                else:
                    bricks -= climb
                    #if bricks is negative and now ladders, end
                if bricks < 0:
                    return i

        return N-1
                
#recudec form less conditional
class Solution(object):
    def furthestBuilding(self, heights, bricks, ladders):
        """
        :type heights: List[int]
        :type bricks: int
        :type ladders: int
        :rtype: int
        """
        '''
        on each iteration we always add the current climb to the hap, 
        forcing the heap for become longer; heap constraint problem\
        notes on time complexty, NlogN, typical heapp
        but note the heap will never contain L+1 climbs at a time, when the heap gets to L, we need to pop
        and replace with bricks and reclaim  aldder
        so its actuall LogL, for each N, in the worst case, N logL
        '''
        N = len(heights)
        ladders_used = []
        for i in range(N-1):
            climb = heights[i+1] - heights[i]
            if climb <= 0:
                continue #no need for a climb
            #otherwise use a ladder for this height
            heappush(ladders_used,climb)
            #check we can still have ladders
            if len(ladders_used) <= ladders:
                continue
            #otherwise we need to use bricsk here
            bricks -= heappop(ladders_used)
            #if we go negative
            if bricks < 0:
                return i
        return N - 1

#max heap
class Solution(object):
    def furthestBuilding(self, heights, bricks, ladders):
        """
        :type heights: List[int]
        :type bricks: int
        :type ladders: int
        :rtype: int
        """
        '''
        we can flip the problem and use a maxheap, but ladders and bricks should now change
        instead of allocatin ladders we allocate bricks
        when we are out of bricks, replace the LONGEST climb with a ladder
        algo notes:
            we need to keep track of how many bricks we've used, 
            the easies way is ot subtract bricks from the input parameter and check when it goes to zero
            the only difference now, is that when bricks goes negative, we donlt simply stop
            we need to try to get more bricks by exchaning it with a ladder
            we do this by removing the largest brick allovation, i.e largest climb from the heap
            why does this work? there should be a previous climb with more bricks to reclaim
            or we've just added the largest climb onto the max heap
            we keep going until ladders run out
        '''
        bricks_used = []
        N = len(heights)
        for i in range(N-1):
            climb = heights[i+1] - heights[i]
            if climb <= 0:
                continue # do nothing
            #MAX HEAP
            heappush(bricks_used, -climb)
            #use up bricks
            bricks -= climb
            #our of bricks and ladders
            if bricks < 0 and ladders == 0:
                return i
            #but if we out of bricks and still have ladders, use a ladder to gain bricks
            if bricks < 0:
                bricks += -heappop(bricks_used)
                ladders -= 1
        return N - 1

#binarys search
class Solution(object):
    def furthestBuilding(self, heights, bricks, ladders):
        """
        :type heights: List[int]
        :type bricks: int
        :type ladders: int
        :rtype: int
        """
        '''
        binary search for final reachable building
        if we can ask the question, given the number of bricks and ladders, can i reach the kth building? can i reach the kth + 1 building
        from the first two solutions, we found an NLogL approach, and doing linear search for each thing would be (N^2logL)
        but if we go binary search search N*(logL)^2
        before doing binary search, we need to see if our question turns the array into an equivalent sorting, if a building is reachable from k, then i can reach builidng [0 to k]. and not k+1 to end
        https://leetcode.com/problems/furthest-building-you-can-reach/solution/
        elegant walk through on binary search best practices
        TAKEAWAY ON POINTERS:
        The short rule to remember is: if you used hi = mid - 1, then use the higher midpoint. If you used lo = mid + 1, then use the lower midpoint.
        '''
        def is_reach(idx):
            #determine if the idx buiuldg is reachable
            climbs = []
            for h1,h2 in zip(heights[:idx], heights[1:idx+1]):
                if h2 - h1 > 0:
                    climbs.append(h2-h1)
            #sort the climbs
            climbs.sort()
            #now check whether or not we have enough bricks and laddders to cover all climbs, use up bircks before ladders
            bricks_rem = bricks
            ladders_rem = ladders
            for c in climbs:
                #not enough bricks
                if bricks_rem >= c:
                    bricks_rem -= c
                elif ladders_rem >= 1:
                    ladders_rem -= 1 #need at least one latter
                else:
                    #i cant make it
                    return False
            return True
        
        #binary search
        lo,hi = 0, len(heights) -1
        #we want a single index, and if we are length two in the final search we want the smaller one
        while lo < hi:
            mid = lo + (hi - lo + 1) // 2 #why + 1?, we want the upper mid to guarentee convergence
            if is_reach(mid):
                #building is reacable, but idk if mid + 1 is
                lo = mid
            else:
                hi = mid - 1 #not reachable
        return lo

#improvmennts to binnary serach
class Solution(object):
    def furthestBuilding(self, heights, bricks, ladders):
        """
        :type heights: List[int]
        :type bricks: int
        :type ladders: int
        :rtype: int
        """
        '''
        we can improve binary search from approach 4
        instead  of min sorting the climbs array every time, just attach on index  to each climb in the climbs lists
        then in  the reachable function, add a condition telling it to skip any climbs with an high  index greather than the one passed into the check
        algo:
            get the sorted climbs once, but all attach an inndex
            min sort
            then in min reachable function, skip  climbs greater  than current index
        '''
        sorted_climbs = []
        N = len(heights)
        for i in range(N-1):
            climb = heights[i+1] - heights[i]
            if climb <= 0:
                continue
            sorted_climbs.append([climb,i+1])
        sorted_climbs.sort(key = lambda x: x[0])
        
        def is_reachable(building_idx,climbs,bricks,ladders):
            for c,idx in climbs:
                if idx > building_idx:
                    continue
                #use bricks first
                if bricks >= c:
                    bricks -= c
                elif ladders >= 1:
                    ladders -= 1
                else:
                    return False
            return True
        #binary search
        #binary search
        lo,hi = 0, len(heights) -1
        #we want a single index, and if we are length two in the final search we want the smaller one
        while lo < hi:
            mid = lo + (hi - lo + 1) // 2 #why + 1?, we want the upper mid to guarentee convergence
            if is_reachable(mid,sorted_climbs,bricks,ladders):
                #building is reacable, but idk if mid + 1 is
                lo = mid
            else:
                hi = mid - 1 #not reachable
        return lo


#################
#Power of Three
#################
class Solution(object):
    def isPowerOfThree(self, n):
        """
        :type n: int
        :rtype: bool
        """
        '''
        ughhh, just log base 3, if its an int or not
        or just check all powers of three
        '''
        if n <= 0:
            return False
        power = 0
        while 3**power <= 2**31 -1:
            if n == 3**power:
                return True
            power += 1
        return False

#binary search
class Solution(object):
    def isPowerOfThree(self, n):
        """
        :type n: int
        :rtype: bool
        """
        '''
        in the last approach, i was  steadyling increasing by 1
        i can use bianry search to speed it up
        '''
        #neagative case
        if n <= 0:
            return False
        #but what power of three one just beflow 2**31 - 1
        # 3^X <= 2**31 - 1
        highest_power = floor(log(2**31 - 1) /log(3))
        lo, hi = 0, int(highest_power)
        while lo <= hi:
            mid = lo + (hi - lo) // 2
            if 3**mid == n:
                return True
            elif 3**mid > n:
                hi = mid -1
            else:
                lo = mid + 1
        return 3**lo == n #bonus points for this return!

class Solution(object):
    def isPowerOfThree(self, n):
        """
        :type n: int
        :rtype: bool
        """
        '''
        could also just keep diving by 3 and so long as we have no reaminder and check we can get to 1
        '''
        if n < 1:
            return False
        while n % 3== 0:
            n /= 3
        return n == 1

#change of base
class Solution(object):
    def isPowerOfThree(self, n):
        """
        :type n: int
        :rtype: bool
        """
        '''
        use change of base to find the exponent, and check that exponenet i int
        '''
        if n < 1:
            return False
        exponent = log(n) / log(3)
        return abs(exponent - round(exponent)) < 1e-11

#modular arithmetic
class Solution(object):
    def isPowerOfThree(self, n):
        """
        :type n: int
        :rtype: bool
        """
        '''
        coindicentally three is a prime number
        the largest power of three that its into signed 32 bit is 3**19
        '''
        if n > 0 and 3**19 % n == 0:
            return True
        else:
            return False

################
#Unique Paths II
################
#using ausx space
class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        '''
        in the original problem, we just accumlate the total paths from a cells top neighbfor and left neighbor
        dp array problem and return the bottomr right answer
        use aux space first
        '''
        rows = len(obstacleGrid)
        cols = len(obstacleGrid[0])
        dp = [[0]*cols for _ in range(rows)]
        #fill top row
        for i in range(cols):
            if obstacleGrid[0][i] == 0:
                dp[0][i] = 1
            else:
                break
        #fill in first col
        for i in range(rows):
            if obstacleGrid[i][0] == 0:
                dp[i][0] = 1
            else:
                break
        
        #starting from (1,1)
        for i in range(1,rows):
            for j in range(1,cols):
                if obstacleGrid[i][j] == 0: #no no obstacle
                    dp[i][j] = dp[i][j-1] + dp[i-1][j]
                else:
                    dp[i][j] = 0
        return dp[-1][-1] 

#now in place
class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        '''
        now do in place,
        since the array is only 1s and zeroz
        '''
        rows = len(obstacleGrid)
        cols = len(obstacleGrid[0])
        obstacleGrid[0][0] = 1
        #fill top row
        for i in range(1,cols):
            if obstacleGrid[0][i] == 0:
                obstacleGrid[0][i] = 1
            else:
                break
        #fill in first col
        for i in range(1,rows):
            if obstacleGrid[i][0] == 0:
                obstacleGrid[i][0] = 1
            else:
                break
        #now start from (1,1)
        for i in range(1,rows):
            for j in range(1,cols):
                if obstacleGrid[i][j] == 0:
                    obstacleGrid[i][j] = obstacleGrid[i][j-1] + obstacleGrid[i-1][j]
                else: 
                    obstacleGrid[i][j] = 0
        return obstacleGrid[-1][-1]

#######################
# Find First and Last Position of Element in Sorted Array
#######################
#linear time
class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        '''
        well we can just scan the array and fill in the spots
        '''
        N = len(nums)
        ans = [-1,-1]
        
        #start from front
        for i in range(N):
            if nums[i] == target:
                ans[0] = i
                break
                
        #start from back
        for j in range(N):
            if nums[N-j-1] == target:
                ans[1] = N-j-1
                break
        return ans

class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        '''
        great, how about log n
        binary search to find the middle, then send out two pointers
        no because in the case all elements are target, you'd up doing two pointers for the whole array
        toughie....
        [1,1,3,4,5,5,5,5,5,7,8,9,10,11]
        we can use two binary search to find the first and last position, but how
        FIRST POSITION IN THE ARRAY:
            case1: if mid is the beginning index of target, our mid is first done
            case2:the element to the left of this index is not equal to the target, (i.e nums[mid] != target, we keep looking on the left side of the array)
            [2,7,7,7,10]
            mid is 7, then check mid-1 == target, discard right and search left
        LAST POSITION IN THE ARRAY:
            [2,7,7,7,8,10]
            mid is 7, and mid+1 == 7, discard left and search right
        algo:
            1 def find bound finctions
            2. two vars, begin and end
            3. binar searchr beg <= end
            4. each step calculate the middle element
                if nums[mid] == target:
                    if looking for first:
                        check mid and mid -1 conditions
                    if looking for end:
                        check mid and mid + 1 conditions
                if nums[mid] > target:
                    end = mid-1
                if nums[mid] < taget:
                    beg = mid + 1
            5. note in the main functino if after finding first it's tsitll -1, retunr [-1,-1] because there is no start
        '''
        def findBound(nums, target,isFirst):
            N = len(nums)
            beg,end = 0,N-1
            while beg <= end:
                mid = beg + (end - beg) //2
                #found target
                if nums[mid] == target:
                    #check finding lowbound
                    if isFirst:
                        #found first, not if we've found our target, at some point mid becomes beg, it has to
                        if mid == beg or nums[mid-1] < target:
                            return mid
                        #seach onf left
                        end = mid -1
                    #finding high bound
                    else:
                        #found last
                        if mid == end or nums[mid+1] > target:
                            return mid
                        beg = mid + 1
                elif nums[mid] > target:
                    end = mid - 1
                else:
                    beg = mid + 1
            return -1
        
        low_bound = findBound(nums,target,isFirst=True)
        if low_bound == -1:
            return [-1,-1]
        up_bound = findBound(nums,target,isFirst = False)
        return [low_bound,up_bound]

#another way
class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        '''
        we can do two binary searches finding left most target
        and right most target
        '''
        if not nums:
            return [-1,-1]
        N = len(nums)
        start,end = -1,-1
        lo,hi = 0,N
        #not setting <= means the pointers can eventaully cross one another
        while lo < hi:
            mid = lo + (hi-lo) // 2
            if nums[mid] >= target:
                hi = mid
            else:
                lo = mid+1
        if lo < N and nums[lo] == target:
            start = lo
            
        #find right bound
        lo,hi = 0,N
        while lo < hi:
            mid = lo + (hi-lo) // 2
            if nums[mid] <= target:
                lo = mid+1
            else:
                hi = mid
        if nums[hi-1] == target:
            end = hi-1
        
        return [start,end]
                
#another way
#https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/discuss/1181832/JS-Python-Java-C%2B%2B-or-Easy-Binary-Search-Solution-w-Explanation
class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        '''
        just some notes in the bisect left and right functions for python
        biseft left returns the idx of the first occurence in the sorted array 
        bisect right returns the idx+1, of the right most target element in the array
        '''
        Tleft = bisect_left(nums,target)
        #not in the array
        if Tleft == len(nums) or nums[Tleft] != target:
            return [-1,-1]
        return [Tleft,bisect_right(nums,target)-1]

#########################
#Meeting Scheduler
########################
#two pointer
class Solution(object):
    def minAvailableDuration(self, slots1, slots2, duration):
        """
        :type slots1: List[List[int]]
        :type slots2: List[List[int]]
        :type duration: int
        :rtype: List[int]
        """
        '''
        if the intervlas are sorted by start
        then the common slot available for each pair is:
        [max(start1,start2),min(end1,end2)]
        we can sort and use two poitners
        why do we sort on the start? if a slot starts earlier, it will end earlier, remmeber for both people there are no overlapping time slows
        how do we decide which pointer should beincrment:
            we always move the one that ends EARLIER!
            if we sorted on the start times
            and is slots1[i][1] > slots2[j][1], we know slots[i+1][0] > slots2[j][1]
            so that there will be no intersction between slots[i+1] and slots2[j]
        TAKEAWAY, SORT ON START, THEN UP EARLIER END
        '''
        slots1.sort()
        slots2.sort()
        
        p1 = p2 = 0
        while p1 < len(slots1) and p2 < len(slots2):
            #find common slot
            common_start = max(slots1[p1][0],slots2[p2][0])
            common_end = min(slots1[p1][1],slots2[p2][1])
            if common_end - common_start >= duration:
                return [common_start,common_start+duration]
            #move pointer to slot ending earlier, DUH! now that I think aboutit
            if slots1[p1][1] < slots2[p2][1]:
                p1 += 1
            else:
                p2 += 1
        return []

#using heap
class Solution(object):
    def minAvailableDuration(self, slots1, slots2, duration):
        """
        :type slots1: List[List[int]]
        :type slots2: List[List[int]]
        :type duration: int
        :rtype: List[int]
        """
        '''
        we can also use a heap, and we only need one heap
        we can push all the time slots into a heap, and if any two time slots are common, we know they are from different people?
        THINK??? WHY IS THIS ALLOWED
        it is guaranteed that no two available time slot of the same person intersect
        for any two time slots [start1, end1] and [start2, end2] of the same person, either start1 > end2 or start2 > end1.
        TAKEWAY, any two existinng time slots must be from different perople
        algo:
            inint heap of timeslots and push time slots that last longer than durationt into it
            keep popping from the heap, so lon as we have 1 element in the heap
            pop first
            peek next
            if we find end1>= start2+draitions, return it
        '''
        all_slots = slots1+slots2
        #now push all slots >= duration on to heap
        heap = []
        for start,end in all_slots:
            if end - start >= duration:
                heappush(heap,(start,end))
        while len(heap) > 1:
            start1,end1 = heappop(heap)
            start2,end2 = heap[0]
            if end1 - start2 >= duration:
                return [start2,start2+duration]
        return []

##########################
# Powerful Integers
##########################
#log(1) results in div by zerp
#TLE
class Solution(object):
    def powerfulIntegers(self, x, y, bound):
        """
        :type x: int
        :type y: int
        :type bound: int
        :rtype: List[int]
        """
        '''
        a powerful integer can be written as:
        x^i + y^j <= bound
        best idea would be to simulate, keep taking powers of i and j >=, but first we need to bound i and j
        the smallest value x and y can be when raised to i and j is zero
        then you have the case when x or y is 1
        '''
        #edge case
        if bound == 0:
            return []
        #find highest i and highest j
        highest_i = 0
        highest_j = 0
        
        while bound - 1 - x**highest_i >= 0:
            highest_i += 1
            
        while bound - 1 - x**highest_j >= 0:
            highest_j += 1
        
        res = set()
        for i in range(highest_i+1):
            for j in range(highest_j+1):
                cand = x**i + y**j
                if cand <= bound:
                    res.add(cand)
        return res

class Solution(object):
    def powerfulIntegers(self, x, y, bound):
        """
        :type x: int
        :type y: int
        :type bound: int
        :rtype: List[int]
        """
        '''
        a powerful integer can be written as:
        x^i + y^j <= bound
        best idea would be to simulate, keep taking powers of i and j >=, but first we need to bound i and j
        the smallest value x and y can be when raised to i and j is zero
        then you have the case when x or y is 1
        '''
        #edge case
        if bound == 0:
            return []
        #find highest i and highest j, watch the case for 1
        highest_i = bound if y == 1 else int(floor(log(bound-1)/log(y)))
        highest_j = bound if x == 1 else int(floor(log(bound-1)/log(x)))
        
        
        res = set()
        for i in range(highest_j+1):
            for j in range(highest_i+1):
                cand = x**i + y**j
                if cand <= bound:
                    res.add(cand)
                #don't forget the break conditions
                if x == 1:
                    break
            if y == 1:
                break
        return res
#using bound-1
class Solution(object):
    def powerfulIntegers(self, x, y, bound):
        """
        :type x: int
        :type y: int
        :type bound: int
        :rtype: List[int]
        """
        '''
        a powerful integer can be written as:
        x^i + y^j <= bound
        best idea would be to simulate, keep taking powers of i and j >=, but first we need to bound i and j
        the smallest value x and y can be when raised to i and j is zero
        then you have the case when x or y is 1
        '''
        #edge case
        if bound == 0:
            return []
        #find highest i and highest j, watch the case for 1
        highest_i = bound if x == 1 else int(round(log(bound-1)/log(x)))
        highest_j = bound if y == 1 else int(round(log(bound-1)/log(y)))
        
        
        res = set()
        for i in range(highest_i+1):
            for j in range(highest_j+1):
                cand = x**i + y**j
                if cand <= bound:
                    res.add(cand)
                #don't forget the break conditions
                if y == 1:
                    break
            if x == 1:
                break
        return res



















