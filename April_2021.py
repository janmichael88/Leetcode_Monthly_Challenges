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
#
#########################