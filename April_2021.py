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