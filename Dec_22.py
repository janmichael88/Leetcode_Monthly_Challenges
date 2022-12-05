####################################
# 1657. Determine if Two Strings Are Close (REVISTED)
# 02DEC22
####################################
#using counts and check counts of chars are the same
class Solution:
    def closeStrings(self, word1: str, word2: str) -> bool:
        '''
        strings are close if we can obtain 1 from the other using the following transformations
        1. swap any two existing chars
        2. transform every occurrence of one char into another existing char
        
        the first case allows me to generate any permutation of the word
        for the first part we can just check if word1 is a perm of word2 or word2 is a perm of word1
        
        characters must be the same
        and the frequncies of characters must be the same
        
        the implication of rule 2 is that if we have u counts of char i
        and v counts of char j
        if we were to swap char i to char j, then the counts must also swap
        and if word1 is a transormation of word2, the freqeusnt of counts should be the same
        '''
        if set(word1) == set(word2):
            #check counts
            counts1 = Counter(word1)
            counts2 = Counter(word2)
            if Counter(counts1.values()) == Counter(counts2.values()):
                return True
            else:
                return False
        else:
            return False


class Solution:
    def closeStrings(self, word1: str, word2: str) -> bool:
        '''
        use integer to store presnce of char in word
        and array to store counts
        '''
        if len(word1) != len(word2):
            return False
        
        counts1 = [0]*26
        counts2 = [0]*26
        
        seen_chars1 = 0
        seen_chars2 = 0
        
        #they must be the same size at this point
        for i,j in zip(word1,word2):
            first = ord(i) - ord('a')
            second = ord(j) - ord('a')
            
            counts1[first] += 1
            seen_chars1 = seen_chars1 | (1 << first)
            
            counts2[second] += 1
            seen_chars2 = seen_chars2 | (1 << second)
        
        
        if seen_chars1 != seen_chars2:
            return False
        
        counts1.sort()
        counts2.sort()
        
        for i in range(26):
            if counts1[i] != counts2[i]:
                return False
        return True

###################################
# 2396. Strictly Palindromic Number
# 02DEC22
###################################
#this shit fucking works?! LMAOOOO
class Solution:
    def isStrictlyPalindromic(self, n: int) -> bool:
        '''
        just get base represenstation of a number
        in log n time
        then check if it is a palindrom
        '''
        def getBaseRep(n,b):
            bits = []
            while n:
                bits.append(n % b)
                n //= b
            return bits
        
        for b in range(2,n-2+1):
            bits = getBaseRep(n,b)
            if bits != bits[::-1]:
                return False
        
        return True

class Solution:
    def isStrictlyPalindromic(self, n: int) -> bool:
        '''
        turns out we just return false

        Intuition
		The condition is extreme hard to satisfy, think about it...
		for every base b between 2 and n - 2...
		4 is not strictly palindromic number
		5 is not strictly palindromic number
		..
		then the bigger, the more impossible.
		Just return false


		Prove
		4 = 100 (base 2), so 4 is not strictly palindromic number
		for n > 4, consider the base n - 2.
		In base n - 1, n = 11.
		In base n - 2, n = 12, so n is not strictly palindromic number.

		There is no strictly palindromic number n where n >= 4


		More
		I think it may make some sense to ask if there a base b
		between 2 and n - 2 that n is palindromic,
		otherwise why it bothers to mention n - 2?

		It's n - 2, not n - 1,
		since for all n > 1,
		n is 11 in base n - 2.
		(Because n = (n - 1) + (1))

		Then it's at least a algorithme problem to solve,
		instead of a brain-teaser.

		Maybe Leetcode just gave a wrong description.


		Complexity
		Time O(1)
	Space O(1)

        '''
        return False

###############################################
# 451. Sort Characters By Frequency (REVISTED)
# 03NOV22
##############################################
class Solution:
    def frequencySort(self, s: str) -> str:
        '''
        we can use a variant of bucket sort
        first find the max frequency among all the char frequencies
        then we have buckets for each of the frequencies up to and including the max freq
        in each bucket we just put the letter
        then re-traverse the buckets and grab each letter that number of times
        
        '''
        #find max frequqency
        counts = Counter(s)
        max_freq = max(counts.values())
        
        buckets = [[] for _ in range(max_freq + 1)]
        
        #each bucket represents a frequency up to max freq, and for each freq add the letter into it
        for k,v in counts.items():
            buckets[v].append(k)
        
        #traverse the buckets and add to ans
        ans = ""
        for i in range(len(buckets)):
            for char in buckets[i]:
                ans += char*i
        
        return ans[::-1]

#############################################
# 1823. Find the Winner of the Circular Game
# 03DEC22
#############################################
#yassss! lets fucking goooooo!
class Node:
    def __init__(self,val,next=None):
        self.val = val
        self.next = next
        
class Solution:
    def findTheWinner(self, n: int, k: int) -> int:
        '''
        i could just simulate the game until there is only one player left
        i can mimic this with a linked list

        '''
        #first make the circular linked list
        dummy = Node(-1)
        curr = dummy
        for i in range(1,n+1):
            newNode = Node(i)
            curr.next = newNode
            curr = curr.next
        
        curr.next = dummy.next
        
        curr = dummy.next
        
        #now we can simulate
        lost = set()
        
        while len(lost) < n-1:
            prev = curr
            for _ in range(k-1):
                prev = curr
                curr = curr.next
            prev.next = curr.next
            lost.add(curr)
            curr = prev.next
            
        return curr.val

#with an array
class Solution:
    def findTheWinner(self, n: int, k: int) -> int:
        '''
        we can also simulate with an array
        '''
        
        circle = [i for i in range(1,n+1)]
        last_idx = 0
        
        while len(circle) > 1:
            #move index
            last_idx = (last_idx + k - 1) % len(circle)
            #remove
            del circle[last_idx]
        
        return circle[0]

class Solution:
    def findTheWinner(self, n: int, k: int) -> int:
        '''
        we can also use a q
        we shuffle array by moving the first element back to the end for each step k
        then once are out of steps k, we remove the first element
        '''
        q = deque([i for i in range(1,n+1)])
        
        while len(q) > 1:
            x = k
            print(q)
            while x > 1:
                r = q[0]
                q.popleft()
                q.append(r)
                x -= 1
            q.popleft()
        
        return q[0]

#dp
class Solution:
    def findTheWinner(self, n: int, k: int) -> int:
        '''
        this is the josephus problem
        to find an O(N) solution we first need to model the recurrent
        then we can build bottom up dp using constant space
        
        recurrence
        dp(n,k) = (dp(n-1,k) + (k-1)) % n +1
        
        After the first person (kth from the beginning) is killed, n-1 persons are left. Make recursive call for Josephus(n – 1, k) to get the position with n-1 persons. But the position returned by Josephus(n – 1, k) will consider the position starting from k%n + 1. So make adjustments to the position returned by Josephus(n – 1, k). 
        '''
        memo = {}
        def dp(n):
            if n == 1:
                return 1
            if (n) in memo:
                return memo[(n,k)]
            ans = (dp(n-1) + k - 1) % n + 1
            memo[(n)] = ans
            return ans
        
        return dp(n)


#constant space
class Solution:
    def findTheWinner(self, n: int, k: int) -> int:
        '''
        constance space
        '''
        prev = 1
        curr = 0
        for x in range(n+1):
            curr = (prev + k -1) % n + 1
            prev = curr
        
        return curr


##################################
# 2256. Minimum Average Difference
# 04DEC22
##################################
class Solution:
    def minimumAverageDifference(self, nums: List[int]) -> int:
        '''
        we need to calculate the average difference fore ach index i, and return the smallest one
        
        averae diferent for an index i is
            abs different between average of first i+1 elements and average of n-i-1 elements
        
        we can use prefix sum array to reduce over head needed then just return the minimum
        
        '''
        pref_sum = [0]
        for num in nums:
            pref_sum.append(pref_sum[-1] + num)
        
        #to get sum for nums between i and j, we want pref_sum[j+1] - pref_sum[i]
        
        avg_diff = float('inf')
        index = 0
        N = len(nums)
        
        for i in range(N):
            #need sum from i+1,N
            right_sum = pref_sum[N] - pref_sum[i+1]
            left_sum = pref_sum[i+1] - pref_sum[0]
            
            #calcualte average difference
            size_right = N-i-1
            size_left = i + 1
            
            #get avg diff
            right_avg_diff = right_sum // size_right if size_right != 0 else 0
            left_avg_diff = left_sum // size_left if size_left != 0 else 0
            
            local_avg_diff = abs(left_avg_diff - right_avg_diff)
            if local_avg_diff < avg_diff:
                avg_diff = local_avg_diff
                index = i
        
        return index

################################
# 942. DI String Match
# 05NOV22
################################
class Solution:
    def diStringMatch(self, s: str) -> List[int]:
        '''
        build the string smartly by taking numbers from lo and hi ends
        i guessed i must alternating taking lo and hi because if i take the lowest low first or the highest hi first
        im guaranteed to have another avaialble smaller of higher number
        
        problem is with logic in taking the last number
        to s, append the opposite of the last digit
        '''
        s += 'I' if s[-1] == 'D' else 'I'
        N = len(s)
        nums = [i for i in range(N)]
        lo = 0
        hi = len(nums) - 1
        
        ans = []
        
        for ch in s:
            if ch == 'I':
                ans.append(nums[lo])
                lo += 1
            else:
                ans.append(nums[hi])
                hi -= 1
        
        return ans