##################
#Distribute Candies
###################
class Solution(object):
    def distributeCandies(self, candyType):
        """
        :type candyType: List[int]
        :rtype: int
        """
        '''
        count the types of candies,
        then traverse the counts and just take the max when you can
        hashset can only be so large, we can keep adding candies until we are in the acorrect size
        then just return the size of the hashset
        '''
        N = len(candyType)
        numAllowed = N // 2
        
        num_types = set()
        
        for c in candyType:
            if len(num_types) >= numAllowed:
                break
            num_types.add(c)
        return len(num_types)
        

        #could also just
        return min(len(candType)//2, len(set(candyType)))

#####################
#Single Row Keyboard
#####################
class Solution(object):
    def calculateTime(self, keyboard, word):
        """
        :type keyboard: str
        :type word: str
        :rtype: int
        """
        '''
        mapp the keyboard then just add it up
        '''
        N = len(word)
        mapp = {k:v for k,v in zip(keyboard,range(26))}
        steps = mapp[word[0]]
        
        for i in range(1,N):
            steps += abs(mapp[word[i]] - mapp[word[i-1]])
        return steps


#######################
# Set Mismatch
####################
#close one!
class Solution(object):
    def findErrorNums(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        '''
        since they are all unique, the ith position should be the ith number
        it it is not, then that number would either have been duplicated or not inlcuded
    
        '''
        #wellp the may not be sorted
        nums.sort()
        #first find the doubled number
        counts = Counter()
        doubled = None
        for num in nums:
            counts[num] += 1
            if counts[num] == 2:
                doubled = num
            
        #then find the delted number
        deleted = None
        for i in range(len(nums)):
            if nums[i] != i + 1:
                deleted = i
        
        return [doubled,deleted+1]

#LETS GOOOOOO
class Solution(object):
    def findErrorNums(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        '''
        what if i add to a hashset 1 by 1
        if the size of the hashset does not increase, i know that was the doubled number
        i can do this i
        using hash avoid the need to sort
        '''
        #first pass, find the doubled number
        first_set = set()
        size = len(first_set)
        
        doubled = None
        for i in range(len(nums)):
            first_set.add(nums[i])
            if len(first_set) == size:
                doubled = nums[i]
            size = len(first_set)
        
        #second pass, i've already hashed everyyint in the first_set
        #go backwards for n
        N = len(nums)
        
        deleted = None
        while N >= 1:
            if N in first_set:
                first_set.remove(N)
            else:
                deleted = N
                break
            N -= 1
        
        return [doubled,deleted]

#brute force
class Solution(object):
    def findErrorNums(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        '''
        again lets go over some of the approaches
        bruteforce
        examine all elements 1 to N and count them up
        '''
        duplicated, missing = -1,-1
        for i in range(1,len(nums)):
            count = 0
            for j in range(0,len(nums)):
                if nums[j] == i:
                    count += 1
            if count == 2:
                duplicated = i
            elif count == 0:
                missing = i
            #to termiante the search stop if found
            if duplicated > 0 and missing > 0:
                break
        return [duplicated,missing]

#sorting
class Solution(object):
    def findErrorNums(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        '''
        using sorting
        we can check missing by looking if consecutive elements are the same
        '''
        duplicated,missing = -1,-1
        nums.sort()
        for i in range(1,len(nums)):
            if nums[i] == nums[i-1]:
                duplicated = nums[i]
            elif nums[i] > nums[i-1] + 1: #no longer consective increase by 1
                missing = nums[i-1] + 1
        if nums[len(nums)-1] != len(nums):
            missing = len(nums) #wtf is going on here
        return [duplicated,missing]

class Solution(object):
    def findErrorNums(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        '''
        using hash map
        and then counting
        '''
        duplicated,missing = -1,-1
        counts = Counter(nums)
        for i in range(1,len(nums)+1):
            if i in counts:
                if counts[i] == 2:
                    duplicated = i
            else:
                missing = i
        
        return [duplicated,missing]

class Solution(object):
    def findErrorNums(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        '''
        using count array instead of hashmap
        
        '''
        count_array = [0]*len(nums)
        duplicated,missing = -1,-1
        
        for i in range(len(nums)):
            count_array[nums[i]-1] += 1
        
        for i in range(len(count_array)):
            if count_array[i] == 0:
                missing = i + 1
            elif count_array[i] == 2:
                duplicated = i + 1
        return [duplicated,missing]

#constant space
class Solution(object):
    def findErrorNums(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        '''
        the big one! using constant space
        we can save space regarding eh duplicacy of an element or absence of by using he nums array it self
        we know that all the elelmetns in the nums array are postive, and lie in the range(1,N)
        so we can pick an ellment i from nums, then we can invert the elment at the index abs(i)
        by doing so, if one of the elemnts j occurs twice, when this number is encountered a second time, the element nums[abs(i)] wel be found to be negative!!!!
        doing inversions we can check if a number if already negative to find the duplicate number
        After the inversions have been done, if all the elements in numsnums are present correctly, the resultant numsnums array will have all the elements as negative now. But, if one of the numbers, jj is missing, the element at the j^{th}j 
th
  index will be positive. This can be used to determine the missing number.
        '''
        duplicated,missing = -1,-1
        for num in nums:
            if nums[abs(num)-1] < 0:
                duplicated = abs(num)
            else:
                nums[abs(num)-1] *= -1
        
        for i in range(len(nums)):
            if nums[i] > 0:
                missing = i + 1
        return [duplicated,missing]

#more mathy solutions
def findErrorNums(self, nums):
	'''
	[1,2,2,4]
	[1,2,3,4]

	to find duplicated, just sum(nums) - sum(set(nums))
	[1,2,2,4] - [1,2,4], that just leaves 2

	to find missing
	sum(1 to N + 1) - sum(set(nums))

	[1,2,3,4] - [1,2,4] = 3
	'''
    return [sum(nums) - sum(set(nums)), sum(range(1, len(nums)+1)) - sum(set(nums))]

class Solution(object):
    def findErrorNums(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        '''
        [1,2,2,4] #act array
        [1,2,3,4] #exp array
        
        sum(act) - sum(exp) = rep - mis
        prod(act) / prod(exp) = rep / mis
        
        rep = (prod(act) / pro(exp))*mis
        mis = sum(act) - sum(exp) - rep
        '''
        N = len(nums)
        sumAct = sum(nums)
        sumExp =  (N*(N+1)) / 2
        prodAct = reduce(mul,nums)
        prodExp = reduce(mul,range(1,N+1))
        
        mis = round((sumAct - sumExp) / ((prodAct / prodExp) - 1))
        rep = (sumAct - sumExp)+mis
        
        print [rep,mis]

####################
#Missing Number
#####################
class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        i could hash the nums and then check each number between 0 to N in hash
        if not that the missing number
        '''
        N = len(nums)
        
        hashed = set(nums)
        for i in range(N+1):
            if i not in hashed:
                return i

#Gauss Strikes again
class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        just get the expected sum
        and deduct from that the current sum
        why? because 1 is missing from the the sum 0 to N
        '''
        N = len(nums)
        expSum = N*(N+1) // 2
        currSum = sum(nums)
        
        return expSum - currSum

####################################
#Intersection of Two Linked Lists
####################################
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        '''
        just pass the contents one one list into a hash
        then check if any fothe nodes in b are in the hash, if so return the node
        '''
        seen = set()
        while headA:
            seen.add(headA)
            headA = headA.next
        
        while headB:
            if headB in seen:
                return headB
            headB = headB.next

#brute force
class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        '''
        lets just code out the brute force solution
        for every node in A travers B and check
        '''
        while headA is not None:
            dummy_B = headB
            while dummy_B is not None:
                if headA == dummy_B:
                    return headA
                dummy_B = dummy_B.next
            headA = headA.next
        
        return None

#O(1) space
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        '''
        if the lengths of headA and headB were the same, then we could just use tow pointers
        and check if the equaled each other, if they didn't adavance both
        varaiant 1:
            get the length of the lists
            set the start pointer for the longer list
            step the pointer though the list togets
        '''
        #get sizeA
        sizeA = 0
        tempA = headA
        while tempA:
            sizeA += 1
            tempA = tempA.next
        #get sizeB
        sizeB = 0 
        tempB = headB
        while tempB:
            sizeB += 1
            tempB = tempB.next
        
        #now move the longer pointer until it gets to shorter
        longer,startLong,startShort = sizeA,headA,headB
        if sizeA > sizeB:
            longer, startLong,startShort = sizeA, headA,headB
        else:
            longer, startLong,startShort = sizeB, headB,headA
        
        #advance longer by abs(dist)
        toAdvance = abs(sizeA-sizeB)
        while toAdvance > 0:
            startLong = startLong.next
            toAdvance -= 1
        
        #now compare
        while startLong:
            if startLong == startShort:
                return startLong
            startLong = startLong.next
            startShort = startShort.next
        return None


#0(1) not getting the lengths
class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        '''
        now using two pointers
        instead of getting the lengths of both lists
        start pointer both off at their respective heads
        the first node that gets to the end first gets sent back to the opposite list it was on
        same thing with the second node
        eventuall they start off at the same and it becomes the same thing as if they had the same length
        why this works? well it ensure both pointers travel the same number of nodes
        '''
        pA = headA
        pB = headB
        
        while pA != pB:
            if pA is None:
                pA = headB
            else:
                pA = pA.next
            if pB is None:
                pB = headA
            else:
                pB = pB.next
        
        return pB

###################################
# Average of Levels in Binary Tree
####################################
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """
        '''
        level order BFS
        once i finish a level, dump the average
        '''
		averages = []

		q = deque([root])

		while q:
		    #get the current size of the level
		    size = len(q)
		    sumLevel = 0
		    for i in range(size):
		        node = q.popleft()
		        sumLevel += node.val
		        if node.left:
		            q.append(node.left)
		        if node.right:
		            q.append(node.right)
		    levelAvg = float(sumLevel) / float(size)
		    averages.append(levelAvg)

		return averages
                
#recursive approach is ugly
class Solution(object):
    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """
        '''
        now how to do this recursively?
        i can use hashmap to store the node vals at each level, which can be done recusrisvely
        then traverse the hash map
        '''
        mapp = defaultdict(list)
        
        def dfs(node,level):
            if not node:
                return
            mapp[level].append(node.val)
            dfs(node.left,level+1)
            dfs(node.right,level+1)
            
        #invoke
        dfs(root,0)
        
        #i dont' wannt sort
        #it just makes sense passing the mapp once to get the max level
        max_level = 0
        for k in mapp.keys():
            max_level = max(max_level,k)
        
        averages = [0]*(max_level+1)
        for k,v in mapp.items():
            averages[k] = float(sum(v))/float(len(v))
        
        return averages

##########################
# Short Encoding of Words
##########################
#oh boy this one was tough, neeeded to use a Trie
class Solution(object):
    def minimumLengthEncoding(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        '''
        we can reduce the length of the encoding if the ith word is a subsequence of any of the i-n words
        the indices of the refernce string mark the start and ends of the words (sorta)
        start of by making a hash
        word : [sebequence of words with the word]
        '''
        mapp = defaultdict(set)
        N = len(words)
        #helper function for subsequence
        def is_sub(word,sub):
            M = len(word)
            N = len(sub)
            i,j = 0,0
            while i < M and j < N:
                if word[i] == sub[j]:
                    j += 1
                i += 1
            return j == N
        
        for i in range(N):
            for j in range(N):
                if i != j:
                    if is_sub(words[i],words[j]):
                        mapp[words[i]].add(words[j])
        #if there are no intersections, then the ecnoding is just each word sepearte by#
        if len(mapp) == 0:
            return 2*len(words)
        
        #now there exists some intersections with the words
        #i can do bfs and increment by the lenghts
        distance = 0
        for w in words:
            if w in mapp:
                
        print mapp

#O(\sum(w_{i}))
#where w_i is the length of word i
class Solution(object):
    def minimumLengthEncoding(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        '''
        im so fucking dumd, just check if a word is suffix of one of the words in it already
        if its a suffix, then that word does not need to be included in the length of the encoding
        '''
        
        setWords = set(words)
        for w in words:
            for i in range(1,len(w)):
                if w[i:] in setWords:
                    setWords.remove(w[i:])
        size = 0
        for w in setWords:
            size += len(w) + 1
        
        return size

#dfs using Trie
class Solution(object):
    def minimumLengthEncoding(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        '''
        https://leetcode.com/problems/short-encoding-of-words/discuss/1096077/Python-Trie
        first build a Trie structure with letters in reverse
        then dfs on the tree getting the total lengths of all root to leaf paths + 1
        this is summed up
        '''
        trie = {}
        for word in words:
            curr_node = trie
            for char in reversed(word):
                if char not in curr_node:
                    curr_node[char] = {}
                curr_node = curr_node[char]
        
        def dfs(trie,level):
            #keep dfsing down the path getting the sum
            size = 0
            if not trie:
                return level
            for child in trie.values():
                size += dfs(child, level+1)
            return size
        return dfs(trie,1)

#
class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode) #dict of TrieNodes, which is just nested dict

class Trie():
    #idk why we made a class for this, series of nested dicts would have been fine
    #fuck itttt
    def __init__(self):
        self.root = TrieNode()
        self.ends = []
    
    def insert(self,word):
        root = self.root
        for char in word:
            root = root.children[char]
        self.ends.append((root,len(word)+1))

class Solution(object):
    def minimumLengthEncoding(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        '''
        https://leetcode.com/problems/short-encoding-of-words/discuss/1095858/Python-Trie-solution-explained
        using a trie, but put them in reverse
        why? if we reverse words and put them in a trie, than for the words ["time", "me", "bell"]
        we get  emit,em,lleb, so if one words was suffix of another, it is now a prefx
        tries are normally done to store prefixes!
        
        in addition to this classical trie structure, we want to keep a list of ends so we can have quick access tot them (in normal tries, we'd have a special char indicating the end of the trie)
        
        '''
        trie = Trie()
        size = 0
        for word in set(words):
            trie.insert(word[::-1])
        
        for node,depth in trie.ends:
            if len(node.children) == 0:
                size += depth
        
        return size

#another Trie
class TrieNode():
    def __init__(self):
        self.children = {}

class Trie():
    def __init__(self):
        self.root = TrieNode()
        self.ends = [] #tuple(TrieNode(),length)
    def insert(self,word):
        root = self.root
        for char in word:
            if char not in root.children: #not at this level
                #make it
                root.children[char] = TrieNode()
            root = root.children[char]
        #we've now move through the root
        self.ends.append((root,len(word)+1))

class Solution(object):
    def minimumLengthEncoding(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        words = list(set(words))
        trie = Trie()
        for w in words:
            trie.insert(w[::-1])
        size = 0
        for node,depth in trie.ends:
            if len(node.children) == 0:
                size += depth
        return size

####################
#Design Hash Map
#####################
#approach 1
'''
inution:
    one of the most common implentatinos is to has the modulo opeartare where the base is the size of the hash
    common to use primes,2069
    in the case of a collision, we use a bucket to hold all values that have the same hash
    bucket can be modeld using an array or linked list

algo:
    *for a given key value, we first apply the hash function to geneate the hash key
    *which corresponds to the address in our main stroage, with this hash key we would find the bucket where the value should be stored
    *now that we have found the bucket, we check if the (key,value) exists
'''

#make bucket object
class Bucket:
    def __init__(self):
        self.bucket = []
    
    def get(self,key):
        for k,v in self.bucket:
            if k == key:
                return v
        return -1
    
    def update(self,key,value):
        #initaliit not found
        self.isFound = False
        for i,kv in enumerate(self.bucket):
            if key == kv[0]:
                self.bucket[i] = (key,value)
                self.isFound = True
                break
        if not self.isFound:
            self.bucket.append((key,value))
    def remove(self,key):
        for i,kv in enumerate(self.bucket):
            if key == kv[0]:
                del self.bucket[i]
        

class MyHashMap(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.space = 2069
        self.hash_table = [Bucket() for i in range(self.space)]

    def put(self, key, value):
        """
        value will always be non-negative.
        :type key: int
        :type value: int
        :rtype: None
        """
        hash_key = key % self.space
        self.hash_table[hash_key].update(key,value)
       
    def get(self, key):
        """
        Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
        :type key: int
        :rtype: int
        """
        hash_key = key % self.space
        return self.hash_table[hash_key].get(key)

    def remove(self, key):
        """
        Removes the mapping of the specified value key if this map contains a mapping for the key
        :type key: int
        :rtype: None
        """
        hash_key = key % self.space
        self.hash_table[hash_key].remove(key)

# Your MyHashMap object will be instantiated and called as such:
# obj = MyHashMap()
# obj.put(key,value)
# param_2 = obj.get(key)
# obj.remove(key)

#an addtional way using linked lists for chaning
#we can also just used a linked list of chaining
class ListNode:
    def __init__(self,key,val):
        self.item = (key,val)
        self.next = None


class MyHashMap(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        #this can vary, but typically use a prime
        #https://cs.stackexchange.com/questions/11029/why-is-it-best-to-use-a-prime-number-as-a-mod-in-a-hashing-function
        #explaining the need for using a prime
        #if the distribution of items to be hashed is uniformly distributed, the size of the hash does not matter so much
        self.size = 2069
        self.hash_table = [None]*self.size
        

    def put(self, key, value):
        """
        value will always be non-negative.
        :type key: int
        :type value: int
        :rtype: None
        """
        idx = key % self.size
        if self.hash_table[idx] == None:
            self.hash_table[idx] = ListNode(key,value)
        else:
            curr = self.hash_table[idx]
            while curr:
                #traverse this bucket
                if curr.item[0] == key:
                    curr.item = (key,value)
                    return
                if curr.next == None:
                    return
                curr = curr.next
            #should now be None if we have gone through the whole thing
            curr.next = ListNode(key,value) 

    def get(self, key):
        """
        Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
        :type key: int
        :rtype: int
        """
        idx = key % self.size
        curr = self.hash_table[idx]
        while curr:
            if curr.item[0] == key:
                return curr.item[1]
            else:
                curr = curr.next
        return -1 #because it now Nonea fter not finding the key in the linekd list
        

    def remove(self, key):
        """
        Removes the mapping of the specified value key if this map contains a mapping for the key
        :type key: int
        :rtype: None
        """
        idx = key % self.size
        curr = prev = self.hash_table[idx]
        if not curr:
            return
        if curr.item[0] == key:
            self.hash_table[idx] = curr.next
        else:
            while curr:
                if curr.item[0] == key:
                    prev.next = curr.next
                    return
                else:
                    curr = curr.next
                    prev = prev.next

# Your MyHashMap object will be instantiated and called as such:
# obj = MyHashMap()
# obj.put(key,value)
# param_2 = obj.get(key)
# obj.remove(key)

####################################
# Remove Palindromic Subsequences
####################################
class Solution(object):
    def removePalindromeSub(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        the key observation is that any seuwnce of one unique char is indeed a plaind drom
        because we only have two unique lettters we can always solve the porblme with at most two steps
        1. remove all as as s single 
        2. remove all b's a s single
        
        remmember this is a subsequence
        if there isn't a palindrome at first
        then we can always just remove the a's and then the b's
        '''
        if not s:
            return 0
        if s == s[::-1]:
            return 1
        return 2
        
class Solution(object):
    def removePalindromeSub(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        using two pointers to check palindrom
        '''
        def is_palindrome(s):
            lo, hi = 0,len(s) -1
            while lo < hi:
                if s[lo] != s[hi]:
                    return False
                lo += 1
                hi -= 1
            return True
        if not s:
            return 0
        if is_palindrome(s):
            return 1
        else:
            return 2
        
############################
# Strobogrammatic Number
############################
class Solution(object):
    def isStrobogrammatic(self, num):
        """
        :type num: str
        :rtype: bool
        """
        '''
        there are only 4 numbers ther when flipped give another number
        0,6,8,9
        if the set of the numbers contains any number out of this, immedialt return false
        '''
        notSB = ['2','3','4','5','7']
        mapp = {'1':'1','0':'0','6':'9','9':'6','8':'8'}
        num = list(num)
        set_num = set(num)
        for n in notSB:
            if n in set_num:
                return False
        #otherwise try and build it
        flipped = []
        for n in reversed(num):
            flipped.append(mapp[n])
        return "".join(flipped) == "".join(num)

class Solution(object):
    def isStrobogrammatic(self, num):
        """
        :type num: str
        :rtype: bool
        """
        '''
        we can treat this like a palindrom problem and use two pointers
        be sure to check the middle as well is stobragmmtic since it gets flipped
        '''
        mapp = {'1':'1','0':'0','6':'9','9':'6','8':'8'}
        
        left,right = 0,len(num)-1
        
        while left <= right:
            if num[left] not in mapp or num[right] not in mapp or mapp[num[left]] != num[right]:
                return False
            left += 1
            right -= 1
        return True

#######################
#Add One Row to Tree
######################
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def addOneRow(self, root, v, d):
        """
        :type root: TreeNode
        :type v: int
        :type d: int
        :rtype: TreeNode
        """
        #edge cae
        if d == 1:
            newNode = TreeNode(v)
            newNode.left = root
            return newNode
        
        def insert(val,node,depth,n):
            #val is the value to be inserted
            #node if the current node
            #depth is the current depth of the current node
            #n is the height at which to insert thew new nodes
            if not node:
                return
            if depth == n-1:
                temp = node.left
                node.left = TreeNode(val)
                node.left.left = temp
                temp = node.right
                node.right = TreeNode(val)
                node.right.right = temp
            else:
                insert(val,node.left,depth+1,n)
                insert(val,node.right,depth+1,n)
        insert(v,root,1,d)
        return root

#iterative stack
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Node():
    def __init__(self,node,depth):
        self.node = node
        self.depth = depth
        
class Solution(object):
    def addOneRow(self, root, v, d):
        """
        :type root: TreeNode
        :type v: int
        :type d: int
        :rtype: TreeNode
        """
        '''
        iterative solution with stack
        but we need a new strcture here, Node, to keep track of the depth of the current node along with its value
        algo:
            push Node on to stack
            for every popped node, check if  its depth == the one prior at which the new node needs to be inserted
            if yes, insert the new nodes appropriately
            if not, push both the left and right  child Node(val+dpeth)
        '''
        #edge case
        if  d == 1:
            temp = TreeNode(v)
            temp.left = root
            return temp
        stack  = [Node(root,1)]
        while stack:
            curr_node = stack.pop()
            if not curr_node.node:
                continue
            if curr_node.depth == d-1:
                temp = curr_node.node.left
                curr_node.node.left = TreeNode(v)
                curr_node.node.left.left = temp
                temp = curr_node.node.right
                curr_node.node.right = TreeNode(v)
                curr_node.node.right.right = temp
            else:
                stack.append(Node(curr_node.node.left, curr_node.depth+1))
                stack.append(Node(curr_node.node.right,curr_node.depth+1))
        return root

#BFS approach
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def addOneRow(self, root, v, d):
        """
        :type root: TreeNode
        :type v: int
        :type d: int
        :rtype: TreeNode
        """
        #using bfs
        if d == 1:
            temp = TreeNode(v)
            temp.left = root
            return temp
        q = deque([(root,1)])
        while q:
            size = len(q)
            for i in range(size):
                #if we get the depth one befroe we need to insert
                curr,level = q.popleft()
                if level == d-1:
                    old_left,old_right = curr.left,curr.right
                    curr.left,curr.right = TreeNode(v),TreeNode(v)
                    curr.left.left, curr.right.right = old_left,old_right
                else:
                    if curr.left:
                        q.append((curr.left,level+1))
                    if curr.right:
                        q.append((curr.right,level+1))
            if level == d-1:
                return root
        return None

#compact dfs
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def addOneRow(self, root, v, d):
        """
        :type root: TreeNode
        :type v: int
        :type d: int
        :rtype: TreeNode
        """
        def dfs(node,depth,d):
            if not node:
                return None
            elif d == 1:
                temp = TreeNode(v,left = node)
                return temp
            elif depth == d - 1:
                node.left = TreeNode(val=v,left=node.left)
                node.right = TreeNode(val=v,right = node.right)
                return node
            dfs(node.left,depth+1,d)
            dfs(node.right,depth+1,d)
            return node
        return dfs(root,1,d)

                    
#####################
#Integer to Roman
#####################
class Solution(object):
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        '''
        i can't believe this shit worked LOL
        just some quikc notes
        the set of all roman ints is constant [1,3999] inclusive
        for the mapp lets  inlcude the other numbers ending in 4 and 9
        we need to check if the largest numeral can go into the nnume
        i would need to start with the greatest roman numberal
        greedily use the largest roman numeral
        '''
        numbers = [1,4,5,9,10,40,50,90,100,400,500,900,1000]
        letters = ['I','IV','V','IX','X','XL','L','XC','C','CD', 'D','CM','M']
        
        output = ""
        start = len(numbers)-1
        while num > 0:
            while num // numbers[start] >= 1:
                output += letters[start]
                num -= numbers[start]
            start -= 1
        return output

class Solution(object):
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        '''
        just another way of doing this using a for loop in order
        order the roman chars and their int values in decreasing order
        then use the divmod function to see how much times it goes and how much is left
        '''
        values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        letters  =  ["M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"]
        
        output = ""
        for i in range(len(values)):
            if num <=0:
                break
            while values[i] <= num:
                num  -= values[i]
                output += letters[i]
            '''
         output = ""
        	for i in range(len(values)):
            if num <=0:
                break
            #could also just use divmod
            count,num = divmod(num,values[i])
            output += letters[i]*count
        
        return output
        '''
        
        return output

############################
# Coin Change
############################
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        '''
        if we are being mathy about this then the optimization problem becomes:
        min_x \sum_{i=0}^{n-1} x_{i} subject to the constraint \sum_{i=0}^{n-1} x_{i} \times c_{i} = S
        where S is the amount, x_{i} is the count of coing with denomination c_{i}
        a trivial solution would be to enumerate all subsets of coin frequenceices [x_{0}..x{n-1}]
        
        we can recusrively use back tracking for this problem
        to the count array [x_{0}...x{n-1}] where each elemnt can be in the range [0,S/c_{i}]
        another way of thinking about this would be to brute force the count array suject to the above constraints
        think about the case we had coins [1] and amount [10]
        we'd go all the way through generating 10 coints total
        then idx falls out of the index and we get to end of the for loop case
        minCost has beenn updated so we reurn it
        if at any point we didn't update, we'd return -1
        
        '''
        
        def calcChange(idx,coins,amount):
            #bottom case
            if amount == 0:
                return 0
            #bonndarycheck
            if idx < len(coins) and amount > 0:
                maxVal = amount // coins[idx] #max coins with this denomination
                minCost = 2**31
                for i in range(maxVal+1):
                    if amount >= i*coins[idx]:
                        result = calcChange(idx+1,coins,amount -i*coins[idx])
                        if result != -1:
                            minCost = min(minCost,result+i)
                if minCost == 2**31:
                    return -1
                else:
                    return minCost
            return -1
        
        return calcChange(0,coins,amount)
        
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        '''
        we can use top down recusrive for this problem
        we define a function F(S), which returns the number of coins neeed to make channge for amount S using coins[c_{0}...c_{n-1}]
        we can split the problem into sub problems
        lets say we new the problem F(S) and now we are adding a new coin S, the recurrence can be written as
        we can san F(S) = F(S-C) + 1,  less C plus using up that coin C
        what we don't konw if the denomination of the last coin C,
        we compute F(S-c_{i}) for each possible denom c_0...c_1...c_{n-1} and chose the minimum amog them
        #draw out the recursion tree
        the actual recurren ise:
         * F(S) = min_{i=0}^{i=n-1} F(S-c_{i}) + 1, subject to S - c_{i} >= 0
         * base, F(S) = 0, when S = 0, F(S) = -1, when n = 0
         and of course we can cache along the way
         
        '''
        #special case
        if amount < 1:
            return 0
        memo = [0]*amount
        def calcChange(coins,remaining):
            if remaining < 0:
                return -1 #we've used too many coins
            if remaining == 0:
                return 0 #just enough so we can't add anymore
            if memo[remaining-1] != 0:
                return memo[remaining-1]
            mini = 2**31
            for c in coins:
                result = calcChange(coins,remaining - c)
                if result >=0 and result < mini:
                    mini = 1 + result
            memo[remaining-1] = -1 if mini == 2**31 else mini
            return memo[remaining -1]
        return calcChange(coins,amount)


#another top down memo offset with -1
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        if amount < 1:
            return 0
        memo = [0]*(amount+1)
        def calcChange(coins,remaining):
            #the path is not valid
            if remaining < 0:
                return -1
            #nothing to contirbute
            if remaining == 0:
                return 0
            if memo[remaining] != 0:
                return  memo[remaining]
            
            minCount = amount+1
            for c in coins:
                count = calcChange(coins,remaining-c)
                if count == -1:
                    continue
                minCount = min(minCount,count+1)
            
            if minCount == amount+1:
                memo[remaining] = -1
            else:
                memo[remaining] = minCount
            return memo[remaining]
        return calcChange(coins,amount)

#bottom up dp araay, i still don't really get it
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        '''
        bottom up dp solution
        for the iterative solution we think in a bottom up manner
        before getting F(i)
        we have to compute all the minimum counts for amounts up to i 
        on each itetation of i of the algorithm F(i) is computd as min_{j=0...n-1} F_{i-c_{j}} + 1
        example:
        coins = [1,2,3]
        amount = 3
        F(3) = min(F(3-c1),F(3-c2),F(3-c1)) + 1
        = min(F(3-1),F(3-2),F(3-3)) + 1
        = min(F(2),F(1),F(0)) + 1
        = min(1,1,0) + 1
        '''
        #1d array holding the min coins needed for that amount
        max_coins = amount + 1
        dp = [2**31]*max_coins
        #zero amount uses no coins
        dp[0] = 0
        for i in range(1,amount+1):
            for j in range(len(coins)):
                #if i can use the jth coin for this ith amount
                if coins[j] <= i:
                    #use it up and take min
                    #at the ith amount or ith - jth coin amoint + plus the for the single jth coint
                    dp[i] = min(dp[i],dp[i-coins[j]]+1) 
        if dp[amount] == 2**31:
            return -1
        else:
            return dp[amount]

#bfs
class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        '''
        using bfs
        the trick now is how to keep the visited set
        we check if we have not seen this amount yet, if we haven't we flipped it to visite again
        '''
		if not amount:
		    return 0
		visited = [False]*(amount+1)
		visited[0] = True

		q = deque([(0,0)])

		while q:
		    curr_amount, coinsUsed = q.popleft()
		    #take coin
		    coinsUsed += 1
		    for c in coins:
		        #incremtn amount by the count
		        next_amount = curr_amount + c
		        if next_amount == amount:
		            return coinsUsed
		        #if we can still keep going
		        if next_amount < amount:
		            #check we havent seen this amount yet
		            if visited[next_amount] == False:
		                visited[next_amount] = True
		                q.append((next_amount,coinsUsed))
		return -1

##########################################################
# Check If a String Contains All Binary Codes of Size K
##########################################################
#probably not the best way but it works
class Solution(object):
    def hasAllCodes(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: bool
        """
        '''
        brute force would be to find all unqiue binary codes of length k in string s
        then pass the string again to see that all binary codes are in it
        try that way first
        '''
        #make hash of all binary codes of length k
        allK = set()
        for i in range(2**k):
            allK.add(bin(i)[2:].zfill(k))
        #traverse s with size k
        count = 0
        for i in range(len(s)-k+1):
            if s[i:i+k] in allK:
                allK.remove(s[i:i+k])
        return len(allK) == 0 

#not generating all binary codes, and checking on the fly
class Solution(object):
    def hasAllCodes(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: bool
        """
        '''
        instead of creating all binary codes of length k
        just check every substring of length k untilwe get all possible binaruy cods
        we can keep the size of the hash equal to 2**k, since there are onlty 2**k substrings
        this way we can add a varaint of early stopping
        '''
        need = 1 << k
        seen = set()
        for i in range(k,len(s)+1):
            temp = s[i-k:i]
            if temp not in seen:
                seen.add(temp)
                need -= 1
            if need == 0:
                return True
        return False

#rolling hash
class Solution(object):
    def hasAllCodes(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: bool
        """
        '''
        using a rolling hash
        we know the total number of binary codes of size k, is just 2**k
        what does that mean?
        we can use a hash! in fact rolling hash
        we map each binary code to a number in the range [0,2**k -1]
        binary number -> decimal is hash vlaue
        because we can direclty apply bitwiseops to binary numbrs,
        we don't need to convert to decimal explicitly
        we just keep getting the hash (which is just the DECIMAL equivlanet) for each slice of range 3
        
        For example, say s="11010110", and k=3, and we just finish calculating the hash of the first substring: "110" (hash is 4+2=6, or 110). Now we want to know the next hash, which is the hash of "101".
        We can start from the binary form of our hash, which is 110. First, we shift left, resulting 1100. We do not need the first digit, so it is a good idea to do 1100 & 111 = 100. The all-one 111 helps us to align the digits. Now we need to apply the lowest digit of "101", which is 1, to our hash, and by using |, we get 100 | last_digit = 100 | 1 = 101.
        
        we can write them together to get:
        new_hash = ((old_hash << 1) & all_one) | last_digit_of_new_hash
        '''
        need = 1 << k
        got = [False]*need #this marks the binary code if we have seen it or not
        all_one = need - 1
        hash_val = 0
        for i in range(len(s)):
            #calculat the hash
            substring = s[i-k+1:i+1]
            #get the hash
            hash_val = ((hash_val << 1) & all_one) | int(s[i])
            #have at least window size k
            if i >= k-1 and got[hash_val] == False:
                got[hash_val] = True
                need -= 1
                if need == 0:
                    return True
        return False
        
########################
# Binary Trees With Factors
#########################
class Solution(object):
    def numFactoredBinaryTrees(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        '''
        a factor binary tree
        is where the children of a node has product equal to the node,
        already we know it has to be >= len(arr)
        now the question then becomes, for each ith elemnet in the array, how many of the len(arr) -ith element pairwise products gives the ith element
        not necessary because i use up the pair
        nah that did not work
        but recall any element can be n number of times
        
        we can define a recusrive function
        let recur(num) by the the answer to the question, how many binary trees exists such that their root is num
        if we look at the left subtree and at the right subtree
        we turn the arr into hash, since each element is unique anayway and we check:
        * if num % candid == 0: it is divisible
        * if num//cand in array, then this can also be a chile
        we add recur(cand)*recur(num//cand to ans)
        
        eample
        [2,4]
        rec(2):
            +1
        rec(4):
            +1
            1*1
        +3
        we an invoke this function for each element in arr but with the cnadida elment removed
        '''
        array = set(arr)
        N = 10**9 + 1 
        memo = {}
        def rec(num):
            if num in memo:
                return memo[num]
            ans = 1
            for candidate in array:
                if num % candidate == 0 and num // candidate in array:
                    ans += rec(candidate)*rec(num//candidate)
            memo[num] = ans
            return memo[num]
        return sum(rec(num) for num in array)


class Solution(object):
    def numFactoredBinaryTrees(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        '''
        we can also use DP to solve this problem with the same logic
        example [2,4,8]
                1  2 5
        1   1+(1*1)  1 + (2*1) + (1*1)
        1    2         ????
        the transiiton function is just
        dp[i] = dp[num]*dp[num/cand]
        
        If you sort the array and build up subtrees, you know those subtrees could have been a subtree of an even larger root (given then the root of the subtree is factor). 
        If we are at element, call it C, and its factors are [b] and [b/a], and we have already found the numbers of subtrees for [b] and [a/b] we multiply them (why? think of the combinations formula). 
        if C can be made up [b] and [b/a] items, then the number of ways C can be made is just num ways at [b] times num ways at [b/a], so l[b]*l[a/b].
        
        '''
        mod = 10**9 + 7
        arr.sort()
        dp = defaultdict(int)
        for a in arr:
            ans = 1
            for b in arr:
                if b > a:
                    break
                ans += (dp[b]*dp[a//b])
            dp[a] = ans
        print dp
        return sum(dp.values()) % mod

#better way and more explicity
class Solution(object):
    def numFactoredBinaryTrees(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        '''
        we can also use DP to solve this problem with the same logic
        example [2,4,8]
                1  2 5
        1   1+(1*1)  1 + (2*1) + (1*1)
        1    2         ????
        the transiiton function is just
        dp[i] = dp[num]*dp[num/cand]
        
        If you sort the array and build up subtrees, you know those subtrees could have been a subtree of an even larger root (given then the root of the subtree is factor). 
        If we are at element, call it C, and its factors are [b] and [b/a], and we have already found the numbers of subtrees for [b] and [a/b] we multiply them (why? think of the combinations formula). 
        if C can be made up [b] and [b/a] items, then the number of ways C can be made is just num ways at [b] times num ways at [b/a], so l[b]*l[a/b].
        
        '''
        mod = 10**9 + 7
        arr.sort()
        N = len(arr)
        idxs = {x:i for i,x in enumerate(arr)} #element : idx
        dp = [1]*N #starting off you can have 1 subtree
        for i,x in enumerate(arr):
            for j in range(i):
                #check if factor
                if x %  arr[j] == 0:
                    complement = x / arr[j] #its pair
                    if complement in idxs:
                        dp[i] += dp[j]*dp[idxs[complement]]
                        dp[i] %= mod
        return sum(dp) % mod

################################
#Swapping Nodes in a Linked List
################################
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def swapNodes(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        '''
        dump the vals into an array
        swap k beg and k end
        recreated the linked list
        do it this way first
        '''
        arr = []
        curr = head
        while curr:
            arr.append(curr.val)
            curr = curr.next
        arr[-k],arr[k-1] = arr[k-1],arr[-k]
        dummy = ListNode()
        curr = dummy
        for num in arr:
            curr.next = ListNode(num)
            curr = curr.next
        return dummy.next

#three pass
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def swapNodes(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        '''
        lets go over all the soluionts from LC
        three pass apporach
        * get length
        * setfont note
        *set enode 
        *swap
        '''
        N = 0
        curr = head
        while curr:
            N += 1
            curr = curr.next
        front = head
        for i in range(1,k):
            front = front.next
        end = head
        for i in range(1,N-k+1):
            end = end.next
        front.val,end.val = end.val, front.val
        return head

#two pass
class Solution(object):
    def swapNodes(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        '''
        two pass
        on the first pass while getting the length, also sent the front node
        '''
        N = 0
        frontNode, endNode = head,head
        curr = head
        while curr:
            N += 1
            if N == k:
                frontNode = curr
            curr = curr.next
        for i in range(1,N-k+1):
            endNode = endNode.next
        
        frontNode.val,endNode.val = endNode.val, frontNode.val
        return head

#single pass
class Solution(object):
    def swapNodes(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        '''
        now can we do it in a single pass?
        we get stuck? how can we know the position of the end node without first geting the length of the linked list (it must have been passed at least once)
        TRICK:
            if endNode is k positions behind a certain node,currNode, when currNode reaches the end of the linked list, the endNode would be at the n-th node
            when currNode is null, endNode is n-k nodes away
        algo:
            start iterating from the head
            keep track of the number of nodes andincrmenet by one for each
            if lenght is k, we know we are the start of front node, we can no create an endneo be the start
            keep moving along advancing both pointers
            swap
            return
        '''
        N = 0
        front,end = None,None
        curr = head
        while curr:
            N += 1
            if end:
                end = end.next
            if N == k:
                front = curr
                end = head
            curr = curr.next
        front.val,end.val = end.val,front.val
        return head

############################
#Encode and Decode TinyURL
############################
#singel counter
class Codec:
    '''
    dumd way is just hash each string as unique
    '''
    def __init__(self):
        self.mapp = {}
        self.idx = 0
    def encode(self, longUrl):
        """Encodes a URL to a shortened URL.
        
        :type longUrl: str
        :rtype: str
        """
        self.mapp[self.idx] = longUrl
        temp = self.idx
        self.idx += 1
        return "http://tinyurl.com/"+str(temp)
        
    def decode(self, shortUrl):
        """Decodes a shortened URL to its original URL.
        
        :type shortUrl: str
        :rtype: str
        """
        size = len("http://tinyurl.com/")
        idx = shortUrl[size:]
        return self.mapp[int(idx)]
        
        

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.decode(codec.encode(url))

#variable length encoding, base62
'''
using variable length encoding
we make use of var leng enocding to enocde given URLS
for every longURL we chosose a variable codelnght for the inputn url, whcuh can be any lenght between 0 to 61
just base 61 then
'''
class Codec:
    def __init__(self):
        self.base = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.size = len(self.base)
        self.mapp = {}
        self.count = 1

    def encode(self, longUrl):
        """Encodes a URL to a shortened URL.
        
        :type longUrl: str
        :rtype: str
        """
        #we are hasing everything to the base chars
        #give refererne to the current count
        temp = self.count
        key = ""
        while temp > 0:
            temp -= 1
            key += self.base[temp % self.size]
            temp /= self.size
        self.mapp[key] = longUrl
        self.count += 1
        return "http://tinyurl.com/" + key
            

    def decode(self, shortUrl):
        """Decodes a shortened URL to its original URL.
        
        :type shortUrl: str
        :rtype: str
        """
        size = len("http://tinyurl.com/")
        idx = shortUrl[size:]
        return self.mapp[idx]

#just use the default hash function     

'''
or just use the builtin hash function
java hash function is:
s[0]∗31 
(n−1)
 +s[1]∗31 
(n−2)
 +...+s[n−1] , where s[i] is the ith character of the string, n is the length of the string.
 python may be differenct but its probaly similay, jsut ust hash
'''
class Codec:
    
    def __init__(self):
        self.mapp = {}
    def encode(self, longUrl):
        """Encodes a URL to a shortened URL.
        
        :type longUrl: str
        :rtype: str
        """
        self.mapp[str(hash(longUrl))] = longUrl
        return "http://tinyurl.com/"+str(hash(longUrl))
        

    def decode(self, shortUrl):
        """Decodes a shortened URL to its original URL.
        
        :type shortUrl: str
        :rtype: str
        """
        size = len("http://tinyurl.com/")
        idx = shortUrl[size:]
        return self.mapp[str(idx)]

'''
we can use a random interger to enocde
in case the genrated code happed to be arleady ampped to some previous longURL, we genreae a new random inter
'''
import random
class Codec:
    def __init__(self):
        self.mapp = {}
        self.key = random.randint(0,2**31)
    def encode(self, longUrl):
        """Encodes a URL to a shortened URL.
        
        :type longUrl: str
        :rtype: str
        """
        while self.key in self.mapp:
            self.key = random.randint(0,2**31)
        self.mapp[self.key] = longUrl
        return "http://tinyurl.com/"+str(self.key)

    def decode(self, shortUrl):
        """Decodes a shortened URL to its original URL.
        
        :type shortUrl: str
        :rtype: str
        """
        size = len("http://tinyurl.com/")
        idx = shortUrl[size:]
        return self.mapp[int(idx)]

#####################################
#Construct Binary Tree from String
#####################################
#FAIL............
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def str2tree(self, s):
        """
        :type s: str
        :rtype: TreeNode
        """
        '''
        i need to use a stack
        its always a left node first
        when i get to a closing
        parse that and make it a node
        and then push back on to the stack
        then make that perivousonce a node with left connections and push back on to stack
        then right
        '''
        stack = []
        for char in s:
            if char == ')':
                number = 0
                mult = 0
                while stack and stack[-1] != '(':
                    if stack[-1] == '-':
                        stack.pop()
                        number *= -1
                    else:
                        number += int(stack.pop())*(10**mult)
                        mult += 1
                stack.append(TreeNode(number))
            else:
                stack.append(char)
        print stack
        
#recursive solution
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def str2tree(self, s):
        """
        :type s: str
        :rtype: TreeNode
        """
        '''
        we can model this is a recursion where () is a recusrive call that returns a TreeNode
        we stop the recursion when we get to a closing bracket
        closing bracket will the most recent subtree
        algo:
            define function to get the number for val of the node
            while loop digit by digit trick
            take care of mins sign
            we define a new function, call it rec for mo
            it takes thes tring and the current index of the current chars as nputs and return a port of the TreNode rep of the current subtree an also the dnex of the next char to be processed in the string
            the idnex manip is importnant becasue we don't want to parse the string twice to figure out the boundaries for the children subtrees
            we can have 1 of three termination conditions
                1. when there are note moe characters to process
                2. Next, we get the value for the root node of this tree. This is an invariant here. We will never find any brackets before we get the value for the root node.
                3. Once we have the value, we form the root node.
                4. Then, we check for an opening bracket (make sure to check for the end of string conditions always). If there is one, we make a recursive call and use the node returned as the left child of the current node.
                5. Finally, we check if there's another opening bracket. If there is one, then it represents our right child and we again make a recursive call to construct that and make the right connection.
                6. another opening brakcte indicates the presnece of a right chidlren, and so we simply make another recursive call and attach the returned node to the right
                7. if another closing brracket, we are done with this node, we move one step forward and return from our current recursion
        NOTES: the helper recursive function returns two values, 1. the node representation 2, the index of the next char
         '''
        def getNumber(s,index):
            #negative check
            is_negative = False
            
            if s[index] == '-':
                is_negative = True
                index += 1
            number = 0
            while index < len(s) and s[index].isdigit():
                number = number*10+int(s[index])
                index += 1
            #returns the number for the node 
            #also the index of the first opening char immedialty following the current number
            return number if not is_negative else -number,index
        def rec(s,index):
            #base case
            if index == len(s):
                return None,index
            #start of the tree will alwasy be a number
            value,index = getNumber(s,index)
            node = TreeNode(value)
            #if there is any data left, we check for the left subtree first
            if index < len(s) and s[index] == '(':
                node.left,index = rec(s,index+1)
            #if there is a right
            if node.left and index < len(s) and s[index] == '(':
                node.right,index = rec(s,index+1)
            
            if index < len(s) and s[index] == ')':
                return node,index+1
            else:
                return node,index
        return rec(s,0)[0]
            
#iteratively with stack
class Solution(object):
    def str2tree(self, s):
        """
        :type s: str
        :rtype: TreeNode
        """
        '''
        iterative stack approach
        intution:
            it's pretty much the same as the recusrive solution, the only thing we need to take care of are the three difference states of a our recusrive function
            1. NOT STARTED: this is the initial state the node always starts in. just take the valye and amke the root
            2. LEFT DONE - weve made the root, and there is a left child to be made, use index to detect presence of right child
            3. RIGHT DONE - final state of any node
        algo:
        1. define the usual helper function to get the number
        2. use stack
        3. initailly push the root node on stack, with state NOT STARTED
        4. iterate through the string doing the following
            a. pop the node, this will be the global root
            b. if the curr char is '-':
                1. means we have not started processing yet, so make call to getNumber, and make the node
                2. then check for presnce of left child by checking if there are remaining chars to process and the one we have to process now is an opening bracket
                    - add the current node to stack, and mark as LEFT DONE
                    - assing node.left a new Tree Now (be sure to take a look at the diagrams)
                    - one we get to a closing bracket we conect the nodes
                3. if it is '('
                    -check for presence of a right child becasue we know that the value of the node is rleady set at this point
                    -we check if the curr char, is there is one left, is '('
                        * we add the curr node back to the queue assuming its state to be LEFT DONE
                        * then we assign node.right to a new TreeNode and also add it to the queue
                        

        '''
        def getNumber(s,index):
            #negative check
            is_negative = False
            
            if s[index] == '-':
                is_negative = True
                index += 1
            number = 0
            while index < len(s) and s[index].isdigit():
                number = number*10+int(s[index])
                index += 1
            #returns the number for the node 
            #also the index of the first opening char immedialty following the current number
            return number if not is_negative else -number,index
        
        if not s:
            return None
        #init the stack with an empty node
        root = TreeNode()
        stack = [root]
        #keeps track of index
        index = 0
        while index < len(s):
            node = stack.pop()
            
            #node startd yest
            if s[index].isdigit() or s[index] == '-':
                value,index = getNumber(s,index)
                node.val = value
                
                #now we need to check for any children left
                if index < len(s) and s[index] == '(':
                    stack.append(node)
                    #assign the current nodes left
                    node.left = TreeNode()
                    stack.append(node.left)
            #LEFT has been done
            elif node.left and s[index] == '(':
                stack.append(node)
                node.right = TreeNode()
                stack.append(node.right)
            
            index += 1
        
        return stack.pop() if stack else root

############################################################
#Best Time to Buy and Sell Stock with Transaction Fee
############################################################
#good recap article
#https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/discuss/108870/Most-consistent-ways-of-dealing-with-the-series-of-stock-problems
class Solution(object):
    def maxProfit(self, prices, fee):
        """
        :type prices: List[int]
        :type fee: int
        :rtype: int
        """
        '''
        think of the first stock problem, where you were allowed to take as many transactions
        well id only take if there was an increase, and is the fee put me above the previous profit
        '''
        N = len(prices)
        #create dp array storing profits
        cash = 0
        hold = -prices[0]
        #starting off we either have zero cash, or take the first 1 at day one
        for i in range(1,N):
            #cash is the amount taken if we arne't holding anything after day i
            cash = max(cash, hold + prices[i]-fee) #sell stock
            #we took no action
            hold = max(hold,cash-prices[i]) #buy stock
        return cash
            

##################################
#Generate Random Point in a Circle
###################
#rejection sampling
'''
The area of the square is (2R)^2 = 4R(2R) 
2
 =4R and the area of the circle is \pi R \approx 3.14RπR≈3.14R. \dfrac{3.14R}{4R} = \dfrac{3.14}{4} = .785 
4R
3.14R
​	
 = 
4
3.14
​	
 =.785. Therefore, we will get a usable sample approximately 78.5\%78.5% of the time and the expected number of times that we will need to sample until we get a usable sample is \dfrac{1}{.785} \approx 1.274 
.785
1
​	
 ≈1.274 times.
 '''
class Solution(object):
    '''
    get the x and y limits of the of the circle
    randomly generate 1 in that range
    use the equation to get the other one
    returnt he point
    '''
    def __init__(self, radius, x_center, y_center):
        """
        :type radius: float
        :type x_center: float
        :type y_center: float
        """
        self.radius = radius
        self.xc = x_center
        self.yc = y_center
    def randPoint(self):
        """
        :rtype: List[float]
        """
        #we can use the rand function and genetare a random point from the smallest x to the alrgest x
        #smae thing for the way
        x0 = self.xc - self.radius
        y0 = self.yc - self.radius
        
        while True:
            #make a new guess
            x1 = x0 + (2*self.radius)*random.random()
            y1 = y0 + (2*self.radius)*random.random()
            if (x1-self.xc)**2 + (y1-self.yc)**2 <= (self.radius)**2: #from the center
                return x1,y1

#inverse transform sampling
'''
we can paramtertize the circle in terms of r and theta
and randomnly get a value in the domina of r and in the domain of theta
'''
class Solution(object):

    def __init__(self, radius, x_center, y_center):
        """
        :type radius: float
        :type x_center: float
        :type y_center: float
        """
        self.r = radius
        self.xc = x_center
        self.yc = y_center

    def randPoint(self):
        """
        :rtype: List[float]
        """
        '''
        area = math.pi * self.r ** 2
		R = math.sqrt(random.uniform(0, area) / math.pi)
		Uniformly choose a point in the circle area, and get the distance between this point and the center.
        '''

        r0 = self.r*random.random()**.5 #inverse transform sampling
        theta0 = random.random()*2*math.pi
        return r0*math.cos(theta0)+self.xc,r0*math.sin(theta0)+self.yc

# Your Solution object will be instantiated and called as such:
# obj = Solution(radius, x_center, y_center)
# param_1 = obj.randPoint()


##########################
#Wiggle Subsequence
##########################
#close one
class Solution(object):
    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        a wiggle subsequence is a sequence where conseuctive differences alternatet between positive and negative
        you can only delete elements, NOT PICK AND CHOOSE
        
        can i just use a stack and only add an element if it less than top of the stack
        but keep alternating
        '''
        def sign(x):
            if x < 0:
                return -1
            else:
                return 1
        if len(nums) < 2:
            return len(nums)
        if len(set(nums)) == 1:
            return 1
        N = len(nums)
        stack = [nums[0],nums[1]]
        first_diff = stack[-1] -stack[-2]
        for i in range(2,N):
            curr_diff = nums[i] - stack[-1]
            #this must be in oppisite sign of the first_diff
            if sign(first_diff)*sign(curr_diff) == -1:
                stack.append(nums[i])
                first_diff = curr_diff
        
        return len(stack)

#brute foce recusrive
#n factorical
#each call in the resursino tree would be the idxth element
#then it branches into another tree for isup and !isup
class Solution(object):
    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        there are quite a few solutions to this problem so lets go through them and type them all out
        #BRUTE force recusion
        we can find the legnth of every possible subseqence and findd the max length of all of them
        we define a recursive function to help us out
        calc(nums,idx,isUp), isUp tells use where to find an increasing wiggle or decreasing wiggle
        if the function is called after an increasing wiggle, we need to fine the next decreasing subseqene
        some for the opposite case
        '''
        @lru_cache(None)
        def calc(nums,idx,isUp):
            maxlength = 0
            for i in range(idx+1,len(nums)):
                if (isUp == True and nums[i] > nums[idx]) or (isUp == False and nums[i] < nums[idx]):
                    #recurse maxlength 
                    maxlength = max(maxlength,1+calc(nums,i,not isUp))
            return maxlength
        
        if len(nums) < 2:
            return len(nums)
        
        return 1 + max(calc(nums,0,True),calc(nums,0,False))

#Dynamic programming N squared
class Solution(object):
    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        dp solution
        allocate two dp arrays, up and down
        whenever we pick an element that could be part of a rising sub or decreasing sub
        up[i] refers to the length of the longest wiggle subsequence obtatins so far considering the ith element as the last element of the wiggle subsequence and ending with a rising wiggle
        down[i] refers to the lenght of thelongest wiggle subsequence obatined so far considering the i'th element as the last element of the wiggle subsequence and ending with a falling wiggle
        up[i] will be updated every time we find a rising wiggle ending with the i^{th}i 
th
  element. 
  Now, to find up[i] we need to consider the maximum out of all the previous wiggle subsequences ending with a falling wiggle 
  i.e. down[j]down[j], for every j<i and nums[i]>nums[j]. Similarly, down[i]down[i] will be updated.


        '''
        if len(nums) < 2:
            return len(nums)
        N = len(nums)
        up = [0]*N
        down = [0]*N
        #start with element at index 1
        for i in range(1,N):
            #we are not looking at the itnervetal from j to i where any j < i
            for j in range(0,i):
                #increasing wiggle
                if nums[i] > nums[j]:
                    up[i] = max(up[i],down[j]+1)
                #decreasing wiggle
                elif nums[i] < nums[j]:
                    down[i] = max(down[i],up[j]+1) #add 1 indicates we have have found a longe subsequene so we include it
        
        return 1 + max(up[N-1],down[N-1])

#Dynamic programing linear time
class Solution(object):
    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        we can do even better doing linear time
        intuition:
        any element in the array can correspond to one of three states
            1. up, nums[i] > nums[i-1]
            2. down, nums[i] < nums[i-1]
            3. equal, nums[i] == nums[i-1]
        
        we update as:
            if up, case 1, the element before it must be a down
            sp up[i] = down[i-1] + 1, down[i] rmeains the same as down[i-1]
            if down, cas2, then elemnet before must wiggle up
            so down[i] = up[i-1] +1, up[i] reamins the same is up[i-1]
            if equal, case 3, then there is no chancge, so down[i] = down[i-1] and up[i] = up[i-1]
        
        at the end, we find the large of the lengths in the last entry
        '''
        N = len(nums)
        if N < 2:
            return N
        up = [0]*N
        down = [0]*N
        up[0],down[0] =1,1 #bottom case, seq oflength 1 is trivially wiggle length1
        for i in range(1,N):
            if nums[i] > nums[i-1]:
                up[i] = down[i-1] + 1
                down[i] = down[i-1]
            elif nums[i] < nums[i-1]:
                down[i] = up[i-1] +1
                up[i] = up[i-1]
            else:
                down[i] = down[i-1]
                up[i] = up[i-1]
        return max(up[N-1],down[N-1])

#of course we can remove the dp array and just do O(1) space
class Solution(object):
    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        O(1) space, just removing the dp arrays
        
        '''
        N = len(nums)
        if N < 2:
            return N
        up,down = 1,1
        for i in range(1,N):
            if nums[i] > nums[i-1]:
                up = down + 1
            elif nums[i] < nums[i-1]:
                down = up + 1
        return max(up,down)

#greedy
class Solution(object):
    def wiggleMaxLength(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        to be honest, i don't think anyone could have gotten to the O(1) space solution
        without first doingt he DP solution
        this one really comes from solving a subproblem and carrying it forward
        
        the greedy approach could have been seen, which is what i initally started first
        lets go over it 
        
        the problme is equivlaent to finding  the number of alternating max and min speaks in the array
        why? if we choose any other intermediate number to be a part of the currnet wiggle subsequence,the maximum length of the wiggle subsequence will always be less than or euqal to the one obtaine by choosing only the intermediate
        '''
        def sign(x):
            if x < 0:
                return -1
            else:
                return 1
            
        N = len(nums)
        if N < 2:
            return N
        
        first_diff = nums[1] - nums[0]
        count = 2 if first_diff != 0 else 1
        for i in range(2,N):
            curr_diff = nums[i]-nums[i-1]
            if sign(first_diff)*sign(curr_diff) == -1 and curr_diff != 0:
                count += 1
                first_diff = curr_diff
        return count

#################
#Keys and Rooms
#################
class Solution(object):
    def canVisitAllRooms(self, rooms):
        """
        :type rooms: List[List[int]]
        :rtype: bool
        """
        '''
        this is just bfs, visit all the rooms and see if we get to a rooms using the keys
        then add the visited 
        and return if have visited all the rooms
        '''
        N = len(rooms)
        visited = set()
        q = deque([0])
        
        while q:
            curr_room = q.popleft()
            visited.add(curr_room)
            for key in rooms[curr_room]:
                if key not in visited:
                    q.append(key)
        return len(visited) == N

#could also use dfs
class Solution(object):
    def canVisitAllRooms(self, rooms):
        """
        :type rooms: List[List[int]]
        :rtype: bool
        """
        visited = set()
        
        def dfs(room):
            visited.add(room)
            for r in rooms[room]:
                if r not in visited:
                    dfs(r)
                    
        dfs(0)
        return len(visited) == len(rooms)

##############################
# Design Underground System
#############################
class UndergroundSystem(object):
    '''
    this is more of a systems design question
    '''

    def __init__(self):
        self.custTimes = {} #(id:t,currStation)
        self.stationCheckouts = defaultdict(list) #(to,from:time)
        

    def checkIn(self, id, stationName, t):
        """
        :type id: int
        :type stationName: str
        :type t: int
        :rtype: None
        """
        self.custTimes[id] = (t,stationName)
        

    def checkOut(self, id, stationName, t):
        """
        :type id: int
        :type stationName: str
        :type t: int
        :rtype: None
        """
        #get current customer infro
        currTime,currStation = self.custTimes[id]
        #update hash of cusTime
        self.custTimes[id] = (t,stationName)
        #update checkoutt imes
        self.stationCheckouts[(currStation,stationName)].append(t-currTime)
        
        

    def getAverageTime(self, startStation, endStation):
        """
        :type startStation: str
        :type endStation: str
        :rtype: float
        """
        #total times
        total = self.stationCheckouts[(startStation,endStation)]
        return float(sum(total)) / float(len(total))
        


# Your UndergroundSystem object will be instantiated and called as such:
# obj = UndergroundSystem()
# obj.checkIn(id,stationName,t)
# obj.checkOut(id,stationName,t)
# param_3 = obj.getAverageTime(startStation,endStation)

#updating average on the fly
class UndergroundSystem(object):
    '''
    this is more of a systems design question
    saving on space, updating average on the fly
    '''

    def __init__(self):
        self.custTimes = {} #(id:t,currStation)
        self.stationCheckouts = {} #(to,from:time)
        

    def checkIn(self, id, stationName, t):
        """
        :type id: int
        :type stationName: str
        :type t: int
        :rtype: None
        """
        self.custTimes[id] = (t,stationName)
        

    def checkOut(self, id, stationName, t):
        """
        :type id: int
        :type stationName: str
        :type t: int
        :rtype: None
        """
        #get current customer infro
        currTime,currStation = self.custTimes[id]
        #update hash of cusTime
        self.custTimes[id] = (t,stationName)
        #update checkoutt imes
        timeStayed = t - currTime
        #instead of appending to list, just update the averages on the fly
        if (currStation,stationName) in self.stationCheckouts:
            size,currAvg =  self.stationCheckouts[(currStation,stationName)]
            total = currAvg*size + timeStayed
            size += 1
            currAvg = float(total) / float(size)
            self.stationCheckouts[(currStation,stationName)] = (size,currAvg)
        else:
            self.stationCheckouts[(currStation,stationName)] = (1,timeStayed)
        
        

    def getAverageTime(self, startStation, endStation):
        """
        :type startStation: str
        :type endStation: str
        :rtype: float
        """
        #total times
        return self.stationCheckouts[(startStation,endStation)][1]


####################
#Reordered Power of 2
####################
#132/135! dang it
class Solution(object):
    def reorderedPowerOf2(self, N):
        """
        :type N: int
        :rtype: bool
        """
        '''
        this is obvie a bitwise trick here
        if there weren't then we' need every permutation of N
        generate powers of two
        dumpe the size (length of) each int into a hash of
        don't do size, just hash int with digits
        
        for the N look if we have those digits at that key, return true
        '''
        mapp = defaultdict(set)
        for i in range(32):
            temp = str(2**i)
            for char in temp:
                mapp[2**i].add(char)
        temp = str(N)
        for k,v in mapp.items():
            #potential candidate
            if len(temp) == len(str(k)):
                #check set
                check = v
                for char in temp:
                    if char in check:
                        check.remove(char)
                if len(check) == 0:
                    return True
        
        return False

#comparing count objects
class Solution(object):
    def reorderedPowerOf2(self, N):
        """
        :type N: int
        :rtype: bool
        """
        '''
        this is obvie a bitwise trick here
        if there weren't then we' need every permutation of N
        generate powers of two
        dumpe the size (length of) each int into a hash of
        don't do size, just hash int with digits
        
        for the N look if we have those digits at that key, return true
        
        
       	well we could just generate count objects nad compre

       	note, if I didn't have access to count objects, i would need tod defince funcitno for out
        '''
        mapp = defaultdict(set)
        countN = Counter(str(N))
        for i in range(31):
            currCount = Counter(str(1 << i))
            if currCount == countN:
                return True
        return False

######################
#Vowel Spellchecker
######################
#fail, but not really, too many moveing pieces i think
class Solution(object):
    def spellchecker(self, wordlist, queries):
        """
        :type wordlist: List[str]
        :type queries: List[str]
        :rtype: List[str]
        """
        '''
        we need to check is query word in the word list, IN ORDER
        and immeadialety add the matched word from the wordList to and outpu
        the problem is we need to check fo an exact match first
        
        this problem sucks......
        '''
        output = []
        for q in queries:
            #check exact match first
            found = False
            for w in wordList:
                if found == True:
                    break
                if q == w:
                    output.append(w)
            #no exact match
            if found == False:
                #check caps for matching
                count = 0
                
class Solution(object):
    def spellchecker(self, wordlist, queries):
        """
        :type wordlist: List[str]
        :type queries: List[str]
        :rtype: List[str]
        """
        '''
        its ok i didn't get this one, but this like a design problem 
        we need to break the problem down into the cases
        there are 3, and for each we need to have a specific return value
        we also make use of a hash table
        cases
            1. Exact Match, hold a set of words to efficient scane whetehr query is eact
            2. Caps Match, we hold a hash table that converts the word from its lowercase version to the oriignal word with correct cap
            3. Vowel replacement, we hold a hash that converts the word from its lowercase versino with the vowles masked out to the originl word, * repalces vowel
        
        the rest of the algo rithm is just careful planning and reading
        its more Design based
        
        '''
        
        words_perfect = set(wordlist)
        words_cap = {}
        words_vow = {}
        
        #helper function, turns vowles in *
        def devowel(word):
            vowels = 'aeiou' #constant O(5)
            output = ""
            for char in word:
                if char in vowels:
                    output += "*"
                else:
                    output += char
            return output
        
        for word in wordlist:
            wordlow = word.lower()
            #if the key already exsits, it won't update
            #this is the wrench here, same as checking if not in dict add it again
            #whey set default? well recall
            #When the query matches a word up to capitlization, you should return the first such match in the wordlist.
            #we only update our hashes once!
            words_cap.setdefault(wordlow, word)
            words_vow.setdefault(devowel(wordlow), word)
        
        def solve(query):
            if query in words_perfect:
                return query

            queryL = query.lower()
            if queryL in words_cap:
                return words_cap[queryL]

            queryLV = devowel(queryL)
            if queryLV in words_vow:
                return words_vow[queryLV]
            return ""

        return map(solve, queries)

class Solution(object):
    def spellchecker(self, wordlist, queries):
        """
        :type wordlist: List[str]
        :type queries: List[str]
        :rtype: List[str]
        """
        '''
        key take aways,
        we mask the vowels for each word with *
        if query word is in the original, just returnt he words
        if query.lower in the caps dictionary, just return the first occurence of the matchs!, this is importnat
        same thing with the devowel
        we could have just traversed the word list backward or check if we have populated our hashes
        '''
        words_perfect = set(wordlist)
        words_lowered = {}
        words_devowel = {}
        
        
        #aux functions
        def helper(word):
            return "".join('*' if char in 'aeiou' else char for char in word.lower())
        
        def query(word):
            if word in words_perfect:
                return word
            if word.lower() in words_lowered:
                return words_lowered[word.lower()]
            if helper(word) in words_devowel:
                return words_devowel[helper(word)]
            return ""
        
        #hash words_lowered
        for w in wordlist:
            if w.lower() not in words_lowered:
                words_lowered[w.lower()] = w
        #devolwed
        for w in wordlist:
            if helper(w) not in words_devowel:
                words_devowel[helper(w)] = w
        #could have also for looped appended here        
        return map(query,queries)

##########################################
#Find Smallest Common Element in All Rows
##########################################
#https://leetcode.com/problems/find-smallest-common-element-in-all-rows/discuss/387204/Python-3-solutions-Binary-Search-Hashmap-and-Set
class Solution(object):
    def smallestCommonElement(self, mat):
        """
        :type mat: List[List[int]]
        :rtype: int
        """
        '''
        get the freq counts of each element
        then find the minimum of num whose counts are >= numw rows
        '''
        rows = len(mat)
        cols = len(mat[0])
        counts = {}
        for i in range(rows):
            for j in range(cols):
                if mat[i][j] in counts:
                    counts[mat[i][j]] += 1
                else:
                    counts[mat[i][j]] = 1
        answer = 10**4
        for k,v in counts.items():
            if v >= rows:
                answer = min(answer,k)
                
        return answer if answer != 10**4 else -1

#optimized with one less pass
class Solution:
    # 652 ms, hashmap
    def smallestCommonElement(self, mat: List[List[int]]) -> int:
        D = collections.defaultdict(int)
        for arr in mat:
            for num in arr:
                D[num] += 1
                if D[num] == len(mat):
                    return num
        return -1
        

#binary search
class Solution(object):
    def smallestCommonElement(self, mat):
        """
        :type mat: List[List[int]]
        :rtype: int
        """
        '''
        instead of counting, we can use binary search
        we can go through each element in the first row, and then us binary serach to check if that elements exists in other rows
        would m rows times n log m
        this is still slower thought
        algo:
            iterate through each element in the first row
            itnit found to be true
            for each row after the first row
            use binary search for the exsitence of that current element
            if it does not,set foun to false, exist the loop
        '''
        rows = len(mat)
        cols = len(mat[0])
        #helepr function for binary search, lets do this recurivley for fun
        def binarySearch(val,arr,start,end):
            if (end < start):
                return 0
            mid = start + (end-start)//2
            if arr[mid] == val: 
                return 1
            if (val < arr[mid]):
                return binarySearch(val, arr, start, mid-1)
            elif (val > arr[mid]):
                return binarySearch(val, arr, mid+1, end)

        for c in range(cols):
            candidate = mat[0][c]
            for r in range(1,rows):
                if binarySearch(candidate,mat[r],0,cols-1) == 0:
                    #the break into else is actuall quite cleve
                    break
            else:
                return candidate
        return -1

###########################
#3Sum With Multiplicity
###########################
#TLE
class Solution(object):
    def threeSumMulti(self, arr, target):
        """
        :type arr: List[int]
        :type target: int
        :rtype: int
        """
        '''
        well n cubed is clrealy out, looks like n suqared might be acceptable witht this case
        warm, code n cubed first
        '''
        N = len(arr)
        count = 0
        for i in range(N):
            for j in range(i+1,N):
                for k in range(j+1,N):
                    if arr[i] + arr[j] + arr[k] == target:
                        count += 1
        return count
        
#FML....
class Solution(object):
    def threeSumMulti(self, arr, target):
        """
        :type arr: List[int]
        :type target: int
        :rtype: int
        """
        '''
        well n cubed is clrealy out, looks like n suqared might be acceptable witht this case
        warm, code n cubed first
        i can degenerate this problem by making it a two sum problem if i find a sum = target - arr[i]
        '''
        N = len(arr)
        count = 0
        for i in range(N):
            first = arr[i]
            twoSum = target - first
            mapp = {}
            #because there could be multiple multiple copes of one of the elements in the tuple
            positions = defaultdict(list)
            for j in range(i+1,N):
                mapp[arr[j]] = twoSum - arr[j]
                positions[arr[j]].append(j) #this should be in increasing order now
            
            for k,v in mapp.items():
                if v in mapp:
                    #a possible match
                    if first + k + v == target:
                        count += len(positions[v])
                        del positions[arr[v]]
        return count
                   
#well it was good to review two sum
class Solution(object):
    def threeSumMulti(self, arr, target):
        """
        :type arr: List[int]
        :type target: int
        :rtype: int
        """
        '''
        recall the two pointer method to solve the Two Sum problem
        givne a sorted array have left and right points, whenever the elementes at those pointes == target, its one and move
        if its short increase the lowest one and check again
        if its large decrease it
        this is the intuion use to solve the probelm in N^2 time
        algo
            * sort the array, indicies don't matter so long as  they are distinct elements
            * for each element i, we can define T = target - array[i]
            * then use the two pointer tech
            * however since elements can be duplicated
            * so we need to keep count of candidate frequencies
            * example, he target is say, 8, and we have a remaining array (A[i+1:]) of [2,2,2,2,3,3,4,4,4,5,5,5,6,6]
            * Whenever A[j] + A[k] == T, we should count the multiplicity of A[j] and A[k]. In this example, if A[j] == 2 and A[k] == 6, the multiplicities are 4 and 2, and the total number of pairs is 4 * 2 = 8
            * then move on to the array [3,3,4,4,4,5,5,5]
            *special case if A[j] == A[k]
            * if we are at the array [4,4,4], there are only three such pairs
            * in genearal array of length M has M*(M-1) // distinct pairs
        '''
        mod = 10**9 + 7
        count = 0 
        N = len(arr)
        arr.sort()
        
        for i in range(N):
            #apply two sum with points j and k
            twoSum = target - arr[i]
            j = i + 1
            k = N -1
            while j < k:
                if arr[j] + arr[k] < twoSum:
                    j += 1
                elif arr[j] + arr[k] > twoSum:
                    k -= 1
                #at some point these two will equal twoSum, and case 1 are differnt
                elif arr[j] != arr[k]:
                    #cound the number of times weve moved left and riight
                    left = 1
                    right = 1
                    while j + 1 < k and arr[j] == arr[j+1]:
                        left += 1
                        j += 1
                    while k - 1 > j and arr[k] == arr[k-1]:
                        right += 1
                        k -= 1
                    
                    #how many pairs have we contirbuted
                    count += left*right #the multiplicities
                    count %= mod
                    #from two sum
                    j += 1
                    k -= 1
                else:
                    #htey are equal and all the same, case 2
                    M = (k-j+1)
                    count += M*(M-1) / 2
                    count %= mod
                    break
        return count

class Solution(object):
    def threeSumMulti(self, arr, target):
        """
        :type arr: List[int]
        :type target: int
        :rtype: int
        """
        '''
        counting with cases
        we can define count[x] be the number of timex x occurs in A
        then for every x+y+z == target, can try to count the correct amount
        there are a few cases
        case
            1. if x,y,and z are all different, then the constribution is count[x]*count[y]*count[z]
            2. if x == y != z, ie two are the same, then the contribution is (n choose k)
            (count[x]_C_2)*count[z]
            3. if x != y == z, then the contribution from these terms is count[x]*(count[y]_C_2)
            4. if x == y == z, the contribution is count[x] C 3
            n! / (n-k)! k!
        '''
        mod = 10**9 + 7
        counts = [0]*101 #only possible numbers in the array
        for x in arr:
            counts[x] += 1
        
        result = 0
        
        #case 1 all differer
        for x in range(101):
            for y in range(x+1,101):
                z = target - x - y
                #constatring
                if y < z <= 100:
                    result += counts[x]*counts[y]*counts[z]
                    result %= mod
        
        #case 2, x and y are teh same, so just 2x
        for x in range(101):
            z = target - 2*x
            if x < x <= 100:
                result += (counts[x]*(counts[x]-1)/2)*counts[z]
                result %= mod
        
        #case 3,x==y, same as case2
        for x in range(101):
            if (target -x) % 2 == 0:
                y = (target -x) / 2
                if x < y <= 100:
                    result += counts[x]*(counts[y]*(counts[y]-1))/2
                    result %= mod
        
        #case 4,x==y==x
        if target % 3 == 0:
            x = target /3
            if 0 <= x <= 100:
                result += (counts[x]*(counts[x]-2)*(counts[x]-2)) / 6
                results %= mod
        
        return result

class Solution(object):
    def threeSumMulti(self, arr, target):
        """
        :type arr: List[int]
        :type target: int
        :rtype: int
        """
        '''
        this can also be naturally extended from Three sum
        again let count[x] be the number of times x occurs in theaaray
        For example, if A = [1,1,2,2,3,3,4,4,5,5] and target = 8, then keys = [1,2,3,4,5]. When doing 3Sum on keys (with i <= j <= k), 
        we will encounter some tuples that sum to the target, like (x,y,z) = (1,2,5), (1,3,4), (2,2,4), (2,3,3)
        '''
        MOD = 10**9 + 7
        count = collections.Counter(arr)
        keys = sorted(count)

        ans = 0

        # Now, let's do a 3sum on "keys", for i <= j <= k.
        # We will use count to add the correct contribution to ans.
        for i, x in enumerate(keys):
            T = target - x
            j, k = i, len(keys) - 1
            while j <= k:
                y, z = keys[j], keys[k]
                if y + z < T:
                    j += 1
                elif y + z > T:
                    k -= 1
                else: # x+y+z == T, now calculate the size of the contribution
                    if i < j < k:
                        ans += count[x] * count[y] * count[z]
                    elif i == j < k:
                        ans += count[x] * (count[x] - 1) / 2 * count[z]
                    elif i < j == k:
                        ans += count[x] * count[y] * (count[y] - 1) / 2
                    else:  # i == j == k
                        ans += count[x] * (count[x] - 1) * (count[x] - 2) / 6

                    j += 1
                    k -= 1

        return ans % MOD

######################
#Advantage Shuffle
######################
class Solution(object):
    def advantageCount(self, A, B):
        """
        :type A: List[int]
        :type B: List[int]
        :rtype: List[int]
        """
        '''
        for a in A, if a > b, we should pair it, otherwise a is uselss for our score, as it can't beat any cards
        if every card in A is larger than every card in B, it doesn't matter what the order ir
        we might as well use the weakest card to pair with b, and whatever is left just pair with the remaning cards
        algo:
            *sort A and B both into new arrays
            *for every elemnt in a compare to B
            *we can maintain a hash of potential pairings to a card b
            *if a cannot beat b, add to another list for remaining
        '''
        A_sort = sorted(A)
        B_sort = sorted(B)
        
        matchesB = {b:[] for b in B}
        remaining = [] #there could be no beating card for card b
        i = 0
        for a in A_sort:
            if a > B_sort[i]:
                matchesB[B_sort[i]].append(a)
                i += 1
            else:
                remaining.append(a)
        output = []
        for b in B:
            if len(matchesB[b]) > 0:
                output.append(matchesB[b].pop())
            else:
                output.append(remaining.pop())
        return output

#two pointer without hashmap
class Solution(object):
    def advantageCount(self, A, B):
        """
        :type A: List[int]
        :type B: List[int]
        :rtype: List[int]
        """
        '''
        another way would be to sort both A and B
        when sorting B, pair element with index value
        start pairing the greatest element in B with the greates element in A
        if A can't beat the current card b, use the smallest one
        
        '''
        N = len(A)
        output = [-1]*N
        A.sort()
        B = sorted([(v,i) for i,v in enumerate(B)])
        
        #now we need to pointers in A, one always pointing to the smallest, one always pointing to the lrgest
        #then we can start pairing
        l,r = 0, N-1
        #go backwards starting from B
        for i in range(N-1,-1,-1):
            #the greater element from A
            if A[r] > B[i][0]:
                output[B[i][1]] = A[r]
                r -= 1
            else:
                output[B[i][1]] = A[l]
                l += 1
        return output

#############################
#Pacific Atlantic Water Flow
############################
#well you almost had it...
class Solution(object):
    def pacificAtlantic(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """
        '''
        we want to find the cooridantes where water can flow,
        so we can say, from an (i,j) point is there a decreasing path to the pacific and atlantic
        so we want to dfs on each point and see if we can get to both the pacific and atlantic
        if we can, then that's a valid point
        so we need two dfs functions one for pacific and one for atlantic
        traverse the matrix and invoke, if both dfs' make it to the respetive sides, its valid point
        '''
        rows = len(matrix)
        cols = len(matrix[0])
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        results = []
        #every time we run dfs, make a new visited set
        def dfs_pacific(i,j):
            if i == -1 or j == -1:
                return True
            if i == rows or j == cols : #got to the atlantic side
                return
            visited_pacific.add((i,j))
            for dx,dy in dirrs:
                next_x,next_y = i + dx, j + dy
                if 0 <= next_x < rows and 0 <= next_y < cols and  (next_x,next_y) not in visited_pacific and matrix[next_x][next_y] <= matrix[i][j]:
                    dfs_pacific(next_x,next_y)
                elif next_x == -1 or next_y == -1:
                    return True
        visited_pacific = set()          
        for i in range(rows):
            for j in range(cols):
                if dfs_pacific(i,j) == True:
                    results.append((i,j))
        print results


class Solution(object):
    def pacificAtlantic(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """
        '''
        BFS, can flip the problem: we start traversing from each ocean and access neighboring node that is only higher
        we complete two BFS's from the pacific and the atlanatic
        algo:
            if empty return emptya rray
            queup, starting from pac and atlantic
            only starting from the shorts
            bfs on each
            find the intersection of points that are in both ques
        '''
        if not matrix or not matrix[0]:
            return []
        rows = len(matrix)
        cols = len(matrix[0])
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        #set up queues
        pac_q = deque()
        atl_q = deque()
        
        #first element of rows
        for i in range(rows):
            pac_q.append((i,0))
            atl_q.append((i,cols-1))
        #first elemetn of cols
        for i in range(cols):
            pac_q.append((0,i))
            atl_q.append((rows-1,i))
        
        def bfs(q):
            can_reach = set()
            while q:
                row,col = q.popleft()
                can_reach.add((row,col))
                for dx,dy in dirrs:
                    new_row,new_col = row + dx, col + dy
                    #check in constraints
                    if 0 <= new_row < rows and 0 <= new_col < cols and (new_row,new_col) not in can_reach and matrix[new_row][new_col] >= matrix[row][col]:
                        q.append((new_row,new_col))
            return can_reach
        
        #bfs from both sides
        pac_side = bfs(pac_q)
        atl_side = bfs(atl_q)
        
        return list(pac_side.intersection(atl_side))

#DFS
class Solution(object):
    def pacificAtlantic(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """
        '''
        DFS, same thing as BFS, but we would invoke on each othe nodes on the pac side and each of the nodes in the atl side
        
        '''
        if not matrix or not matrix[0]:
            return []
        rows = len(matrix)
        cols = len(matrix[0])
        dirrs = [(1,0),(-1,0),(0,1),(0,-1)]
        
        #set up queues
        pac_reach = set()
        atl_reach = set()
        
        def dfs(row,col,reached):
            reached.add((row,col))
            for dx,dy in dirrs:
                new_row,new_col = row + dx, col + dy
                #check in constraints
                if 0 <= new_row < rows and 0 <= new_col < cols and (new_row,new_col) not in reached and matrix[new_row][new_col] >= matrix[row][col]:
                    dfs(new_row,new_col,reached)
        
        #dfs invocations,first element of rows
        for i in range(rows):
            dfs(i,0,pac_reach)
            dfs(i,cols-1,atl_reach)
        #first elemetn of cols
        for i in range(cols):
            dfs(0,i,pac_reach)
            dfs(rows-1,i,atl_reach)
        return list(pac_reach.intersection(atl_reach))

#########################
#Word Subsets
#########################
#fuck this problem again...
#almost
class Solution(object):
    def wordSubsets(self, A, B):
        """
        :type A: List[str]
        :type B: List[str]
        :rtype: List[str]
        """
        '''
        if b is a subet of a, then a is a superset of b
        the ath word in A is the count of the number of chars in the ath words
        when we check the ath word in a is a superset of the bth word in B we are asking char in ath word >= char in both word
        now if we check wheter a word in a ia a superset of wordB_[I]
        we weill check for eah letter and each i such that N_{letter}(wordA) >= N_letter (wordB_{i})
        this is the KEY!!!! make sure to fucking remember this!
        For example, when checking whether "warrior" is a superset of words B = ["wrr", "wa", "or"], we can combine these words in B to form a "maximum" word "arrow", that has the maximum count of every letter in each word in B.
        which in python can be done with |=
        reduce B the to the largest superset (i.e the set containing all b in B)
        thenc check if we can make the a'th word in a from this superset
        '''
        #construct superset B
        supersetB = {}
        for i in range(ord('a'),ord('z')+1):
            supersetB[chr(i)] = 0
        for b in B:
            countB = Counter(b)
            for char,count in countB.items():
                supersetB[char] = max(supersetB[char],count)
        print supersetB
        
        #now see if i can make a using the supersetB
        universals = []
        for a in A:
            countA = Counter(a)
            constraint = 0
            for char,count in countA.items():
                if supersetB[char] >= count:
                    constraint += 1
            if constraint > len(B):
                universals.append(a)
        return universals
        

class Solution(object):
    def wordSubsets(self, A, B):
        """
        :type A: List[str]
        :type B: List[str]
        :rtype: List[str]
        """
        def count(word):
            ans = [0] * 26
            for letter in word:
                ans[ord(letter) - ord('a')] += 1
            return ans

        bmax = [0] * 26
        for b in B:
            for i, c in enumerate(count(b)):
                bmax[i] = max(bmax[i], c)

        ans = []
        for a in A:
            if all(x >= y for x, y in zip(count(a), bmax)):
                ans.append(a)
        return ans

class Solution(object):
    def wordSubsets(self, A, B):
        """
        :type A: List[str]
        :type B: List[str]
        :rtype: List[str]
        """
        '''
        if b is a subet of a, then a is a superset of b
        the ath word in A is the count of the number of chars in the ath words
        when we check the ath word in a is a superset of the bth word in B we are asking char in ath word >= char in both word
        now if we check wheter a word in a ia a superset of wordB_[I]
        we weill check for eah letter and each i such that N_{letter}(wordA) >= N_letter (wordB_{i})
        this is the KEY!!!! make sure to fucking remember this!
        For example, when checking whether "warrior" is a superset of words B = ["wrr", "wa", "or"], we can combine these words in B to form a "maximum" word "arrow", that has the maximum count of every letter in each word in B.
        which in python can be done with |=
        reduce B the to the largest superset (i.e the set containing all b in B)
        thenc check if we can make the a'th word in a from this superset
        '''
        #cheeky way using |=
        count = Counter()
        for b in B:
            count |= Counter(b)
            
        results = []
        for a in A:
            if not count - Counter(a):
                results.append(a)
        return results
        
class Solution(object):
    def wordSubsets(self, A, B):
        """
        :type A: List[str]
        :type B: List[str]
        :rtype: List[str]
        """
        '''
        making superset of B that includes all multiplicites of each b in B
        instead of making entries for all a to z, check if in hash and update on the fly
        '''
        superset_B = {}
        for b in B:
            temp = Counter(b)
            for char,count in temp.items():
                if char not in superset_B:
                    superset_B[char] = count
                else:
                    superset_B[char] = max(supserset_B[char],count)
                    
        output = []
        for a in A:
            temp = Counter(a)
            if all([char in temp and temp[char] >= count for char,count in superset_B.items() ]):
                output.append(a)
        return output

##########################
#Palindromic Substrings
###########################
#brute force TLE
class Solution(object):
    def countSubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        brute force
        check all starts and ends for palindromes
        '''
        def is_pal(string):
            N = len(string)
            left,right = 0,N-1
            while left <= right:
                if string[left] != string[right]:
                    return False
                left += 1
                right -= 1
            return True
        
        ans = 0
        for i in range(len(s)):
            for j in range(i+1,len(s)+1):
                if is_pal(s[i:j]) == True:
                    ans += 1
        return ans
            
class Solution(object):
    def countSubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        dynamic programming, if we know a string is a palindrom, then its inner parts must also be a palindrome
        axbobxa is one
        and so is xbobx
        and so is bob
        and so is o
        each can be a sub problem, and checking one subproblem solves the larger one, in fact checking them all
        intuition:
            while checking all substrings of a large string for palindromciity, we check smallers one repeatedly
            if we store the result of processing those smaller substrings, we can resuse them
            example:
            for the string "axbobxa", the substring "bob" needs to checked f
            or the substring "xbobx" and the string "axbobxa". 
            In fact, to check all three of these strings, the single character string "o" needs to be checked.
        
        algo:
            1. defines dp state that gets update
            2. dp(i,j) tells us where the string from i to j is a palindrome
            3. thus the answer to our problem lies in couting all subtrings whose state is True
            4. the base cases:
                a. sinlge letter dp(i,i) = Trye
                b. double leter of the smae char, dp(i,i+1) True if s[i] == s[i+1] false if other wise
            5. identify the optimal substructure: a string is a palindrom if:
                a. its first and last are equal
                b. and the rest inside is also apalindrom
            6.The optimal substructure can be formualted inot a recurence:
                a. dp(i,j) = Trye if all dp(i+1,j-1) where s[i]=s[j] else false
            7. idenitfy all overlapping sub problems only once. the optimal substructue ensures that the state for a string only depends on the state for a single substring. if we copute and save the sates for all smaller substrings first, larger strings can be procesed by reusing saved states. use the bases states
            8. the answer is found by counting all state that evaluate to True. since each state tells whether a unique substring is a palindrome or not
        '''
        N = len(s)
        ans = 0
        
        if N <= 0:
            return 0
        
        dp = [[0]*N for _ in range(N)]
        
        #base case, single letter substrings
        for i in range(N):
            dp[i][i] = 1
        #base case, double of same char
        for i in range(N-1):
            if s[i] == s[i+1]:
                dp[i][i+1] = 1
            #start increamineting answer
            if dp[i][i+1] == 1:
                ans += 1
            else:
                ans += 0
        #for all other caes, substrings of length 3 to N
        for size in range(3,N+1):
            for i in range(0,N-size+1):
                dp[i][i+size-1] = dp[i+1][i+size-1-1] and s[i] == s[i+size-1]
                if dp[i][i+size-1] == 1:
                    ans += 1
                else: 
                    ans += 0
                print s[i:i+size],s[i],s[i+size-1]
        return ans

#two pointer instead of dp
class Solution(object):
    def countSubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        '''
        examine each positions in thes string and expand from the center
        but we also need to do this in this case the the len(s) parity is evern
        #could have also wrapped this into a function
        for an odd length palindrom, every chary position is a center
        for an even length palindrome, every pair is a center
        '''
        output = 0
        N = len(s)
        for start in range(N):
            left,right = start,start
            while 0 <= left < N and 0 <= right < N and s[left] == s[right]:
                output += 1
                left -=1
                right += 1
            #start ands start + 1
            left,right = start,start+1
            while 0 <= left < N and 0 <= right < N and s[left] == s[right]:
                output += 1
                left -=1
                right += 1
        return output
                

#better dp way
class Solution(object):
    def countSubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        #https://leetcode.com/problems/palindromic-substrings/discuss/449719/Python-DP-Solution-with-Explanation
        '''
        define dp(i,j) as a boolean if s[i:j+1] is a palindrome
        then we can model the recurrence
        dp(i,j) = dp(i+1,j-1) if s[i] == s[j]
        '''
        if not s:
            return 0
        N = len(s)
        dp = [[False]*N for _ in range(N)]
        result = 0
        #revere N to make indexing easier
        for i in reversed(range(N)):
            dp[i][i] = True
            result += 1
            for j in range(i+1,N):
                #pairs, check chars are same
                if j - i == 1:
                    dp[i][j] = s[i] == s[j]
                else:
                    #is the recurrence
                    dp[i][j] = (s[i]==s[j]) and (dp[i+1][j-1])
                if dp[i][j] == True:
                    result += 1
        return result

#############################################
# Reconstruct Original Digits from English
#############################################
#so fucking close! 13/24
#i feel like this involes a backtracking approach....
#this approach reggidle choosese ones,
#i need to go down this path, and see if i crash
#if this path crashes, abamdom
class Solution(object):
    def originalDigits(self, s):
        """
        :type s: str
        :rtype: str
        """
        '''
        the string is concated to one string, and has the numbers written out in enlgith
        if the string had all the letters it owuld bne
        "zeroonetwothreefourfivesixseveneightnine"
        i can hash each written englgih number to a counter
        greedy
        keep substring in order until there is a change
        '''
        nums = ['zero','one','two','three','four','five','six','seven','eight','nine']
        nums_strs = ['0','1','2','3','4','5','6','7','8','9']
        allcounts = Counter(s)
        
        #helper function to see of char ounts in all counts and returns True or not
        def helper(candidate,currcount):
            if all([char in currcount and currcount[char] >= count for char,count in candidate.items()]):
                return True
            else:
                return False
        
        output = ""
        i = 0
        while len(allcounts) > 0 and i < len(nums):
            if helper(Counter(nums[i]),allcounts):
                output += nums_strs[i]
                allcounts -= Counter(nums[i])

            elif helper(Counter(nums[i]),allcounts) == False:
                i += 1
        return output

class Solution(object):
    def originalDigits(self, s):
        """
        :type s: str
        :rtype: str
        """
        '''
        first appraoch greedily takes the smallest english word numbers frist
        example
        twonine
        the ans is '29'
        but we could also make one
        whihc would leave twni, which is nohting, and the constraints say all chars can be used
        
        inution:
            * "z" is unique to zero
            * "w" is uniqute to two
            * "u" is unique to four
            * "x" is unique to six
            * "g" is unique to eight
        we can hash evens to their unique chars and count em up, what about odds?
            * "h" is present to in three and eight
            * "f" is present in five and four
            * "s" is present in seven and six
        now that leaves nines and ones
            * "i" is present in nine, five,six,and eight
            * "n" is present in one,sevenm and nine
            
        algo:
            1. find count evens
            2. find couns, threes, fives,seves
            3. find counts 1 and nines
        '''
        counts = Counter(s)
        #output digit:fred
        mapp = {}
        
        #evens
        evens = ['0','2','4','6','8']
        evens_chars = ['z','w','u','x','g']
        for digit,char in zip(evens,evens_chars):
            mapp[digit] = counts[char]
        
        #three,five,and sevens
        odds = ['3','5','7']
        odds_comps = ['8','4','6']
        odds_chars = ['h','f','s']
        
        for digit,comp,char in zip(odds,odds_comps,odds_chars):
            mapp[digit] = counts[char] - mapp[comp]
            
        #count9
        mapp['9'] = counts['i'] - mapp['5'] - mapp['6'] - mapp['8']
        #count1
        mapp['1'] = counts['n'] - mapp['7'] - 2*mapp['9'] #two n's in nine
        
        return "".join([key*mapp[key] for key in sorted(mapp.keys())])

class Solution(object):
    def originalDigits(self, s):
        """
        :type s: str
        :rtype: str
        """
        '''
        straight cascading
        '''
        c = Counter(s)
        n = [0]*10
        
        n[0] = c['z']
        n[2] = c['w']
        n[4] = c['u']
        n[6] = c['x']
        n[8] = c['g']
        n[1] = c['o'] - n[0] - n[2] - n[4]
        n[3] = c['h'] - n[8]
        n[5] = c['f']- n[4]
        n[7] = c['s'] - n[6]
        n[9] = c['i'] - n[5] - n[6] - n[8]
        
        output = []
        for i in range(10):
            output.append(str(i)*n[i])
        return ''.join(output)

########################
#Flip Binary Tree to Match Preorder Traversal
#########################
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution(object):
    def flipMatchVoyage(self, root, voyage):
        """
        :type root: TreeNode
        :type voyage: List[int]
        :rtype: List[int]
        """
        '''
        pre order traversal is root, left, right
        we want to return the nodes that we flip such that the the flipped nodes match the pre order traversal, or the voyage
        ughhhh, this one sucks....
        root, is an actuall Tree Note, voyage is a list
        https://leetcode.com/problems/flip-binary-tree-to-match-preorder-traversal/discuss/214216/JavaC%2B%2BPython-DFS-Solution
        dfs recurisve solution
        global integer idx pointing to spot in voyage
        if the current node we are is None, just return
        if the curret node we are does not match the voyage, no amound of swapping can be done to change, return -1
        if there is a let, butw e dont have the voyage value, flip the node with its right
        '''
        self.result = []
        self.idx = 0
        
        def dfs(node):
            if node:
                #not matching the voyage, we are doomed
                if node.val != voyage[self.idx]:
                    self.result =[-1]
                    return
                #otherwise keep going
                self.idx += 1
                #if we can swap left with right
                if node.left and node.left.val != voyage[self.idx]:
                    self.result.append(node.val)
                    #but since we swapped do down right now
                    dfs(node.right)
                    dfs(node.left)
                else:
                    dfs(node.left)
                    dfs(node.right)
        dfs(root)
        if self.result and self.result[0] == -1:
            return [-1]
        return self.result

class Solution(object):
    def flipMatchVoyage(self, root, voyage):
        """
        :type root: TreeNode
        :type voyage: List[int]
        :rtype: List[int]
        """
        '''
        iterative stack
        '''
        results = []
        stack = [root]
        idx = 0
        while stack:
            node = stack.pop()
            if not node:
                continue
            #not matching, we are doomed
            if node and node.val != voyage[idx]:
                return [-1]
            #on to the enxt
            idx += 1
            #potential swap
            if node.right and node.right.val == voyage[idx]:
                #making sure its not a leaf
                if node.left:
                    results.append(node.val)
                stack.append(node.left)
                stack.append(node.right)
            else:
                stack.append(node.right)
                stack.append(node.left)
        return results

########################
# Parallel Courses
#########################
#BFS using Q
class Solution(object):
    def minimumSemesters(self, n, relations):
        """
        :type n: int
        :type relations: List[List[int]]
        :rtype: int
        """
        '''
        graph problme using bfs
        intuition:
            learn all courses avaialbe in a single semester to minmize the number of semesters
            we start from courses taht have no prereqs
            traverse and marked as learned, and we keep going until there are no more available course to learn
            we add to the quea all possible classes we can take
            is the case where we cannot take any more courses, i.e if the number of nodes we visited is strucly less than the number of total nodes
        algo:
            1. build the directed graph of relations
            2. record the in-degree of each node (i.e the number of edges, in this case the num of prereqs)
            3. inint q, and push into the q, those nodes with 0 edges
            4. init step and visint count
            5. bfs
                a. inint next que to record nodes needed in next pass
                b. incrment step
                c. for each node in q:
                    1. increment visited count
                    2. for each end node reachable frmo node
                        a. decremant the in-degre of end node
                        b. if in indegree of end_node reaches 0, push it
                    3. assing q to next q
            6. if visited == N,return step, other wise retunr -1
        '''
        adj_list = {i:[] for i in range(1,n+1)}
        in_degree = {i:0 for i in range(1,n+1)}
        for start,end in relations:
            adj_list[start].append(end)
            in_degree[end] += 1
            
        q = deque()
        #mark class completition
        taken = set()
        semesters = 0
        #we need to add to this q all starting classes with 0 in degree
        for node,preq in in_degree.items():
            if preq == 0:
                taken.add(node)
                q.append(node)
                
        while q:
            N = len(q)
            #bfs for each class in q
            for i in range(N):
                curr_course = q.popleft()
                for next_course in adj_list[curr_course]:
                    #use up a preq, we added the ones with zero prereqs, so these should be > 0
                    in_degree[next_course] -= 1
                    #now for all the next courses, take if zero
                    if in_degree[next_course] == 0:
                        q.append(next_course)
                        taken.add(next_course)
            #incrment the semester count
            semesters += 1
        #we only return the semses if we have taken all classes, otherwise we didnt because there was a cycle
        if len(taken) == n:
            return semesters
        else:
            return -1

#dfs with cycle detection
#note, if i this were an interview, id go with BFS
#dfs DAG cycle detection are way too hard
class Solution(object):
    def minimumSemesters(self, n, relations):
        """
        :type n: int
        :type relations: List[List[int]]
        :rtype: int
        """
        '''
        dfs, from the BFS we notice one thing,
        the number of semesters needed is equal to the length of the longet path in the graph
        now think about this? why?
        if we are allowed to take as many classes as we can in one semester, then the only constraint would be the number of classes that we cant ake that have a prerequsite!
        we cannot take a class without taking its prereq, so in the graph, the min number of semesters is the longest path!
        i.e take class one by one satisfying the prereqs
        the problem happens if there is a cycle
        
        intuition:
            * check for cycle in graph
            * find the longest path
            each of these parts can be done with dfs, approach 3 combines them in one approach
            * cycle detection:
                each node has one of three states, unvisited, visiting, visited
                before dfs we inint all nodes to their state
                while dfsing we mark the current node as visiting, until we search all paths out from the node
                if we meed a node marked with processing, it must have come from upstream, i.e cycle present so abandon
                if dfs finishes, and all ndoes are marked as visited, no ccyle
            * longest path:
                * typical dfs, return max out of all recusrive calls for its child nodes
                * to prevent reducnancies, we cache
        
        algo:
            Step 1: Build a directed graph from relations.
            Step 2: Implement a function dfsCheckCycle to check whether the graph has a cycle.
            Step 3: Implement a function dfsMaxPath to calculate the length of the longest path in the graph.
            Step 4: Call dfsCheckCycle, return -1 if the graph has a cycle.
            Step 5: Otherwise, call dfsMaxPath. Return the length of the longest path in the graph.
        '''
        adj_list = {i:[] for i in range(1,n+1)}
        for start,end in relations:
            adj_list[start].append(end)
            
        #check if graph has cycle 
        visited = {}
        
        def dfs_has_cycle(node):
            if node in visited:
                return visited[node]
            else:
                visited[node] = -1 #mark as visiting
            for end in adj_list[node]:
                if dfs_has_cycle(end) == True:
                    return True #cycle
            #mark as visited
            visited[node] = False
            return False
        for node in adj_list.keys():
            if dfs_has_cycle(node) == True:
                return -1
        
        #no cylce present, find longest path, markes the longest path for each node
        visited_length = {}
        
        def dfs_max_path(node):
            if node in visited_length:
                return visited_length[node]
            max_length = 1
            for end in adj_list[node]:
                #get thhe currenth length
                length = dfs_max_path(end)
                max_length = max(max_length,length+1)
            #cache it
            visited_length[node] = max_length
            return max_length
        
        return max(dfs_max_path(node) for node in adj_list.keys())

#######################
#Russian Doll Envelopes
#######################
#near end of the month hard problem, no sweat if i cant get it
#brute force won't work in all settings
class Solution(object):
    def maxEnvelopes(self, envelopes):
        """
        :type envelopes: List[List[int]]
        :rtype: int
        """
        '''
        an envolope can fit into another evenvolope if its w,h are less than the current envolope under consideration
        brute force is easy
        for each enveolope examine every other envolope and see if it can go inside it
        update maxlength on the floy
        '''
        N = len(envelopes)
        max_nesting = 0
        for i in range(N):
            curr_envelope = envelopes[i]
            curr_nesting = 1
            for j in range(N):
                if curr_envelope[0] > envelopes[j][0] and curr_envelope[1] > envelopes[j][1]:
                    curr_nesting += 1
            print i,curr_nesting
            max_nesting  = max(max_nesting,curr_nesting)
        return max_nesting

class Solution(object):
    def maxEnvelopes(self, envelopes):
        """
        :type envelopes: List[List[int]]
        :rtype: int
        """
        '''
        this can be broken to the longest increasing subsequence problem (LIS)
        we need to find a sequence such that seq[i+1] > seq[i], at some point i need to reivew
        brief review of LIS problem:
            we hold a dp array such that dp[i] is the smallest element that ends in increasing subsequence of legnth i + 1
            whenver we get to a new element, call it e, we binary search inside dp to find the largest index i, such that e can end that subsequence
            we then update dp[i] with e
            the length of the LIS is the same as the length of dp, as if dp has index i, then it must have subsequence of length i+1
        
        algorithm:
            let's pretend that we found the ebst arragnemnet of envs
            we know that each env must be increasing in w
            this our ebst arrangement has to be a subseq of all our envs sorted on w
            
            after sorting, we can simply find the length of the lognest inceasing subseq on thes second dimensino
        example:
            consider input: [1, 3], [1, 4], [1, 5], [2, 3], would imply we enst envelopes with heights 3,4,5
            but this cannot be because widths for all those three with those heights are 1
            in order to fix this, we don't just sort increaisng in first dim, but we sort decreasing second dim
            
        [[5,4],[6,4],[6,7],[2,3]] sort be asneding wdith and descending width
        [2,3],[5,4],[6,4],[6,7]
        now just look at the hgiths
        [3,4,7]
        which implies [2,3] -> [5,4] -> [6,7]
        now we are just looking for the lognest increasing subsequence
        '''
        env = sorted(envelopes, key = lambda x: (x[0],-x[1]))
        
        dp = [] #this stores the indices the of the lonest icnreasin subsequege
        for w,h in env:
            i, N = 0,len(dp)
            while i < N:
                #no longe an increasing subsequence on h
                if h <= dp[i]:
                    break
                i += 1
            #gotten ot the current end
            #meaning we encounter a new element that resulted in last increasing
            if i == N:
                dp.append(h)
            #the last new element encountered did not result in an increasing subsequege
            else:
                dp[i] = h
        
        return len(dp)

#binary search
class Solution(object):
    def maxEnvelopes(self, envelopes):
        """
        :type envelopes: List[List[int]]
        :rtype: int
        """
        '''
        using binary search
        '''
        env = sorted(envelopes, key = lambda x : (x[0],-x[1]))
        
        dp = []
        for w,h in env:
            i = bisect_left(dp,h)
            
            if i == len(dp):
                dp.append(h)
            else:
                dp[i] = h
        return len(dp)

#binary search of LIS variant
class Solution(object):
    def maxEnvelopes(self, envelopes):
        """
        :type envelopes: List[List[int]]
        :rtype: int
        """
        '''
        binary search
        '''
        N = len(envelopes)
        if N<= 1:
            return N
        
        #sort ascneding width, and descending height
        envelopes.sort(key = lambda x: (x[0],-x[1]))
        #find LIS
        size = 0
        tails = [0]*N
        
        for w,h in envelopes:
            #binary search
            left,right = 0,size -1 #won't go into the first pass
            while left <= right:
                mid = left + (right - left) //2 
                if tails[mid] >= h:
                    #no longer increasing
                    right = mid -1
                else:
                    left = mid + 1
            #we want the position to the left? why? idk
            tails[left] = h
            size = max(size,left+1)
        return size

########################
#Stamping the sequence
########################
class Solution(object):
    def movesToStamp(self, stamp, target):
        """
        :type stamp: str
        :type target: str
        :rtype: List[int]
        """
        '''
        this one is doozy, lets go over the official LC solution
        work backwards
        image we stamped the seq with moves m1,m2...mN 
        now form the target, make the moves in reverse order to to get the starting string ??????
        
        now lets call the ith window, a subarray of target of lengthe stamp.lenghtm that starts at i
        each move at position i is possible of the ith window matches the stamp
        after every char in the windwo becomes a wildcard that can match any char in the stamp
        example:
        stamp = 'abca', target = 'aabcaca'
        working backwards
        starting at i == 1, we can indo the chars in the window
        'a????ca'
        then undo starting at i == 3
        'a??????'
        then undo at 0
        '???????'
        algo:
            1. keep track of every window. we want to know how many cells initally match the stamp, call this made, and which ones don't call todo
            2. any window that are ready get enqued
            3. more specifically, we enqueue the positions of each char (to save time we enqueue by char not by window). this represents that the char is ready to turn into a ? int out working target string
            4. for each char, look at all the windows that intesect it and updater their todo lists
            5. if any todo lists become empty in this amnner, then we enque the chars in window made that hve not been processed yet
        '''
        #cache
        M, N = len(stamp), len(target)
        q = deque()
        done = [False]*N
        ans = []
        A = []
        #go through all windows
        for i in range(N-M+1):
            #for each window [i,i+M), A[i] will contain infor on what needs to change befor we can reverse it
            made, todo = set(),set()
            for j,c in enumerate(stamp):
                #just coparing ith window with chars in stamp
                a = target[i+j]
                #matc
                if a == c:
                    made.add(i+j) #adding indices
                else:
                    todo.add(i+j)
            #to examine
            A.append((made,todo))
            #if we can reverse stamp at i immediatly, enqueue letters from this window
            if not todo: #i.e all made
                ans.append(i)
                for j in range(i,i+M):
                    if not done[j]:
                        q.append(j)
                        done[j] = True
        #bfs
        while q: #for each enqued letter, i.e it could not revered just yet
            i = q.popleft()
            #for each window that is potentially affected, j is tart
            for j in range(max(0,i-M+1),min(N-M,i)+1):
                if i in A[j][1]: #the affected window
                    A[j][1].discard(i) #remove from todolist of this window
                    if not A[j][1]: #to di list of this window might be empty
                        ans.append(j)
                        for m in A[j][0]: #for each letter to potentaily enqueu
                            if not done[m]:
                                q.append(m)
                                done[m] = True
        return ans[::-1] if all(done) else []

            