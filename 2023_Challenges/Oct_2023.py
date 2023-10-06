####################################################################
# 2038. Remove Colored Pieces if Both Neighbors are the Same Color
# 02OCT23
#####################################################################
class Solution:
    def winnerOfGame(self, colors: str) -> bool:
        '''
        alice can only remove A pieces, if an A is surrounded by two other As and is not on the edge
        same thing for Bob, but with B pieces
        if a player cannot make a move on their turn, they lose and the other player wins
        in order for Alice to win, she has to have more moves than Bob
        count the number of moves and check if Alice has more moves than Bob
        
        count streaks of As and Bs and calculate the number of moves
        '''
        def countMoves(colors, char):
            moves = 0
            curr_streak = 0
            for c in colors:
                if c == char:
                    curr_streak += 1
                else:
                    if curr_streak > 2:
                        moves += curr_streak - 2
                    curr_streak = 0
            if curr_streak > 2:
                moves += curr_streak - 2
                curr_streak = 0
            return moves
        
        
        aliceMoves = countMoves(colors,'A')
        bobMoves = countMoves(colors, 'B')
        print(aliceMoves,bobMoves)
        return aliceMoves > bobMoves
    
class Solution:
    def winnerOfGame(self, colors: str) -> bool:
        '''
        just a review on two key intutions
        1. when one player removes a letter, it wil never create a new removal oppuruntiy for another player
            i.e deleting an A or B on alice or bob's turn doesn't give the opposing player any advantage
        2. the order in which removals happens is irrelevant
            if we have AAAAA, removing any of the middle As just brings it to one less A
        
        
        '''
        A,B,N = 0,0,len(colors)
        #dont go to the edges
        for i in range(1,N-1):
            if colors[i-1] == colors[i] == colors[i+1]:
                if colors[i] == 'A':
                    A += 1
                else:
                    B += 1
        
        return A > B
    
##############################################
# 1804. Implement Trie II (Prefix Tree)
# 02OCT23
##############################################
#yessss!
class Node:
    def __init__(self):
        self.children = defaultdict()
        self.word_counts = 0
        self.is_end = False
        #keep track of ending words
        self.words_ending = 0
        

class Trie:
    '''
    this is a design problem, think about what we want to support for the methods
    '''

    def __init__(self):
        self.trie = Node()

    def insert(self, word: str) -> None:
        curr = self.trie
        for ch in word:
            if ch in curr.children:
                curr = curr.children[ch]
            else:
                curr.children[ch] = Node()
                curr = curr.children[ch]
            #update counts
            curr.word_counts += 1
        
        curr.is_end = True #maark
        curr.words_ending += 1

    def countWordsEqualTo(self, word: str) -> int:
        #this means we could have multiple occurrences for a word in the tree
        #for the end of a word, store the counts
        curr = self.trie
        for ch in word:
            if ch in curr.children:
                curr = curr.children[ch]
            else:
                return 0
        
        if curr.is_end and curr.words_ending >= 0:
            return curr.words_ending
        return 0
        

    def countWordsStartingWith(self, prefix: str) -> int:
        #each node should have a count as well as marking the end of word
        #then we just descend the tree until the end and grab a count
        curr = self.trie
        for ch in prefix:
            if ch in curr.children:
                curr = curr.children[ch]
            else:
                return 0
        
        if curr.word_counts >= 0:
            return curr.word_counts
        return 0

    def erase(self, word: str) -> None:
        #this part is tought
        #remove one ocrrucne. just decrese by 1
        curr = self.trie
        for ch in word:
            if ch in curr.children:
                curr = curr.children[ch]
            else:
                return False
            curr.word_counts -= 1
        
        curr.words_ending -= 1


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.countWordsEqualTo(word)
# param_3 = obj.countWordsStartingWith(prefix)
# obj.erase(word)

######################################################
# 2001. Number of Pairs of Interchangeable Rectangles
# 03OCT23
######################################################
class Solution:
    def interchangeableRectangles(self, rectangles: List[List[int]]) -> int:
        '''
        two rectanlge are interchangable if their ratios of widht/heights are the same
        '''
        counts = Counter()
        ans = 0
        
        for w,h in rectangles:
            ratio = w/h
            ans += counts[ratio]
            counts[ratio] += 1
        
        return ans
    
class Solution:
    def interchangeableRectangles(self, rectangles: List[List[int]]) -> int:
        '''
        the real way would be to use gcd, then reduce fraction to lowest terms
        then use this fraction as the key
        '''
        
        def gcd(a,b):
            if b == 0:
                return a
            return gcd(b, a % b)
            
        counts = Counter()
        ans = 0
        
        for w,h in rectangles:
            GCD = gcd(w,h)
            #store key is tuple 
            w = w // GCD
            h = h // GCD
            ratio = (w,h)
            ans += counts[ratio]
            counts[ratio] += 1
        
        return ans
    
##########################################################
# 2083. Substrings That Begin and End With the Same Letter
# 03OCT23
##########################################################
class Solution:
    def numberOfSubstrings(self, s: str) -> int:
        '''
        look at some examples
        abcba, examine subtrings starting and ending with each letter
        a: a, abcba,a
        b: b, bcb, b
        c: a
        
        in general if a char i, appears n times, then there are n*(n+1)/2 substrings that start and being with that letter
        '''
        ans = 0
        counts = defaultdict()
        
        for ch in s:
            counts[ch] = counts.get(ch,0) + 1
        
        for count in counts.values():
            ans += count*(count+1) // 2
        
        return ans

#######################################
# 229. Majority Element II
# 05OCT23
#######################################
class Solution:
    def majorityElement(self, nums: List[int]) -> List[int]:
        '''
        take the idea from Majority Element I,where we keep track of the current candiate
        essentially we are finding majority nums[:i] for all i
        for the case when n//2:
            keep count of occurrences of current candidate and current candidate variables
            when count if zero, we assign new candidate as the current element
            otherwise +=1 if num is the cnadiate, else -1
        heres the snippet
        class Solution:
            def majorityElement(self, nums):
                count = 0
                candidate = None

                for num in nums:
                    if count == 0:
                        candidate = num
                    count += (1 if num == candidate else -1)

                return candidate
        
        now for the case n//3
        intuition
            there can be at most one majority element with count more than n//2
            there can be as most two elements with count more than n//3
            three for count more than n//4
        
        for some count n // k, there can be at most k-1, with k being >= 1
        we adopt the same intution for the n//2 case to the n // 3 case, but keep track of two candidates instead of 1 
        '''
        if not nums:
            return []
        
        cand1,cand2 = None,None
        count1,count2 = 0,0
        
        for num in nums:
            #matching candidates
            if num == cand1:
                count1 += 1
            elif num == cand2:
                count2 += 1
            #check counts
            elif count1 == 0:
                cand1 = num
                count1 += 1
            elif count2 == 0:
                cand2 = num
                count2 += 1
            else:
                count1 -= 1
                count2 -= 1
        
        #check
        ans = []
        count1 = 0
        count2 = 0
        for num in nums:
            if num == cand1:
                count1 += 1
            if num == cand2:
                count2 += 1
        N = len(nums)
        if count1 > N//3:
            ans.append(cand1)
        if count2 > N//3:
            ans.append(cand2)
        
        return ans
        
#################################################
# 1206. Design Skiplist
# 05OCT23
#################################################
'''
reading on skiplists
https://brilliant.org/wiki/skip-lists/#height-of-the-skip-list
https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2015/resources/mit6_046js15_lec07/
be sure to check out implementaion frmo video
actual implementation
https://leetcode.com/problems/design-skiplist/discuss/1573487/Clean-Python
'''
import random

class Node:
    def __init__(self, val = -1, right = None, bottom = None):
        #nodes for only going right and bottom
        self.val = val
        self.right = right
        self.bottom = bottom

class Skiplist:
    def __init__(self):
        self.head = Node()

    def flip_coin(self): #coin flip for promoting when element is added
        return random.randrange(0, 2)

    def search(self, target: int) -> bool:
        node = self.head

        #go right and down
        while node:
            while node.right and target > node.right.val:
                node = node.right
            if node.right and target == node.right.val:
                return True
            node = node.bottom

        return False

    def add(self, num: int) -> None:
        node = self.head
        #store all the nodes we went down on while search
        record_levels = []

        while node:
            while node.right and num > node.right.val:
                node = node.right

            record_levels.append(node) #all the nodes we went down on are in the array, with the most recent ones (i.e the bottom) are at the end of the array
            node = node.bottom
        #insertion prep
        new_node = None
        
        #while we don't have a new node or while we can promote (get a heads)
        while not new_node or self.flip_coin():
            #if we are at the top level
            if len(record_levels) == 0:
                #just jeep adding to head and point to isetself,
                #in the case where we keep getting heads and we have to promote
                self.head = Node(-1, None, self.head)
                prev_level = self.head
            #we need to promote and we have levles
            else:
                prev_level = record_levels.pop()
            #make new node
            new_node = Node(num, prev_level.right, new_node)
            #connect
            prev_level.right = new_node

    def erase(self, num: int) -> bool:
        #easy peeze insert
        node = self.head
        boolean = False

        while node:
            while node.right and num > node.right.val:
                node = node.right
            #erase all nodes with that num value doing down levels
            if node.right and num == node.right.val:
                node.right = node.right.right
                boolean = True
            node = node.bottom

        return boolean
    
#another way
'''
https://cw.fel.cvut.cz/old/_media/courses/a4b36acm/maraton2015skiplist.pdf
https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2015/resources/mit6_046js15_lec07/
https://leetcode.com/problems/design-skiplist/discuss/1082053/simple-solution-with-dynamic-levels-%2B-references
'''
import random


class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None
        self.down = None

class Skiplist:
    def __init__(self):
        # sentinel nodes to keep code simple
        node = ListNode(float('-inf'))
        node.next = ListNode(float('inf'))
        self.levels = [node]

    def search(self, target: int) -> bool:
        level = self.levels[-1]
        while level:
            node = level
            while node.next.val < target:
                node = node.next
            if node.next.val == target:
                return True
            level = node.down
        return False

    def add(self, num: int) -> None:
        stack = []
        level = self.levels[-1]
        while level:
            node = level
            while node.next.val < num:
                node = node.next
            stack.append(node)
            level = node.down

        heads = True
        down = None
        while stack and heads:
            prev = stack.pop()
            node = ListNode(num)
            node.next = prev.next
            node.down = down
            prev.next = node
            down = node
            # flip a coin to stop or continue with the next level
            heads = random.randint(0, 1)

        # add a new level if we got to the top with heads
        if not stack and heads:
            node = ListNode(float('-inf'))
            node.next = ListNode(num)
            node.down = self.levels[-1]
            node.next.next = ListNode(float('inf'))
            node.next.down = down
            #this is in reverse, 
            #top left node is actually at the bottom of the list
            self.levels.append(node)

    def erase(self, num: int) -> bool:
        stack = []
        level = self.levels[-1]
        while level:
            node = level
            while node.next.val < num:
                node = node.next
            if node.next.val == num:
                stack.append(node)
            level = node.down

        if not stack:
            return False

        for node in stack:
            node.next = node.next.next

        # remove the top level if it's empty
        while len(self.levels) > 1 and self.levels[-1].next.next is None:
            self.levels.pop()

        return True

# Your Skiplist object will be instantiated and called as such:
# obj = Skiplist()
# param_1 = obj.search(target)
# obj.add(num)
# param_3 = obj.erase(num)

