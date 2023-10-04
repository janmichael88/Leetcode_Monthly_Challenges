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
    
