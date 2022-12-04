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

