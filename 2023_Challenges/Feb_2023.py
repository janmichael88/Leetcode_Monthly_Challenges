###########################################
# 1071. Greatest Common Divisor of Strings
# 01FEB23
###########################################
#yessssssssss
class Solution:
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        '''
        for two strings s and t, t can divide s if s = t + t + ... t,
        rather t is just a concatenation of s
        return the largest string x such that x divides both str1 and str2
        
        brute force would be to check all possible substrings of str2, and check that it can be a substring of str1
        
		slow but it works
        '''
        def test_prefix(s,t):
            #s is str1 and t is what we are testing
            while t != "" and s.startswith(t):
                s = s.replace(t,"")
            
            return s == ""
        
        ans = ""
        for i in range(len(str2)+1):
            cand = str2[:i]
            #test both ways, it must divide both
            if test_prefix(str1,cand) and test_prefix(str2,cand):
                ans = cand

        
        return ans


class Solution:
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        '''
        we know that for a string to be a divisor of another string, it must be a prefix in both strings
        so we can just check all prefixes of one in the other
        
        in order for it to evenly divide, the smallest string can at most go into the larger string
        so we only need to generate all prefixies starting with the smaller string
        
        we need to find the base string in each str1 and str2, if the base string can divide str1 and str2
        it is a valid candidate
        the base string concatenated n times will yeild str1 and the base stringe concatenated m times will yeild str2
        for any valid interger n and m
        
        we actually don't need to compare potential prefixes as bases, just get their lengths
        the prefix can be any size from 0 to min(len(str1),len(str2))
        
        this is actually a really good review
        
        '''
        size1 = len(str1)
        size2 = len(str2)
        
        def valid_divisor(k):
            #must evenly divide both strings
            if size1 % k != 0 or size2 % k != 0:
                return False
            #count the number of parts for each string
            parts1,parts2 = size1 // k, size2 // k
            #find the base string
            base = str1[:k]
            #check that we can make conctenations
            return str1 == base*parts1 and str2 == base*parts2
        
        #we want the longest string, so start with the largest length
        #dont go past 0
        for size in range(min(size1,size2),0,-1):
            if valid_divisor(size):
                return str1[:size]
        
        return ""

#from chatgpt
class Solution:
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        '''
        just compute the gcd of len(str1) and len(str2)
        and return that prefix with that length
        '''
        def gcd_of_strings(str1, str2):
            if len(str1) < len(str2):
                str1, str2 = str2, str1
            if str1 == str2:
                return str1
            if str1[:len(str2)] != str2:
                return ""
            return gcd_of_strings(str1[len(str2):], str2)

        
        return gcd_of_strings(str1,str2)


#iteratively
class Solution:
	def gcd_of_strings(str1, str2):
	    if len(str1) < len(str2):
	        str1, str2 = str2, str1
	    if str1 == str2:
	        return str1
	    for i in range(len(str2), 0, -1):
	        if len(str2) % i == 0 and str1[:i] * (len(str2) // i) == str2:
	            if str1[:i] * (len(str1) // len(str2)) == str1:
	                return str1[:i]
	    return ""




#using gcd on lengths
class Solution:
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        '''
        since both strings need to be concatentations of each other, we need to check:
        str1 + str2 == str2 + str1
        
        otherwise we need to find the gcd of the two lengths and check
        
        contradiction:
            it is not possible for the GCD string to be shorter than the gcdbase
            
        if there is a shorter string that divides both str1 and str2, then gcdBase is also a divisible string, so a divisible string shorter than gcdBase can never be the GCD string
        '''
        def gcd(x,y):
            if y == 0:
                return x
            else:
                return gcd(y,  x % y)
            
        
        #check if the have the non-zero GCD string
        if str1 + str2 != str2 + str1:
            return ""

        k = gcd(len(str1),len(str2))
        return str1[:k]
        