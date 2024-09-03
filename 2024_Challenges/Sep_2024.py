#######################################
# 2022. Convert 1D Array Into 2D Array
# 01SEP24
######################################
class Solution:
    def construct2DArray(self, original: List[int], m: int, n: int) -> List[List[int]]:
        '''
        i can either precompute the 2d array and fill in the sports
        if im at cell (i,j) i can get its 1d index as i*
        '''
        if len(original) != m*n:
            return []
        
        ans = [[0]*n for _ in range(m)]
        for i in range(len(original)):
            #could also do:
            #row,col = divmod(i)
            row = i // n
            col = i % n
            ans[row][col] = original[i]
        
        return ans
        
#####################################################
# 1894. Find the Student that Will Replace the Chalk
# 02SEP24
#####################################################
class Solution:
    def chalkReplacer(self, chalk: List[int], k: int) -> int:
        '''
        the sum of the whole array repeats itself
        find the point where it exceeds k
        its just modulu sum(chalk) for the remainder of chalk
        '''
        sum_chalk = sum(chalk)
        times_around = k // sum_chalk
        #chalk_left = max(0,k - sum_chalk*times_around)
        chalk_left = k % sum_chalk

        for i,student in enumerate(chalk):
            if student > chalk_left:
                return i
            chalk_left -= student
        
        return i
    
#binary search
class Solution:
    def chalkReplacer(self, chalk: List[int], k: int) -> int:
        '''
        another solution is to do binary search on the pref_sum array of chalk
        the idea is to look for the index in pref_sum array that is < remaining chalk
        '''
        pref_chalk = [0]
        for c in chalk:
            pref_chalk.append(pref_chalk[-1] + c)
        
        chalk_left = k % pref_chalk[-1]
        #print(pref_chalk)
        #print(chalk_left)
        #look for upper bound
        left = 1
        right = len(pref_chalk) - 1
        ans = right
        while left < right:
            mid = left + (right - left) // 2
            if pref_chalk[mid] <= chalk_left:
                left = mid + 1
            else:
                ans = mid
                right = mid
                
        return ans - 1
    
##############################################
# 1945. Sum of Digits of String After Convert
# 03SEP24
#############################################
class Solution:
    def getLucky(self, s: str, k: int) -> int:
        '''
        generate starting number and repeat k times
        '''
        starting_number = 0
        for ch in s:
            pos = (ord(ch) - ord('a')) + 1
            if pos // 10 > 0:
                starting_number *= 100
            else:
                starting_number *= 10
            starting_number += pos
        
        #reapeat k times
        while k > 0:
            starting_number = self.sum_digits(starting_number)
            k -= 1
        
        return starting_number
    
    def sum_digits(self,num):
        ans = 0
        while num:
            ans += num % 10
            num = num // 10
        
        return ans
    
class Solution:
    def getLucky(self, s: str, k: int) -> int:
        '''
        instead of checking for multiple of 1 and 10, just grab the gitis
        '''
        starting_number = 0
        for ch in s:
            pos = (ord(ch) - ord('a')) + 1
            while pos > 0:
                starting_number = starting_number*10 + pos % 10
                pos = pos // 10
        
        #reapeat k times
        while k > 0:
            starting_number = self.sum_digits(starting_number)
            k -= 1
        
        return starting_number
    
    def sum_digits(self,num):
        ans = 0
        while num:
            ans += num % 10
            num = num // 10
        
        return ans