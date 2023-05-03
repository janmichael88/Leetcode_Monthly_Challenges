#############################################
# 1065. Index Pairs of a String
# 01MAY23
############################################
class Solution:
    def indexPairs(self, text: str, words: List[str]) -> List[List[int]]:
        '''
        for each word in words, check if in text
        '''
        ans = []
        
        for word in words:
            N = len(word)
            for i in range(0,len(text) - N + 1):
                check = text[i:i+N]
                if check == word:
                    ans.append([i, i+N-1])
        
        ans.sort(key = lambda x: (x[0],x[1]))
        return ans
    
#hashmap
class Solution:
    def indexPairs(self, text: str, words: List[str]) -> List[List[int]]:
        '''
        hashset each word and just check all substrings
        '''
        words = set(words)
        N = len(text)
        ans = []
        for i in range(N):
            for j in range(i,N):
                substring = text[i:j+1]
                if substring in words:
                    ans.append([i,j])
        
        return ans
    
########################################
# 1822. Sign of the Product of an Array
# 01MAY23
#######################################
class Solution:
    def arraySign(self, nums: List[int]) -> int:
        '''
        just check pairty of + signs, -signs, and count zeros
        if there is a zero:
            return 0
        positive signes don't matter
        if there are an even number of negative signes, its positive
        otherwise negative
        '''
        count_zeros = 0
        count_negatives = 0
        
        for num in nums:
            count_zeros += num == 0
            count_negatives += (num < 0)
        
        if count_zeros:
            return 0
        elif count_negatives % 2 == 0:
            return 1
        else:
            return -1
        
class Solution(object):
    def arraySign(self, nums):
        sign = 1
        for num in nums:
            if num == 0:
                return 0
            if num < 0:
                sign = -1 * sign

        return sign
    
##################
# 475. Heaters
# 02APR23
##################
#YASSSS
#careful with the max and min bounds
#binary search using workable solution
class Solution:
    def findRadius(self, houses: List[int], heaters: List[int]) -> int:
        '''
        we are given arrays of houses and heaters,
        return the minimum heater radius so that all heaters can cover all the houses
        
        houses = [1,2,3,4]
        heaters = [1,4]
        
        let N = len(houses) - 1 = 3
        start with the first heater 1 advance all the way so that works
        try smaller, 2
        
        smallest answer can be 0
        largest answer can be max(houses) - min(houses)
        
        i can use binary search to find a workable solution, but I need an O(N) solution to check if the current workable solution works
        function to check if the current heater radius works
        '''
        left = 0
        right = max(max(houses),max(heaters)) - min(min(houses),min(heaters))
        #sort
        houses.sort()
        heaters.sort()
        
        def check_radius(heaters,houses,radius):
            i = 0
            for j in range(len(heaters)):
                curr_heater = heaters[j]
                #advance if within radius
                while i < len(houses) and abs(curr_heater - houses[i]) <= radius:
                    i += 1
            
            return i == len(houses)
        
        while left < right:
            #guess
            mid = left + (right - left) // 2
            if check_radius(heaters,houses,mid):
                right = mid
            else:
                left = mid + 1
        
        return right
                
#another binary search
class Solution:
    def findRadius(self, houses: List[int], heaters: List[int]) -> int:
        '''
        sort housea and heaters in ascending order
        then for each house find the closest heater to the left and to the right (lower and upper bounds)
        if the house if to left of all the heaters, well this house has to use the first heater, heaters[0]
        if house is to the right of all the heaters, use the end
            in these cases, we need to take the maximum
        otherwise a house is in between two heaters, so take the minimum, then the max
        
        we turn a minimum problem to a maximum problem
        summary:
            for each house find the two closest heaters
            for this house the local minimum is just the close of the two heaters, and although it will work for this house
            it might not work for other houses, but we know that it cannot be any smaller than this current mimum
        '''
        #sort heaters
        heaters.sort()

        # Initialize the result to 0
        result = 0

        # For each house, find the closest heater to the left and the right
        for house in houses:
            left = bisect.bisect_right(heaters, house) - 1
            right = bisect.bisect_left(heaters, house)

            # If the house is to the left of all heaters, use the closest heater to the left
            if left < 0:
                result = max(result, heaters[0] - house)
            # If the house is to the right of all heaters, use the closest heater to the right
            elif right >= len(heaters):
                result = max(result, house - heaters[-1])
            # If the house is between two heaters, use the closer of the two
            else:
                result = max(result, min(house - heaters[left], heaters[right] - house))

        # Return the result
        return result
    
#one more
class Solution:
    def findRadius(self, houses: List[int], heaters: List[int]) -> int:
        houses.sort()
        heaters.sort()
        res = 0
        heater = 0
        
        for h in houses:
            while heater + 1 < len(heaters) and heaters[heater + 1] == heaters[heater]:                  # Avoid duplicates
                heater += 1
            while heater + 1 < len(heaters) and abs(heaters[heater + 1] - h) < abs(heaters[heater] - h): # If using next heater is more efficient
                heater += 1                                                                              # Then use next heater
            
            res = max(res, abs(heaters[heater] - h))        # Update its range to house
        
        return res
