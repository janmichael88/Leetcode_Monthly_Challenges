######################################
# 1230. Toss Strange Coins
# 02JUN23
#####################################
class Solution:
    def probabilityOfHeads(self, prob: List[float], target: int) -> float:
        '''
        if i just had 1 coin and flipped it, this would just have been a bernoulli trial
        but we are given an array of coins
        keep track of count heads, count tails, position, and curr probabilty
            
        
        '''
        n = len(prob)
        memo = {}
        def dp(i,count):
            if count > target:
                return 0
            if i == n:
                if count == target:
                    return 1
                else:
                    return 0
            
            if (i,count) in memo:
                return memo[(i,count)]
            
            #getting a head
            get_head = prob[i]*dp(i+1,count+1)
            #getting tails
            get_tails = (1-prob[i])*dp(i+1,count) #doesn't go up because its a tail
            ans = get_head + get_tails
            memo[(i,count)] = ans
            return ans
        
        
        return dp(0,0)
    

#bottom up
class Solution:
    def probabilityOfHeads(self, prob: List[float], target: int) -> float:
        '''
        if i just had 1 coin and flipped it, this would just have been a bernoulli trial
        but we are given an array of coins
        keep track of count heads, count tails, position, and curr probabilty
            
        
        '''
        n = len(prob)
        dp = [[0]*(target+1) for _ in range(n+1)]
        
        #base case fill
        for i in range(n,-1,-1):
            for count in range(target,-1,-1):
                if count > target:
                    dp[i][count] = 0
                if i == n:
                    if count == target:
                        dp[i][count] = 1
                    else:
                        dp[i][count] = 0
                        
        
        #one away from base case
        for i in range(n-1,-1,-1):
            for count in range(target-1,-1,-1):
                #getting a head
                get_head = prob[i]*dp[i+1][count+1]
                #getting tails
                get_tails = (1-prob[i])*dp[i+1][count] #doesn't go up because its a tail
                ans = get_head + get_tails
                dp[i][count] = ans
        
        print(dp)
        return dp[0][0]

class Solution:
    def probabilityOfHeads(self, prob: List[float], target: int) -> float:
        n = len(prob)
        dp = [[0] * (target + 1) for _ in range(n + 1)]
        dp[0][0] = 1

        for i in range(1, n + 1):
            dp[i][0] = dp[i - 1][0] * (1 - prob[i - 1])
            for j in range(1, target + 1):
                if j > i:
                    break
                dp[i][j] = dp[i - 1][j - 1] * prob[i - 1] + dp[i - 1][j] * (1 - prob[i - 1])

        return dp[n][target]

#######################################################
# 1502. Can Make Arithmetic Progression From Sequence
# 06JUN23
######################################################
class Solution:
    def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
        '''
        there should only be one difference
        try sorting increasnly, then check,
        then decresilngly then check
        '''
        N = len(arr)
        if N == 2:
            return True
        
        arr_inc = sorted(arr)
        for i in range(1,N-1):
            x = arr_inc[i-1]
            y = arr_inc[i]
            z = arr_inc[i+1]
            
            if y - x != z - y:
                return False
            
        return True
    
#no sorting
class Solution:
    def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
        '''
        we don't  need to sort
        assume the array is arithmetic, with common differecne diif
        let the first term by min+valie
            then the differeence between each eleement arr[i] and min_value must be a multiple of diff
        example
        [1,4,7,10]
        10 - 1 / 3 = 3
        4 -  1 % 3 == 3
        
        we just check that (arr[i] - min_value) % diff is a multiple
            i.e modulo == 0
        
        careful with duplicate elements in the array, like [1,2,3,2,5]
        this will fail the first algo since each arr[i] - min is divisible by diff
        use set, and check size of set == len(arr)
        
        '''
        min_value,max_value = min(arr),max(arr)
        N = len(arr)
        
        #all the same, return true
        if max_value - min_value == 0:
            return True
        #range should be also divisible by steps
        if (max_value - min_value) % (N-1):
            return False
        
        diff = (max_value - min_value) // (N-1)
        seen = set()
        for num in arr:
            if (num - min_value) % diff:
                return False
            seen.add(num)
        
        return len(seen) == N
    
#constant space
class Solution:
    def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
        '''
        in order to use O(1) space we need to modifiy the input
        say an array is arithmetic, with range_diff == diff,tecnically diff / (n-1) should be the step size
        we can difine its position in the array as
            j = (arr[i] - min_value) / diff
        
        if j == i, then this number is in the right place, othrwise swap
        
        '''
        min_value,max_value = min(arr),max(arr)
        N = len(arr)
        
        #all the same, return true
        if max_value - min_value == 0:
            return True
        #range should be also divisible by steps
        if (max_value - min_value) % (N-1):
            return False
        
        diff = (max_value - min_value) // (N-1)
        
        i = 0
        while i < N:
            #arr[i] is correctly place
            if arr[i] == min_value + i*diff:
                i += 1
            
            #doesn't belong to this arithemtic sequence
            elif (arr[i] - min_value) % diff:
                return False
            #otherwise swap
            else:
                j = (arr[i] - min_value) // diff
                #dupliate?
                if arr[i] == arr[j]:
                    return False
                arr[i],arr[j] = arr[j], arr[i]
        
        return True
