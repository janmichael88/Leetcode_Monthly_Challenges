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
