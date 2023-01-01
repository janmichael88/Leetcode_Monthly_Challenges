###################################
# 818. Race Car
# 27DEC22
###################################
#not close at all in thiking....
class Solution:
    def racecar(self, target: int) -> int:
        '''
        car starts at position 0 and speed +1
        we can either A (acelerate) or R (reverse)
        
        when A
            position += speed
            speed *= 2
        
        when R:
            if speed is positive
                speed = -
            else:
                speed = 1
            
            position stays the same
            
        example
            pos = 0
            speed = 1
            
            first instruction:
            A
            
            pos = 1
            speed = 2
            
            A
            
            pos = 3
            speed = 4
            
            R
            
            pos = 3
            speed = - 1
            
        given target position, return length of shortest sequecne to get there
        
        this is dp
        the transitions are already there
        
        if dp(pos,speed) represents the smallest sequence of A and R to get to pos
        then dp(pos,speed) =  1 + min ({
            if previous was an A:
                dp(pos - speed//2, speed//2)
            if previous was an R
                dp(pos,1)
        })
        '''