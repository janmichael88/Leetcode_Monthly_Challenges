################################
#Number of Recent Calls 10/01/20
################################
#TLE exceed
class RecentCounter(object):

    def __init__(self):
        #store pings
        self.all_pings = []
        

    def ping(self, t):
        """
        :type t: int
        :rtype: int
        """
        #first append
        self.all_pings.append(t)
        #get the range
        start,end = t-3000,t
        #return call
        in_range = 0
        #traverse pings
        for ping in self.all_pings:
            if start <= ping <=end:
                in_range += 1
        return in_range

from collections import deque
class RecentCounter(object):

    def __init__(self):
        #store pings.using q, add right, popleft
        self.all_pings = deque()
        self.count = 0
        

    def ping(self, t):
        """
        :type t: int
        :rtype: int
        """
        #use q, and keep popping off pings that aren't in the arange
        self.all_pings.append(t)
        self.count += 1
        while self.all_pings and self.all_pings[0] < t - 3000:
            self.all_pings.popleft()
            self.count -= 1
        return self.count
        

