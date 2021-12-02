##########################
# 198. House Robber
# 01DEC21
##########################
#top down recursion with memozation
class Solution:
    def rob(self, nums: List[int]) -> int:
        '''
        this is just a 0/1 knapsack
        we can either chose to rob this house  and take its value
        or skip this house
        dp(i) represents the amount of profit i have atfter robbing house i
        need to hold if we robbed or not
        rather dp(i) gives the max profit i can get if start robbing at house i
        
        take = dp(i+2) + nums[i]
        no_take = dp(i+1) then advance 
        dp(i) = max(take,notake)
        '''
        memo = {}
        N = len(nums)
        def dp(i):
            if i >= N:
                return 0 #no profit
            if i in memo:
                return memo[i]
            
            rob = nums[i] + dp(i+2)
            no_rob = dp(i+1)
            ans = max(rob,no_rob)
            memo[i] = ans
            return ans
        
        return dp(0)

#bottom up dp
class Solution:
    def rob(self, nums: List[int]) -> int:
        '''
        to get dp(0)
        we need dp(1)
        to get dp(1) we need dp(2)
        so start from the end of the array
        '''
        if not nums:
            return 0
        
        N = len(nums)
        dp = [0]*(N+2)
        #edge case for zero beyond 2
        
        for i in range(N-1,-1,-1):
            rob = nums[i] + dp[i+2]
            no_rob = dp[i+1]
            dp[i] = max(rob,no_rob)
            
        return dp[0]
        
#bottom up constant space
class Solution:
    def rob(self, nums: List[int]) -> int:
        '''
        to get dp(0)
        we need dp(1)
        to get dp(1) we need dp(2)
        so start from the end of the array
        '''
        # Special handling for empty case.
        if not nums:
            return 0
        
        N = len(nums)
        
        rob_next_plus_one = 0
        rob_next = nums[N - 1]
        
        # DP table calculations.
        for i in range(N - 2, -1, -1):
            
            # Same as recursive solution.
            current = max(rob_next, rob_next_plus_one + nums[i])
            
            # Update the variables
            rob_next_plus_one = rob_next
            rob_next = current
            
        return rob_next

###################################
# 328. Odd Even Linked List
# 02DEC21
###################################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        i can keep two pointers
        one odd and one even
        if i odds on one side and evens on one side
        i would want to make odd.next = even.next
        then just move the odd and even pointers
        
        at the end, connect odds tail to evens head
        
        to maintain the invariant, we make sure even and its next exists
        '''
        if not head:
            return None
        odd = head
        even = head.next
        evenHead = even
        
        while even and even.next:
            odd.next = even.next
            odd = odd.next
            even.next = odd.next
            even = even.next
        #tails
        odd.next = evenHead
        return head

        
##################################################
# 708. Insert into a Sorted Circular Linked List
# 01DEC21
##################################################
#close one, but you pretty much had it on an interview
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, next=None):
        self.val = val
        self.next = next
"""

class Solution:
    def insert(self, head: 'Node', insertVal: int) -> 'Node':
        '''
        i can find the break in the circular list
        before finding the break, pull all values into a list
        find place where insert val can be inserted, then put it there
        rebuild
        '''
        if not head:
            dummy = Node(val = insertVal)
            dummy.next = dummy
            return dummy
        seen = set()
        values = []
        
        curr = head
        while curr not in seen:
            values.append(curr.val)
            seen.add(curr)
            curr = curr.next
            
        #now scan values to find where to match, find the left bound
        left_bound = len(values)
        for i in range(len(values)-1):
            if values[i] <= insertVal <= values[i+1]:
                left_bound = i+1
                break
                
        lower_half = values[:left_bound]
        lower_half.append(insertVal)
        values = lower_half + values[left_bound:]
        
        #now build a new one and return it
        dummy = Node()
        temp = dummy
        for num in values:
            temp.next = Node(val = num)
            temp = temp.next
        print(values)
        print(dummy.next.val)
        print(temp.val)
        #reconnect
        temp.next = dummy.next
        return dummy.next


