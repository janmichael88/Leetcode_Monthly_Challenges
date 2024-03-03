###########################################
# 2864. Maximum Odd Binary Number
# 01MAR24
###########################################
class Solution:
    def maximumOddBinaryNumber(self, s: str) -> str:
        '''
        odds only have 1 set bit at the 0th posiition
        count up all bits, place 1 on the 0th pos, then load the 1 bits on the MSB positions
        '''
        N = len(s)
        count_ones = 0
        for ch in s:
            count_ones += ch == '1'
        
        return '1'*(count_ones-1)+'0'*(N - count_ones)+'1'
    
#################################################
# 3062. Winner of the Linked List Game
# 01MAR24
##################################################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def gameResult(self, head: Optional[ListNode]) -> str:
        '''
        just advance two steps at a time and check
        curr and curr.next
        '''
        odd = 0
        even = 0
        
        curr = head
        while curr:
            if curr.next.val > curr.val:
                odd += 1
            else:
                even += 1
            curr = curr.next.next
        
        if odd > even:
            return 'Odd'
        elif even > odd:
            return 'Even'
        else:
            return 'Tie'
        
#we can just maintain point difference too
        # Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def gameResult(self, head: Optional[ListNode]) -> str:
        '''
        just advance two steps at a time and check
        curr and curr.next
        '''
        score = 0
        
        curr = head
        while curr:
            if curr.next.val > curr.val:
                score += 1
            else:
                score -= 1
            curr = curr.next.next
        
        if score > 0:
            return 'Odd'
        elif score < 0:
            return 'Even'
        else:
            return 'Tie'

#####################################################
# 1474. Delete N Nodes After M Nodes of a Linked List
# 03MAR24
#####################################################
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteNodes(self, head: ListNode, m: int, n: int) -> ListNode:
        curr = head
        prev = head
        
        while curr:
            curr_m = m
            curr_n = n
            while curr and curr_m > 0:
                prev = curr
                curr = curr.next
                curr_m -= 1
            
            while curr and curr_n > 0:
                curr = curr.next
                curr_n -= 1
            
            prev.next = curr
        
        return head