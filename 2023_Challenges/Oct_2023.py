####################################################################
# 2038. Remove Colored Pieces if Both Neighbors are the Same Color
# 02OCT23
#####################################################################
class Solution:
    def winnerOfGame(self, colors: str) -> bool:
        '''
        alice can only remove A pieces, if an A is surrounded by two other As and is not on the edge
        same thing for Bob, but with B pieces
        if a player cannot make a move on their turn, they lose and the other player wins
        in order for Alice to win, she has to have more moves than Bob
        count the number of moves and check if Alice has more moves than Bob
        
        count streaks of As and Bs and calculate the number of moves
        '''
        def countMoves(colors, char):
            moves = 0
            curr_streak = 0
            for c in colors:
                if c == char:
                    curr_streak += 1
                else:
                    if curr_streak > 2:
                        moves += curr_streak - 2
                    curr_streak = 0
            if curr_streak > 2:
                moves += curr_streak - 2
                curr_streak = 0
            return moves
        
        
        aliceMoves = countMoves(colors,'A')
        bobMoves = countMoves(colors, 'B')
        print(aliceMoves,bobMoves)
        return aliceMoves > bobMoves
    
class Solution:
    def winnerOfGame(self, colors: str) -> bool:
        '''
        just a review on two key intutions
        1. when one player removes a letter, it wil never create a new removal oppuruntiy for another player
            i.e deleting an A or B on alice or bob's turn doesn't give the opposing player any advantage
        2. the order in which removals happens is irrelevant
            if we have AAAAA, removing any of the middle As just brings it to one less A
        
        
        '''
        A,B,N = 0,0,len(colors)
        #dont go to the edges
        for i in range(1,N-1):
            if colors[i-1] == colors[i] == colors[i+1]:
                if colors[i] == 'A':
                    A += 1
                else:
                    B += 1
        
        return A > B