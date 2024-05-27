from game import Game
import math
class MiniMax(Game):

    def __init__(self):
        self.EMPTY = None
        # self.board = self.initial_state()
        self.X = "X"
        self.O = "O"

    def minimax(self, board):
        """
        Returns the optimal action for the current player on the board.
        """
        if self.terminal(board):
            return None

        if self.player(board) == self.X:
            value = -math.inf
            action = None
            for a in self.actions(board):
                v = self.min_value(self.result(board, a))
                if v > value:
                    value = v
                    action = a

        if self.player(board) == self.O:
            value = math.inf
            action = None
            for a in self.actions(board):
                v = self.max_value(self.result(board, a))
                if v < value:
                    value = v
                    action = a
        return action

    def max_value(self, board):
        if self.terminal(board):
            return self.utility(board)

        v = -math.inf
        for a in self.actions(board):
            v = max(v, self.min_value(self.result(board, a)))
        return v

    def min_value(self, board):
        if self.terminal(board):
            return self.utility(board)

        v = math.inf
        for a in self.actions(board):
            v = min(v, self.max_value(self.result(board, a)))
        return v