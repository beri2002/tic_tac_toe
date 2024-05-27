"""
Tic Tac Toe Player
"""
from reinforcement_learning import ReinforcementLearning
from deep_reinforcement_learning import DeepReinforcementLearning
from minimax import MiniMax
from game import Game

class TicTacToe(Game):
    def __init__(self, classic=True, reinforcement=False, deep_reinforcement=False):
        self.EMPTY = None
        # self.board = self.initial_state()
        self.X = "X"
        self.O = "O"
        
        self.classic = classic
        self.reinforcement = reinforcement
        self.deep_reinforcement = deep_reinforcement

    def algorithm(self, board):
        """
        Returns the appropriate algorithm based on the current game mode.
        """
        if self.classic:
            ttt = MiniMax()
            return ttt.minimax(board)
        if self.reinforcement:
            ttt = ReinforcementLearning()
            return ttt.reinforcement_learning(board)
        if self.deep_reinforcement:
            ttt = DeepReinforcementLearning()
            return ttt.deep_reinforcement_learning(board)
