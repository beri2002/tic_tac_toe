import copy

class Game():
    def __init__(self):
        self.EMPTY = None
        self.board = self.initial_state()
        self.X = "X"
        self.O = "O"

    def initial_state(self):
        """
        Returns starting state of the board.
        """
        return [[self.EMPTY, self.EMPTY, self.EMPTY],
                [self.EMPTY, self.EMPTY, self.EMPTY],
                [self.EMPTY, self.EMPTY, self.EMPTY]]


    def player(self, board):
        """
        Returns player who has the next turn on a board.
        """
        x = 0
        o = 0

        if self.terminal(board):
            return self.X
        
        if all(all(x != self.EMPTY for x in row) for row in board):
            return self.X
        
        for i in range(3):
            for j in range(3):
                if board[i][j] == self.X:
                    x += 1
                elif board[i][j] == self.O:
                    o += 1

        if x > o:
            return self.O
        
        return self.X


    def actions(self,board):
        """
        Returns set of all possible actions (i, j) available on the board.
        """
        # Initialize an empty set
        actions = set()

        if self.terminal(board):
            return self.X

        for i in range(3):
            for j in range(3):
                if board[i][j] == self.EMPTY:
                    actions.add((i, j))

        return actions


    def result(self, board, action):
        """
        Returns the board that results from making move (i, j) on the board.
        """
        i = action[0]
        j = action[1]

        if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]):
            raise IndexError("Out of bounds move")

        if board[i][j] != self.EMPTY:
            raise Exception("Invalid move")
        
        # Deep copy of the original table
        copied_board = copy.deepcopy(board)

        copied_board[i][j] = self.player(board)

        return copied_board

    def winner(self, board):
        """
        Returns the winner of the game, if there is one.
        """
        # Check for X wins
        x_wins = self.check_win(board, self.X)

        # Check for O wins
        o_wins = self.check_win(board, self.O)

        if x_wins:
            return self.X
        if o_wins:
            return self.O
        
        else:
            return None


    def terminal(self, board):
        """
        Returns True if game is over, False otherwise.
        """
        if all(all(x != self.EMPTY for x in row) for row in board):
            return True
        
        state = self.winner(board)

        if state is not None:
            return True
        
        return False


    def utility(self, board):
        """
        Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
        """
        state = self.winner(board)

        if state == self.X:
            return 1

        if state == self.O:
            return -1
        
        return 0

    # Define a function to check for wins
    def check_win(self, board, symbol):
        # Check horizontal and vertical wins
        for i in range(3):
            if all(board[i][j] == symbol for j in range(3)) or all(board[j][i] == symbol for j in range(3)):
                return True
        # Check diagonal wins
        if all(board[i][i] == symbol for i in range(3)) or all(board[i][2 - i] == symbol for i in range(3)):
            return True
        return False

