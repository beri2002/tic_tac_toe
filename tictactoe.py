"""
Tic Tac Toe Player
"""

import math
import copy


X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x = 0
    o = 0

    if terminal(board):
        return X
    
    if all(all(x != EMPTY for x in row) for row in board):
        return X
    
    for i in range(3):
        for j in range(3):
            if board[i][j] == X:
                x += 1
            elif board[i][j] == O:
                o += 1

    if x > o:
        return O
    
    return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    # Initialize an empty set
    actions = set()

    if terminal(board):
        return X

    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                actions.add((i, j))

    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i = action[0]
    j = action[1]

    if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]):
        raise IndexError("Out of bounds move")

    if board[i][j] != EMPTY:
        raise Exception("Invalid move")
    
    # Deep copy of the original table
    copied_board = copy.deepcopy(board)

    copied_board[i][j] = player(board)

    return copied_board

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Check for X wins
    x_wins = check_win(board, X)

    # Check for O wins
    o_wins = check_win(board, O)

    if x_wins:
        return X
    if o_wins:
        return O
    
    else:
        return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if all(all(x != EMPTY for x in row) for row in board):
        return True
    
    state = winner(board)

    if state is not None:
        return True
    
    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    state = winner(board)

    if state == X:
        return 1

    if state == O:
        return -1
    
    return 0

def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    if player(board) == X:
        value = -math.inf
        action = None
        for a in actions(board):
            v = min_value(result(board, a))
            if v > value:
                value = v
                action = a

    if player(board) == O:
        value = math.inf
        action = None
        for a in actions(board):
            v = max_value(result(board, a))
            if v < value:
                value = v
                action = a
    return action

def max_value(board):
    if terminal(board):
        return utility(board)

    v = -math.inf
    for a in actions(board):
        v = max(v, min_value(result(board, a)))
    return v

def min_value(board):
    if terminal(board):
        return utility(board)

    v = math.inf
    for a in actions(board):
        v = min(v, max_value(result(board, a)))
    return v

# Define a function to check for wins
def check_win(board, symbol):
    # Check horizontal and vertical wins
    for i in range(3):
        if all(board[i][j] == symbol for j in range(3)) or all(board[j][i] == symbol for j in range(3)):
            return True
    # Check diagonal wins
    if all(board[i][i] == symbol for i in range(3)) or all(board[i][2 - i] == symbol for i in range(3)):
        return True
    return False