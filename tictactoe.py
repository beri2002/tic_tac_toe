"""
Tic Tac Toe Player
"""

import math
import copy
import random

# Step 1: Instantiate the TicTacToe class
tic_tac_toe_agent = TicTacToe()

# Step 2: Train the agent
tic_tac_toe_agent.train(num_episodes=1000)  # Train for 1000 episodes

# Step 3: Use the trained agent
board = tic_tac_toe_agent.initial_state()
while not tic_tac_toe_agent.terminal(board):
    # Display the current state of the board
    print("Current board state:")
    for row in board:
        print(" ".join(cell if cell else "-" for cell in row))
    
    # Get the agent's action
    action = tic_tac_toe_agent.reinforcement_learning(board)
    
    # Apply the action to the board
    board = tic_tac_toe_agent.result(board, action)

# Display the final state of the board
print("Final board state:")
for row in board:
    print(" ".join(cell if cell else "-" for cell in row))

# Check the winner of the game
winner = tic_tac_toe_agent.winner(board)
if winner:
    print("Winner:", winner)
else:
    print("It's a draw!")


class TicTacToe:
    def __init__(self, classic=True, reinforcement=False, deep_reinforcement=False):
        self.board = self.initial_state()
        self.X = "X"
        self.O = "O"
        self.EMPTY = None

        self.classic = classic
        self.reinforcement = reinforcement
        self.deep_reinforcement = deep_reinforcement

        self.Q_values = {}

        # Define hyperparameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # Epsilon for epsilon-greedy strategy


    def algorithm(self):
        """
        Returns the appropriate algorithm based on the current game mode.
        """
        if self.classic:
            return self.minimax
        if self.reinforcement:
            return self.reinforcement_learning
        if self.deep_reinforcement:
            return self.deep_reinforcement_learning
        
    def reinforcement_learning(self, board):
        # Convert the board to a hashable tuple for dictionary lookup
        state = tuple(map(tuple, board))

        if self.terminal(board):
            return None

        if random.random() < self.epsilon:
            # Explore: choose a random action
            return random.choice(list(self.actions(board)))

        # Exploit: choose the action with the highest Q-value
        action_values = {}
        for action in self.actions(board):
            next_state = self.result(board, action)
            action_values[action] = self.Q_values.get(tuple(map(tuple, next_state)), 0)

        # Select the action with the highest Q-value
        best_action = max(action_values, key=action_values.get)
        return best_action
    
    def update_Q_values(self, state, action, reward, next_state):
        # Convert states to hashable tuples for dictionary lookup
        state = tuple(map(tuple, state))
        next_state = tuple(map(tuple, next_state))

        # Update Q-value using Q-learning update rule
        old_Q = self.Q_values.get((state, action), 0)
        max_next_Q = max([self.Q_values.get((next_state, a), 0) for a in self.actions(next_state)], default=0)
        new_Q = old_Q + self.learning_rate * (reward + self.discount_factor * max_next_Q - old_Q)
        self.Q_values[(state, action)] = new_Q

    def train(self, num_episodes=1000):
        for _ in range(num_episodes):
            # Reset the environment
            board = self.initial_state()
            done = False

            while not done:
                # Player X's turn
                action = self.reinforcement_learning(board)
                next_state = self.result(board, action)
                reward = self.reward(next_state)
                self.update_Q_values(board, action, reward, next_state)
                board = next_state
                done = self.terminal(board)

    def reward(self, board):
        winner = self.winner(board)
        if winner == self.X:
            return 1
        elif winner == self.O:
            return -1
        else:
            return 0

    def deep_reinforcement_learning(self, board):
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