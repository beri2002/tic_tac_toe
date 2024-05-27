from game import Game
import random

class ReinforcementLearning(Game):
    def __init__(self):
        self.EMPTY = None
        # self.board = self.initial_state()
        self.X = "X"
        self.O = "O"
        self.Q_values = {}
        
        # Define hyperparameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # Epsilon for epsilon-greedy strategy
        
        # rain the agent
        self.train()
        
        win_rate = self.evaluate()
        print("Win rate against random player:", win_rate)

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

    def train(self, num_episodes=10000):
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

    def evaluate(self, num_episodes=1000, opponent=None):
        """
        Evaluate the performance of the agent by playing against an opponent for a specified number of episodes.

        Args:
            num_episodes (int): Number of evaluation episodes.
            opponent (function): Function that returns an action given the current board state. If None, a random player is used.

        Returns:
            float: Win rate of the agent.
        """
        wins = 0

        for _ in range(num_episodes):
            board = self.initial_state()
            done = False
            turn = self.X

            while not done:
                if turn == self.X:
                    action = self.reinforcement_learning(board)
                else:
                    if opponent:
                        action = opponent(board)
                    else:
                        action = random.choice(list(self.actions(board)))

                next_state = self.result(board, action)
                done = self.terminal(next_state)

                if done:
                    winner = self.winner(next_state)
                    if winner == self.X:
                        wins += 1
                    break

                board = next_state
                turn = self.player(board)

        return wins / num_episodes


    def reward(self, board):
        winner = self.winner(board)
        if winner == self.X:
            return 1
        elif winner == self.O:
            return -1
        else:
            return 0