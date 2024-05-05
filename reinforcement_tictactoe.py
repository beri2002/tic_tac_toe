import numpy as np
from sklearn.linear_model import SGDRegressor
from collections import defaultdict

class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9
        self.current_player = 'X'

    def reset(self):
        self.board = [' '] * 9
        self.current_player = 'X'

    def is_winner(self, player):
        winning_combos = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]              # diagonals
        ]
        for combo in winning_combos:
            if all(self.board[i] == player for i in combo):
                return True
        return False

    def is_draw(self):
        return ' ' not in self.board

    def is_game_over(self):
        return self.is_winner('X') or self.is_winner('O') or self.is_draw()

    def get_valid_moves(self):
        return [i for i in range(9) if self.board[i] == ' ']

    def make_move(self, move):
        self.board[move] = self.current_player
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def get_state(self):
        return tuple(self.board)

class RLAgent:
    def __init__(self, epsilon=0.1, alpha=0.01, gamma=0.9):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_values = defaultdict(lambda: defaultdict(int))

    def choose_action(self, state, valid_moves):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(valid_moves)
        else:
            q_values = [self.q_values[state][move] for move in valid_moves]
            return valid_moves[np.argmax(q_values)]

    def update_q_values(self, state, action, reward, next_state):
        max_next_q = max(self.q_values[next_state].values()) if next_state in self.q_values else 0
        self.q_values[state][action] += self.alpha * (reward + self.gamma * max_next_q - self.q_values[state][action])

def play_game(env, agent):
    state_history = []
    while not env.is_game_over():
        current_state = env.get_state()
        valid_moves = env.get_valid_moves()
        action = agent.choose_action(current_state, valid_moves)
        state_history.append((current_state, action))
        env.make_move(action)

    return state_history

def evaluate_win_rate(agent, opponent, num_games=100):
    agent_wins = 0
    opponent_wins = 0

    for _ in range(num_games):
        env = TicTacToe()  # Assuming TicTacToe environment is used
        current_player = 'X'  # Assuming agent plays as 'X' and opponent plays as 'O'

        while not env.is_game_over():
            if current_player == 'X':
                state = env.get_state()
                action = agent.choose_action(state, env.get_valid_moves())
                env.make_move(action)
                current_player = 'O'
            else:
                state = env.get_state()
                action = opponent.choose_action(state, env.get_valid_moves())
                env.make_move(action)
                current_player = 'X'

        if env.is_winner('X'):  # Agent wins
            agent_wins += 1
        elif env.is_winner('O'):  # Opponent wins
            opponent_wins += 1

    agent_win_rate = agent_wins / num_games * 100
    opponent_win_rate = opponent_wins / num_games * 100

    print(f"Agent win rate: {agent_win_rate:.2f}%")
    print(f"Opponent win rate: {opponent_win_rate:.2f}%")

# Training Loop
epsilon = 0.1
alpha = 0.01
gamma = 0.9
agent = RLAgent(epsilon, alpha, gamma)
opponent = RLAgent(epsilon, alpha, gamma)
num_episodes = 10000

for episode in range(num_episodes):
    env = TicTacToe()
    state_history = play_game(env, agent)
    # Update Q-values based on game outcome
    winner = 'X' if env.is_winner('X') else 'O' if env.is_winner('O') else None
    if winner:
        reward = 1 if winner == 'X' else -1
    else:
        reward = 0

    for state, action in state_history:
        next_state = env.get_state()
        agent.update_q_values(state, action, reward, next_state)
    board = [[' ' for _ in range(2)] for _ in range(2)]
    running = True
    turn = 'X'

evaluate_win_rate(agent, opponent, num_games=1000)
