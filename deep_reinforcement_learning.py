from game import Game
from model import TicTacToeNet
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

class DeepReinforcementLearning(Game):
    def __init__(self):
        super().__init__()
        self.EMPTY = None
        self.X = "X"
        self.O = "O"
        self.net = TicTacToeNet()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.model_path = "tic_tac_toe/TicTacToeNet.pt"

        # Load model weights if file exists
        if os.path.exists(self.model_path):
            self.net.load_state_dict(torch.load(self.model_path))
            self.train(saved=True)
            self.evaluate()

        else:
            self.train()
            self.evaluate()

    def state_to_tensor(self, state):
        tensor = torch.zeros(9)
        for i in range(3):
            for j in range(3):
                if state[i][j] == self.X:
                    tensor[i * 3 + j] = 1
                elif state[i][j] == self.O:
                    tensor[i * 3 + j] = -1
        return tensor

    def tensor_to_action(self, tensor):
        actions = set()
        for i in range(3):
            for j in range(3):
                if tensor[0, i * 3 + j] > 0:
                    actions.add((i, j))
        return actions

    def deep_reinforcement_learning(self, board):
        state = self.state_to_tensor(board)
        prediction = self.net(state.unsqueeze(0))
        return self.tensor_to_action(prediction)

    def train(self, num_episodes=10000, print_interval=1000, saved=False):
        if saved:
            num_episodes = 1000
            print_interval = 100

        for episode in range(num_episodes):
            # Play a game
            board = self.initial_state()
            player = self.X
            while not self.terminal(board):
                action = self.deep_reinforcement_learning(board)
                board = self.result(board, action)
                player = self.player(board)
            
            # Update the neural network
            target = torch.zeros(9)
            if self.winner(board) == self.X:
                target[action] = 1
            elif self.winner(board) == self.O:
                target[action] = -1
            else:
                target[action] = 0.5
            target = target.unsqueeze(0)
            
            state = self.state_to_tensor(board).unsqueeze(0)
            prediction = self.net(state)
            loss = self.loss_fn(prediction, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Print training progress
            if episode % print_interval == 0:
                print(f"Episode: {episode}/{num_episodes}, Loss: {loss.item()}")

        # Save model weights
        torch.save(self.net.state_dict(), self.model_path)

    def evaluate(self, num_games=100, opponent=None):
        wins = 0
        draws = 0
        losses = 0

        for _ in range(num_games):
            board = self.initial_state()
            player = self.X

            # Alternate between the model and a random player
            while not self.terminal(board):
                if player == self.X:
                    action = self.deep_reinforcement_learning(board)
                else:
                    if opponent:
                        action = opponent(board)
                    else:
                        action = random.choice(list(self.actions(board)))
                
                board = self.result(board, action)
                player = self.player(board)

            if self.winner(board) == self.X:
                wins += 1
            elif self.winner(board) == self.O:
                losses += 1
            else:
                draws += 1

        print(f"Wins: {wins}, Draws: {draws}, Losses: {losses}")

