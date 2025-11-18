from copy import copy, deepcopy
import numpy as np
import pygame
import torch
from piece import BODIES, Piece
from board import Board
from random import randint
from genetic_helpers import *

class NeuralNetwork:
    def __init__(self, input_size=10, hidden_size1=32, hidden_size2=16, weights=None, device='cpu'):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.device = device

        if weights is None:
            # initialize weights (xavier initialization from geeks4geeks)
            lim1 = np.sqrt(6 / (input_size + hidden_size1))
            lim2 = np.sqrt(6 / (hidden_size1 + hidden_size2))
            lim3 = np.sqrt(6 / (hidden_size2 + 1))

            self.W1 = torch.FloatTensor(input_size, hidden_size1).uniform_(-lim1, lim1).to(device)
            self.W2 = torch.FloatTensor(hidden_size1, hidden_size2).uniform_(-lim2, lim2).to(device)
            self.W3 = torch.FloatTensor(hidden_size2, 1).uniform_(-lim3, lim3).to(device)

            self.b1 = torch.zeros(hidden_size1, device=device)
            self.b2 = torch.zeros(hidden_size2, device=device)
            self.b3 = torch.zeros(1, device=device)

        else:
            # rebuild weights from array
            self.set_weights(weights) 
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x).to(self.device)

        h1 = torch.nn.functional.leaky_relu(torch.matmul(x, self.W1) + self.b1, negative_slope=0.01)
        h2 = torch.nn.functional.leaky_relu(torch.matmul(h1, self.W2) + self.b2, negative_slope=0.01)
        out = torch.matmul(h2, self.W3) + self.b3
        return out.item() 
    
    def get_f_weights(self):
        # flatten weights into one vector
        return np.concatenate([
            self.W1.cpu().detach().numpy().flatten(),
            self.b1.cpu().detach().numpy().flatten(),
            self.W2.cpu().detach().numpy().flatten(),
            self.b2.cpu().detach().numpy().flatten(),
            self.W3.cpu().detach().numpy().flatten(),
            self.b3.cpu().detach().numpy().flatten()
        ])
    
    def set_weights(self, weights):
        idx = 0

        w1_size = self.input_size * self.hidden_size1
        self.W1 = torch.FloatTensor(weights[idx:idx+w1_size].reshape(
            self.input_size, self.hidden_size1)).to(self.device)
        idx += w1_size

        self.b1 = torch.FloatTensor(weights[idx:idx+self.hidden_size1]).to(self.device)
        idx += self.hidden_size1

        w2_size = self.hidden_size1 * self.hidden_size2
        self.W2 = torch.FloatTensor(weights[idx:idx+w2_size].reshape(
            self.hidden_size1, self.hidden_size2)).to(self.device)
        idx += w2_size

        self.b2 = torch.FloatTensor(weights[idx:idx+self.hidden_size2]).to(self.device)
        idx += self.hidden_size2

        w3_size = self.hidden_size2
        self.W3 = torch.FloatTensor(weights[idx:idx+w3_size].reshape(
            self.hidden_size2, 1)).to(self.device)
        idx += w3_size

        self.b3 = torch.FloatTensor(weights[idx:idx+1]).to(self.device)

    def get_weight_count(self):
        w1 = self.input_size * self.hidden_size1 + self.hidden_size1
        w2 = self.hidden_size1 * self.hidden_size2 + self.hidden_size2
        w3 = self.hidden_size2 + 1
        return w1 + w2 + w3


class CUSTOM_AI_MODEL:
    def __init__(self, genotype=None, input_size=10, hidden_size1=32, hidden_size2=16, device='cpu'):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2

        self.device = device

        if genotype is not None:
            self.net = NeuralNetwork(self.input_size, self.hidden_size1, self.hidden_size2, genotype, device)
        else: self.net = NeuralNetwork(self.input_size, self.hidden_size1, self.hidden_size2, device=device)

    def extract_features(self, board):
        peaks = get_peaks(board)
        h_peaks = np.max(peaks)
        holes = get_holes(peaks, board)
        wells = get_wells(peaks)

        feats = {
            'agg_height': np.sum(peaks),
            'n_holes': np.sum(holes),
            'bumpiness': get_bumpiness(peaks),
            'num_pits': np.count_nonzero(np.count_nonzero(board, axis=0) == 0),
            'max_wells': np.max(wells),
            'n_cols_w_holes': np.count_nonzero(np.array(holes) > 0),
            'row_trans': get_row_transition(board, h_peaks),
            'col_trans': get_col_transition(board, peaks),
            'cleared': np.count_nonzero(np.mean(board, axis=1)),
            'rows_to_clear': np.count_nonzero(np.all(board == 1, axis=1))
        }

        feat_array = np.array(list(feats.values()), dtype=float)
        norm_feats = feat_array / (np.abs(feat_array).max() + 1e-8)
        return norm_feats 
    
    def eval_board(self, board):
        feats = self.extract_features(board)
        score = self.net.forward(feats)
        return score

    def get_best_move(self, board, piece, depth=1):
        best_x = 0
        max_val = -np.inf
        best_piece = piece

        for _ in range(4):
            piece = piece.get_next_rotation() 

            for x in range(board.width):
                try: y = board.drop_height(piece, x)
                except: continue
            
                board_cp = deepcopy(board.board)
                for pos in piece.body: 
                    board_cp[y + pos[1]][x + pos[0]] = True
            
                board_np = bool_to_np(board_cp)
                score = self.eval_board(board_np)

                if score > max_val:
                    max_val = score
                    best_x = x
                    best_piece = piece
        
        return best_x, best_piece
            
    def get_genotype(self):
        return self.net.get_f_weights()
    
    def set_genotype(self, genotype):
        self.net.set_weights(genotype)

def load_weights(path='src/custom_data/best_weights.pth', device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint = torch.load(path, map_location=device)

    agent = CUSTOM_AI_MODEL(
        input_size=checkpoint.get('input_size', 10),
        hidden_size1=checkpoint.get('hidden_size1', 32),
        hidden_size2=checkpoint.get('hidden_size2', 16),
        device=device
    )

    agent.net.W1 = checkpoint['W1'].to(device)
    agent.net.b1 = checkpoint['b1'].to(device)
    agent.net.W2 = checkpoint['W2'].to(device)
    agent.net.b2 = checkpoint['b2'].to(device)
    agent.net.W3 = checkpoint['W3'].to(device)
    agent.net.b3 = checkpoint['b3'].to(device)

    return agent