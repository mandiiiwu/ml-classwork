from copy import copy, deepcopy
import numpy as np
import pygame
import torch
from piece import BODIES, Piece
from board import Board
from random import randint
from genetic_helpers import *

"""
Add your code here 
"""

class NeuralNetwork: 
    def __init__(self, input_size=9, hidden_size=16, weights=None, device='cpu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device 

        if weights is None:
            # xavier initialization for weights
            lim1 = np.sqrt(6 / (input_size + hidden_size))
            lim2 = np.sqrt(6 / (hidden_size + 1))

            self.W1 = torch.FloatTensor(input_size, hidden_size).uniform_(-lim1, lim1).to(device)
            self.W2 = torch.FloatTensor(hidden_size, 1).uniform_(-lim2, lim2).to(device)

            self.b1 = torch.zeros(hidden_size, device=device)
            self.b2 = torch.zeros(1, device=device)

        else:
            # reconstruct from flattened like in typical genetic algs 
            self.set_weights(weights) 
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x).to(self.device)

        h = torch.nn.functional.leaky_relu(torch.matmul(x, self.W1) + self.b1, negative_slope=0.01)
        out = torch.matmul(h, self.W2) + self.b2
        return out.item() 
    
    def get_f_weights(self):
        # flatten weights into one vector 
        return np.concatenate([
            self.W1.cpu().detach().numpy().flatten(),
            self.b1.cpu().detach().numpy().flatten(),
            self.W2.cpu().detach().numpy().flatten(),
            self.b2.cpu().detach().numpy().flatten()
        ])
    
    def set_weights(self, weights):
        idx = 0

        w1_size = self.input_size * self.hidden_size
        self.W1 = torch.FloatTensor(weights[idx:idx+w1_size].reshape(
            self.input_size, self.hidden_size)).to(self.device)
        idx += w1_size

        self.b1 = torch.FloatTensor(weights[idx:idx+self.hidden_size]).to(self.device)
        idx += self.hidden_size

        w2_size = self.hidden_size
        self.W2 = torch.FloatTensor(weights[idx:idx+w2_size].reshape(
            self.hidden_size, 1)).to(self.device)
        idx += w2_size

        self.b2 = torch.FloatTensor(weights[idx:idx+1]).to(self.device)

    def get_weight_count(self):
        return(self.input_size * self.hidden_size + 
               2*(self.hidden_size) + 1)


class CUSTOM_AI_MODEL:
    def __init__(self, genotype=None, input_size=9, hidden_size=16, lr=0.001, momentum=0.9, device='cpu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.momentum = momentum

        self.device = device

        if genotype is not None:
            self.net = NeuralNetwork(self.input_size, self.hidden_size, genotype, device)
        else: self.net = NeuralNetwork(self.input_size, self.hidden_size, device=device)

        self.vel = np.zeros(self.net.get_weight_count())
        self.best_sc = -np.inf 
        self.curr_sc = 0
        self.move_hist = [] 

        self.recent_scs = []
        self.adap_thresh = 5 # adaptive threshold 

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
            'cleared': np.count_nonzero(np.mean(board, axis=1))
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
        
        self.move_hist.append({
            'position': best_x,
            'piece': best_piece,
            'score': max_val
        })
        
        # if len(self.move_hist) >= self.adap_thresh:
        #     self.online_adapt()

        return best_x, best_piece
    
    def online_adapt(self):
        if len(self.move_hist) < 2:
            return
        
        rec_moves = self.move_hist[-self.adap_thresh:]
        scores = [x['score'] for x in rec_moves]

        # if performance is getting worse then try to adapt 
        if len(scores) >= 2:
            trend = np.mean(np.diff(scores))

            if trend < 0: # only adapt if perf is not improving 
                curr_weights = self.net.get_f_weights()
                ptb = np.random.normal(0, 0.01, len(curr_weights)) # perturbation

                self.vel = self.momentum * self.vel + self.lr * ptb

                new_weights = curr_weights + self.vel
                self.net.set_weights(new_weights)
        
    def get_genotype(self):
        return self.net.get_f_weights()
    
    def set_genotype(self, genotype):
        self.net.set_weights(genotype)
        self.vel = np.zeros(len(genotype))

def load_weights(path='src/custom_data/best_weights.pth', device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoint = torch.load(path, map_location=device)

    agent = CUSTOM_AI_MODEL(
        input_size=checkpoint.get('input_size', 9),
        hidden_size=checkpoint.get('hidden_size', 16),
        device=device
    )

    agent.net.W1 = checkpoint['W1'].to(device)
    agent.net.b1 = checkpoint['b1'].to(device)
    agent.net.W2 = checkpoint['W2'].to(device)
    agent.net.b2 = checkpoint['b2'].to(device)

    return agent