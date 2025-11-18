from custom_model import CUSTOM_AI_MODEL
import numpy as np
import torch
from game import Game
import time

def inference():
    checkpoint = torch.load('src/custom_data/best_weights.pth')

    agent = CUSTOM_AI_MODEL(
        input_size=checkpoint.get('input_size', 9),
        hidden_size=checkpoint.get('hidden_size', 16),
        device='cpu'
    )

    agent.net.W1 = checkpoint['W1']
    agent.net.b1 = checkpoint['b1']
    agent.net.W2 = checkpoint['W2']
    agent.net.b2 = checkpoint['b2']
    
    game = Game('student', agent=agent)
    p_dropped, r_cleared = game.run()

    print(f'rows cleared: {r_cleared}, pieces dropped: {p_dropped}')

inference()