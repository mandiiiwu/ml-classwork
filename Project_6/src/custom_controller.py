import numpy as np
from game import Game
from custom_model import CUSTOM_AI_MODEL
import random
import pandas as pd
import os

def compute_fitness(agent, num_trials=3):
    fitness_scores = []

    for trial in range(num_trials):
        game = Game('student', agent=agent)
        p_dropped, rows_cleared = game.run_no_visual()

        fitness = p_dropped + (rows_cleared * 2.0)
        fitness_scores.append(fitness)

        print(f"      trial {trial+1}/{num_trials}: pieces={p_dropped}, rows={rows_cleared}, fitness={fitness:.2f}")

    avg_fitness = np.mean(fitness_scores)
    return avg_fitness


def crossover(parent1, parent2):
    ws1 = parent1.get_genotype()
    ws2 = parent2.get_genotype()

    i = random.randint(1, len(ws1) - 1)

    child_ws = np.concatenate([
        ws1[:i],
        ws2[i:]
    ])

    return CUSTOM_AI_MODEL(genotype=child_ws)


def mutate(agent, mutation_rate=0.15, mutation_scale=0.3):
    # apply random mutations

    weights = agent.get_genotype()
    mask = np.random.random(len(weights)) < mutation_rate

    mutations = np.random.normal(0, mutation_scale, len(weights))
    weights[mask] += mutations[mask]

    agent.set_genotype(weights)

    return agent


def train(num_gens, pop_size, num_trials, num_elite, surv_rate, log_file):
    print('=' * 70)
    print('TRAINING NEURAL NETWORK')
    print('=' * 70)
    print(f"generations: {num_gens}")
    print(f"pop. size: {pop_size}")
    print(f"trials per agent: {num_trials}")
    print(f"elite agents: {num_elite}")
    print(f"surv rate: {surv_rate}")
    print('=' * 70)

    headers = ['generation', 'best_fitness', 'avg_fitness', 'elite_fitness',
               'best_pieces', 'best_rows']
    data_log = []

    population = [CUSTOM_AI_MODEL() for _ in range(pop_size)]

    # track best agent
    bestever_fit = -np.inf
    bestever_weights = None
    bestever_pieces = 0
    bestever_rows = 0

    for gen in range(num_gens):
        print('\n'); print('=' * 70)
        print(f"GENERATION {gen+1}/{num_gens}")
        print('=' * 70)

        fit_data = []
        for i, agent in enumerate(population):
            fitness = compute_fitness(agent, num_trials)
            fit_data.append((fitness, agent, i))
            print(f"    agent {i+1} average fitness: {fitness:.2f}")

        fit_data.sort(reverse=True, key=lambda x: x[0])

        best_fit = fit_data[0][0]
        avg_fit = np.mean([f for f, _, _ in fit_data])
        elite_fit = np.mean([f for f, _, _ in fit_data[:num_elite]])

        print(f"\testing best agent of gen {gen+1}...")
        best_agent = fit_data[0][1]
        test_game = Game('student', agent=best_agent)
        best_pieces, best_rows = test_game.run_no_visual()

        if best_fit > bestever_fit:
            bestever_fit = best_fit
            bestever_weights = best_agent.get_genotype()
            bestever_pieces = best_pieces
            bestever_rows = best_rows

            save_path = 'src/custom_data/best_weights.npy'
            os.makedirs('src/custom_data', exist_ok=True)
            np.save(save_path, bestever_weights)
            print(f"good model!!! saved to {save_path}")

        print('=' * 70)
        print(f"gen {gen+1} summary:")
        print(f"  best fitness:    {best_fit:.2f} (pieces={best_pieces}, rows={best_rows})")
        print(f"  avg fitness: {avg_fit:.2f}")
        print(f"  elite fitness:   {elite_fit:.2f}")
        print(f"  best ever:       {bestever_fit:.2f} (pieces={bestever_pieces}, rows={bestever_rows})")
        print('=' * 70)

        data_log.append([gen+1, best_fit, avg_fit, elite_fit, best_pieces, best_rows])

        if (gen + 1) % 5 == 0 or gen == num_gens - 1:
            df = pd.DataFrame(data_log, columns=headers)
            os.makedirs('src/custom_data', exist_ok=True)
            df.to_csv(f'src/custom_data/{log_file}', index=False)
            print(f"progress saved to src/custom_data/{log_file}!")

        if gen < num_gens - 1:
            next_gen = []

            for i in range(num_elite):
                elite_weights = fit_data[i][1].get_genotype()
                elite_agent = CUSTOM_AI_MODEL(genotype=elite_weights.copy())
                next_gen.append(elite_agent)

            num_parents = max(2, int(pop_size * surv_rate))
            parents = [agent for _, agent, _ in fit_data[:num_parents]]

            while len(next_gen) < pop_size:
                parent1, parent2 = random.sample(parents, 2)

                child = crossover(parent1, parent2)
                child = mutate(child, mutation_rate=0.15, mutation_scale=0.3)

                next_gen.append(child)

            population = next_gen

    print('\n')
    print('*' * 70)
    print("training is complete!!!")
    print('*' * 70)
    print(f"best fitness: {bestever_fit:.2f}")
    print(f"best pieces: {bestever_pieces}")
    print(f"best rows: {bestever_rows}")
    print(f"best weights saved to: src/custom_data/best_weights.npy")
    print(f"training log saved to: src/custom_data/{log_file}")
    print('*' * 70)

    return bestever_weights, data_log


if __name__ == '__main__':
    best_weights, log = train(
        num_gens=30,            # rec: 30-100 
        pop_size=20,            # rec: 20-50
        num_trials=3,           # games per fitness eval
        num_elite=3,            # top n agents to keep
        surv_rate=0.3, 
        log_file='custom_trial1.csv'
    )