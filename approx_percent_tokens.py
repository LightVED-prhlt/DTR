import numpy as np
import pandas as pd
from termcolor import colored

def get_tokens_per_combination(pruning_ratio, revive_ratio):
    N_TOKENS_INIT = 197
    n_tokens_alive = N_TOKENS_INIT
    n_tokens_dead = 0
    tokens_per_block = []
    for _ in range(12):
        # Pruning
        n_tokens_alive = int(((1 - pruning_ratio) * n_tokens_alive).round())
        n_tokens_dead = N_TOKENS_INIT - n_tokens_alive

        # Revival
        n_tokens_alive += int((revive_ratio * n_tokens_dead).round())
        n_tokens_dead = N_TOKENS_INIT - n_tokens_alive

        tokens_per_block.append(int(n_tokens_alive))
    return tokens_per_block

def print_colored_df(df, target, margin):
    columns = df.columns.tolist()
    columns = [f"{col:.1f}" for col in columns]

    print('', *columns, sep="\t", end="\n")
    for pruning_ratio in df.index:
        print(f"{pruning_ratio:.1f}\t", end="")
        for revive_ratio in df.columns:
            value = df.at[pruning_ratio, revive_ratio]
            if value >= target - margin and value <= target + margin:
                color = 'green'
            else:
                color = 'red'
            print(colored(f"{value:.4f}", color), end="\t")
        print()

df = pd.DataFrame(index=np.arange(0., 1.01, 0.1), columns=np.arange(0., 1.01, 0.1))

for pruning_ratio in np.arange(0., 1.01, 0.1):
    for revive_ratio in np.arange(0., 1.01, 0.1):
        n_tokens = get_tokens_per_combination(pruning_ratio, revive_ratio)
        n_tokens_mean = np.mean(n_tokens)
        n_tokens_percentage = n_tokens_mean / 197.
        df.at[pruning_ratio, revive_ratio] = n_tokens_percentage

print_colored_df(df, target=0.6345, margin=0.03)