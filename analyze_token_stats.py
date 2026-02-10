import os
import ast

from typing import List
import numpy as np
import pandas as pd

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'

def build_event_matrix(df: pd.DataFrame):
    event_matrix : List[List[int]] = []

    for i in range(len(df)):
        row = []
        attn_mask = df.at[i, 'attn_mask']
        revived_tokens = df.at[i, 'revived_tokens']

        for a, r in zip(attn_mask, revived_tokens):
            if not a and not r:
                row.append(0)  # Sigue vivo
            elif a and not r:
                row.append(1)  # Muerto y no revive
            elif not a and r:
                row.append(2)  # Caso imposible
            elif a and r:
                row.append(3)  # Revivido

        event_matrix.append(row)

    return event_matrix, len(event_matrix), len(event_matrix[0])

# def plot_event_matrix(event_matrix: List[List[int]], n_blocks: int, n_tokens: int, out_path: str):
#     # Crear un mapa de colores
#     cmap = ListedColormap(["#FFFFFF", "#FFAAAA", "#AAFFAA"])

#     # Crear la figura y el eje
#     fig, ax = plt.subplots(figsize=(n_tokens, n_blocks))

#     # Establecer explícitamente los límites de los ejes para asegurar que todos los parches se vean
#     ax.set_xlim(0, n_tokens)
#     ax.set_ylim(0, n_blocks)

#     # Dibujar la matriz de eventos
#     for i in range(n_blocks):
#         for j in range(n_tokens):
#             ax.add_patch(plt.Rectangle((j, i), 1, 1, color=cmap(event_matrix[i][j])))

#     # Configurar los ejes
#     ax.grid(False)
#     ax.set_xticks(np.arange(n_tokens) + 0.5)
#     ax.set_yticks(np.arange(n_blocks) + 0.5)
#     ax.set_xticklabels(np.arange(1, n_tokens + 1))
#     ax.set_yticklabels(np.arange(1, n_blocks + 1))
#     ax.set_xlabel("Tokens")
#     ax.set_ylabel("Block")
#     # ax.set_title("Matriz de Eventos")

#     # 2. Dibujar las líneas verticales manualmente
#     # Se dibujará una línea desde x=grid_x_start, x=1+grid_x_start, etc.
#     vertical_lines = np.arange(n_tokens) + 0.0
#     ax.vlines(vertical_lines, ymin=0, ymax=n_blocks, color='#D3D3D3', linestyle='-', linewidth=1)

#     # 3. Dibujar las líneas horizontales manualmente
#     # Se dibujará una línea desde y=grid_y_start, y=1+grid_y_start, etc.
#     horizontal_lines = np.arange(n_blocks) + 0.0
#     ax.hlines(horizontal_lines, xmin=0, xmax=n_tokens, color='#D3D3D3', linestyle='-', linewidth=1)

#     # Añadir leyenda
#     legend_elements = [
#         Patch(color="#FFFFFF", label="Alive"),
#         Patch(color="#FFAAAA", label="Dead"),
#         # Patch(color="#000000", label="Impossible case"),
#         Patch(color="#7BEB7B", label="Revived"),
#     ]
#     ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)

#     # Guardar la figura
#     plt.savefig(out_path, bbox_inches="tight")
#     plt.close(fig)

def plot_event_matrix(event_matrix: List[List[int]], n_blocks: int, n_tokens: int, out_path: str):
    """
    Dibuja una matriz de eventos, dividiéndola en dos partes si hay muchos tokens.
    Genera dos figuras: out_path_part1.png y out_path_part2.png (u otro formato).
    """
    # Crear mapa de colores
    cmap = ListedColormap(["#FFFFFF", "#FFAAAA", "#AAFFAA"])

    # Calcular mitad para partir la matriz
    mitad = n_tokens // 2
    partes = [
        (0, mitad, "part1"),
        (mitad, n_tokens, "part2")
    ]

    # Crear ambas figuras
    for inicio, fin, sufijo in partes:
        sub_matrix = [fila[inicio:fin] for fila in event_matrix]
        n_subtokens = fin - inicio

        fig, ax = plt.subplots(figsize=(n_subtokens * 0.5, n_blocks * 0.5))  # escala visual más razonable

        # Dibujar celdas
        for i in range(n_blocks):
            for j in range(n_subtokens):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=cmap(sub_matrix[i][j])))

        # Configurar ejes
        ax.set_xlim(0, n_subtokens)
        ax.set_ylim(0, n_blocks)
        ax.grid(False)
        ax.set_xticks(np.arange(n_subtokens) + 0.5)
        ax.set_yticks(np.arange(n_blocks) + 0.5)
        ax.set_xticklabels(np.arange(inicio + 1, fin + 1))
        ax.set_yticklabels(np.arange(1, n_blocks + 1))
        ax.set_xlabel("Tokens")
        ax.set_ylabel("Block")

        # Dibujar líneas de cuadrícula
        ax.vlines(np.arange(n_subtokens), ymin=0, ymax=n_blocks, color='#D3D3D3', linewidth=1)
        ax.hlines(np.arange(n_blocks), xmin=0, xmax=n_subtokens, color='#D3D3D3', linewidth=1)

        # Leyenda
        legend_elements = [
            Patch(color="#FFFFFF", label="Alive"),
            Patch(color="#FFAAAA", label="Pruned"),
            Patch(color="#7BEB7B", label="Revived"),
        ]
        ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)

        # Guardar con nombre diferenciado
        base, ext = os.path.splitext(out_path)
        out_file = f"{base}_{sufijo}{ext}"
        plt.savefig(out_file, bbox_inches="tight")
        plt.close(fig)

        print(f"✅ Figura guardada: {out_file}")

def main():
    # Define paths
    experiment_id = 0
    csv_path = f"token_stats_{experiment_id}.csv"
    out_path = f"out_{experiment_id}.pdf"

    # Crear dataframe
    df = pd.read_csv(csv_path)
    df['revived_tokens'] = df['revived_tokens'].apply(ast.literal_eval)
    df['attn_mask'] = df['attn_mask'].apply(ast.literal_eval)

    # Construir matriz de eventos
    events, n_blocks, n_tokens = build_event_matrix(df)
    plot_event_matrix(events, n_blocks, n_tokens, out_path)

if __name__ == "__main__":
        main()