import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import deque

# To LaTeX
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "font.family": "serif",
    "pgf.rcfonts": False,
})

FONTSIZE = 16

# plt.rcParams.update({
#     # "font.size": 16,
#     "axes.labelsize": 28,
#     "xtick.labelsize": 22,
#     "ytick.labelsize": 22,
#     "legend.fontsize": 28,
#     "pdf.fonttype": 42,
#     "ps.fonttype": 42,
# })

###############
### FIGURE 1: CURVAS DE PRUNING SCHEDULES
###############

# Datos simulados para 3 Keep Ratios (0.5, 0.6, 0.7) y 3 Schedulers
# Estructura: ratios[keep_ratio][scheduler] = list_of_12_tokens
data = {
    0.5: {"Stepwise": [196, 196, 196, 98, 98, 98, 49, 49, 49, 25, 25, 25], 
          "Linear": [196, 180, 165, 149, 134, 118, 103, 87, 72, 56, 41, 25], 
          "Exponential": [196, 163, 135, 112, 93, 77, 64, 53, 44, 36, 30, 25]},
    0.6: {"Stepwise": [196, 196, 196, 118, 118, 118, 71, 71, 71, 43, 43, 43], 
          "Linear": [196, 182, 168, 154, 140, 126, 113, 99, 85, 71, 57, 43], 
          "Exponential": [196, 171, 149, 130, 113, 98, 86, 75, 65, 57, 49, 43]},
    0.7: {"Stepwise": [196, 196, 196, 137, 137, 137, 96, 96, 96, 67, 67, 67], 
          "Linear": [196, 184, 173, 161, 149, 137, 126, 114, 102, 90, 79, 67], 
          "Exponential": [196, 178, 161, 146, 133, 120, 109, 99, 90, 81, 74, 67]}
}

keep_ratios = [0.5, 0.6, 0.7]
schedulers = ["Stepwise", "Linear", "Exponential"]

# Dibujar todas las curvas en la misma figura
fig, ax = plt.subplots(figsize=(10, 6))
ax.tick_params(axis='both', labelsize=12)

# Paleta de colores por keep_ratio y estilos por scheduler
colors = deque(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'])
linestyles = ['-', '--', ':']

for i, kr in enumerate(keep_ratios):
    for j, sch in enumerate(schedulers):
        label = f"{sch} (Budget Target={kr})"
        ax.plot(
            range(1, 13),
            data[kr][sch],
            marker='o',
            label=label,
            color=colors.popleft(),
            linestyle=linestyles[j],
        )

ax.set_xticks(range(1, 13))
ax.set_xlabel("Block Index", fontsize=FONTSIZE)
ax.set_ylabel("Active Tokens ($N_{active}$)", fontsize=FONTSIZE)
ax.grid(True)
# ax.legend(ncol=3, bbox_to_anchor=(0.5, -0.1), loc='upper center', frameon=False)
ax.legend(loc='best')

plt.savefig("pruning_schedules_2.pdf", bbox_inches='tight')
plt.close()

###############
### FIGURE 2: BARRAS 80/20 CON LÍNEA
###############

# Valores solicitados
values = [196, 178, 161, 146, 133, 120, 109, 99, 90, 81, 74, 67]
x = list(range(1, len(values) + 1))

# Particiones 80/20 de cada barra (en altura)
base = [round(v * 0.8) for v in values]
top = [v - b for v, b in zip(values, base)]  # v*0.2

# Hacer que la primera barra sea totalmente de la parte base (sin top)
base[0] = values[0]
top[0] = 0

fig, ax = plt.subplots(figsize=(10, 6))
ax.tick_params(axis='both', labelsize=12)

# Dibujar barras apiladas: base (k=80) y top (r=20)
ax.bar(x, base, color='C0', edgecolor='black', width=0.9)
ax.bar(x, top, bottom=base, color='C1', edgecolor='black', width=0.9)

# Añadir etiquetas dentro de cada segmento: k=80 (tokens) en la parte base, r=20 (tokens) en la parte superior
offset = max(values) * 0.015
for xi, v, b, t in zip(x, values, base, top):
    # Texto en la porción 80% (k)
    if b < 196:
        ax.text(xi, b / 2, f'k=80\n({b})', ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    # Texto en la porción 20% (r) — solo si hay porción superior
    if t > 0:
        ax.text(xi, b + t / 2, f'r=20\n({t})', ha='center', va='center', fontsize=10, color='black', fontweight='bold')

    # Valor total por encima del punto
    ax.text(xi, v + offset, str(v), ha='center', va='bottom', fontsize=12, color='black')

# Línea que une los puntos (valores totales)
ax.plot(x, values, marker='o', color='black', linewidth=1, label='Budget Target: 0.7\nPruning Schedule: Exponential', markersize=4)

ax.set_ylim(0, max(values) * 1.06)
ax.set_xticks(x)
ax.set_xlabel('Block Index', fontsize=FONTSIZE)
ax.set_ylabel('Active Tokens ($N_{active}$)', fontsize=FONTSIZE)
ax.grid(True, axis='y', linestyle='--', alpha=0.5)
ax.legend(loc='best', fontsize=12)

plt.savefig('pruning_schedules.pdf', bbox_inches='tight')
plt.close()