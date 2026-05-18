import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Path relativo alla cartella dello script stesso
OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# FIGURA 1 — Ablation Study: NDCG@K (4 varianti)
# ─────────────────────────────────────────────

variants   = ['full', 'no_tags', 'no_overview', 'genres_only']
labels_nice = ['Full\n(completa)', 'No tags', 'No overview', 'Genres\nonly']
K_vals     = [5, 10, 20]

# valori NDCG@K dal documento
ndcg = {
    'full':         [0.0061, 0.0081, 0.0106],
    'no_tags':      [0.0060, 0.0086, 0.0106],
    'no_overview':  [0.0055, 0.0071, 0.0125],
    'genres_only':  [0.0045, 0.0061, 0.0081],
}

# colori accademici, distinguibili in B&W
colors = ['#2C5F8A', '#4DAF4A', '#FF7F00', '#E41A1C']
hatches = ['', '///', '...', 'xxx']

fig, ax = plt.subplots(figsize=(9, 5))
x    = np.arange(len(K_vals))
n    = len(variants)
w    = 0.18
offsets = np.linspace(-(n-1)*w/2, (n-1)*w/2, n)

bars_list = []
for i, (var, col, hatch) in enumerate(zip(variants, colors, hatches)):
    bars = ax.bar(x + offsets[i], ndcg[var], width=w,
                  color=col, hatch=hatch, edgecolor='white',
                  linewidth=0.6, alpha=0.88, label=labels_nice[i])
    bars_list.append(bars)
    # etichetta valore sopra ogni barra
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.00015,
                f'{h:.4f}', ha='center', va='bottom',
                fontsize=7.5, color='#333333')

ax.set_xticks(x)
ax.set_xticklabels([f'NDCG@{k}' for k in K_vals], fontsize=12)
ax.set_ylabel('NDCG@K', fontsize=12)
ax.set_title('Ablation Study — Composizione Text Representation\n'
             'Impatto di ciascuna variante su NDCG@K (test set)',
             fontsize=13, fontweight='bold', pad=14)
ax.set_ylim(0, 0.016)
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.4f'))
ax.grid(axis='y', linestyle='--', alpha=0.5, linewidth=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# annotazione delta chiave su NDCG@10
# no_tags vs full: +5.9%
ax.annotate('+5.9%\nvs full', xy=(x[1]+offsets[1], ndcg['no_tags'][1]),
            xytext=(x[1]+offsets[1]+0.02, ndcg['no_tags'][1]+0.0012),
            fontsize=7, color='#4DAF4A', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#4DAF4A', lw=1.2))
# genres_only vs full: -24.7%
ax.annotate('−24.7%\nvs full', xy=(x[1]+offsets[3], ndcg['genres_only'][1]),
            xytext=(x[1]+offsets[3]+0.25, ndcg['genres_only'][1]+0.0015),
            fontsize=7, color='#E41A1C', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#E41A1C', lw=1.2))

ax.legend(loc='upper left', fontsize=9.5, framealpha=0.9,
          edgecolor='#cccccc', ncol=2)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'fig_ablation_ndcg.pdf', dpi=200, bbox_inches='tight')
fig.savefig(OUTPUT_DIR /'fig_ablation_ndcg.png', dpi=200, bbox_inches='tight')
print("Fig 1 OK")
plt.close()

# ─────────────────────────────────────────────
# FIGURA 2 — Confronto modelli: HR@K e NDCG@K
# ─────────────────────────────────────────────

models = ['Popularity\nbaseline', 'ItemKNN\nCF', 'PureSVD\n(50 fatt.)',
          'Content-\nBased', 'Hybrid\nSVD+CB\n(γ=0.7)']

hr = {
    5:  [0.0197, 0.0066, 0.0361, 0.0131, 0.0328],
    10: [0.0328, 0.0098, 0.0590, 0.0213, 0.0607],
    20: [0.0475, 0.0197, 0.0902, 0.0377, 0.0869],
}
ndcg2 = {
    5:  [0.0144, 0.0051, 0.0218, 0.0073, 0.0201],
    10: [0.0186, 0.0062, 0.0292, 0.0098, 0.0292],
    20: [0.0223, 0.0086, 0.0371, 0.0137, 0.0356],
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
x2 = np.arange(len(models))
w2 = 0.22
k_colors = ['#2C5F8A', '#4DAF4A', '#E41A1C']

for ax_idx, (metric_dict, metric_name) in enumerate(
        [(hr, 'HR@K'), (ndcg2, 'NDCG@K')]):
    ax2 = axes[ax_idx]
    k_offsets = [-w2, 0, w2]
    for j, (k, kc) in enumerate(zip(K_vals, k_colors)):
        vals = metric_dict[k]
        bars2 = ax2.bar(x2 + k_offsets[j], vals, width=w2,
                        color=kc, alpha=0.82,
                        edgecolor='white', linewidth=0.5,
                        label=f'K={k}')
        # highlight best per K
        best_idx = int(np.argmax(vals))
        ax2.bar(x2[best_idx] + k_offsets[j], vals[best_idx],
                width=w2, color=kc, alpha=1.0,
                edgecolor='gold', linewidth=1.8)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(models, fontsize=9)
    ax2.set_ylabel(metric_name, fontsize=11)
    ax2.set_title(f'{metric_name} per modello e K', fontsize=12,
                  fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.45, linewidth=0.7)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(fontsize=9, framealpha=0.85, edgecolor='#cccccc')
    ax2.set_ylim(0, max(max(v) for v in metric_dict.values()) * 1.18)

fig.suptitle('Confronto tra modelli — HR@K e NDCG@K (test set)\n'
             'Il bordo dorato indica il modello migliore per ciascun K',
             fontsize=12, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(OUTPUT_DIR /'fig_confronto_modelli.pdf', dpi=200, bbox_inches='tight')
fig.savefig(OUTPUT_DIR/'fig_confronto_modelli.png', dpi=200, bbox_inches='tight')
print("Fig 2 OK")
plt.close()

# ─────────────────────────────────────────────
# FIGURA 3 — Grid search gamma (PureSVD+CB)
# ─────────────────────────────────────────────

gammas     = [0.20, 0.40, 0.50, 0.60, 0.70, 0.80]
hr10_val   = [0.0279, 0.0361, 0.0475, 0.0557, 0.0590, 0.0574]
ndcg10_val = [0.0164, 0.0222, 0.0274, 0.0299, 0.0316, 0.0315]
hr10_test  = [0.0361, 0.0393, 0.0393, 0.0508, 0.0607, 0.0607]
ndcg10_test= [0.0183, 0.0193, 0.0195, 0.0226, 0.0292, 0.0288]

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(gammas, hr10_val,   'o--', color='#2C5F8A', label='HR@10 (val)',   lw=1.6)
ax.plot(gammas, ndcg10_val, 's--', color='#FF7F00', label='NDCG@10 (val)', lw=1.6)
ax.plot(gammas, hr10_test,  'o-',  color='#2C5F8A', label='HR@10 (test)',  lw=2.2)
ax.plot(gammas, ndcg10_test,'s-',  color='#FF7F00', label='NDCG@10 (test)',lw=2.2)

# linea verticale gamma*
ax.axvline(0.70, color='gray', linestyle=':', linewidth=1.2)
ax.text(0.71, 0.001, r'$\gamma^*=0.70$', fontsize=9, color='gray')

# punti ottimali
ax.scatter([0.70], [0.0316], s=80, color='#FF7F00', zorder=5)
ax.scatter([0.70], [0.0590], s=80, color='#2C5F8A', zorder=5)

ax.set_xlabel(r'Peso SVD — $\gamma$', fontsize=11)
ax.set_ylabel('Metrica', fontsize=11)
ax.set_title(r'Grid search su $\gamma$ — Hybrid PureSVD+Content-Based', fontsize=12,
             fontweight='bold')
ax.legend(fontsize=9, framealpha=0.9, edgecolor='#cccccc', ncol=2)
ax.grid(linestyle='--', alpha=0.45, linewidth=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR /'fig_gamma_gridsearch.pdf', dpi=200, bbox_inches='tight')
fig.savefig(OUTPUT_DIR/'fig_gamma_gridsearch.png', dpi=200, bbox_inches='tight')
print("Fig 3 OK")
plt.close()

# ─────────────────────────────────────────────
# FIGURA 4 — Coverage baseline vs ibrido
# ─────────────────────────────────────────────

mod_labels   = ['Popularity\nbaseline', 'PureSVD', 'Content-\nBased', 'Hybrid\nSVD+CB']
coverage_vals= [2.21, None, None, 33.32]  # None = da calcolare

fig, ax = plt.subplots(figsize=(6, 4))
x_cov = [0, 3]  # solo valori noti
vals_cov = [2.21, 33.32]
bar_colors = ['#E41A1C', '#4DAF4A']
bars3 = ax.bar(x_cov, vals_cov, color=bar_colors, width=0.55,
               edgecolor='white', linewidth=0.5, alpha=0.88)
for bar, v in zip(bars3, vals_cov):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.3, f'{v:.2f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(mod_labels, fontsize=10)
ax.set_ylabel('Coverage@10 (%)', fontsize=11)
ax.set_title('Coverage@10 — Trade-off Accuratezza / Diversità\n'
             'Frazione del catalogo raccomandata almeno a un utente',
             fontsize=11, fontweight='bold', pad=10)
ax.set_ylim(0, 40)
# barre grigie per valori mancanti
for xi in [1, 2]:
    ax.bar(xi, 38, color='#dddddd', width=0.55, edgecolor='white')
    ax.text(xi, 19, 'da\ncalcolare', ha='center', va='center',
            fontsize=9, color='#888888')
ax.grid(axis='y', linestyle='--', alpha=0.45, linewidth=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR /'fig_coverage.pdf', dpi=200, bbox_inches='tight')
fig.savefig(OUTPUT_DIR /'fig_coverage.png', dpi=200, bbox_inches='tight')
print("Fig 4 OK")
plt.close()

# ─────────────────────────────────────────────
# FIGURA 5 — Cold-start stratificato per quartile
# (valori plausibili basati sul comportamento atteso dei modelli;
#  da sostituire con dati reali dall'esperimento)
# ─────────────────────────────────────────────

quartiles  = ['Q1\n(cold)\n≤31 rating', 'Q2\n≤65 rating',
              'Q3\n≤149 rating', 'Q4\n(warm)\n>149 rating']
hr10_q = {
    'Popularity':  [0.030, 0.031, 0.033, 0.035],
    'PureSVD':     [0.025, 0.045, 0.065, 0.085],
    'Content-Based': [0.018, 0.020, 0.022, 0.024],
    'Hybrid SVD+CB': [0.028, 0.050, 0.068, 0.088],
}
col_cold = ['#FF7F00', '#2C5F8A', '#4DAF4A', '#984EA3']

fig, ax = plt.subplots(figsize=(9, 4.8))
x_q = np.arange(len(quartiles))
w_q = 0.19
offsets_q = np.linspace(-(len(hr10_q)-1)*w_q/2,
                         (len(hr10_q)-1)*w_q/2,
                         len(hr10_q))
for (mod, vals), col, off in zip(hr10_q.items(), col_cold, offsets_q):
    ax.bar(x_q + off, vals, width=w_q, color=col, alpha=0.85,
           edgecolor='white', linewidth=0.5, label=mod)
    ax.plot(x_q + off, vals, 'o-', color=col, lw=1.2, ms=4, alpha=0.7)

ax.set_xticks(x_q)
ax.set_xticklabels(quartiles, fontsize=10)
ax.set_ylabel('HR@10', fontsize=11)
ax.set_title('Analisi Cold-Start — HR@10 per quartile di attività utente\n'
             r'\textit{Nota: valori simulativi — sostituire con risultati sperimentali reali}',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.9, edgecolor='#cccccc', ncol=2)
ax.grid(axis='y', linestyle='--', alpha=0.45, linewidth=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# watermark "SIMULATIVO"
ax.text(0.5, 0.5, 'DATI SIMULATIVI\nDA SOSTITUIRE',
        transform=ax.transAxes, fontsize=22, color='gray',
        alpha=0.18, ha='center', va='center', rotation=30,
        fontweight='bold')
fig.tight_layout()
fig.savefig(OUTPUT_DIR /'fig_coldstart.pdf', dpi=200, bbox_inches='tight')
fig.savefig(OUTPUT_DIR /'fig_coldstart.png', dpi=200, bbox_inches='tight')
print("Fig 5 OK")
plt.close()

print("\nTutti i grafici generati correttamente.")
