import matplotlib.pyplot as plt
import numpy as np

# =========================
# DATA
# =========================
models = ["U-Net", "SwinIR", "Restormer", "RepLKNet", "MR-LKV"]

psnr = [34.89, 34.40, 36.97, 35.33, 37.94]
ssim = [0.8664, 0.8625, 0.8991, 0.8786, 0.9024]
lpips = [0.1400, 0.1139, 0.1101, 0.1405, 0.1027]

rmse = [0.03840, 0.04070, 0.03315, 0.03829, 0.03097]
mae  = [0.01612, 0.01682, 0.01322, 0.01642, 0.01234]

flops = [88.78, 430.26, 378.40, 97.64, 45.62]
time  = [6.09, 2514.62, 227.13, 193.40, 60.21]
params = [7.76, 1.04, 10.66, 2.73, 8.06]

# =========================
# STYLE (PROFESSIONAL)
# =========================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12
})

# Professional color palette
baseline_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
highlight_color = "#2ca02c"

colors = baseline_colors + [highlight_color]

# =========================
# GRAPH 1 — QUANTITATIVE
# =========================
fig, ax = plt.subplots(figsize=(9,5))

x = np.arange(len(models))
width = 0.25

ax.bar(x - width, psnr, width, label="PSNR ↑", color=colors)
ax.bar(x, ssim, width, label="SSIM ↑", color=colors, alpha=0.75)
ax.bar(x + width, lpips, width, label="LPIPS ↓", color=colors, alpha=0.5)

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=20)
ax.set_title("Quantitative Performance Comparison")

ax.legend(frameon=False)
ax.grid(axis='y', linestyle='--', alpha=0.3)

for spine in ["top","right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("graph1_quantitative.pdf", dpi=300)
plt.show()

# =========================
# GRAPH 2 — ERROR METRICS
# =========================
fig, ax = plt.subplots(figsize=(8,5))

ax.bar(x - width/2, rmse, width, label="RMSE ↓", color=colors)
ax.bar(x + width/2, mae, width, label="MAE ↓", color=colors, alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=20)
ax.set_title("Error Metrics Comparison")

ax.legend(frameon=False)
ax.grid(axis='y', linestyle='--', alpha=0.3)

for spine in ["top","right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("graph2_error.pdf", dpi=300)
plt.show()

# =========================
# GRAPH 3 — FLOPs vs PSNR
# =========================
plt.figure(figsize=(7,5))

for i, model in enumerate(models):
    size = 180 if model == "MR-LKV" else 120
    edge = "black"
    
    plt.scatter(flops[i], psnr[i],
                s=size,
                color=colors[i],
                edgecolor=edge)

    plt.text(flops[i]*1.02, psnr[i]*1.002, model)

plt.xlabel("FLOPs (GFLOPs)")
plt.ylabel("PSNR (dB)")
plt.title("Performance vs Computational Cost")

plt.grid(True, linestyle='--', alpha=0.3)

for spine in ["top","right"]:
    plt.gca().spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("graph3_tradeoff_psnr.pdf", dpi=300)
plt.show()

# =========================
# GRAPH 4 — FLOPs vs TIME
# =========================
plt.figure(figsize=(7,5))

for i, model in enumerate(models):
    size = 180 if model == "MR-LKV" else 120
    
    plt.scatter(flops[i], time[i],
                s=size,
                color=colors[i],
                edgecolor='black')

    plt.text(flops[i]*1.02, time[i]*1.1, model)

plt.xlabel("FLOPs (GFLOPs)")
plt.ylabel("Inference Time (ms)")
plt.title("Efficiency Trade-off")

plt.yscale("log")

plt.grid(True, linestyle='--', alpha=0.3)

for spine in ["top","right"]:
    plt.gca().spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("graph4_tradeoff_time.pdf", dpi=300)
plt.show()

# =========================
# GRAPH 5 — COMBINED
# =========================
fig, axs = plt.subplots(1, 3, figsize=(15,4.5))

axs[0].bar(models, flops, color=colors)
axs[0].set_title("FLOPs")

axs[1].bar(models, time, color=colors)
axs[1].set_title("Inference Time")
axs[1].set_yscale("log")

axs[2].bar(models, params, color=colors)
axs[2].set_title("Parameters")

for ax in axs:
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.tick_params(axis='x', rotation=20)
    for spine in ["top","right"]:
        ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("graph5_combined.pdf", dpi=300)
plt.show()