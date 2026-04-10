import matplotlib.pyplot as plt
import numpy as np

# =========================
# GLOBAL IEEE STYLE
# =========================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300
})

# =========================
# DATA (UPDATED FROM TABLES)
# =========================
models = ["U-Net", "SwinIR", "Restormer", "RepLKNet", "MR-LKV"]

# Quantitative Metrics
psnr = [29.14, 29.07, 29.80, 29.11, 29.48]
ssim = [0.9002, 0.8907, 0.9148, 0.9020, 0.8904]
lpips = [0.1059, 0.1008, 0.0878, 0.1090, 0.1063]

# Error Metrics
rmse = [0.06183, 0.06268, 0.05828, 0.06156, 0.05843]
mae  = [0.03315, 0.03613, 0.02887, 0.03133, 0.03366]

# Computational Metrics
flops = [54.82, 87.67, 229.62, 59.35, 26.76]
time  = [7.52, 236.41, 117.82, 105.11, 28.66]
params = [7.76, 0.33, 10.66, 2.73, 7.69]

# IEEE grayscale palette (MR-LKV highlighted)
#colors = ["#4d4d4d", "#7f7f7f", "#a6a6a6", "#595959", "#000000"]
colors = [
    "#1f77b4",  # U-Net (blue)
    "#2ca02c",  # SwinIR (green)
    "#d62728",  # Restormer (red)
    "#9467bd",  # RepLKNet (purple)
    "#2ca02c"   # MR-LKV (same green but will highlight separately)
]

# =========================
# FIGURE 1 — QUANTITATIVE METRICS
# =========================
fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))

metrics = [psnr, ssim, lpips]
titles = ["PSNR ↑", "SSIM ↑", "LPIPS ↓"]

for i, ax in enumerate(axs):
    ax.bar(models, metrics[i], color=colors)
    ax.set_title(titles[i])
    ax.tick_params(axis='x', rotation=20)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("figure1_quantitative_ieee.pdf")
plt.show()

# =========================
# FIGURE 2 — ERROR METRICS
# =========================
x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(6,4))

ax.bar(x - width/2, rmse, width, label="RMSE ↓", color="#6e6e6e")
ax.bar(x + width/2, mae, width, label="MAE ↓", color="#000000")

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=20)
ax.set_title("Error Metrics Comparison")

ax.legend(frameon=False)
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("figure2_error_ieee.pdf")
plt.show()

# =========================
# FIGURE 3 — FLOPs vs PSNR
# =========================
plt.figure(figsize=(5.5,4))

for i, model in enumerate(models):
    size = 180 if model == "MR-LKV" else 120
    
    plt.scatter(flops[i], psnr[i],
                s=size,
                color=colors[i],
                edgecolor='black')

    plt.text(flops[i]*1.01, psnr[i]*1.001, model, fontsize=8)

plt.xlabel("FLOPs (G)")
plt.ylabel("PSNR (dB)")
plt.title("Performance vs Computational Cost")

plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("figure3_tradeoff_psnr_ieee.pdf")
plt.show()

# =========================
# FIGURE 4 — FLOPs vs TIME
# =========================
plt.figure(figsize=(5.5,4))

for i, model in enumerate(models):
    size = 180 if model == "MR-LKV" else 120
    
    plt.scatter(flops[i], time[i],
                s=size,
                color=colors[i],
                edgecolor='black')

    plt.text(flops[i]*1.01, time[i]*1.05, model, fontsize=8)

plt.xlabel("FLOPs (G)")
plt.ylabel("Inference Time (ms)")
plt.title("Efficiency Trade-off")

plt.yscale("log")

plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("figure4_tradeoff_time_ieee.pdf")
plt.show()

# =========================
# FIGURE 5 — COMPLEXITY SUMMARY
# =========================
fig, axs = plt.subplots(1, 3, figsize=(12,3.5))

data = [flops, time, params]
titles = ["FLOPs (G)", "Inference Time (ms)", "Parameters (M)"]

for i, ax in enumerate(axs):
    ax.bar(models, data[i], color=colors)
    ax.set_title(titles[i])
    ax.tick_params(axis='x', rotation=20)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)

    if i == 1:
        ax.set_yscale("log")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("figure5_complexity_ieee.pdf")
plt.show()