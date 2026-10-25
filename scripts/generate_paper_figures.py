"""
Generate high-resolution publication-quality figures for the VLSP 2026 paper:
1. fig_mos_scatter.pdf / .png: Scatter plot of MOS predictions vs Ground Truth (Private Test 302 clips)
2. fig_temporal_localization.pdf / .png: Qualitative timeline case study of temporal fault localization
3. fig_architecture.pdf / .png: Visual system architecture diagram
"""

import os
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import signal

os.makedirs('paper', exist_ok=True)

# -----------------------------------------------------------------------------
# FIGURE 1: ARCHITECTURE DIAGRAM
# -----------------------------------------------------------------------------
def plot_architecture():
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis('off')

    # Color palette (modern publication tones)
    c_blue = '#E3F2FD'
    c_blue_b = '#1976D2'
    c_green = '#E8F5E9'
    c_green_b = '#388E3C'
    c_orange = '#FFF3E0'
    c_orange_b = '#F57C00'
    c_purple = '#F3E5F5'
    c_purple_b = '#7B1FA2'
    c_gray = '#F5F5F5'
    c_gray_b = '#616161'

    def draw_box(x, y, w, h, bg, border, title, subtitle="", radius=1.5):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.5,rounding_size={radius}",
            ec=border, fc=bg, lw=1.5, zorder=2
        )
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w/2, y + h*0.62, title, ha='center', va='center', fontsize=9, fontweight='bold', color='#212121')
            ax.text(x + w/2, y + h*0.30, subtitle, ha='center', va='center', fontsize=7.5, color='#424242')
        else:
            ax.text(x + w/2, y + h/2, title, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#212121')

    def draw_arrow(x1, y1, x2, y2, color='#455A64', lw=1.5):
        ax.annotate(
            '', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=2, shrinkB=2, mutation_scale=12)
        )

    # 1. Input Audio
    draw_box(2, 20, 12, 10, c_gray, c_gray_b, "Raw Audio", "16 kHz Mono Waveform")

    # 2. Dual Foundation Backbones
    draw_arrow(14.5, 25, 19, 25)
    draw_box(19, 28, 20, 14, c_blue, c_blue_b, "Microsoft WavLM-Large", "1024-d Denoising Latents\n(Frozen 24 Layers, 316M)")
    draw_box(19, 8, 20, 14, c_blue, c_blue_b, "Meta XLS-R-300M", "1024-d Phonetic Latents\n(Frozen 128 Langs, 317M)")

    # Fused Latents Box
    draw_arrow(39.5, 35, 43, 30)
    draw_arrow(39.5, 15, 43, 20)
    draw_box(43, 21, 13, 8, c_purple, c_purple_b, "Dual Fused", "2048-d Frame Latents")

    # Branch A: SLSQP Tri-Ensemble MOS Head (Top)
    draw_arrow(39.5, 38, 59, 41)
    draw_box(59, 36, 23, 11, c_green, c_green_b, "SLSQP Tri-Ensemble MOS Head", "- WavLM LoRA ($w_1=0.247$)\n- Frozen WavLM ASP ($w_2=0.364$)\n- XLS-R-300M ASP ($w_3=0.389$)")
    draw_arrow(82.5, 41.5, 87, 41.5)
    draw_box(87, 37.5, 11, 8, '#C8E6C9', c_green_b, "Quality Score", r"$\hat{Q} \in [1.0, 5.0]$")

    # Branch B: Temporal Fault Localization Ensemble (Bottom)
    draw_arrow(56.5, 25, 59, 20)
    draw_box(59, 5, 23, 25, c_orange, c_orange_b, "Complementary Fault Decoders", "- Dual Conformer (2048-d, k=15)\n- Conformer Peak (1024-d, 25ep)\n- Conformer-Pro (k=5, 15, 31)\n- Decoupled Tri-Branch (TCN+LSTM)\nLoss: Focal + 0.5 Soft-IoU + 0.2 Presence")

    draw_arrow(82.5, 17.5, 87, 17.5)
    draw_box(87, 12, 11, 11, '#FFE0B2', c_orange_b, "Fault Intervals", "Noise, Distortion, Pause\n(Two-Tier Hysteresis)")

    plt.tight_layout()
    plt.savefig('paper/fig_architecture.pdf', bbox_inches='tight')
    plt.savefig('paper/fig_architecture.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Generated paper/fig_architecture.pdf and .png")

# -----------------------------------------------------------------------------
# FIGURE 2: SCATTER PLOT MOS PREDICTIONS VS GROUND TRUTH (302 Private Test clips)
# -----------------------------------------------------------------------------
def plot_mos_scatter():
    with open('data/private_test_ground_truth.json') as f:
        gt_data = json.load(f)
    with open('submissions/submission_v11.json') as f:
        sub_data = json.load(f)

    gt_dict = {x['file_id']: x for x in gt_data}
    sub_dict = {x['file_id']: x for x in sub_data}

    fids = [x['file_id'] for x in gt_data]
    y_true = np.array([gt_dict[fid]['quality_score'] for fid in fids])
    y_pred = np.array([sub_dict[fid]['quality_score'] for fid in fids])

    # Metrics
    r = np.corrcoef(y_true, y_pred)[0, 1]
    mse = np.mean((y_true - y_pred) ** 2)

    # Classify error types (following EM-VSQA Top-2 convention)
    # Over-predicted: y_pred > 3.5 and y_true <= y_pred - 1.0
    # Under-predicted: y_true >= 4.0 and y_pred < 3.2
    is_over = (y_pred > 3.5) & (y_true <= y_pred - 1.0)
    is_under = (y_true >= 4.0) & (y_pred < 3.2)
    is_normal = ~is_over & ~is_under

    fig, ax = plt.subplots(figsize=(5.5, 4.4), dpi=300)

    # Scatter points
    ax.scatter(y_pred[is_normal], y_true[is_normal], color='#1976D2', alpha=0.65, s=32, edgecolors='none', label=f'Normal ({np.sum(is_normal)})')
    ax.scatter(y_pred[is_over], y_true[is_over], color='#D32F2F', alpha=0.85, s=40, edgecolors='k', linewidths=0.5, label=f'Over-predicted ({np.sum(is_over)})')
    ax.scatter(y_pred[is_under], y_true[is_under], color='#388E3C', alpha=0.85, s=40, edgecolors='k', linewidths=0.5, label=f'Under-predicted ({np.sum(is_under)})')

    # Regression line
    poly = np.polyfit(y_pred, y_true, 1)
    x_grid = np.linspace(2.0, 4.3, 100)
    ax.plot(x_grid, np.polyval(poly, x_grid), color='#B71C1C', linestyle='--', linewidth=1.8, label=f'Linear Fit (Slope={poly[0]:.2f})')

    # Ideal diagonal
    ax.plot([1.0, 5.0], [1.0, 5.0], color='#9E9E9E', linestyle=':', linewidth=1.2, label='Ideal Identity')

    ax.set_xlabel(r'Predicted MOS ($\hat{Q}$)', fontsize=10, fontweight='bold')
    ax.set_ylabel(r'Ground Truth MOS ($Q^*$)', fontsize=10, fontweight='bold')
    ax.set_xlim(1.8, 4.4)
    ax.set_ylim(0.8, 5.0)

    # Info box
    text_str = f"Private Test Set (302 clips)\n$\\mathbf{{PCC}} = \\mathbf{{{r:.4f}}}$\n$\\mathrm{{MSE}} = {mse:.4f}$"
    props = dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5', edgecolor='#BDBDBD', alpha=0.95)
    ax.text(0.04, 0.96, text_str, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=props)

    ax.legend(loc='lower right', fontsize=8, framealpha=0.92)
    ax.grid(True, linestyle='--', alpha=0.45)
    plt.tight_layout()
    plt.savefig('paper/fig_mos_scatter.pdf', bbox_inches='tight')
    plt.savefig('paper/fig_mos_scatter.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Generated paper/fig_mos_scatter.pdf and .png")

# -----------------------------------------------------------------------------
# FIGURE 3: TEMPORAL FAULT LOCALIZATION CASE STUDY (Waveform + Intervals)
# -----------------------------------------------------------------------------
def plot_temporal_localization():
    fid = 'pt_00027.wav'
    wav_path = f'data/private_test/audio/{fid}'
    audio, sr = sf.read(wav_path)
    dur = len(audio) / sr
    t_audio = np.linspace(0, dur, len(audio))

    with open('data/private_test_ground_truth.json') as f:
        gt_entry = next(x for x in json.load(f) if x['file_id'] == fid)
    with open('submissions/submission_v11.json') as f:
        sub_entry = next(x for x in json.load(f) if x['file_id'] == fid)

    fig, (ax_wave, ax_bar) = plt.subplots(2, 1, figsize=(7.2, 3.2), sharex=True, gridspec_kw={'height_ratios': [1.3, 1.0]}, dpi=300)

    # 1. Waveform
    ax_wave.plot(t_audio, audio, color='#37474F', lw=0.6, alpha=0.85)
    ax_wave.set_ylabel('Amplitude', fontsize=9, fontweight='bold')
    ax_wave.set_title(f'Temporal Fault Localization Case Study: Utterance {fid} ($D = {dur:.2f}$ s, Ground Truth MOS = {gt_entry["quality_score"]:.2f})', fontsize=10, fontweight='bold')
    ax_wave.grid(True, linestyle=':', alpha=0.5)
    ax_wave.set_xlim(0, dur)

    # Highlight distortion zone on waveform
    gt_dist = gt_entry['faults']['distortion']
    for s, e in gt_dist:
        ax_wave.axvspan(s, e, color='#E53935', alpha=0.18, label='True Distortion Region')

    # 2. Interval comparison bar
    # Y-axis rows: Ground Truth Distortion, Predicted Distortion
    ax_bar.set_ylim(-0.5, 2.5)
    ax_bar.set_yticks([0.5, 1.5])
    ax_bar.set_yticklabels(['Prediction (Run 4)', 'Ground Truth'], fontsize=9, fontweight='bold')
    ax_bar.set_xlabel('Time (seconds)', fontsize=9, fontweight='bold')

    # Ground Truth intervals
    for s, e in gt_dist:
        rect = patches.Rectangle((s, 1.2), e - s, 0.6, facecolor='#E53935', edgecolor='#B71C1C', lw=1.2, zorder=3)
        ax_bar.add_patch(rect)
        ax_bar.text((s + e)/2, 1.5, f"Distortion [{s:.2f}s - {e:.2f}s]", ha='center', va='center', color='white', fontsize=7.5, fontweight='bold', zorder=4)

    # Predicted intervals
    pred_dist = sub_entry['faults']['distortion']
    for s, e in pred_dist:
        rect = patches.Rectangle((s, 0.2), e - s, 0.6, facecolor='#1E88E5', edgecolor='#0D47A1', lw=1.2, zorder=3)
        ax_bar.add_patch(rect)
        ax_bar.text((s + e)/2, 0.5, f"Distortion [{s:.2f}s - {e:.2f}s]", ha='center', va='center', color='white', fontsize=7.5, fontweight='bold', zorder=4)

    # IoU annotation (using displayed 2-decimal rounded interval bounds for exact arithmetic consistency with paper text and caption)
    s_gt_r, e_gt_r = round(gt_dist[0][0], 2), round(gt_dist[0][1], 2)
    s_pred_r, e_pred_r = round(pred_dist[0][0], 2), round(pred_dist[0][1], 2)
    clip_iou = (min(e_gt_r, e_pred_r) - max(s_gt_r, s_pred_r)) / (max(e_gt_r, e_pred_r) - min(s_gt_r, s_pred_r))
    ax_bar.text(dur - 0.05, 2.1, f"Clip Distortion IoU = {clip_iou*100:.1f}%", ha='right', va='center', fontsize=8.5, fontweight='bold', color='#0D47A1', bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor='#90CAF9'))

    ax_bar.grid(True, axis='x', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig('paper/fig_temporal_localization.pdf', bbox_inches='tight')
    plt.savefig('paper/fig_temporal_localization.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Generated paper/fig_temporal_localization.pdf and .png")

if __name__ == '__main__':
    # Preserve modern draw.io architecture diagram if fig_architecture.drawio exists
    drawio_file = Path('paper/fig_architecture.drawio')
    if not drawio_file.exists():
        plot_architecture()
    else:
        print("Preserving high-resolution draw.io architecture diagram (paper/fig_architecture.pdf / .png)")

    plot_mos_scatter()
    plot_temporal_localization()
    print("All publication figures synchronized successfully!")
