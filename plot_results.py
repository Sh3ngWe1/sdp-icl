"""
plot_results.py - Phase 4: SDP-ICL 實驗數據視覺化

讀取 dp_analysis_report.json，並繪製：
1. Privacy-Utility Trade-off Curve (Epsilon vs F1)
2. Cost-Benefit Curve (N vs F1)
3. Time vs Utility Curve
將所有圖表儲存至 results/charts/ 目錄。
"""

import json
import os
import matplotlib.pyplot as plt

# ================= 1. 設定區 =================
# INPUT_REPORT = "results/dp_analysis_report_5000.json"
# OUTPUT_DIR = "results/charts_5000"
INPUT_REPORT = "results/dp_analysis_report.json"
OUTPUT_DIR = "results/charts"

# 圖表樣式設定 (學術風格)
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'qwen_sanitized': '#1f77b4',  # 藍色
    'qwen_standard': '#aec7e8',   # 淺藍
    'llama_sanitized': '#ff7f0e', # 橘色
    'llama_standard': '#ffbb78'   # 淺橘
}
MARKERS = {
    'sanitized': 'o', # 圓形
    'standard': 's'   # 方形
}

def ensure_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def load_data():
    if not os.path.exists(INPUT_REPORT):
        print(f"找不到檔案 {INPUT_REPORT}，請先執行 analyze_sdp_voting.py")
        return None
    with open(INPUT_REPORT, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_experiment_name(file_name):
    """從檔名解析模型與資料類型"""
    model = "qwen" if "qwen" in file_name else "llama"
    data_type = "sanitized" if "sanitized" in file_name else "standard"
    return model, data_type

# ================= 2. 繪圖函數 =================

def plot_privacy_utility_tradeoff(data, fixed_n=20):
    """繪製 Epsilon vs F1 曲線"""
    plt.figure(figsize=(10, 6))
    
    for report in data:
        model, data_type = parse_experiment_name(report['file_name'])
        label = f"{model.capitalize()} ({data_type.capitalize()})"
        color_key = f"{model}_{data_type}"
        
        # 篩選固定 N 的數據，並排除 inf (稍後畫成水平線)
        metrics = [m for m in report['metrics'] if m['N'] == fixed_n]
        finite_metrics = [m for m in metrics if m['epsilon'] != "inf"]
        inf_metric = next((m for m in metrics if m['epsilon'] == "inf"), None)
        
        if not finite_metrics:
            continue
            
        # 排序 epsilon 確保畫線正確
        finite_metrics.sort(key=lambda x: float(x['epsilon']))
        eps_values = [str(m['epsilon']) for m in finite_metrics]
        f1_values = [m['F1'] for m in finite_metrics]
        
        # 畫折線圖
        plt.plot(eps_values, f1_values, marker=MARKERS[data_type], 
                 color=COLORS.get(color_key, 'black'), linewidth=2, markersize=8, label=label)
        
        # 畫無隱私 (inf) 的基準水平線
        if inf_metric:
            plt.axhline(y=inf_metric['F1'], color=COLORS.get(color_key, 'black'), 
                        linestyle='--', alpha=0.5, 
                        label=f"{label} (Upper Bound, eps=inf)")

    plt.title(f'Privacy-Utility Trade-off (Ensemble N={fixed_n})', fontsize=16, fontweight='bold')
    plt.xlabel('Privacy Budget ($\epsilon$)', fontsize=14)
    plt.ylabel('F1 Score (%)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, f"privacy_utility_tradeoff_N{fixed_n}.png")
    plt.savefig(output_path, dpi=300)
    print(f"✅ 已儲存: {output_path}")
    plt.close()

def plot_cost_benefit(data, fixed_eps=3.0):
    """繪製 N vs F1 曲線"""
    plt.figure(figsize=(10, 6))
    
    for report in data:
        model, data_type = parse_experiment_name(report['file_name'])
        label = f"{model.capitalize()} ({data_type.capitalize()})"
        color_key = f"{model}_{data_type}"
        
        # 篩選固定 epsilon 的數據
        metrics = [m for m in report['metrics'] if str(m['epsilon']) == str(fixed_eps)]
        if not metrics:
            continue
            
        metrics.sort(key=lambda x: int(x['N']))
        n_values = [str(m['N']) for m in metrics]
        f1_values = [m['F1'] for m in metrics]
        
        plt.plot(n_values, f1_values, marker=MARKERS[data_type], 
                 color=COLORS.get(color_key, 'black'), linewidth=2, markersize=8, label=label)

    plt.title(f'Ensemble Cost-Benefit (Privacy Budget $\epsilon$={fixed_eps})', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Ensembles (N)', fontsize=14)
    plt.ylabel('F1 Score (%)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, f"cost_benefit_N_eps{fixed_eps}.png")
    plt.savefig(output_path, dpi=300)
    print(f"✅ 已儲存: {output_path}")
    plt.close()

def plot_time_vs_utility(data, fixed_eps=3.0):
    """繪製 Time vs F1 曲線"""
    plt.figure(figsize=(10, 6))
    
    for report in data:
        model, data_type = parse_experiment_name(report['file_name'])
        label = f"{model.capitalize()} ({data_type.capitalize()})"
        color_key = f"{model}_{data_type}"
        
        metrics = [m for m in report['metrics'] if str(m['epsilon']) == str(fixed_eps)]
        if not metrics:
            continue
            
        metrics.sort(key=lambda x: int(x['N']))
        time_values = [m['avg_time_sec'] for m in metrics]
        f1_values = [m['F1'] for m in metrics]
        
        plt.plot(time_values, f1_values, marker=MARKERS[data_type], 
                 color=COLORS.get(color_key, 'black'), linewidth=2, markersize=8, label=label)
        
        # 在每個點旁邊標註 N 的數值
        for i, m in enumerate(metrics):
            plt.annotate(f"N={m['N']}", (time_values[i], f1_values[i]), 
                         textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    plt.title(f'Inference Time vs Utility ($\epsilon$={fixed_eps})', fontsize=16, fontweight='bold')
    plt.xlabel('Average Inference Time (Seconds)', fontsize=14)
    plt.ylabel('F1 Score (%)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, f"time_vs_utility_eps{fixed_eps}.png")
    plt.savefig(output_path, dpi=300)
    print(f"✅ 已儲存: {output_path}")
    plt.close()

# ================= 3. 論文專用圖表 (main2) =================

INPUT_REPORT_5000 = "results/dp_analysis_report_5000.json"
OUTPUT_DIR_5000 = "results/charts_5000_paper"

def _get_qwen_sanitized(data_5000):
    """從 5000 筆報告中取出 Qwen Sanitized 的 metrics"""
    for report in data_5000:
        if "qwen_sanitized" in report['file_name']:
            return report['metrics']
    return []

def plot_f1_vs_epsilon(data_5000):
    """
    f1_vs_epsilon.png
    X 軸: epsilon (0.1, 0.5, 1.0, 3.0, 5.0, 10.0)
    Y 軸: F1 (Qwen Sanitized, N=5 / N=20 / N=100)
    """
    metrics = _get_qwen_sanitized(data_5000)
    eps_labels = [0.1, 0.5, 1.0, 3.0, 5.0, 10.0]
    target_ns = [5, 20, 100]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', '^']
    linestyles = ['-', '--', ':']

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for idx, n in enumerate(target_ns):
        f1_vals = []
        for eps in eps_labels:
            entry = next(
                (m for m in metrics if m['N'] == n and float(m['epsilon']) == eps),
                None
            )
            f1_vals.append(entry['F1'] if entry else None)

        x_pos = list(range(len(eps_labels)))
        ax.plot(x_pos, f1_vals,
                marker=markers[idx], color=colors[idx],
                linestyle=linestyles[idx],
                linewidth=2.2, markersize=8,
                label=f'N={n}')

    ax.set_xticks(range(len(eps_labels)))
    ax.set_xticklabels([str(e) for e in eps_labels], fontsize=12)
    ax.set_ylim(85, 91)
    ax.set_xlabel(r'Privacy Budget ($\epsilon$)', fontsize=14)
    ax.set_ylabel('F1 Score (%)', fontsize=14)
    ax.set_title('F1 Score vs. Privacy Budget $\\epsilon$ (Qwen, Sanitized)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()

    out = os.path.join(OUTPUT_DIR_5000, "f1_vs_epsilon.png")
    fig.savefig(out, dpi=300)
    print(f"✅ 已儲存: {out}")
    plt.close(fig)


def plot_latency_f1_vs_n(data_5000):
    """
    latency_f1_vs_n.png
    X 軸: N (5, 10, 20, 50, 100)
    左 Y 軸 (藍): F1 (Qwen Sanitized, eps=0.1)
    右 Y 軸 (紅): avg_time_sec (Qwen Sanitized, eps=0.1)
    並在 N=20 加上 "Sweet Spot" 標註
    """
    metrics = _get_qwen_sanitized(data_5000)
    target_ns = [5, 10, 20, 50, 100]
    target_eps = 0.1

    f1_vals, time_vals = [], []
    for n in target_ns:
        entry = next(
            (m for m in metrics if m['N'] == n and float(m['epsilon']) == target_eps),
            None
        )
        f1_vals.append(entry['F1'] if entry else None)
        time_vals.append(entry['avg_time_sec'] if entry else None)

    x = list(range(len(target_ns)))
    x_labels = [str(n) for n in target_ns]

    plt.rcParams.update({'font.family': 'DejaVu Sans'})

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('#f8f9fa')

    color_f1   = '#2563EB'
    color_time = '#DC2626'
    color_star = '#15803d'

    sweet_idx = target_ns.index(20)

    # ── F1 折線（左軸）───────────────────────────
    # 先畫普通圓點（排除 N=20）
    x_other = [xi for xi in x if xi != sweet_idx]
    f1_other = [f1_vals[xi] for xi in x_other]
    line1, = ax1.plot(x, f1_vals,
                      color=color_f1, linewidth=2.5, zorder=3,
                      marker='o', markersize=0, label='F1 Score (left)')
    ax1.plot(x_other, f1_other,
             'o', color=color_f1, markersize=9, zorder=4)
    # N=20 換成星號
    star_handle, = ax1.plot(sweet_idx, f1_vals[sweet_idx],
                            '*', color=color_star, markersize=18,
                            zorder=5, label='Sweet Spot ($N$=20)')

    ax1.set_xlabel('Number of Ensembles ($N$)', fontsize=13, labelpad=8)
    ax1.set_ylabel('F1 Score (%)', color=color_f1, fontsize=13, labelpad=8)
    ax1.tick_params(axis='y', labelcolor=color_f1, labelsize=11)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    f1_min = min(v for v in f1_vals if v)
    f1_max = max(v for v in f1_vals if v)
    ax1.set_ylim(f1_min - 0.4, f1_max + 0.55)
    ax1.spines['top'].set_visible(False)
    ax1.spines['left'].set_color(color_f1)
    ax1.spines['left'].set_linewidth(1.6)
    ax1.spines['bottom'].set_color('#94a3b8')
    ax1.spines['right'].set_visible(False)

    # ── 延遲折線（右軸）─────────────────────────
    ax2 = ax1.twinx()
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    line2, = ax2.plot(x, time_vals,
                      color=color_time, marker='D', linestyle='--',
                      linewidth=2.3, markersize=8,
                      zorder=3, label='Avg Latency (right)')
    ax2.set_ylabel('Average Latency (s)', color=color_time, fontsize=13, labelpad=8)
    ax2.tick_params(axis='y', labelcolor=color_time, labelsize=11)
    ax2.set_ylim(0, max(t for t in time_vals if t) * 1.22)
    ax2.spines['right'].set_color(color_time)
    ax2.spines['right'].set_linewidth(1.6)

    # ── 在每個點旁標數值（字小一點）───────────────
    for i, (f1, t) in enumerate(zip(f1_vals, time_vals)):
        ax1.annotate(f'{f1:.2f}', xy=(i, f1),
                     xytext=(0, 9), textcoords='offset points',
                     ha='center', va='bottom', fontsize=8,
                     color=color_f1)
        ax2.annotate(f'{t:.1f}s', xy=(i, t),
                     xytext=(0, -15), textcoords='offset points',
                     ha='center', va='top', fontsize=8,
                     color=color_time)

    # ── 圖例（三條一起放左上）───────────────────
    handles = [line1, line2, star_handle]
    ax1.legend(handles, [h.get_label() for h in handles],
               fontsize=11, loc='upper left',
               framealpha=0.9, edgecolor='#cbd5e1', frameon=True)

    ax1.set_title(
        r'F1 Score & Latency vs. $N$  (Qwen-Sanitized, $\epsilon=0.1$)',
        fontsize=14, fontweight='bold', pad=12
    )
    ax1.grid(axis='y', linestyle='--', alpha=0.45, zorder=1)
    ax1.grid(axis='x', linestyle=':', alpha=0.3, zorder=1)
    fig.tight_layout()

    out = os.path.join(OUTPUT_DIR_5000, "latency_f1_vs_n.png")
    fig.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 已儲存: {out}")
    plt.close(fig)


def main2():
    """論文用兩張圖：f1_vs_epsilon.png 與 latency_f1_vs_n.png"""
    if not os.path.exists(OUTPUT_DIR_5000):
        os.makedirs(OUTPUT_DIR_5000)

    if not os.path.exists(INPUT_REPORT_5000):
        print(f"找不到檔案 {INPUT_REPORT_5000}，請確認路徑。")
        return

    with open(INPUT_REPORT_5000, 'r', encoding='utf-8') as f:
        data_5000 = json.load(f)

    print("📊 開始繪製論文用圖表 (main2)...")
    plot_f1_vs_epsilon(data_5000)
    plot_latency_f1_vs_n(data_5000)
    print(f"\n🎉 完成！請前往 {OUTPUT_DIR_5000}/ 查看圖表。")


# ================= 4. 主程式 =================
def main():
    ensure_dir()
    data = load_data()
    if not data:
        return
        
    print(f"📊 開始繪製圖表...")
    
    # 畫出各種圖表
    plot_privacy_utility_tradeoff(data, fixed_n=5)
    plot_privacy_utility_tradeoff(data, fixed_n=10)
    plot_privacy_utility_tradeoff(data, fixed_n=20)
    plot_privacy_utility_tradeoff(data, fixed_n=100)

    plot_cost_benefit(data, fixed_eps=0.1)
    plot_cost_benefit(data, fixed_eps=1.0)
    plot_cost_benefit(data, fixed_eps=3.0)
    plot_cost_benefit(data, fixed_eps=float('inf'))
    
    plot_time_vs_utility(data, fixed_eps=0.1)
    plot_time_vs_utility(data, fixed_eps=1.0)
    plot_time_vs_utility(data, fixed_eps=3.0)
    plot_time_vs_utility(data, fixed_eps=float('inf'))
    
    print(f"\n🎉 繪圖完成！請前往 {OUTPUT_DIR}/ 查看圖表。")

if __name__ == "__main__":
    # main()
    main2()