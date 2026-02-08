import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_results(txt_path):
    rows = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith('K |') or ln.startswith('-'):
            continue
        parts = [p.strip() for p in ln.split('|')]
        if len(parts) < 15:
            continue
        try:
            # Format: K | Pt(dBm) | Model | RMSE_2D | 90%_2D | ...
            K = int(parts[0])
            pt = int(parts[1])
            model = parts[2]
            
            # Index mapping based on header in results_k_sweep.txt:
            # 0:K, 1:Pt, 2:Model, 3:RMSE_2D, 4:90%_2D, 5:95%_2D,
            # 6:RMSE_Ang, 7:90%_Ang, 8:95%_Ang, 
            # 9:RMSE_Rng, 10:90%_Rng, 11:95%_Rng, 
            # 12:RMSE_Vel, 13:90%_Vel, 14:95%_Vel
            
            p90_ang = float(parts[7])
            p90_rng = float(parts[10])
            p90_vel = float(parts[13])
        except Exception as e:
            # print(f"Skipping line: {ln} due to {e}")
            continue
        rows.append({'K': K, 'pt': pt, 'model': model, 'p90_ang': p90_ang, 'p90_rng': p90_rng, 'p90_vel': p90_vel})
    return rows

def style_for(model):
    """Return (color, linestyle, marker) for model."""
    base = model.upper()
    if 'YOLO' in base:
        color = 'blue'
        linestyle = '--'
        marker = 's' # Square for YOLO
    else:
        color = 'red'
        linestyle = '-'
        marker = 'o' # Circle for CFARNet
    return color, linestyle, marker

def plot_k_sweep(rows, out_path):
    # Extract unique K values
    ks = sorted({r['K'] for r in rows})
    
    # Separate data by model
    # Assumption: We are plotting for a specific Pt, or aggregating?
    # Usually K sweep is done at a fixed Pt (e.g. 50dBm).
    # If multiple Pts exist, we should filter or plot separately.
    # Let's check available Pts.
    pts = sorted({r['pt'] for r in rows})
    if not pts:
        print("No data found.")
        return

    # Use the first available Pt if multiple, or specific one if needed.
    # The user request implies a single plot.
    target_pt = pts[0] 
    if len(pts) > 1:
        print(f"Multiple Pt found in results: {pts}. Using Pt={target_pt} for plotting.")
    
    # Filter for target Pt
    rows = [r for r in rows if r['pt'] == target_pt]
    
    # Models
    models = sorted({r['model'] for r in rows})
    # Sort order: CFAR, YOLO
    models = sorted(models, key=lambda m: 0 if 'CFAR' in m.upper() else 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(16,5))
    
    # Metrics to plot
    metrics = [
        ('p90_ang', 'angle error ($\\degree$)', '(a) 90th percentile angle error vs. $K$'),
        ('p90_rng', 'range error (m)', '(b) 90th percentile range error vs. $K$'),
        ('p90_vel', 'radial velocity error (m/s)', '(c) 90th percentile radial velocity error vs. $K$')
    ]
    
    handles = []
    labels = []
    
    for i, (key, ylabel, title) in enumerate(metrics):
        ax = axes[i]
        
        for model in models:
            # Extract (x, y)
            model_rows = [r for r in rows if r['model'] == model]
            # Ensure sorting by K
            model_rows.sort(key=lambda r: r['K'])
            
            x = [r['K'] for r in model_rows]
            y = [r[key] for r in model_rows]
            
            color, ls, mk = style_for(model)
            disp_model = "CFARNet" if "CFAR" in model.upper() else "YOLO"
            
            l, = ax.plot(x, y, color=color, linestyle=ls, marker=mk, markersize=5, linewidth=1.5, label=disp_model)
            
            # Collect handles only from first subplot to avoid duplicates
            if i == 0:
                handles.append(l)
                labels.append(disp_model)
        
        ax.set_xlabel('Number of targets')
        ax.set_ylabel(f'90th percentile {ylabel}')
        ax.set_xticks(ks)
        ax.grid(True, linestyle='--', alpha=0.4)
        
        # Add sub-caption below. Adjusted y to -0.21 to reduce gap with x-axis label.
        ax.text(0.5, -0.2, title, transform=ax.transAxes, ha='center', va='top', fontsize=11)

    # Legend at top center
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=True)
    
    # Use standard tight layout with tight bbox saving to remove whitespace
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f'Wrote {out_path}')
    plt.close()

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(script_dir, 'results_k_sweep.txt')
    out_dir = os.path.join(script_dir, 'plots')
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(txt_path):
        print(f"Error: {txt_path} not found.")
    else:
        rows = parse_results(txt_path)
        if not rows:
            print(f'No data parsed from {txt_path}')
        else:
            out_path = os.path.join(out_dir, 'plot_k_sweep_p90.png')
            plot_k_sweep(rows, out_path)
