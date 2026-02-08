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
            K = int(parts[0])
            delta = float(parts[1])
            pt = int(parts[2])
            model = parts[3]
            # Using positional mapping based on header format
            p90_ang = float(parts[8])
            p90_rng = float(parts[11])
            p90_vel = float(parts[14])
        except Exception:
            continue
        rows.append({'K': K, 'delta': delta, 'pt': pt, 'model': model, 'p90_ang': p90_ang, 'p90_rng': p90_rng, 'p90_vel': p90_vel})
    return rows


def style_for(model, idx):
    """Return (color, linestyle, marker) for model and index (delta index)."""
    # Colors for models: CFARNet (red), YOLO (blue)
    base = model.upper()
    if 'YOLO' in base:
        color = 'blue'  # YOLO -> Blue
        linestyle = '--'
    else:
        color = 'red'   # CFARNet -> Red
        linestyle = '-'
    
    # Markers depending on Delta index
    # 0 -> o, 1 -> v, 2 -> ^, 3 -> s, 4 -> x
    markers = ['o', 'v', '^', 's', 'x', 'D']
    mk = markers[idx % len(markers)]
    
    return color, linestyle, mk


def plot_multi_styled(rows, out_path):
    deltas = sorted({r['delta'] for r in rows})
    pts = sorted({r['pt'] for r in rows})

    # Adjust figsize for 3 rectangular subplots (approx 4:3 aspect ratio per plot)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Handles map to store line objects for legend
    handles_map = {} 

    # Plotting loop
    # We iterate through all deltas and models found in data
    # But we want consistent styles.
    
    for idx, delta in enumerate(deltas):
        # Filter rows for this delta
        delta_rows = [r for r in rows if r['delta'] == delta]
        
        # We expect CFAR and YOLO for each delta. 
        # Let's explicitly look for them.
        for disp_model in ['CFARNet', 'YOLO']:
            # Find matching data
            # disp_model 'CFARNet' matches 'CFAR' or 'CFARNet' in data
            # disp_model 'YOLO' matches 'YOLO' in data
            
            target_rows = []
            for r in delta_rows:
                m_upper = r['model'].upper()
                if disp_model == 'CFARNet' and 'CFAR' in m_upper:
                    target_rows.append(r)
                elif disp_model == 'YOLO' and 'YOLO' in m_upper:
                    target_rows.append(r)
            
            # Prepare data arrays
            y_ang = []
            y_rng = []
            y_vel = []
            
            # Determine color/style
            # Keep consistent style for (model, delta)
            if disp_model == 'CFARNet':
                color = 'red'
                ls = '-'
            else:
                color = 'blue'
                ls = '--'
            
            # Marker depends on delta index
            markers = ['o', 'v', '^', 's', 'x', 'D']
            mk = markers[idx % len(markers)]
            
            # Retrieve data for each pt
            # If no data found for this model/delta, we still want to plot NaNs or nothing
            # but we need a handle for the legend if we want to show it.
            
            has_data = False
            for pt in pts:
                # find row for this pt
                r_pt = [r for r in target_rows if r['pt'] == pt]
                if r_pt:
                    y_ang.append(r_pt[0]['p90_ang'])
                    y_rng.append(r_pt[0]['p90_rng'])
                    y_vel.append(r_pt[0]['p90_vel'])
                    has_data = True
                else:
                    y_ang.append(float('nan'))
                    y_rng.append(float('nan'))
                    y_vel.append(float('nan'))
            
            label = f"{disp_model} ($\\Delta\\phi={delta}^\\circ$)"
            
            if has_data:
                # Plot
                l1, = axes[0].plot(pts, y_ang, color=color, linestyle=ls, marker=mk, 
                             linewidth=1.5, markersize=5, label=label)
                axes[1].plot(pts, y_rng, color=color, linestyle=ls, marker=mk, 
                             linewidth=1.5, markersize=5, label=label)
                axes[2].plot(pts, y_vel, color=color, linestyle=ls, marker=mk, 
                             linewidth=1.5, markersize=5, label=label)
                handles_map[(disp_model, delta)] = l1
            else:
                # Creates a proxy artist if data is missing but we want it in legend?
                # Usually better not to show if data is completely missing.
                pass

    # Configure Axes
    # (a) Angle
    axes[0].set_xlabel('Transmit Power $P_t$ (dBm)')
    axes[0].set_ylabel('90th percentile angle error ($^\\circ$)')
    # Moved y to -0.26 to account for shorter axes height
    axes[0].text(0.5, -0.2, '(a) 90th percentile angle error vs. $P_t$', 
                 transform=axes[0].transAxes, ha='center', va='top', fontsize=11)

    # (b) Range
    axes[1].set_xlabel('Transmit Power $P_t$ (dBm)')
    axes[1].set_ylabel('90th percentile range error (m)')
    axes[1].text(0.5, -0.2, '(b) 90th percentile range error vs. $P_t$', 
                 transform=axes[1].transAxes, ha='center', va='top', fontsize=11)

    # (c) Velocity
    axes[2].set_xlabel('Transmit Power $P_t$ (dBm)')
    axes[2].set_ylabel('90th percentile radial velocity error (m/s)')
    axes[2].text(0.5, -0.2, '(c) 90th percentile velocity error vs. $P_t$', 
                 transform=axes[2].transAxes, ha='center', va='top', fontsize=11)

    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xticks(pts)
        ax.set_title('')
        # Force square aspect ratio
        # ax.set_box_aspect(1)

    # Construct Legend: Row 1 CFAR, Row 2 YOLO
    # Matplotlib fills legend column-by-column by default.
    # To get Row 1 = All CFAR and Row 2 = All YOLO, we must interleave them in the list:
    # [C1, Y1, C2, Y2, C3, Y3, ...]
    # So that Col 1 gets (C1, Y1), Col 2 gets (C2, Y2), etc.
    
    cfar_handles = []
    yolo_handles = []
    
    # 1. Collect CFARNet handles
    for d in deltas:
        h = handles_map.get(('CFARNet', d))
        if h:
            cfar_handles.append(h)
    
    # 2. Collect YOLO handles
    for d in deltas:
        h = handles_map.get(('YOLO', d))
        if h:
            yolo_handles.append(h)
            
    # Interleave them
    legend_handles = []
    legend_labels = []
    
    # Assuming equal length, otherwise zip truncates.
    # Given the loop structure above, they should match as long as data exists.
    for ch, yh in zip(cfar_handles, yolo_handles):
        legend_handles.append(ch)
        legend_labels.append(ch.get_label())
        legend_handles.append(yh)
        legend_labels.append(yh.get_label())

    # Debug print
    print("Legend Handles Order (Interleaved):")
    for l in legend_labels:
        print(" - ", l)

    ncol = len(deltas)
    # Use global figure legend
    fig.legend(legend_handles, legend_labels, loc='upper center', 
               bbox_to_anchor=(0.5, 0.98), ncol=ncol, frameon=True, 
               fontsize=9, columnspacing=1.0, handletextpad=0.5)

    # Adjusted top/bottom margins. Removed bottom reservation (0) because bbox_inches='tight' will handle it.
    plt.tight_layout(rect=[0, 0, 1, 0.85])
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print('Wrote', out_path)
    plt.close()




if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    txt_path = os.path.join(script_dir, 'results_2d_rmse.txt')
    out_dir = os.path.join(script_dir, 'plots')
    os.makedirs(out_dir, exist_ok=True)

    rows = parse_results(txt_path)
    if not rows:
        print('No data parsed from', txt_path)
        raise SystemExit(1)

    out_path = os.path.join(out_dir, '90th_p90_styled.png')
    plot_multi_styled(rows, out_path)
