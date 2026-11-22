"""Plot the IEEE 118 bus network and resource locations.

Resource bus assignments come from the bundled ``ieee118_resources.json``.
Running ``python visualization/plot_topology.py`` saves PDF and PNG files to
``figures/reproduced``.
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd
from types import SimpleNamespace
from paths import DATA

plt.style.use('default')
plt.rcParams.update({
    'font.size': 13,
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Liberation Serif', 'serif'],
    'legend.fontsize': 13,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'mathtext.fontset': 'cm',
})


DEFAULT_DEVICE_CONFIG = {
    'case24_ieee_rts': dict(num_gen=38, num_WT=10, num_Solar=5, num_ESS=6),
    'case118':         dict(num_gen=38, num_WT=10, num_Solar=5, num_ESS=6),
}

def load_network():
    return SimpleNamespace(
        bus=pd.read_csv(DATA/"ieee118_buses.csv", index_col=0),
        bus_geodata=pd.read_csv(DATA/"ieee118_coordinates.csv", index_col=0),
        line=pd.read_csv(DATA/"ieee118_lines.csv", index_col=0),
        trafo=pd.read_csv(DATA/"ieee118_transformers.csv", index_col=0),
    )

NETWORK_LOADER = {"case118": load_network}


STYLE = {
    'Generator':    dict(color='#2196F3', marker='o', s=70,  alpha=0.85, label='Generator'),
    'Wind Turbine': dict(color='#4CAF50', marker='^', s=120, alpha=0.90, label='Wind Turbine (WT)'),
    'Solar PV':     dict(color='#FF9800', marker='D', s=120, alpha=0.90, label='Solar PV'),
    'ESS':          dict(color='#F44336', marker='s', s=150, alpha=0.90, label='ESS'),
}


def get_bus_assignments(net, cfg):
    resources=json.loads((DATA/"ieee118_resources.json").read_text())
    return tuple(np.asarray(resources[k],dtype=int)-1
                 for k in ("thermal","storage","wind","solar"))


def jitter(arr, scale=0.08, seed=42):
    rng = np.random.RandomState(seed)
    return arr + rng.uniform(-scale, scale, len(arr))


def get_bus_coordinates(net):
    if hasattr(net, 'bus_geodata'):
        geo = net.bus_geodata.sort_index()
        return geo['x'].values, geo['y'].values

    if 'geo' not in net.bus.columns:
        raise AttributeError('network has neither bus_geodata nor bus.geo coordinates')

    coords = []
    for raw_geo in net.bus.sort_index()['geo']:
        point = json.loads(raw_geo) if isinstance(raw_geo, str) else raw_geo
        coords.append(point['coordinates'])
    coords = np.asarray(coords, dtype=float)
    return coords[:, 0], coords[:, 1]


def plot_topology(network_name='case118', out_dir='figure', cfg=None):
    os.makedirs(out_dir, exist_ok=True)

    net    = NETWORK_LOADER[network_name]()
    if cfg is None:
        cfg = DEFAULT_DEVICE_CONFIG[network_name]
    gen_b, ess_b, wt_b, solar_b = get_bus_assignments(net, cfg)

    x, y = get_bus_coordinates(net)
    n_bus = len(net.bus)

    print(f'[{network_name}] buses={n_bus}  lines={len(net.line)}  '
          f'trafos={len(net.trafo)}')
    print(f'  Gen  buses (unique): {sorted(set(gen_b.tolist()))}')
    print(f'  WT   buses: {wt_b.tolist()}')
    print(f'  PV   buses: {solar_b.tolist()}')
    print(f'  ESS  buses: {ess_b.tolist()}')

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.set_facecolor('#f5f5f5')

    voltage_styles = {
        345.0: dict(color='#1a1a1a', lw=1.5, alpha=0.85),
        138.0: dict(color='#888888', lw=1.1, alpha=0.50),
    }
    line_voltage = net.line.merge(net.bus[['vn_kv']],
                                  left_on='from_bus', right_index=True)['vn_kv']
    for idx, row in net.line.iterrows():
        fb, tb = int(row.from_bus), int(row.to_bus)
        v = line_voltage.iloc[idx]
        style = voltage_styles.get(v, voltage_styles[138.0])
        ax.plot([x[fb], x[tb]], [y[fb], y[tb]],
                color=style['color'], lw=style['lw'],
                alpha=style['alpha'], zorder=1)
    for _, row in net.trafo.iterrows():
        fb, tb = int(row.hv_bus), int(row.lv_bus)
        ax.plot([x[fb], x[tb]], [y[fb], y[tb]],
                color='#888', lw=1.2, ls='--', alpha=0.45, zorder=1)

    ax.scatter(x, y, s=95, c='white', edgecolors='#333',
               linewidths=1.25, zorder=2)
    center_x, center_y = np.mean(x), np.mean(y)
    label_radius = np.hypot(x - center_x, y - center_y)
    label_radius[label_radius == 0] = 1.0
    label_dx = 0.38 * (x - center_x) / label_radius
    label_dy = 0.31 * (y - center_y) / label_radius
    labeled_buses = sorted(set(np.concatenate([gen_b, ess_b, wt_b, solar_b]).astype(int).tolist()))
    for i in labeled_buses:
        ax.text(x[i] + label_dx[i], y[i] + label_dy[i], str(i + 1),
                fontsize=5.8, fontweight='bold', ha='center', va='center',
                color='#222', zorder=6,
                bbox=dict(boxstyle='round,pad=0.05', facecolor='white',
                          edgecolor='none', alpha=0.62))

    devices = [
        ('Generator',    gen_b,   0.06, 1,  4),
        ('Wind Turbine', wt_b,    0.10, 3,  5),
        ('Solar PV',     solar_b, 0.12, 5,  5),
        ('ESS',          ess_b,   0.09, 7,  5),
    ]
    for name, buses, scale, seed_x, zorder in devices:
        if len(buses) == 0:
            continue
        st = STYLE[name]
        ax.scatter(jitter(x[buses], scale, seed_x),
                   jitter(y[buses], scale, seed_x + 1),
                   s=st['s'], c=st['color'], marker=st['marker'],
                   alpha=st['alpha'], zorder=zorder)

    legend_elems = []
    for name, buses, *_ in devices:
        if len(buses) == 0:
            continue
        st = STYLE[name]
        legend_elems.append(
            Line2D([0], [0], marker=st['marker'], color='w',
                   markerfacecolor=st['color'], markersize=10,
                   label=f'{st["label"]} (×{len(buses)})'))
    legend_elems.append(
        Line2D([0], [0], color='#1a1a1a', lw=1.5, label='345 kV'))
    legend_elems.append(
        Line2D([0], [0], color='#888888', lw=1.1, label='138 kV'))
    legend_elems.append(
        Line2D([0], [0], color='#888', lw=1.5, ls='--', label='Transformer'))
    ax.legend(handles=legend_elems,
              loc='upper right',
              bbox_to_anchor=(0.31, 0.83),
              bbox_transform=ax.transAxes,
              framealpha=0.78, labelspacing=0.35, handlelength=1.7,
              borderpad=0.45, prop={'weight': 'bold', 'size': 6.5})

    title_map = {
        'case24_ieee_rts': 'IEEE RTS-24 Bus System',
        'case118':         'IEEE 118-Bus System',
    }
    ax.set_title(
        title_map[network_name],
        fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()

    stem = f'Topology_network_{network_name}'
    for ext in ('pdf', 'png'):
        path = os.path.join(out_dir, f'{stem}.{ext}')
        plt.savefig(path, dpi=200, bbox_inches='tight')
        print(f'  Saved: {path}')
    plt.close()



if __name__ == "__main__":
    from paths import OUTPUT
    plot_topology("case118", str(OUTPUT), DEFAULT_DEVICE_CONFIG["case118"])
