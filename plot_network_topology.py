"""
Plot IEEE RTS-24 and IEEE 118-bus network topologies with device placements.

The bus assignments are kept consistent with ``solve_PD_instance`` in ``Ess.py``
by using the same fixed random seed and assignment order:
generator -> ESS -> WT -> solar.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandapower.networks as ppnw


DEFAULT_DEVICE_CONFIG = {
    'case24_ieee_rts': dict(num_gen=38, num_WT=10, num_Solar=5, num_ESS=6),
    'case118':         dict(num_gen=38, num_WT=10, num_Solar=5, num_ESS=6),
}

NETWORK_LOADER = {
    'case24_ieee_rts': ppnw.case24_ieee_rts,
    'case118':         ppnw.case118,
}

STYLE = {
    'Generator':    dict(color='#2196F3', marker='o', s=70,  alpha=0.85, label='Generator'),
    'Wind Turbine': dict(color='#4CAF50', marker='^', s=120, alpha=0.90, label='Wind Turbine (WT)'),
    'Solar PV':     dict(color='#FF9800', marker='D', s=120, alpha=0.90, label='Solar PV'),
    'ESS':          dict(color='#F44336', marker='s', s=150, alpha=0.90, label='ESS'),
}


def get_bus_assignments(net, cfg):
    """
    Reproduce the same device-to-bus assignments used in ``Ess.py``.
    The assignment order is generator -> ESS -> WT -> solar.
    """
    rng = np.random.RandomState(0)
    bus_list = list(range(len(net.bus)))
    gen_bus   = rng.choice(bus_list, cfg['num_gen'],   replace=True)
    ess_bus   = rng.choice(bus_list, cfg['num_ESS'],   replace=True)
    wt_bus    = rng.choice(bus_list, cfg['num_WT'],    replace=True)
    solar_bus = rng.choice(bus_list, cfg['num_Solar'], replace=True) \
                if cfg['num_Solar'] > 0 else np.array([], dtype=int)
    return gen_bus, ess_bus, wt_bus, solar_bus


def jitter(arr, scale=0.08, seed=42):
    rng = np.random.RandomState(seed)
    return arr + rng.uniform(-scale, scale, len(arr))


def plot_topology(network_name='case24_ieee_rts', out_dir='figure', cfg=None):
    os.makedirs(out_dir, exist_ok=True)

    net    = NETWORK_LOADER[network_name]()
    if cfg is None:
        cfg = DEFAULT_DEVICE_CONFIG[network_name]
    gen_b, ess_b, wt_b, solar_b = get_bus_assignments(net, cfg)

    geo = net.bus_geodata.sort_index()
    x   = geo['x'].values
    y   = geo['y'].values
    n_bus = len(net.bus)

    print(f'[{network_name}] buses={n_bus}  lines={len(net.line)}  '
          f'trafos={len(net.trafo)}')
    print(f'  Gen  buses (unique): {sorted(set(gen_b.tolist()))}')
    print(f'  WT   buses: {wt_b.tolist()}')
    print(f'  PV   buses: {solar_b.tolist()}')
    print(f'  ESS  buses: {ess_b.tolist()}')

    fig, ax = plt.subplots(figsize=(14, 9))
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

    ax.scatter(x, y, s=160, c='white', edgecolors='#333',
               linewidths=1.5, zorder=2)
    for i in range(n_bus):
        ax.text(x[i] + 0.18, y[i] + 0.15, str(i + 1),
                fontsize=7.5, fontweight='bold', ha='center', va='center', color='#222', zorder=3)

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
              bbox_to_anchor=(0.35, 0.85),
              bbox_transform=ax.transAxes,
              framealpha=0.92, fontsize=11, prop={'weight': 'bold'})

    title_map = {
        'case24_ieee_rts': 'IEEE RTS-24 Bus System',
        'case118':         'IEEE 118-Bus System',
    }
    ax.set_title(
        f'{title_map[network_name]}: Device Allocation\n'
        f'Gen={cfg["num_gen"]}  WT={cfg["num_WT"]}  '
        f'Solar={cfg["num_Solar"]}  ESS={cfg["num_ESS"]}',
        fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()

    stem = f'Topology_network_{network_name}'
    for ext in ('pdf', 'png'):
        path = os.path.join(out_dir, f'{stem}.{ext}')
        plt.savefig(path, dpi=200, bbox_inches='tight')
        print(f'  Saved: {path}')
    plt.close()


if __name__ == '__main__':
    network_name = 'case118'
    out_dir = 'figure/topology'
    
    cfg = DEFAULT_DEVICE_CONFIG[network_name].copy()
    cfg['num_gen'] = 38
    cfg['num_WT'] = 10
    cfg['num_Solar'] = 5
    cfg['num_ESS'] = 6

    plot_topology(network_name, out_dir, cfg)
