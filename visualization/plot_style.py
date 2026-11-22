import os
import numpy as np
import matplotlib.pyplot as plt


# Shared palette and figure dimensions.
PALETTE = {
    'plan':        'steelblue',
    'actual':      'darkorange',
    'agc':         'violet',
    'limit':       'red',
    'bound':       'black',
}
FIG_WIDTH = 10.0
ROW_HEIGHT = 2.52


# Font configuration for manuscript figures.
def apply_style(base_size=16, verbose=True):
    """Apply the manuscript font style before creating figures."""
    plt.style.use('default')
    serif_stack = ['STIXGeneral', 'Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'serif']
    plt.rcParams.update({
        'font.size': base_size,
        'font.family': 'serif',
        'font.serif': serif_stack,
        'font.weight': 'bold',
        'axes.titleweight': 'bold',
        'axes.titlesize': 20,
        'axes.labelweight': 'bold',
        'axes.labelsize': 16,
        'legend.fontsize': 12,
        'xtick.labelsize': base_size,
        'ytick.labelsize': base_size,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.8,
        'lines.linewidth': 2.5,
        'mathtext.fontset': 'stix',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    if verbose:
        from matplotlib.font_manager import findfont, FontProperties
        got = findfont(FontProperties(family='serif'))
        name = os.path.basename(got)
        if 'stix' not in name.lower():
            print(f"[plot_style] STIXGeneral unavailable; using {name} instead")
        else:
            print(f"[plot_style] serif -> {name}")


def finalize_figure(fig):
    """Apply the manuscript typography to titles, labels, ticks, and legends."""
    for ax in fig.axes:
        compact = getattr(ax, "_paper_style_compact", False)
        tick_size = 9 if compact else 16
        ax.tick_params(axis='both', which='major', labelsize=tick_size, width=0.8)
        plt.setp(ax.get_xticklabels(), fontfamily='STIXGeneral', fontweight='bold')
        plt.setp(ax.get_yticklabels(), fontfamily='STIXGeneral', fontweight='bold')
        if not compact:
            ax.title.set_fontfamily('STIXGeneral')
            ax.title.set_fontsize(20)
            ax.title.set_fontweight('bold')
            ax.title.set_fontstyle('italic')
            for label in (ax.xaxis.label, ax.yaxis.label):
                label.set_fontfamily('STIXGeneral')
                label.set_fontsize(16)
                label.set_fontweight('bold')
                label.set_fontstyle('italic')
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontfamily('STIXGeneral')
                text.set_fontsize(12 if not compact else 9)
                text.set_fontweight('bold')
                text.set_fontstyle('italic')
    if fig._suptitle is not None:
        fig._suptitle.set_fontfamily('STIXGeneral')
        fig._suptitle.set_fontsize(20)
        fig._suptitle.set_fontweight('bold')
        fig._suptitle.set_fontstyle('italic')


# Common axis styling.
def style_axis(ax, ylabel, T=24, grid_alpha=0.3):
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_xticks(np.arange(T))
    ax.set_xticklabels(np.arange(T))
    ax.set_xlabel('Hour')
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='both', which='major', labelsize=16, width=0.8)
    plt.setp(ax.get_xticklabels(), fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontweight='bold')
    ax.grid(True, linestyle='--', alpha=grid_alpha, linewidth=0.8)
