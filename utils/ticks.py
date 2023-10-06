import numpy as np
import matplotlib.figure
import matplotlib.pyplot as plt
plt.rcParams["axes.autolimit_mode"] = "round_numbers"
plt.rcParams["axes.xmargin"] = 0.0
plt.rcParams["axes.ymargin"] = 0.0


def get_nice_lim_and_ticks(ymin, ymax, size_scale=1., force_min=None, force_max=None):
    fig = matplotlib.figure.Figure(figsize=(6.4 * size_scale, 4.8 * size_scale))
    ax = fig.add_subplot()
    ax.plot([ymin, ymax])

    if force_min is not None and force_max is not None:
        ax.plot([force_min, force_max])
        ax.set_ylim(force_min, force_max)
    elif force_min is not None:
        ax.plot([force_min])
        ax.set_ylim(ymin=force_min)
    elif force_max is not None:
        ax.plot([force_max])
        ax.set_ylim(ymax=force_max)

    lims = np.array(ax.get_ylim())
    ticks = np.array(ax.get_yticks())
    ticks = ticks[(ticks >= lims[0]) & (ticks <= lims[1])]

    if ticks[0] != lims[0] or ticks[-1] != lims[-1]:
        ticks = get_ticks_even(ymin, ymax)

    if np.all(lims == lims.astype(np.int32)):
        lims = lims.astype(np.int32)
    if np.all(ticks == ticks.astype(np.int32)):
        ticks = ticks.astype(np.int32)

    return lims, ticks

def get_ticks_even(ymin, ymax):
    candidates = (3, 4, 5)
    current = -2
    res = None
    for c in candidates:
        ticks = np.linspace(ymin, ymax, c)
        isallint = np.all(ticks == ticks.astype(np.int32))
        countzeros = -1 if not isallint else str(ticks).count("0")
        if countzeros > current:
            res = ticks
            current = countzeros
    return res
