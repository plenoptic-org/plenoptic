"""various helpful utilities for plotting or displaying information
"""


def clean_up_axes(ax, ylim=None, spines_to_remove=['top', 'right', 'bottom']):
    """CLean up an axis, as desired when making a stem plot of the representation

    This helper function takes in an axis

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis to clean up
    ylim : tuple or None
        If not None, the y-limits to use for this plot. If None, we
        use the default, slightly adjusted so that the minimum is 0
    spines_to_remove : list
        Some combination of 'top', 'right', 'bottom', and 'left'. The
        spines we remove from the axis

    Returns
    -------
    ax : matplotlib.pyplot.axis
        The cleaned-up axis

    """
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim((0, ax.get_ylim()[1]))
    ax.xaxis.set_visible(False)
    for s in spines_to_remove:
        ax.spines[s].set_visible(False)
    return ax
