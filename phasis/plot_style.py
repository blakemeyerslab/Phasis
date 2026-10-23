"""Portable fonts for Phasis-owned plots, without changing caller settings."""

from functools import wraps

import matplotlib as mpl


def portable_plot_fonts(render):
    """Use bundled fonts during rendering, then restore the caller's rcParams."""
    @wraps(render)
    def wrapped(*args, **kwargs):
        with mpl.rc_context({
            "font.family": "DejaVu Sans",
            "font.style": "normal",
            "font.variant": "normal",
            "font.stretch": "normal",
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "figure.titleweight": "normal",
            "figure.labelweight": "normal",
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "it",
            "text.usetex": False,
            "pdf.use14corefonts": False,
            "ps.useafm": False,
        }):
            return render(*args, **kwargs)

    return wrapped
