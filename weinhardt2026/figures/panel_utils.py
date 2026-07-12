"""Shared utilities for saving figure panels in detailed and clean versions."""

import os
import matplotlib.pyplot as plt


def strip_text(ax):
    """Remove all text elements from an axes: titles, labels, tick labels,
    legends, annotations, and free text objects."""
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Remove legend
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    # Remove annotations
    for ann in list(ax.texts):
        ann.remove()

    # Remove any child annotations (arrows + text boxes)
    for child in list(ax.get_children()):
        if isinstance(child, plt.Annotation):
            child.remove()


def strip_figure(fig):
    """Remove all text from every axes in a figure, plus suptitle."""
    fig.suptitle('')
    for ax in fig.get_axes():
        strip_text(ax)
    # Remove figure-level legends
    for legend in list(fig.legends):
        legend.remove()


def save_panel(fig, output_dir, panel_name, dpi=300):
    """Save a figure as both detailed and clean versions.

    Saves:
        {output_dir}/{panel_name}.png          — detailed (with all text)
        {output_dir}/{panel_name}.pdf          — detailed
        {output_dir}/{panel_name}_clean.png    — clean (no text)
        {output_dir}/{panel_name}_clean.pdf    — clean
    """
    os.makedirs(output_dir, exist_ok=True)

    # Detailed version
    fig.savefig(os.path.join(output_dir, f'{panel_name}.png'),
                dpi=dpi, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, f'{panel_name}.pdf'),
                dpi=dpi, bbox_inches='tight')

    # Clean version: strip all text, then save
    strip_figure(fig)
    fig.savefig(os.path.join(output_dir, f'{panel_name}_clean.png'),
                dpi=dpi, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, f'{panel_name}_clean.pdf'),
                dpi=dpi, bbox_inches='tight')

    plt.close(fig)
    print(f"  Saved {panel_name} (detailed + clean) → {output_dir}/")
