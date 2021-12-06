#!/usr/bin/env python
"""Programm to plot the output of evaluate.py.

Basic usage:
./sensitivity_plot.py --files path1 path2 ... --output outpath
"""
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import os
import logging
import h5py

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
seconds_per_month = 30 * 24 * 60 * 60

def figsize_type(inp):
    inp = inp.replace(' ', '')
    if not inp.startswith('(') or not inp.endswith(')'):
        raise ValueError('Wrong format')
    inp = inp[1:-1]
    tmp = inp.split(',')
    if len(tmp) != 2:
        raise ValueError('Wrong format')
    x, y = tmp
    x = float(x)
    y = float(y)
    return x, y

def none_type(inp):
    if inp.lower() == 'none':
        return None
    return inp

def color_type(inp):
    if inp.lower() == 'none':
        return None
    if inp[0] == '#':
        return inp
    return colors[inp]

def limit_type(inp):
    inp = inp.replace(' ', '')
    if not inp.startswith('(') or not inp.endswith(')'):
        raise ValueError('Wrong format')
    inp = inp[1:-1]
    tmp = inp.split(',')
    if len(tmp) != 2:
        raise ValueError('Wrong format')
    lower, upper = tmp
    if lower.lower() == 'none':
        lower = None
    else:
        lower = float(lower)
    if upper.lower() == 'none':
        upper = None
    else:
        upper = float(upper)
    return lower, upper

def main(desc):
    print(desc)
    parser = ArgumentParser(description=desc)
    
    parser.add_argument('--files', type=str, nargs='+', required=True,
                        help="Path to the result file(s) as produced by `evaluate.py`")
    parser.add_argument('--output', type=str, required=True,
                        help="Path at which to store the output plot.")
    parser.add_argument('--far-scaling-factor', type=float, default=seconds_per_month,
                        help="The factor by which to scale the FAR-values. The far-values (which are in units [samples/second]) are multiplied by this value.")
    
    parser.add_argument('--figsize', type=figsize_type, default=(19.2, 10.8),
                        help="The size of the final plot in inches.")
    parser.add_argument('--dpi', type=int, default=100,
                        help="The DPI at which to generate the figure.")
    parser.add_argument('--labels', type=none_type, nargs='+',
                        help="Labels to add to the different files. Type `none` for no label. If any label is given the number of arguments must match the number of `--files` given.")
    parser.add_argument('--colors', type=color_type, nargs='+',
                        help="Colors to use for the different files. Type `none` to use the default color cycle for a particular color. All named colors as well as all hex-colors are allowed. If any color is given the number of arguments must match the number of `--files` given. (Hex-colors have to be entered as `\\#000000`)")
    parser.add_argument('--no-legend', action='store_true',
                        help="Put no legend into the plot.")
    parser.add_argument('--no-grid', action='store_true',
                        help="Display no grid.")
    parser.add_argument('--no-tex', action='store_true',
                        help="Do not use LaTeX to render text.")
    parser.add_argument('--xlabel', type=str, default='False alarms [1/month]',
                        help="The x-label to apply to the plot.")
    parser.add_argument('--ylabel', type=str, default="Sensitive distance [Mpc]")
    parser.add_argument('--title', type=str,
                        help="Set a title to the plot.")
    parser.add_argument('--xlim', type=limit_type, default=(None, None),
                        help="Set the limit on the x-axis. Format `(lower,upper)`. Either of the two value `lower` or `upper` or both may be `none`.")
    parser.add_argument('--ylim', type=limit_type, default=(None, None),
                        help="Set the limit on the y-axis. Format `(lower,upper)`. Either of the two value `lower` or `upper` or both may be `none`.")
    
    parser.add_argument('--show', action='store_true',
                        help="Show the resulting plot. Otherwise it will only be stored.")
    parser.add_argument('--verbose', action='store_true',
                        help="Print status updates.")
    parser.add_argument('--force', action='store_true',
                        help="Overwrite existing files.")
    
    args = parser.parse_args()
    
    log_level = logging.INFO if args.verbose else logging.WARN
    
    logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s',
                        level=log_level, datefmt='%d-%m-%Y %H:%M:%S')
    logging.info('Starting')
    
    if os.path.isfile(args.output) and not args.force:
        raise IOError(f'The file {args.output} already exists. Set the flag `--force` to overwrite it.')
    
    if args.labels is None:
        args.labels = args.files
    if len(args.labels) != len(args.files):
        raise ValueError('Length of files and labels must match if any label is given.')
    
    if args.colors is None:
        args.colors = [None] * len(args.files)
    if len(args.colors) != len(args.files):
        raise ValueError('Length of files and colors must match if any color is given.')
    
    plt.rcParams.update({'text.usetex': not args.no_tex})
    fig, ax = plt.subplots(figsize=args.figsize, dpi=args.dpi)
    
    for path, label, color in zip(args.files, args.labels, args.colors):
        logging.info(f'Loading data from {path}')
        with h5py.File(path, 'r') as fp:
            far = fp['far'][()]
            sens = fp['sensitive-distance'][()]
            sidxs = far.argsort()
            far = far[sidxs][1:] * args.far_scaling_factor
            sens = sens[sidxs][1:]
        logging.info(f'Plotting data from {path}')
        ax.semilogx(far, sens, label=label, color=color)
    
    xmin, xmax = ax.get_xlim()
    if args.xlim[0] is not None:
        xmin = args.xlim[0]
    if args.xlim[1] is not None:
        xmax = args.xlim[1]
    ax.set_xlim(xmax, xmin)
    
    ymin, ymax = ax.get_ylim()
    if args.ylim[0] is not None:
        ymin = args.ylim[0]
    if args.ylim[1] is not None:
        ymax = args.ylim[1]
    ax.set_ylim(ymin, ymax)
    
    if not args.no_legend and args.labels.count(None) < len(args.labels):
        ax.legend()
    if not args.no_grid:
        ax.grid()
    if args.title is not None:
        ax.title(args.title)
    
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    
    fig.savefig(args.output)
    logging.info(f'Plot saved at {args.output}')
    if args.show:
        plt.show()
    plt.cla()
    plt.clf()
    logging.info('Finished.')
    return

if __name__ == "__main__":
    main(__doc__)
