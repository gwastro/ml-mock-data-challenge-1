#!/usr/bin/env python
"""A program to plot the found and missed injections.
"""
from argparse import ArgumentParser
import h5py
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib import colors as mcolors
import matplotlib.gridspec as mplgrid

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)


def find_injection_times(fgfiles, injfile, padding_start=0, padding_end=0):
    """Determine injections which are contained in the file.
    
    Arguments
    ---------
    fgfiles : list of str
        Paths to the files containing the foreground data (noise +
        injections).
    injfile : str
        Path to the file containing information on the injections in the
        foreground files.
    padding_start : {float, 0}
        The amount of time (in seconds) at the start of each segment
        where no injections are present.
    padding_end : {float, 0}
        The amount of time (in seconds) at the end of each segment
        where no injections are present.
    
    Returns
    -------
    duration:
        A float representing the total duration (in seconds) of all
        foreground files.
    bool-indices:
        A 1D array containing bools that specify which injections are
        contained in the provided foreground files.
    """
    duration = 0
    times = []
    for fpath in fgfiles:
        with h5py.File(fpath, 'r') as fp:
            det = list(fp.keys())[0]
            
            for key in fp[det].keys():
                ds = fp[f'{det}/{key}']
                start = ds.attrs['start_time']
                end = start + len(ds) * ds.attrs['delta_t']
                duration += end - start
                start += padding_start
                end -= padding_end
                if end > start:
                    times.append([start, end])
    
    with h5py.File(injfile, 'r') as fp:
        injtimes = fp['tc'][()]
    
    ret = np.full((len(times), len(injtimes)), False)
    for i, (start, end) in enumerate(times):
        ret[i] = np.logical_and(start <= injtimes, injtimes <= end)
    
    return duration, np.any(ret, axis=0)


def find_closest_index(array, value, assume_sorted=False):
    """Find the index of the closest element in the array for the given
    value(s).
    
    Arguments
    ---------
    array : np.array
        1D numpy array.
    value : number or np.array
        The value(s) of which the closest array element should be found.
    assume_sorted : {bool, False}
        Assume that the array is sorted. May improve evaluation speed.
    
    Returns
    -------
    indices:
        Array of indices. The length is determined by the length of
        value. Each index specifies the element in array that is closest
        to the value at the same position.
    """
    if len(array) == 0:
        raise ValueError('Cannot find closest index for empty input array.')
    if not assume_sorted:
        array = array.copy()
        array.sort()
    ridxs = np.searchsorted(array, value, side='right')
    lidxs = np.maximum(ridxs - 1, 0)
    comp = np.fabs(array[lidxs] - value) < \
           np.fabs(array[np.minimum(ridxs, len(array) - 1)] - value)  # noqa: E127, E501
    lisbetter = np.logical_or((ridxs == len(array)), comp)
    ridxs[lisbetter] -= 1
    return ridxs


class CornerPlot(object):
    def __init__(self, figsize=None, dpi=None, facecolor=None, edgecolor=None,
                 frameon=None, tight_layout=None, constrained_layout=None,
                 nsamples=None):
        self.samples = {}
        self._to_plot = {}
        self.figoptions = {}
        self.nsamples = nsamples
        self.zvalue = None
        self.zvalue_name = None
        self.priors = {}
        self.set_figoption('figsize', figsize,
                           default=plt.rcParams['figure.figsize'])
        self.set_figoption('dpi', dpi,
                           default=plt.rcParams['figure.dpi'])
        if isinstance(facecolor, str):
            facecolor = colors[facecolor]
        self.set_figoption('facecolor', facecolor,
                           default=plt.rcParams['figure.facecolor'])
        if isinstance(edgecolor, str):
            edgecolor = colors[edgecolor]
        self.set_figoption('edgecolor', edgecolor,
                           default=plt.rcParams['figure.edgecolor'])
        self.set_figoption('frameon', frameon,
                           default=plt.rcParams['figure.frameon'])
        self.set_figoption('tight_layout', tight_layout,
                           default=plt.rcParams['figure.autolayout'])
        self.set_figoption('constrained_layout', constrained_layout,
                           default=plt.rcParams['figure.constrained_layout.use'])  # noqa: E501
        return
    
    def set_figoption(self, name, value, default=None):
        if value is None:
            value = default
        self.figoptions[name] = value
    
    def __len__(self):
        return len(self.samples)
    
    def add_distribution(self, name, samples, plot=True):
        if name in self.samples:
            raise ValueError('Parameter names must be unique!')
        
        samples = np.asarray(samples)
        if samples.ndim != 1:
            raise ValueError(('Samples must be a 1 dimensional array-like '
                              'object.'))
        if self.nsamples is None:
            self.nsamples = len(samples)
        if len(samples) != self.nsamples:
            raise ValueError('Trying to add samples with mismatching length.')
        self.samples[name] = samples
        self._to_plot[name] = plot
    
    def remove_distribution(self, name):
        return self.samples.pop(name, None)
    
    def update_distribution(self, name, samples=None, plot=True):
        if name not in self.samples and samples is None:
            raise ValueError(f'No samples are attributed to {name} yet.')
        if samples is not None:
            samples = np.asarray(samples)
            if self.nsamples is None:
                self.nsamples = len(samples)
            if len(samples) != self.nsamples:
                raise ValueError(('Trying to add samples with mismatching '
                                  'length.'))
            self.samples[name] = samples
        self._to_plot[name] = plot
    
    def set_prior(self, name, samples):
        samples = np.asarray(samples)
        if samples.ndim != 1:
            raise ValueError(('Priors must be a 1 dimensional array-like '
                              'object.'))
        self.priors[name] = samples
    
    def remove_prior(self, name):
        return self.priors.pop(name, None)
    
    def activate_plot(self, name):
        if name in self._to_plot:
            self._to_plot[name] = True
        else:
            raise KeyError(f'Unknown distribution {name}')
    
    def deactivate_plot(self, name):
        if name in self._to_plot:
            self._to_plot[name] = False
        else:
            raise KeyError(f'Unknown distribution {name}')
    
    def add_zvalue(self, samples, name=None):
        if self.zvalue is not None:
            raise RuntimeError(('zvalue was already set. Use `update_zvalue` '
                                'instead.'))
        samples = np.asarray(samples)
        if self.nsamples is None:
            self.nsamples = len(samples)
        if len(samples) != self.nsamples:
            raise ValueError('Trying to add zvalue with mismatching length.')
        self.zvalue = samples
        self.zvalue_name = name
    
    def update_zvalue(self, samples=None, name=None):
        if samples is None:
            self.zvalue_name = name
            return
        if self.zvalue is None:
            self.add_zvalue(samples, name=name)
        samples = np.asarray(samples)
        if len(samples) != self.nsamples:
            raise ValueError(('Trying to update zvalue with mismatching '
                              'length.'))
        self.zvalue = samples
        if name is not None:
            self.zvalue_name = name
    
    def delete_zvalue(self):
        self.zvalue = None
        self.zvalue_name = None
    
    @property
    def plotting(self):
        ret = []
        for name, to_plot in self._to_plot.items():
            if to_plot:
                ret.append(name)
        return ret
    
    def plot(self, names=None, bins=None, normalize=True, grid=True,
             cmap='viridis', marker=None, s=None, limits=None,
             scatter_color=None, hist_color=None):
        if names is None:
            names = self.plotting
        if limits is None:
            limits = {}
        
        for name in names:
            if name not in limits:
                limits[name] = None
        
        if isinstance(names, (list, tuple, np.ndarray)):
            names = list(names)
        else:
            names = [names]
        if isinstance(bins, (list, tuple, np.ndarray)):
            bins = list(bins)
        else:
            bins = [bins] * len(names)
        
        if len(names) != len(bins):
            if len(bins) == 1:
                bins = bins * len(names)
            else:
                raise ValueError(('The length of `bins` passed to this method '
                                  'has to be equal to the number of '
                                  'distributions plotted.'))
        cbar = None
        if len(names) < 0:
            raise ValueError('Nothing to plot.')
        elif len(names) == 1:
            fig, axs = plt.subplots(nrows=1, ncols=1, **self.figoptions)
            axs.hist(self.samples[names[0]], density=normalize, bins=bins[0])
            
            axs.set_title(names[0])
        else:
            fig = plt.figure(**self.figoptions)
            master = mplgrid.GridSpec(1, 2, width_ratios=[len(names), 0.2],
                                      figure=fig)
            gridspec = mplgrid.GridSpecFromSubplotSpec(len(names),
                                                       len(names),
                                                       subplot_spec=master[0],
                                                       hspace=0,
                                                       wspace=0)
            cbarspec = mplgrid.GridSpecFromSubplotSpec(1, 1,
                                                       subplot_spec=master[1])
            idxs = [(i, i) for i in range(len(names))]
            for col in range(len(names)):
                for row in range(col+1, len(names)):
                    idxs.append((row, col))
            axs = {}
            for row, col in idxs:
                name1 = names[row]
                name2 = names[col]
                if name1 == name2:  # Histograms
                    xaxtick_params = {'axis': 'x',
                                      'direction': 'in'}
                    ax = fig.add_subplot(gridspec[row, col])
                    if row != len(names) - 1:  # Bottom right corner
                        xaxtick_params['labelbottom'] = False
                    ax.tick_params(**xaxtick_params)
                    ax.tick_params(axis='y', labelleft=False)
                    
                    if name1 in self.priors:
                        cbins = plt.rcParams['hist.bins']
                        if bins[row] is not None:
                            cbins = bins[row]
                        cbins = np.histogram_bin_edges(self.priors[name1],
                                                       bins=cbins)
                        
                        psamps = np.digitize(self.priors[name1], cbins)
                        psamps[psamps == len(cbins)] -= 1
                        psamps = np.bincount(psamps, minlength=len(cbins))
                        samps = np.digitize(self.samples[name1], cbins)
                        samps[samps == len(cbins)] -= 1
                        samps = np.bincount(samps, minlength=len(cbins))
                        
                        width = (cbins.max() - cbins.min()) / len(cbins)
                        
                        if normalize:
                            norm = np.sum(psamps)
                            samps = samps / norm
                            psamps = psamps / norm
                        
                        ax.bar(cbins, psamps, color='grey', alpha=0.5,
                               align='edge', width=width)
                        ax.bar(cbins, samps, color=hist_color,
                               align='edge', width=width)
                    else:
                        ax.hist(self.samples[name1], density=normalize,
                                bins=bins[row], color=hist_color)
                    ax.set_title(name1)
                else:  # Scatter plots
                    xaxtick_params = {'axis': 'x'}
                    yaxtick_params = {'axis': 'y'}
                    if col == 0:
                        ax = fig.add_subplot(gridspec[row, col],
                                             sharex=axs[(col, col)])
                    else:
                        ax = fig.add_subplot(gridspec[row, col],
                                             sharex=axs[(col, col)],
                                             sharey=axs[(row, 0)])
                    if row != len(names) - 1:
                        xaxtick_params['labelbottom'] = False
                        xaxtick_params['direction'] = 'in'
                    if col != 0:
                        yaxtick_params['labelleft'] = False
                        yaxtick_params['direction'] = 'in'
                    ax.tick_params(**xaxtick_params)
                    ax.tick_params(**yaxtick_params)
                    
                    vmin = None if self.zvalue is None else min(self.zvalue)
                    vmax = None if self.zvalue is None else max(self.zvalue)
                    if self.zvalue is None:
                        ccmap = None
                    else:
                        ccmap = cmap
                    color = self.zvalue
                    if scatter_color is not None:
                        if isinstance(scatter_color, str):
                            color = colors[scatter_color]
                        else:
                            color = scatter_color
                    ax.scatter(self.samples[name2],
                               self.samples[name1],
                               c=color,
                               vmin=vmin,
                               vmax=vmax,
                               marker=marker,
                               s=s,
                               cmap=ccmap)
                axs[(row, col)] = ax
                if grid:
                    ax.grid()
            
            if self.zvalue is not None:
                cax = fig.add_subplot(cbarspec[0])
                norm = Normalize(vmin=vmin, vmax=vmax)
                cbar = fig.colorbar(ScalarMappable(norm=norm,
                                                   cmap=cmap),
                                    cax=cax)
            
            for i, name in enumerate(names):
                ax = axs[(i, i)]
                if limits[name] is None:
                    limits[name] = ax.get_xlim()
                ax.set_xlim(limits[name])
            
            idxs = []
            for row in range(len(names)):
                for col in range(row+1):
                    idxs.append((row, col))
            
            axs = [axs[idx] for idx in idxs]
            
        return fig, axs, cbar
    
    def show(self, names=None, bins=None):
        fig, axs, cbar = self.plot(names=names, bins=bins)
        fig.tight_layout()
        fig.show()
    
    def save(self, fpath, names=None):
        fig, axs, cbar = self.plot(names=names)
        fig.tight_layout()
        fig.savefig(fpath)


def main(desc):
    parser = ArgumentParser(description=desc)
    
    parser.add_argument('--foreground', type=str, nargs='+', required=True,
                        help=("Path to the file(s) containing the analyzed "
                              "foreground data output by"
                              "`generate_data.py --output-foreground-file`."))
    parser.add_argument('--events', type=str, nargs='+', required=True,
                        help=("Path to the file(s) containing the events "
                              "returned by the search on the foreground "
                              "data set as returned by "
                              "`generate_data.py --output-foreground-file`."))
    parser.add_argument('--injections', type=str, required=True,
                        help=("Path to the file containing information "
                              "on the injections. (The file returned by"
                              "`generate_data.py --output-injection-file`"))
    parser.add_argument('--parameters', type=str, nargs='+',
                        help=("Which injection parameters to plot. Default: "
                              "All parameters are plotted."))
    parser.add_argument('--output-found', type=str,
                        help=("Path at which to store the corner plot for the "
                              "found injections."))
    parser.add_argument('--output-missed', type=str,
                        help=("Path at which to store the corner plot for the "
                              "missed injections."))
    parser.add_argument('--store-data', type=str,
                        help=("Path at which to store the missed and found "
                              "injection parameters."))
    parser.add_argument('--verbose', action='store_true',
                        help="Print status updates.")
    parser.add_argument('--force', action='store_true',
                        help="Overwrite existing files.")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARN
    logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s',
                        level=log_level, datefmt='%d-%m-%Y %H:%M:%S')
    
    if args.output_found is None and args.output_missed is None:
        logging.info('Nothing to save.')
        return
    
    if args.output_found is not None and os.path.isfile(args.output_found):
        if not args.force:
            raise IOError(('Cannot overwrite the file at '
                           f'{args.output_found}. Set the flag `--force` to '
                           'do so.'))
    if args.output_missed is not None and os.path.isfile(args.output_missed):
        if not args.force:
            raise IOError(('Cannot overwrite the file at '
                           f'{args.output_missed}. Set the flag `--force` to '
                           'do so.'))
    if args.store_data is not None and os.path.isfile(args.store_data):
        if not args.force:
            raise IOError(('Cannot overwrite the file at '
                           f'{args.store_data}. Set the flag `--force` to '
                           'do so.'))
    
    # Find indices contained in foreground file
    logging.info('Finding injections contained in data')
    padding_start, padding_end = 30, 30
    dur, idxs = find_injection_times(args.foreground,
                                     args.injections,
                                     padding_start=padding_start,
                                     padding_end=padding_end)
    
    # Read injection parameters
    logging.info(f'Reading injections from {args.injections}')
    injparams = {}
    with h5py.File(args.injections, 'r') as fp:
        if args.parameters is None:
            args.parameters = list(fp.keys())
        for key in args.parameters:
            injparams[key] = fp[key][idxs]
        injtimes = fp['tc'][idxs]
    
    # Read foreground events
    logging.info(f'Reading events from {args.events}')
    fgevents = []
    for fpath in args.events:
        with h5py.File(fpath, 'r') as fp:
            fgevents.append(np.vstack([fp['time'][()],
                                       fp['stat'][()],
                                       fp['var'][()]]))
    fgevents = np.concatenate(fgevents, axis=-1)
    
    # Sort events by time
    sidxs = fgevents[0].argsort()
    fgevents = fgevents.T[sidxs].T
    
    logging.info('Finding injection times closest to event times')
    idxs = find_closest_index(injtimes, fgevents[0])
    diff = np.abs(injtimes[idxs] - fgevents[0])
    
    logging.info('Determining found injections')
    tpidxs = np.arange(len(fgevents[0]))[diff <= fgevents[2]]
    fidxs = idxs[tpidxs]
    midxs = np.setdiff1d(np.arange(len(injtimes)), fidxs)
    
    if args.store_data:
        logging.info('Storing found and missed injections')
        mode = 'w' if args.force else 'x'
        with h5py.File(args.store_data, mode) as fp:
            found = fp.create_group('found')
            missed = fp.create_group('missed')
            for key, val in injparams.items():
                found.create_dataset(key, data=val[fidxs])
                missed.create_dataset(key, data=val[midxs])
            found.create_dataset('stat', data=fgevents[1][tpidxs])
        logging.info(('Stored found and missed injections to '
                      f'{args.store_data}'))
    
    if args.output_found is not None:
        logging.info('Plotting found injections')
        fplot = CornerPlot(figsize=(19.2, 10.8))
        for key, val in injparams.items():
            fplot.add_distribution(key, val[fidxs])
            fplot.set_prior(key, injparams[key])
        fplot.add_zvalue(name='Stat', samples=fgevents[1][tpidxs])
        fplot.save(args.output_found)
        logging.info(f'Stored plot of found injections at {args.output_found}')
    
    if args.output_missed is not None:
        logging.info('Plotting missed injections')
        mplot = CornerPlot(figsize=(19.2, 10.8))
        for key, val in injparams.items():
            mplot.add_distribution(key, val[midxs])
            mplot.set_prior(key, injparams[key])
        fig, axs, cbar = mplot.plot(marker='x', scatter_color='red')
        fig.tight_layout()
        fig.savefig(args.output_missed)
        logging.info(('Stored plot of missed injections at '
                      f'{args.output_missed}'))
    
    logging.info('Finished')
    return


if __name__ == "__main__":
    main(__doc__)
