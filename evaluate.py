#!/usr/bin/env python
"""Docstring
"""
import argparse
import numpy as np
import h5py
import os
import logging

def find_injection_times(fgfiles, injfile, padding_start=0, padding_end=0):
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

def get_stats(fgevents, bgevents, injparams, duration=None):
    ret = {}
    injtimes = injparams['tc']
    dist = injparams['distance']
    if duration is None:
        duration = injtime.max() - injtimes.min()
    logging.info('Sorting foreground event times')
    sidxs = fgevents[0].argsort()
    fgevents = fgevents.T[sidxs].T
    
    logging.info('Finding injection times closest to event times')
    maxidxs = np.searchsorted(injtimes, fgevents[0], side='right')
    minidxs = np.maximum(maxidxs - 1, 0)
    maxidxs = np.minimum(maxidxs, len(injtimes) - 1)
    lowdiff = np.abs(injtimes[minidxs] - fgevents[0])
    highdiff = np.abs(injtimes[maxidxs] - fgevents[0])
    
    mmidxs = np.vstack([minidxs, maxidxs])
    lhdiff = np.vstack([lowdiff, highdiff])
    tmpidxs = np.argmin(lhdiff, axis=0)
    boolidxs = np.vstack([tmpidxs == 0, tmpidxs == 1])
    #The indices below are the indices of the closest injections
    idxs = mmidxs[boolidxs]
    diff = lhdiff[boolidxs]
    
    logging.info('Finding true- and false-positives')
    tpidxs = np.arange(len(fgevents[0]))[diff <= fgevents[2]]
    fpidxs = np.arange(len(fgevents[0]))[diff > fgevents[2]]
    
    tpevents = fgevents.T[tpidxs].T
    fpevents = fgevents.T[fpidxs].T
    ufpvals = np.unique(fpevents[1])
    
    ret['fg-events'] = fgevents
    ret['found-indices'] = np.arange(len(injtimes))[idxs]
    ret['missed-indices'] = np.setdiff1d(np.arange(len(injtimes)),
                                         ret['found-indices'])
    ret['true-positive-event-indices'] = tpidxs
    ret['false-positive-event-indices'] = fpidxs
    ret['sorting-indices'] = sidxs
    ret['true-positive-diffs'] = diff[tpidxs]
    ret['false-positive-diffs'] = diff[fpidxs]
    ret['true-positives'] = tpevents
    ret['false-positives'] = fpevents
    
    # ret['average-separation'] = np.mean(ret['true-positive-diffs'])
    
    #Calculate foreground FAR
    logging.info('Calculating foreground FAR')
    noise_stats = fpevents[1].copy()
    noise_stats.sort()
    fgfar = len(noise_stats) - np.searchsorted(noise_stats, tpevents[1],
                                               side='left')
    fgfar = fgfar / duration
    ret['fg-far'] = fgfar
    sfaridxs = fgfar.argsort()
    
    #Calculate background FAR
    logging.info('Calculating background FAR')
    noise_stats = bgevents[1].copy()
    noise_stats.sort()
    far = len(noise_stats) - np.searchsorted(noise_stats, tpevents[1],
                                             side='left')
    far = far / duration
    ret['far'] = far
    
    #Calculate sensitivity
    #CARE! THIS APPLIES ONLY WHEN THE DISTRIBUTION IS CHOSEN CORRECTLY
    logging.info('Calculating sensitivity')
    tp_sort = tpevents[1].copy()
    tp_sort.sort()
    max_distance = dist.max()
    vtot = (4. / 3.) * np.pi * max_distance**3.
    Ninj = len(dist)
    prefactor = vtot / Ninj
    
    nfound = len(tp_sort) - np.searchsorted(tp_sort, tpevents[1],
                                            side='left')
    sample_variance = nfound / Ninj - (nfound / Ninj) ** 2
    vol = prefactor * nfound
    vol_err = prefactor * (Ninj * sample_variance) ** 0.5
    rad = (3 * vol / (4 * np.pi))**(1. / 3.)
    
    ret['sensitive-volume'] = vol
    ret['sensitive-distance'] = rad
    ret['sensitive-volume-error'] = vol_err
    ret['sensitive-fraction'] = nfound / Ninj
        
    return ret

def main(doc):
    parser = argparse.ArgumentParser(description=doc)
    
    parser.add_argument('--injection-file', type=str, required=True,
                        help=("Path to the file containing information "
                              "on the injections. (The file returned by"
                              "`generate_data.py --output-injection-file`"))
    parser.add_argument('--foreground-events', type=str, nargs='+', required=True,
                        help=("Path to the file containing the events "
                              "returned by the search on the foreground "
                              "data set as returned by "
                              "`generate_data.py --output-foreground-file`."))
    parser.add_argument('--foreground-files', type=str, nargs='+', required=True,
                        help=("Path to the file containing the analyzed "
                              "foreground data output by"
                              "`generate_data.py --output-foreground-file`."))
    parser.add_argument('--background-events', type=str, required=True, nargs='+',
                        help=("Path to the file containing the events "
                              "returned by the search on the background"
                              "data set as returned by "
                              "`generate_data.py --output-background-file`."))
    parser.add_argument('--output-file', type=str, required=True,
                        help=("Path at which to store the output HDF5 "
                              "file. (Path must end in `.df`)"))
    # parser.add_argument('--duration', type=float,
    #                     help="Set the duration of analyzed data.")
    
    parser.add_argument('--verbose', action='store_true',
                        help="Print update messages.")
    parser.add_argument('--force', action='store_true',
                        help="Overwrite existing files.")
    
    args = parser.parse_args()
    
    #Setup logging
    log_level = logging.INFO if args.verbose else logging.WARN
    logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s',
                        level=log_level, datefmt='%d-%m-%Y %H:%M:%S')
    
    #Sanity check arguments here
    if os.path.splitext(args.output_file)[1] != '.hdf':
        raise ValueError(f'The output file must have the extension `.hdf`.')
    
    if os.path.isfile(args.output_file) and not args.force:
        raise IOError(f'The file {args.output_file} already exists. Set the flag `force` to overwrite it.')
    
    #Find indices contained in foreground file
    logging.info(f'Finding injections contained in data')
    dur, idxs = find_injection_times(args.foreground_files,
                                     args.injection_file,
                                     padding_start=30,
                                     padding_end=30)
    
    #Read injection parameters
    logging.info(f'Reading injections from {args.injection_file}')
    injparams = {}
    with h5py.File(args.injection_file, 'r') as fp:
        injparams['tc'] = fp['tc'][()][idxs]
        injparams['distance'] = fp['distance'][()][idxs]
    
    #Read foreground events
    logging.info(f'Reading foreground events from {args.foreground_events}')
    fg_events = []
    for fpath in args.foreground_events:
        with h5py.File(fpath, 'r') as fp:
            fg_events.append(np.vstack([fp['time'],
                                        fp['stat'],
                                        fp['var']]))
    fg_events = np.concatenate(fg_events, axis=-1)
    
    #Read background events
    logging.info(f'Reading background events from {args.background_events}')
    bg_events = []
    for fpath in args.background_events:
        with h5py.File(fpath, 'r') as fp:
            bg_events.append(np.vstack([fp['time'],
                                        fp['stat'],
                                        fp['var']]))
    bg_events = np.concatenate(bg_events, axis=-1)
    
    stats = get_stats(fg_events, bg_events, injparams,
                      duration=dur)
    
    
    #Store results
    logging.info(f'Writing output to {args.output_file}')
    mode = 'w' if args.force else 'x'
    with h5py.File(args.output_file, mode) as fp:
        for key, val in stats.items():
            fp.create_dataset(key, data=np.array(val))
    return

if __name__ == "__main__":
    main(__doc__)
