#!/usr/bin/env python
"""Docstring
"""
import argparse
import numpy as np
import h5py

def get_stats(events, injparams):
    ret = {}
    injtimes = injparams['tc']
    sidxs = events[0].argsort()
    events = events.T[sidxs].T
    
    maxidxs = np.searchsorted(injtimes, events[0], side='right')
    minidxs = np.maximum(minidxs - 1, 0)
    maxidxs = np.minimum(maxidxs, len(injtimes) - 1)
    lowdiff = np.abs(injtimes[minidxs] - events[0])
    highdiff = np.abs(injtimes[maxidxs] - events[0])
    
    mmidxs = np.vstack([minidxs, maxidxs])
    lhdiff = np.vstack([lowdiff, highdiff])
    tmpidxs = np.argmin(lhdiff, axis=0)
    boolidxs = np.vstack([tmpidxs == 0, tmpidxs == 1])
    #The indices below are the indices of the closest injections
    idxs = mmidxs[boolidxs]
    diff = lhdiff[boolidxs]
    
    tpidxs = np.arange(len(events[0]))[diff <= events[2]]
    fpidxs = np.arange(len(events[0]))[diff > events[2]]
    
    tpevents = events.T[tpidxs].T
    fpevents = events.T[fpidxs].T
    
    ret['found-indices'] = np.arange(len(injtimes))[idxs]
    ret['missed-indices'] = np.setdiff1d(np.arange(len(injtimes)), ret['found-indices'])
    ret['true-positive-event-indices'] = tpidxs
    ret['false-positive-event-indices'] = fpidxs
    ret['true-positive-diffs'] = diff[tpidxs]
    ret['false-positive-diffs'] = diff[fpidxs]
    ret['true-positives'] = tpevents
    ret['false-positives'] = fpevents
    
    ret['average-separation'] = np.mean(ret['true-positive-diffs'])
    
    #Do FAR calculation
    #Do sensitivity calculation
    

def main(doc):
    parser = argparse.ArgumentParser(description=doc)
    
    parser.add_argument('--injection-file', type=str, required=True,
                        help=("Path to the file containing information "
                              "on the injections. (The file returned by"
                              "`generate_data.py --output-injection-file`"))
    parser.add_argument('--foreground-events', type=str, required=True,
                        help=("Path to the file containing the events "
                              "returned by the search on the foreground "
                              "data set as returned by "
                              "`generate_data.py --output-foreground-file`. <Describe file content>"))
    parser.add_argument('--background-events', type=str, required=True,
                        help=("Path to the file containing the events "
                              "returned by the search on the background"
                              "data set as returned by "
                              "`generate_data.py --output-background-file`. <Describe file content>"))
    
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
    
    #Read injection parameters
    logging.info(f'Reading injections from {args.injection_file}')
    injparams = {}
    with h5py.File(args.injection_file, 'r') as fp:
        for key in fp.keys():
            injparams[key] = fp[key][()]
    
    #Read foreground events
    logging.info(f'Reading foreground events from {args.foreground_events}')
    with h5py.File(args.foreground_events, 'r') as fp:
        fg_events = np.vstack([fp['time'],
                               fp['stat'],
                               fp['var']])
    
    #Read background events
    logging.info(f'Reading background events from {args.background_events}')
    with h5py.File(args.background_events, 'r') as fp:
        bg_events = np.vstack([fp['time'],
                               fp['stat'],
                               fp['var']])
    
    fg_stats = get_stats(fg_events, injparams)
    return

if __name__ == "__main__":
    main(__doc__)
