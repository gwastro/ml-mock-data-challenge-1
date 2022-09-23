#!/usr/bin/env python
from argparse import ArgumentParser
from tensorflow import keras
import numpy as np
import h5py
import os
import time as timemodule
from pycbc.events import time_coincidence
from pycbc.types import TimeSeries
from tqdm import tqdm
import multiprocessing as mp

from reg_loss import reg_loss
from slice_generator import SliceGenerator


def combined_statistic(stat1, stat2):
    return stat1 + stat2 + np.log(1 + np.exp(-stat1) + np.exp(-stat2))


def get_model():
    inp1 = keras.layers.Input((4 * 2048, 1))
    inp2 = keras.layers.Input((4 * 2048, 1))
    inputs = [inp1, inp2]
    main_model = keras.models.load_model('model',
                                         custom_objects={'reg_loss': reg_loss},
                                         compile=False)
    main_model.layers[-1].activation = keras.activations.linear
    out1 = main_model(inp1)
    out2 = main_model(inp2)
    outputs = [out1, out2]
    return keras.models.Model(inputs=inputs, outputs=outputs)


def worker(arg):
    inpath, det, key = arg
    with h5py.File(inpath, 'r') as fp:
        ds = fp[f'{det}/{key}']
        st = ds.attrs['start_time']
        dt = ds.attrs['delta_t']
        ts = TimeSeries(ds[:], epoch=st, delta_t=dt)
        ts = ts.whiten(4, 4)
    return det, key, ts.numpy(), float(ts.start_time), ts.delta_t


def evaluate(times, stat1, stat2, trigger_threshold=3.4,
             cluster_threshold=0.2, coinc_window=0.1):
    rawtimes = times
    res = [stat1, stat2]
    # Single detector events
    single_det_events = []
    for i in range(len(res)):
        # Create triggers
        rawstat = res[i]
        idxs = np.where(rawstat > trigger_threshold)[0]
        
        # Cluster triggers
        clusters = [[rawtimes[idxs[0]], rawtimes[idxs[0]]]]
        clusteridxs = [[idxs[0], idxs[0]]]
        for idx in idxs:
            time = rawtimes[idx]
            bound = clusters[-1][-1]
            if time - bound > cluster_threshold:
                clusters.append([time, time])
                clusteridxs.append([idx, idx])
            else:
                clusters[-1][-1] = time
                clusteridxs[-1][-1] = idx
        
        # Create single detector events
        events = []
        for lidx, ridx in clusteridxs:
            cluster_times = rawtimes[lidx:ridx+1]
            cluster_vals = rawstat[lidx:ridx+1]
            cidx = np.argmax(cluster_vals)
            events.append([cluster_times[cidx], cluster_vals[cidx]])
        
        single_det_events.append(events)
    
    # Coincident events
    times1 = np.array([pt[0] for pt in single_det_events[0]])
    times2 = np.array([pt[0] for pt in single_det_events[1]])
    vals1 = np.array([pt[1] for pt in single_det_events[0]])
    vals2 = np.array([pt[1] for pt in single_det_events[1]])
    idxs1, idxs2, _ = time_coincidence(times1,
                                       times2,
                                       window=coinc_window)
    rettimes = times1[idxs1]
    retvalues = combined_statistic(vals1[idxs1], vals2[idxs2])
    
    return rettimes, retvalues


def main():
    parser = ArgumentParser()
    
    parser.add_argument('inpath', type=str,
                        help="Path of the data file to analyze.")
    parser.add_argument('outpath', type=str,
                        help="Path at which to store the output file.")
    
    args = parser.parse_args()
    
    model = get_model()
    
    tmpname = f'white-{int(timemodule.time())}.hdf'
    tmppath = os.path.join(os.path.split(args.outpath)[0], tmpname)
    with h5py.File(args.inpath, 'r') as infile, h5py.File(tmppath, 'w') as outfile:  # noqa: E501
        dets = list(infile.keys())
        keys = list(sorted(infile[dets[0]].keys(), key=int))
        for key, val in dict(infile.attrs).items():
            outfile.attrs[key] = val
        
        for det in dets:
            outfile.create_group(det)
        
        poolargs = []
        for det in dets:
            for key in keys:
                poolargs.append((args.inpath, det, key))
        total = len(dets) * len(keys)
        
        with mp.Pool() as pool:
            for res in tqdm(pool.imap_unordered(worker, poolargs),
                            total=total,
                            ascii=True,
                            desc='Whitening data'):
                det, key, data, st, dt = res
                outfile[det].create_dataset(key, data=data)
                outfile[det][key].attrs['start_time'] = st
                outfile[det][key].attrs['delta_t'] = dt

    gen = SliceGenerator(tmppath,
                         workers=16,
                         prefetch=10,
                         window_size=4 * 2048,
                         pre_whitened=True,
                         batch_size=512,
                         time_offset=3)
    with gen:
        res = model.predict(gen, verbose=1, steps=len(gen), workers=0,
                            use_multiprocessing=False)
    
    # Store output if requested
    rawtimes = gen.sample_times()
    
    os.remove(tmppath)
    
    # Calculate events
    times, values = evaluate(rawtimes,
                             res[0].T[0] - res[0].T[1],
                             res[1].T[0] - res[1].T[1])
    
    # Store output
    with h5py.File(args.outpath, 'w') as fp:
        fp.create_dataset('time', data=times)
        fp.create_dataset('stat', data=values)
        fp.create_dataset('var', data=np.full(len(times), 0.3))
    return


if __name__ == "__main__":
    main()
