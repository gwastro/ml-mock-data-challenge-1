#!/usr/bin/env python
"""The docstring describing this program.
"""
import argparse
import numpy as np
import h5py
import os
import sys
import logging
import warnings
from shutil import copy
import subprocess
import time
import urllib.request

from pycbc.noise.reproduceable import colored_noise
import pycbc.psd
from pycbc.types import FrequencySeries, TimeSeries
from pycbc.inject import InjectionSet

from segments import OverlapSegment, SegmentList

#ToDos:
#-Implement storage of command into final file
#-Add a docstring
#-Add reference to what kind of data the different sets contain

TIME_STEP = 16
TIME_WINDOW = 6

def check_file_existence(fpath, force):
    if fpath is not None:
        if os.path.isfile(fpath) and not force:
            msg = f'The file {fpath} already exists. Set the flag '
            msg += '`--force` to overwrite existing files.'
            raise IOError(msg)

def base_path():
    return os.path.split(os.path.abspath(__file__))[0]

def get_default_path():
    return os.path.join(base_path(), 'real_noise_file.hdf')

def download_data(path):
    """Download noise data from the central server.
    
    Arguments
    ---------
    path : str
        Path at which to store the file. Must end in `.hdf`.
    """
    url = 'MISSING'
    urllib.request.urlretrieve(url, path)

def get_real_noise(path=None, min_segment_duration=None, start=0,
                   duration=2592000, slide_buffer=None):
    """Get noise from a file as a SegmentList.
    
    Arguments
    ---------
    path : {str or None, None}
        Path from which to load the noise. If None a default path will
        be used. If no file is found at the path, the data will be
        downloaded from a central server.
    min_segment_duration : {float or None, None}
        The minimum duration each segment should have (in seconds).
    start : {float, 0}
        The abstract start time. This is the amount of time to skip in
        at the beginning of the data.
    duration : {float, 2592000}
        The minimum duration of the noise to grab. May be exceeded by
        up to min_segment_duration.
    slide_buffer : {float or None, None}
        The amount of time for each segment that is not usable but
        reserved to shift data between detectors.
    
    Returns
    -------
    SegmentList:
        The SegmentList containing the noise.
    """
    if path is None:
        path = get_default_path()
    if slide_buffer is None:
        slide_buffer = 0
    
    if not os.path.isfile(path):
        download_data(path)
    
    seglist = SegmentList()
    with h5py.File(path, 'r') as fp:
        dets = list(fp.keys())
        start_times = set([])
        for det in dets:
            start_times = start_times.union(set(fp[det].keys()))
    
        start_times = sorted(list(start_times), key=lambda inp: int(inp))
        
        curr_dur = -start
        for st in start_times:
            #If duration requirement is met, stop adding more data
            if curr_dur >= duration:
                break;
            
            #Check where to start
            dt = fp[f'{dets[0]}/{st}'].attrs['delta_t']
            segsamp = len(fp[f'{dets[0]}/{st}'])
            segdur = dt * segsamp - slide_buffer #Segment duration
            
            #If segment is not sufficient to reach start point skip ahead
            if curr_dur + segdur < 0:
                curr_dur += segdur
                continue
            else:
                #Check if start of segment lies before the requested start
                if curr_dur < 0:
                    sidx = int(-curr_dur // dt)
                    stoffset = -curr_dur
                else:
                    sidx = None
                    stoffset = 0
            
            #Check if with this segment the requested duration is exceeded
            if curr_dur + segdur > duration:
                eidx = max(int((curr_dur + segdur - duration + slide_buffer) // dt),
                           int((curr_dur + min_segment_duration + slide_buffer) // dt))
                if eidx >= segsamp: #Assert that index is valid
                    eidx = None
            else:
                eidx = None
            
            #Build segment
            seg = OverlapSegment()
            for det in dets:
                key = f'{det}/{st}'
                ts = TimeSeries(fp[key][sidx:eidx],
                                delta_t=fp[key].attrs['delta_t'],
                                epoch=fp[key].attrs['start_time']+stoffset)
                seg.add_timeseries((det, ts.astype(np.float64)))
            if seg.duration is None:
                seg.duration = float(seg.end_time) - float(seg.start_time) - slide_buffer
            
            #Check if segment fulfills minimum duration requirements
            if min_segment_duration is None:
                seglist.add_segment(seg)
                curr_dur += seg.duration
            else:
                segdur = seg.duration
                if segdur < min_segment_duration:
                    curr_dur -= segdur
                    #If segment crossing the required start is too short
                    #after truncation, the start requirement is still
                    #fulfilled.
                    if curr_dur < 0:
                        curr_dur = 0
                else:
                    seglist.add_segment(seg)
                    curr_dur += seg.duration
    
    if seglist.duration < duration:
        warnings.warn('Not enough data to return the requested amount of data.',
                      RuntimeWarning)
    
    return seglist

def get_noise(dataset, start=0, duration=2592000, seed=0, psds=None,
              low_frequency_cutoff=9, sample_rate=2048,
              filter_duration=128, min_segment_duration=7200,
              slide_buffer=240, real_noise_path=None):
    """A function to generate real or fake noise.
    
    Arguments
    ---------
    dataset : 1 or 2 or 3 or 4
        Specifies the kind of noise to return. If dataset is in
        [1, 2, 3], noise will be simulated. If dataset == 4, real noise
        will be used.
    start : {int, 0}
        <Description>
    duration : {int, 2592000}
        The duration of noise to generate (in seconds).
    seed : {int or dict, 0}
        The seed to use for noise-generation. This seed will be used
        both in the case that noise is simulated as well as when real
        noise is used. In the latter case it will determin which parts
        of the noise will be used and by how much the individual
        detectors are shifted by. If given a dictionary, the keys must
        line up with the keys given by the psds-option. If the
        psds-option is no dictionary, the keys `H1` and `L1` are
        required.
    psds : {str or pycbc.FrequencySeries or dict or None, None}
        The PSD(s) to use. Generally, a dict is expected. The dict has
        to have keys representing the individual detectors and its
        values must be either a str or a pycbc.FrequencySeries. If only
        a single string or pycbc.FrequencySeries is given, the value
        will be used for both detectors, where the detectors are
        specified by `H1` and `L1`. If the a value is a str and a file
        exists at the location specified by the string, the code will
        try to load the file from that location as an ASD file. If no
        file exists at the location the str will instead be interpreted
        as a PSD name as understood by pycbc.psd.from_string. If psds is
        set to None, the PSD `aLIGOZeroDetHighPower` will be used for
        the two detectors `H1` and `L1`.
    low_frequency_cutoff : {float, 9}
        The low frequency cutoff for the noise. (Only noise with
        frequencies larger than the specified value will be generated)
    sample_rate : {int, 2048}
        The sample rate used for the time domain data.
    filter_duration : {float, 0}
        <Description>
    min_segment_duration : {float, 7200}
        The minimum duration in seconds any segment of the data must
        have.
    slide_buffer : {float, 240}
        The amount of time outside of each segment that is used for
        relative time shifts between detectors. Only used for real
        noise, i.e. dataset == 2. (If this is set to 0, two different
        seeds will produce the same output on real noise.)
    real_noise_path : {str or None, None}
        Path from which to read the real noise data. A default location
        will be queried if no value is provided. If the file does not
        exist, it will be downloaded.
    """
    psd_names = {0: 'aLIGOZeroDetHighPower'}
    pdf = 1.0 / filter_duration
    plen = int(sample_rate / pdf) // 2 + 1
    seglist = SegmentList()
    if dataset in [1, 2]:
        if psds is None:
            if dataset == 1:
                psds = {'H1': 'aLIGOZeroDetHighPower',
                        'L1': 'aLIGOZeroDetHighPower'}
            elif dataset == 2:
                rs = np.random.RandomState(seed=seed)
                psds = {'H1': psd_names[rs.randint(0, len(psd_names))],
                        'L1': psd_names[rs.randint(0, len(psd_names))]}
        if isinstance(psds, (FrequencySeries, str)):
            psds = {'H1': psds, 'L1': psds}
        
        if isinstance(seed, int):
            keys = sorted(list(psds.keys()))
            seed = {key: seed+i for (i, key) in enumerate(keys)}
        
        seg = OverlapSegment(duration=duration)
        for key, val in psds.items():
            if isinstance(val, str):
                if os.path.isfile(val):
                    psds[key] = pycbc.psd.from_txt(val,
                                                   plen, pdf,
                                                   low_frequency_cutoff,
                                                   is_asd_file=True)
                else:
                    psds[key] = pycbc.psd.from_string(val, plen, pdf,
                                                      low_frequency_cutoff)
            noise = colored_noise(psds[key],
                                  start,
                                  start+duration,
                                  seed=seed[key],
                                  sample_rate=sample_rate,
                                  low_frequency_cutoff=low_frequency_cutoff)
            seg.add_timeseries((key, noise))
        seglist.add_segment(seg)
        return seglist.get_full_seglist(shift=False)
    elif dataset == 3:
        if isinstance(seed, int):
            rs = np.random.RandomState(seed=seed)
            seed = {'H1': seed, 'L1': seed+1}
        else:
            rs = np.random.RandomState(seed=min(list(seed.values())))
        dur = 0
        times = []
        while dur < duration:
            tmp = rs.randint(min_segment_duration, duration)
            if duration - dur - tmp < min_segment_duration:
                tmp = duration - dur
            if len(times) == 0:
                times.append([0, tmp])
            else:
                times.append([times[-1][1], times[-1][1]+tmp])
            dur += tmp
        
        psds = {key: pycbc.psd.from_string(val, plen, pdf,
                                           low_frequency_cutoff)
                for (key, val) in psd_names.items()}
        
        for stime, etime in times:
            seg = OverlapSegment(duration=etime-stime)
            for key in seed.keys():
                psdnum = rs.randint(0, len(psd_names))
                noise = colored_noise(psds[psdnum],
                                      start+stime,
                                      start+etime,
                                      seed=seed[key],
                                      sample_rate=sample_rate,
                                      low_frequency_cutoff=low_frequency_cutoff)
                seg.add_timeseries((key, noise))
            seglist.add_segment(seg)
        return seglist.get_full_seglist(shift=False)
    elif dataset == 4:
        seglist = get_real_noise(path=real_noise_path, start=start,
                                 duration=duration, slide_buffer=slide_buffer,
                                 min_segment_duration=min_segment_duration)
        
        if isinstance(seed, dict):
            seed = min(list(seed.values()))
        
        return seglist.get_full_seglist(shift=True, seed=seed)
    else:
        raise ValueError(f'Unknown data set {dataset}')

def make_injections(strain, injection_file, f_lower=10, padding_start=0,
                    padding_end=0):
    """Inject waveforms into background.
    
    Arguments
    ---------
    strain : SegmentList
        The SegmentList containing the background.
    injection_file : str
        Path to the file containing the injections. It has to be
        understood by pycbc.inject.InjectionSet.
    f_lower : {float, 10}
        The lower frequency cutoff at with which to create injections.
    padding_start : {float, 0}
        The amount of time in the beginning of each segment to not put
        injections into.
    padding_end : {float, 0}
        The amount of time in the end of each segment to not put
        injections into.
    
    Returns
    -------
    strain:
        A dictionary, where the keys are detector names and the values
        are lists containing PyCBC TimeSeries. The TimeSeries are the
        background segments plus the added injections.
    injtimes:
        Array containing the (shifted) injection times that were
        actually made.
    injidxs:
        Array containing the indices of the injection-parameters that
        were actually made.
    """
    dets = strain.detectors
    ret = {det: [] for det in dets}
    inj, injtimes, injidxs = strain.apply_injections(injection_file,
                                                     shift_data=False,
                                                     padding_start=padding_start,
                                                     padding_end=padding_end,
                                                     f_lower=f_lower,
                                                     return_times=True,
                                                     return_indices=True)
    
    for dic in inj:
        seglen = len(list(dic.values())[0])
        dt = list(dic.values())[0].delta_t
        epoch = float(list(dic.values())[0].start_time)
        for det in dets:
            if det in dic:
                ret[det].append(dic[det])
            else:
                ts = TimeSeries(np.zeros(seglen),
                                delta_t=dt,
                                epoch=epoch)
                ret[det].append(ts)
    return ret, injtimes, injidxs

def save_strain_dict(strain_dict, path, force=False):
    """Save dictionary of strain as produced by the code to a file.
    
    Arguments
    ---------
    strain_dict : dict
        Dictionary where the keys correspond to detector names and the
        values are lists. Each entry in the list is expected to be a
        PyCBC TimeSeries.
    path : str
        Path at which to store the strain.
    force : {bool, False}
        Whether or not to overwrite existing files.
    """
    if os.path.isfile(path):
        if force:
            os.remove(path)
        else:
            raise IOError(f'Strain file at {path} already exists.')
    for det, ts_list in strain_dict.items():
        for ts in ts_list:
            group = f'{det}/{int(ts.start_time)}'
            ts.save(path, group=group)

def main(doc):
    parser = argparse.ArgumentParser(description=doc)
    
    parser.add_argument('-d', '--data-set', type=int, choices=[1, 2, 3, 4], default=1,
                        help="The data set type that should be generated. <add reference to what these sets contain>")
    parser.add_argument('-i', '--output-injection-file', type=str,
                        help=("Path at which the generated injections "
                              "should be stored. If an injection file is "
                              "loaded by setting the option "
                              "`--injection-file` and this option is "
                              "set, the loaded injection file will be "
                              "copied to the specified location. If this "
                              "option is not set a injection file may be "
                              "temporarily stored in the execution "
                              "directory and deleted after use. "
                              "(Extension must be `.hdf`)"))
    parser.add_argument('-f', '--output-foreground-file', type=str,
                        help=("Path at which to store the foreground "
                              "data. The foreground data is the pure "
                              "noise plus additive signals. If this "
                              "option is not specified no foreground "
                              "data will be generated and stored."))
    parser.add_argument('-b', '--output-background-file', type=str,
                        help=("Path at which to store the background "
                              "data. The background data is the pure "
                              "noise without additive signals. If this "
                              "option is not specified no background "
                              "data will be generated and stored."))
    
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help=("The seed to use for data generation. "
                              "Default: 0"))
    parser.add_argument('--start', type=int, default=0,
                        help=("An integer specifying the start time. "
                              "This option is meant to enable the "
                              "generation of multiple parts of a single "
                              "datastream. It sets the internal time "
                              "and always starts at 0. "
                              "It is not to be confused with the GPS "
                              "start time of real data. The GPS start "
                              "time will be set automatically by the "
                              "code. Default: 0"))
    parser.add_argument('--duration', type=int, default=2592000,
                        help=("The duration of data to generate in "
                              "seconds. Default: 2,592,000"))
    parser.add_argument('--slide-buffer', type=int, default=240,
                        help=("The amount of time in seconds up to "
                              "which the real detector can be shifted "
                              "by to produce different noise "
                              "realizations."))
    #Do we want to expose this option?
    parser.add_argument('--padding-start', type=float, default=20,
                        help=("Time at the beginning of the strain "
                              "segment that should not contain "
                              "injections. Default: 20"))
    #Do we want to expose this option?
    parser.add_argument('--padding-end', type=float, default=20,
                        help=("Time at the end of the strain "
                              "segment that should not contain "
                              "injections. Default: 20"))
    
    parser.add_argument('--injection-file', type=str,
                        help=("Path to an injection file that should be "
                              "used. If this option is not set "
                              "injections will be generated automatically."))
    
    parser.add_argument('--verbose', action='store_true',
                        help="Print update messages.")
    parser.add_argument('--force', action='store_true',
                        help="Overwrite existing files.")
    
    args = parser.parse_args()
    
    #Setup logging
    log_level = logging.INFO if args.verbose else logging.WARN
    logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s',
                        level=log_level, datefmt='%d-%m-%Y %H:%M:%S')
    
    #Sanity checks of provided options
    if args.output_foreground_file is None:
        msg = ("The option `--output-foreground-file` was not set and"
               "thus no foreground file will be generated or stored!")
        warnings.warn(msg, RuntimeWarning)
    
    if args.output_background_file is None:
        msg = ("The option `--output-background-file` was not set and"
               "thus no background file will be generated or stored!")
        warnings.warn(msg, RuntimeWarning)
    
    #Test if files already exist
    fpath = args.output_injection_file
    if fpath is not None:
        assert os.path.splitext(fpath)[1] == '.hdf', 'File path must end in `.hdf`'
        check_file_existence(fpath, args.force)
    fpath = args.output_foreground_file
    if fpath is not None:
        assert os.path.splitext(fpath)[1] == '.hdf', 'File path must end in `.hdf`'
        check_file_existence(fpath, args.force)
    fpath = args.output_background_file
    if fpath is not None:
        assert os.path.splitext(fpath)[1] == '.hdf', 'File path must end in `.hdf`'
        check_file_existence(fpath, args.force)
    
    #Generate noise background
    logging.info('Getting noise')
    strain = get_noise(args.data_set, start=args.start,
                       duration=args.duration, seed=args.seed)
    strain_dict = strain.get_full(shift=False)
    
    if args.output_background_file is not None:
        save_strain_dict(strain_dict,
                         args.output_background_file,
                         force=args.force)
        logging.info(f'Saved background to {args.output_background_file}')
    
    tstart = int(np.ceil(strain.start_time))
    tend = int(strain.end_time)
    
    #Take care of injections
    if args.injection_file is None:
        #Create injections
        logging.info('Generating injections')
        inj_config_paths = {1: os.path.join(base_path(), 'ds1.ini'),
                            2: os.path.join(base_path(), 'ds2.ini'),
                            3: os.path.join(base_path(), 'ds3.ini'),
                            4: os.path.join(base_path(), 'ds4.ini')}
        cmd = ['pycbc_create_injections']
        cmd += ['--config-files', str(inj_config_paths[args.data_set])]
        cmd += ['--gps-start-time', str(tstart)]
        cmd += ['--gps-end-time', str(tend)]
        cmd += ['--time-step', str(TIME_STEP)]
        cmd += ['--time-window', str(TIME_WINDOW)]
        if args.output_injection_file is None:
            args.injection_file = os.path.join(base_path(),
                                               f'TMP-{time.time()}.hdf')
        else:
            args.injection_file = args.output_injection_file
        cmd += ['--output-file', args.injection_file]
        if args.verbose:
            cmd += ['--verbose']
        subprocess.call(cmd)
    elif args.output_injection_file is not None:
        #Copy injection file
        copy(args.injection_file, args.output_injection_file)
    
    if args.output_foreground_file is None:
        logging.info('No output for the foreground file was specified. Skipping injections.')
        return
    
    fg_dict, injtimes, injidxs = make_injections(strain,
                                                 args.injection_file,
                                                 f_lower=10,
                                                 padding_start=args.padding_start,
                                                 padding_end=args.padding_end)
    
    if args.output_injection_file is not None:
        with h5py.File(args.output_injection_file, 'a') as fp:
            fp['shift-tc'] = injtimes
            fp['shift-indices'] = injidxs
    
    save_strain_dict(fg_dict,
                     args.output_foreground_file,
                     force=args.force)
    logging.info(f'Saved foreground to {args.output_foreground_file}')
    return

if __name__ == "__main__":
    main(__doc__)
