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

from pycbc.noise.reproduceable import colored_noise
import pycbc.psd
from pycbc.types import FrequencySeries
from pycbc.inject import InjectionSet

#ToDos:
#-Implement storage of command into final file
#-Add option to not cache real-noise data (default should be caching the data)
#-Add a docstring
#-Add reference to what kind of data the different sets contain
#-Add padding option both for real noise and space in beginning and end of segment where no injections are placed

TIME_STEP = 16
TIME_WINDOW = 6

def check_file_existence(fpath, force):
    if fpath is not None:
        if os.path.isfile(fpath) and not force:
            raise IOError((f'The file {fpath} already exists. Set the '
                          'flag `--force` to overwrite existing files.'))

def base_path():
    return os.path.abspath(__file__)

def get_noise(dataset, start=0, duration=2592000, seed=0, psds=None,
              low_frequency_cutoff=9, sample_rate=2048,
              filter_duration=128, padding=0, min_segment_duration=7200):
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
    padding : {int, 0}
        <Description>
    """
    psd_names = {0: 'aLIGOZeroDetHighPower'}
    pdf = 1.0 / filter_duration
    plen = int(sample_rate / pdf) // 2 + 1
    ret = {}
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
            ret[key] = [colored_noise(psds[key],
                                      start-padding,
                                      start+duration+padding,
                                      seed=seed[key],
                                      sample_rate=sample_rate,
                                      low_frequency_cutoff=low_frequency_cutoff)]
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
        
        psds = {key: pycbc.psd.from_string(val, plen, pdf,
                                           low_frequency_cutoff)
                for (key, val) in psd_names}
        
        for key in seed.keys():
            ret[key] = []
        for start, end in times:
            for key in ret.keys():
                psdnum = rs.randint(0, len(psd_names))
                ret[key].append(colored_noise(psds[psdnum],
                                              start,
                                              end,
                                              seed=seed[key],
                                              sample_rate=sample_rate,
                                              low_frequency_cutoff=low_frequency_cutoff))
        return ret
    elif dataset == 4:
        #TODO:
        #-Download data file if not existent
        #-Load data from data file and put into overlap segments
        #-Use `get` to apply seeded time-shifts
        #-Return the resulting time shifts
        raise NotImplementedError
    else:
        raise ValueError(f'Unknown data set {dataset}')
    return ret

def make_injections(strain, injection_file, f_lower=10):
    #TODO: Use SegmentList to apply injections
    injector = InjectionSet(injection_file)
    for det, noise in strain.items():
        injector.apply(noise, det, f_lower=f_lower)

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
    #Do we want to expose this option?
    parser.add_argument('--clear-start', type=float, default=20,
                        help=("Time at the beginning of the strain "
                              "segment that should not contain "
                              "injections. Default: 20"))
    #Do we want to expose this option?
    parser.add_argument('--clear-end', type=float, default=20,
                        help=("Time at the end of the strain "
                              "segment that should not contain "
                              "injections. Default: 20"))
    #Do we want to expose this option?
    parser.add_argument('--noise-padding', type=int, default=0,
                        help=("The amount of time to pad the noise "
                              "segment by for processing reasons."))
    
    parser.add_argument('--injection-file', type=str,
                        help=("Path to an injection file that should be "
                              "used. If this option is not set "
                              "injections will be generated automatically."))
    
    parser.add_argument('--no-cache', action='store_true',
                        help=("Do not permanently store downloaded real "
                              "noise data."))
    
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
    
    #TODO: Assert that output files end in .hdf
    
    #Test if files already exist
    fpath = args.output_injection_file
    check_file_existence(fpath, args.force)
    fpath = args.output_foreground_file
    check_file_existence(fpath, args.force)
    fpath = args.output_background_file
    check_file_existence(fpath, args.force)
    
    #Generate noise background
    logging.info('Getting noise')
    strain = get_noise(args.data_set, start=args.start,
                       duration=args.duration, seed=args.seed,
                        padding=args.noise_padding)
    tstart = []
    tend = []
    for key, val in strain.items():
        tstart.append(float(val.start_time))
        tend.append(float(val.end_time))
        if args.output_background_file is not None:
            val.save(args.output_background_file, group=key)
    tstart = int(np.ceil(max(tstart) + args.noise_padding))
    tend = int(min(tend) - args.noise_padding)
    
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
    
    make_injections(strain, args.injection_file, f_lower=10)
    
    for det, fg in strain.items():
        fg.save(args.output_foreground_file, group=det)
    return

if __name__ == "__main__":
    main(__doc__)
