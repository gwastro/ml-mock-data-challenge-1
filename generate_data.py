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
import requests
import tqdm
import csv

from pycbc.noise.reproduceable import colored_noise
import pycbc.psd
from pycbc.types import FrequencySeries, TimeSeries, \
                        load_frequencyseries, load_timeseries
from pycbc.inject import InjectionSet
from pycbc import DYN_RANGE_FAC

from segments import OverlapSegment, SegmentList
import ligo.segments

#ToDos:
#-Implement storage of command into final file
#-Add a docstring
#-Add reference to what kind of data the different sets contain

TIME_STEP = 16
TIME_WINDOW = 6

def check_file_existence(fpath, force, delete=False):
    if fpath is not None:
        if os.path.isfile(fpath):
            if force:
                if delete:
                    os.remove(fpath)
            else:
                msg = f'The file {fpath} already exists. Set the flag '
                msg += '`--force` to overwrite existing files.'
                raise IOError(msg)

def base_path():
    return os.path.split(os.path.abspath(__file__))[0]

def get_default_path():
    return os.path.join(base_path(), 'real_noise_file.hdf')

def download_data(path=None, resume=True):
    """Download noise data from the central server.
    
    Arguments
    ---------
    path : {str or None, None}
        Path at which to store the file. Must end in `.hdf`. If set to
        None a default path will be used.
    resume : {bool, True}
        Resume the file download if it was interrupted.
    """
    if path is None:
        path = get_default_path()
    assert os.path.splitext(path)[1] == '.hdf'
    url = 'https://www.atlas.aei.uni-hannover.de/work/marlin.schaefer/MDC/real_noise_file.hdf'
    header = {}
    resume_size = 0
    if os.path.isfile(path) and resume:
        mode = 'ab'
        resume_size = os.path.getsize(path)
        header['Range'] = f'bytes={resume_size}-'
    else:
        mode = 'wb'
    with open(path, mode) as fp:
        response = requests.get(url, stream=True, headers=header)
        total_size = response.headers.get('content-length')

        if total_size is None:
            print("No file length found")
            fp.write(response.content)
        else:
            total_size = int(total_size)
            desc = f"Downloading real_noise_file.hdf to {path}"
            print(desc)
            with tqdm.tqdm(total=int(total_size),
                           unit='B',
                           unit_scale=True,
                           dynamic_ncols=True,
                           desc="Progress: ",
                           initial=resume_size) as progbar:
                for data in response.iter_content(chunk_size=4000):
                    fp.write(data)
                    progbar.update(4000)

def load_segments(path=None):
    if path is None:
        path = os.path.join(base_path(), 'segments.csv')
    #Download data if it does not exist
    if not os.path.isfile(path):
        url = 'https://www.atlas.aei.uni-hannover.de/work/marlin.schaefer/MDC/segments.csv'
        response = requests.get(url)
        with open(path, 'wb') as fp:
            fp.write(response.content)
    
    #Load data from CSV file
    segs = ligo.segments.segmentlist([])
    with open(path, 'r') as fp:
        reader = csv.reader(fp)
        for i, row in enumerate(reader):
            if i == 0:
                continue
            idx, start, end = row
            segs.append(ligo.segments.segment([int(start), int(end)]))
    
    return segs

def restrict_segments(segments=None, start_offset=0, duration=2592000,
                      min_segment_duration=None, path=None,
                      slide_buffer=None):
    """Select segments which adhear to the parameters given to the
    function.
    
    Arguments
    ---------
    segments : {ligo.segments.segmentlist or None, None}
        The segmentlist to restrict. The contents are expected to be
        non-overlapping segments with integer valued borders sorted in
        ascending order. If None, load_segments will be called with the
        given path.
    start_offset : {int, 0}
        The amount of time to skip in the beginning. Segments which do
        not fulfill the min_segment_duration are ignored.
    duration : {int, 2592000}
        The total duration the returned segments should span.
    min_segment_duration : {int or None, None}
        The minimum duration of any input segment to be considered and
        also the minimum duration of any output segment. The final
        returned segment may disregard this limit. Duration takes
        precedence over min_segment_duration, meaning that the function
        makes sure that the requested duration is returned before it
        asserts that all segments are of the correct minimum duration.
    path : {str or None, None}
        Only used when segments is None. The path from which to load the
        segments. If None a standard path will be queried. For more
        information please refer to the documentation of load_segments.
    slide_buffer : {int or None, None}
        The amount of time to elongate each return segment by without
        counting it towards the duration. Needed to slide the data from
        one detector with respect to the others.
    
    Returns
    -------
    ligo.segments.segmentlist:
        A segment list containing the segments that sum up to the
        desired duration.
    """
    if slide_buffer is None:
        slide_buffer = 0
    
    if segments is None:
        segments = load_segments(path=path)
    
    past_duration = 0
    ret = ligo.segments.segmentlist([])
    for seg in segments:
        #Check if enough data has been generated
        if past_duration - start_offset >= duration:
            continue
        start, end = seg
        segduration = end - start
        #Check if segment fulfills minimum duration requirements
        if min_segment_duration is not None and segduration - slide_buffer < min_segment_duration:
            continue
        #Check if segment does not cut into start_offset
        if past_duration + segduration < start_offset:
            past_duration += segduration - slide_buffer
            continue
        #Check if segment is only partially required to cover previous time
        if past_duration < start_offset:
            start += start_offset - past_duration
            segduration = end - start
            past_duration = start_offset
            #Check if remainder of segment fulfills minimum duration requirements
            if min_segment_duration is not None and segduration - slide_buffer < min_segment_duration:
                continue
        
        #Check if entire segment is too long to be used completely
        if past_duration + segduration - slide_buffer > start_offset + duration:
            end -= past_duration + segduration - (start_offset + duration + slide_buffer)
            segduration = end - start

        ret.append(ligo.segments.segment([start, end]))
        past_duration += segduration - slide_buffer
    ret.coalesce()
    if past_duration < start_offset + duration:
        warnings.warn("Not enough segments to generate the entire requested duration.")
    return ret

def store_ts(path, det, ts, force=False):
    """Utility function to save a time series.
    
    Arguments
    ---------
    path : str or None
        The path at which to store the time series. If None the function
        will return immediately and not save anything.
    det : str
        The detector of the time series that should be saved.
    ts : TimeSeries
        The time series to save.
    force : {bool, False}
        Overwrite existing files.
    """
    if path is None:
        return
    
    group = f'{det}/{int(ts.start_time)}'
    ts.save(path, group=group)

def get_real_noise(path=None, min_segment_duration=None, start_offset=0,
                   duration=2592000, slide_buffer=None,
                   segment_path=None, detectors=['H1', 'L1'],
                   dyn_range_factor=None, store=None, seed=None,
                   force=False):
    """Get noise from a file as a SegmentList.
    
    Arguments
    ---------
    path : {str or None, None}
        Path from which to load the noise. If None a default path will
        be used. If no file is found at the path, the data will be
        downloaded from a central server.
    min_segment_duration : {float or None, None}
        The minimum duration each segment should have (in seconds).
    start_offset : {float, 0}
        The abstract start time. This is the amount of time to skip in
        at the beginning of the data.
    duration : {float, 2592000}
        The minimum duration of the noise to grab. May be exceeded by
        up to min_segment_duration.
    slide_buffer : {float or None, None}
        The amount of time for each segment that is not usable but
        reserved to shift data between detectors.
    detectors : {list of str, [`H1`, `L1`]}
        The list of detector data to grab. (Also fixes the order in
        which the data is stored in the segments)
    dyn_range_factor : {float or None, None}
        The factor by which the loaded data will be divided. This is
        done such that the loaded data can be stored in single
        precision. If set to None the default value from PyCBC will be
        used.
    store : {str or None, None}
        Store the output directly to the path given as argument. If set
        to None does not store immediately and returns a SegmentList.
    seed : {int or None, None}
        The seed to use when applying shifts. Only used when the option
        `store` is not None.
    force : {bool, False}
        Overwrite existing files. (Only used when store is not None)
    
    Returns
    -------
    SegmentList:
        The SegmentList containing the noise. (Only returns when store
        is None)
    """
    #Grab default values for options
    if path is None:
        path = get_default_path()
    if slide_buffer is None:
        slide_buffer = 0
    if dyn_range_factor is None:
        dyn_range_factor = DYN_RANGE_FAC
    
    if not os.path.isfile(path):
        download_data(path)
    
    #If file can't be opened it is probably not done downloading.
    try:
        with h5py.File(path, 'r') as fp:
            fp.attrs
    except:
        download_data(path, resume=True)
    
    raw_segments = load_segments(path=segment_path)
    
    segments = restrict_segments(start_offset=start_offset,
                                 duration=duration,
                                 min_segment_duration=min_segment_duration,
                                 path=segment_path)
    
    load_times = {}
    for seg in segments:
        for rawseg in raw_segments:
            if seg in rawseg:
                load_times[seg] = rawseg
                break;
        if seg not in load_times:
            raise RuntimeError
    
    seglist = SegmentList()
    if store is not None:
        rs = np.random.RandomState(seed)
    with h5py.File(path, 'r') as fp:
        for seg in segments:
            start_time = load_times[seg][0]
            segdur = seg[1] - seg[0] - slide_buffer
            overlap_seg = OverlapSegment(duration=segdur)
            for det in detectors:
                key = f'{det}/{start_time}'
                epoch = fp[key].attrs['start_time']
                dt = fp[key].attrs['delta_t']
                sidx = int((seg[0] - epoch) / dt)
                eidx = int((seg[1] - epoch) / dt)
                ts = TimeSeries(fp[key][sidx:eidx],
                                delta_t=dt,
                                epoch=float(seg[0]))
                ts = ts.astype(np.float64) / dyn_range_factor
                overlap_seg.add_timeseries((det, ts))
            if store is None:
                seglist.add_segment(overlap_seg)
            else:
                tmpseed = rs.randint(0, int(1e6))
                data = overlap_seg.get(shift=True, seed=tmpseed)
                for det, ts in zip(overlap_seg.detectors, data):
                    store_ts(store, det, ts, force=force)
    if store is None:
        return seglist
    else:
        return

class NoiseGenerator(object):
    psd_options = {'H1': ['aLIGOZeroDetHighPower',
                          'aLIGOZeroDetLowPower',
                          'aLIGOLateHighSensitivityP1200087',
                          'aLIGOMidHighSensitivityP1200087'],
                   'L1': ['aLIGOZeroDetHighPower',
                          'aLIGOZeroDetLowPower',
                          'aLIGOLateHighSensitivityP1200087',
                          'aLIGOMidHighSensitivityP1200087']}
    def __init__(self, dataset, seed=0, filter_duration=128,
                 sample_rate=2048, low_frequency_cutoff=15,
                 detectors=['H1', 'L1']):
        if dataset not in [1, 2, 3]:
            raise ValueError(f'PsdGenerator is only defined for datasets 1, 2, and 3.')
        self.dataset = dataset
        self.seed = seed
        self.filter_duration = filter_duration
        self.sample_rate = sample_rate
        self.low_frequency_cutoff = low_frequency_cutoff
        self.detectors = detectors
        self.fixed_psds = {det: None for det in self.detectors}
        self.delta_f = 1.0 / self.filter_duration
        self.plen = int(self.sample_rate / self.delta_f) // 2 + 1
        self.rs = np.random.RandomState(seed=self.seed)
    
    def __call__(self, start, end, generate_duration=3600):
        return self.get(start, end, generate_duration=generate_duration)
    
    def get(self, start, end, generate_duration=3600):
        keys = {}
        if self.dataset == 1:
            logging.debug(f'Called with dataset 1')
            for det in self.detectors:
                keys[det] = 'aLIGOZeroDetHighPower'
        elif self.dataset == 2:
            logging.debug(f'Called with dataset 2')
            for det in self.detectors:
                if self.fixed_psds[det] is None:
                    key = self.rs.randint(0, len(self.psd_options[det]))
                    self.fixed_psds[det] = self.psd_options[det][key]
                keys[det] = self.fixed_psds[det]
        elif self.dataset == 3:
            logging.debug(f'Called with dataset 3')
            for det in self.detectors:
                key = self.rs.randint(0, len(self.psd_options[det]))
                keys[det] = self.psd_options[det][key]
        else:
            raise RuntimeError(f'Unkown dataset {self.dataset}.')
        
        logging.debug(f'Generated keys {keys}')
        ret = {}
        for det, key in keys.items():
            logging.debug(f'Starting generating process for detector {det} and key {key}')
            if isinstance(key, str): #Normal case
                if os.path.isfile(key): #Check if we have to load PSD
                    try:
                        #Try loading from frequency series
                        psd = load_frequencyseries(key)
                    except:
                        #Try loading ASD from txt file
                        psd = pycbc.psd.from_txt(key,
                                                 self.plen,
                                                 self.delta_f,
                                                 self.low_frequency_cutoff,
                                                 is_asd_file=True)
                else:
                    #Try to interpret string as key known to PyCBC
                    logging.debug(f'Now generating PSD from string {key}')
                    psd = pycbc.psd.from_string(key,
                                                self.plen,
                                                self.delta_f,
                                                self.low_frequency_cutoff)
            
            if generate_duration is None:
                generate_duration = end - start
                logging.debug(f'Generate duration was None')
            logging.debug(f'Generate duration set to {generate_duration}')
            done_duration = 0
            noise = None
            #Generate time series noise in chunks
            while done_duration < end - start:
                logging.debug(f'Start of loop with done_duration: {done_duration}')
                segstart = start + done_duration
                segend = min(end, segstart + generate_duration)
                logging.debug(f'Generation segment: {(segstart, segend)} of duration {segend - segstart}')
                tmp = colored_noise(psd,
                                    segstart,
                                    segend,
                                    seed=self.seed,
                                    sample_rate=self.sample_rate,
                                    low_frequency_cutoff=self.low_frequency_cutoff)
                logging.debug(f'Succsessfully generated time domain noise')
                if noise is None:
                    logging.debug('Setting noise to tmp')
                    noise = tmp
                else:
                    logging.debug('Appending tmp to noise')
                    noise.append_zeros(len(tmp))
                    noise.data[-len(tmp):] = tmp.data[:]
                done_duration += segend - segstart
            logging.debug(f'Exited while loop with done_duration: {done_duration}')
            ret[det] = noise
        return ret

def get_noise(dataset, start_offset=0, duration=2592000, seed=0,
              low_frequency_cutoff=15, sample_rate=2048,
              filter_duration=128, min_segment_duration=7200,
              slide_buffer=240, real_noise_path=None,
              generate_duration=3600, segment_path=None,
              detectors=['H1', 'L1'], store=None, force=False):
    """A function to generate real or fake noise.
    
    Arguments
    ---------
    dataset : 1 or 2 or 3 or 4
        Specifies the kind of noise to return. If dataset is in
        [1, 2, 3], noise will be simulated. If dataset == 4, real noise
        will be used.
    start_offset : {int, 0}
        <Description>
    duration : {int, 2592000}
        The duration of noise to generate (in seconds).
    seed : {int, 0}
        The seed to use for noise-generation. This seed will be used
        both in the case that noise is simulated as well as when real
        noise is used. In the latter case it will determin by how much
        the individual detectors are shifted by.
    low_frequency_cutoff : {float, 15}
        The low frequency cutoff for the noise. (Only noise with
        frequencies larger than the specified value will be generated)
    sample_rate : {int, 2048}
        The sample rate used for the time domain data.
    filter_duration : {float, 128}
        <Description>
    min_segment_duration : {float, 7200}
        The minimum duration in seconds any segment of the data must
        have.
    slide_buffer : {float, 240}
        The amount of time outside of each segment that is used for
        relative time shifts between detectors. Only used for real
        noise, i.e. dataset == 4. (If this is set to 0, two different
        seeds will produce the same output on real noise.)
    real_noise_path : {str or None, None}
        Path from which to read the real noise data. A default location
        will be queried if no value is provided. If the file does not
        exist, it will be downloaded.
    generate_duration : {int, 3600}
        Only used when simulating noise. The maximum duration of noise
        to generate at once. (Setting this number higher may increase
        speed at the cost of larger memory requirements)
    segment_path : {str or None, None}
        The path at which to find the segment file. See load_segments
        for more information.
    detectors : {list of str, [`H1`, `L1`]}
        The detectors for which to grab the noise.
    store : {str or None, None}
        Store the time series at the given path. If set to None the data
        will be returned instead of being stored immediately.
    force : {bool, False}
        Overwrite existing files. (Only used when store is not None)
    """  
    segments = restrict_segments(start_offset=start_offset,
                                 duration=duration,
                                 min_segment_duration=min_segment_duration,
                                 path=segment_path)
    return_segs = SegmentList()
    if dataset in [1, 2, 3]:
        noi_gen = NoiseGenerator(dataset,
                                 seed=seed,
                                 filter_duration=filter_duration,
                                 sample_rate=sample_rate,
                                 low_frequency_cutoff=low_frequency_cutoff,
                                 detectors=detectors)
        for seg in segments:
            logging.debug(f'Now processing segment {seg} of duration {seg[1] - seg[0]} and generating noise for that')
            noise = noi_gen(seg[0], seg[1],
                            generate_duration=generate_duration)
            logging.debug(f"Finished generating this noise. It is of duration {noise['H1'].duration} and has {len(noise['H1'])} samples.")
            #TODO: Store these segments here
            ret_seg = OverlapSegment(duration=seg[1] - seg[0])
            for det in detectors:
                ret_seg.add_timeseries((det, noise[det]))
            if store is None:
                return_segs.add_segment(ret_seg)
            else:
                logging.debug(f'Trying to store data to file {store}')
                data = ret_seg.get(shift=False)
                logging.debug(f'Segment detectors are: {ret_seg.detectors}')
                for det, ts in zip(ret_seg.detectors, data):
                    logging.debug(f'Storing time series of duration {ts.duration} for detector {det} at {store}')
                    store_ts(store, det, ts, force=force)
        if store is None:
            return return_segs.get_full_seglist(shift=False)
        else:
            return
    elif dataset == 4:
        seglist = get_real_noise(path=real_noise_path,
                                 start_offset=start_offset,
                                 duration=duration,
                                 slide_buffer=slide_buffer,
                                 min_segment_duration=min_segment_duration,
                                 detectors=detectors,
                                 store=store)
        if store is None:
            return seglist.get_full_seglist(shift=True, seed=seed)
        else:
            return
    else:
        raise ValueError(f'Unknown data set {dataset}')

def make_injections(fpath, injection_file, f_lower=20, padding_start=0,
                    padding_end=0, store=None, force=False):
    """Inject waveforms into background.
    
    Arguments
    ---------
    fpath : str
        Path at which the background data is stored.
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
    store : {str or None, None}
        Path at which to store the output. If set to None the output
        will not stored but returned instead.
    force : {bool, False}
        Overwrite existing files.
    
    Returns
    -------
    strain:
        A dictionary, where the keys are detector names and the values
        are lists containing PyCBC TimeSeries. The TimeSeries are the
        background segments plus the added injections.
    """
    with h5py.File(fpath, 'r') as fp:
        dets = list(fp.keys())
        times = list(fp[dets[0]].keys())
    
    injector = InjectionSet(injection_file)
    injtable = injector.table
    
    ret = {}
    for t in times:
        for det in dets:
            if det not in ret:
                ret[det] = []
            group = f'{det}/{t}'
            ts = load_timeseries(fpath, group=group)
            idxs = np.where(np.logical_and(float(ts.start_time) + padding_start <= injtable['tc'],
                                           injtable['tc'] <= float(ts.end_time) - padding_end))[0]
            injector.apply(ts, det, f_lower=f_lower,
                           simulation_ids=list(idxs))
            store_ts(store, det, ts, force=force)
            if store is None:
                ret[det].append(ts)
    
    if store is None:
        return ret
    else:
        return

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
    parser.add_argument('--start-offset', type=int, default=0,
                        help=("An integer specifying the start time offset. "
                              "This option is meant to enable the "
                              "generation of multiple parts of a single "
                              "datastream. It sets the internal time "
                              "which always starts at 0. "
                              "It is not to be confused with the GPS "
                              "start time of real data. The GPS start "
                              "time will be set automatically by the "
                              "code. Default: 0"))
    parser.add_argument('--duration', type=int, default=2592000,
                        help=("The duration of data to generate in "
                              "seconds. Default: 2,592,000"))
    parser.add_argument('--generate-duration', type=int, default=3600,
                        help=("When generating noise this amount is "
                              "generated at a time and the results are "
                              "concatenated. Lower numbers reduce "
                              "memory requirements."))
    
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
    
    if args.output_injection_file is None and \
       args.output_background_file is None and \
       args.output_foreground_file is None:
       raise ValueError(f'No options to store data were set.')
    
    #Sanity checks of provided options
    if args.output_foreground_file is None:
        msg = ("The option `--output-foreground-file` was not set and"
               "thus no foreground file will be generated or stored!")
        warnings.warn(msg, RuntimeWarning)
    
    tmp_bg = False
    if args.output_background_file is None:
        msg = ("The option `--output-background-file` was not set and"
               "thus no background file will be generated or stored!")
        warnings.warn(msg, RuntimeWarning)
        tmp_bg = True
        args.output_background_file = os.path.join(base_path(),
                                                   f'TMP-{time.time()}-BG.hdf')
    
    #Test if files already exist
    fpath = args.output_injection_file
    if fpath is not None:
        assert os.path.splitext(fpath)[1] == '.hdf', 'File path must end in `.hdf`'
        check_file_existence(fpath, args.force)
    fpath = args.output_foreground_file
    if fpath is not None:
        assert os.path.splitext(fpath)[1] == '.hdf', 'File path must end in `.hdf`'
        check_file_existence(fpath, args.force, delete=True)
    fpath = args.output_background_file
    if fpath is not None:
        assert os.path.splitext(fpath)[1] == '.hdf', 'File path must end in `.hdf`'
        check_file_existence(fpath, args.force, delete=True)
    
    tmp_inj = False
    try:
        #Generate noise background
        logging.info('Getting noise')
        get_noise(args.data_set, start_offset=args.start_offset,
                  duration=args.duration, seed=args.seed,
                  store=args.output_background_file, force=args.force)
        
        segs = load_segments()
        tstart, tend = segs.extent()
        
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
            cmd += ['--seed', str(args.seed)]
            if args.output_injection_file is None:
                args.injection_file = os.path.join(base_path(),
                                                   f'TMP-{time.time()}-INJ.hdf')
                tmp_inj = True
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
        
        make_injections(args.output_background_file,
                        args.injection_file,
                        f_lower=20,
                        padding_start=30,
                        padding_end=30,
                        store=args.output_foreground_file,
                        force=args.force)
        logging.info(f'Saved foreground to {args.output_foreground_file}')
    except Exception as e:
        if tmp_bg and args.output_background_file is not None:
            if os.path.isfile(args.output_background_file):
                os.remove(args.output_background_file)
        if tmp_inj and args.injection_file is not None:
            if os.path.isfile(args.injection_file):
                os.remove(args.injection_file)
        raise e
    if tmp_bg and args.output_background_file is not None:
        if os.path.isfile(args.output_background_file):
            os.remove(args.output_background_file)
    if tmp_inj and args.injection_file is not None:
        if os.path.isfile(args.injection_file):
            os.remove(args.injection_file)
    return

if __name__ == "__main__":
    main(__doc__)
