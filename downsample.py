from argparse import ArgumentParser
import numpy as np
import os
import urllib.request
import json
import logging
import tqdm

from pycbc.filter import resample_to_delta_t, highpass
from pycbc.frame import query_and_read_frame
from pycbc.catalog import Catalog
from pycbc import dq
from pycbc import DYN_RANGE_FAC

from ligo.segments import segmentlist, segment

def downsample(strain, sample_rate, low_freq_cutoff=None, dtype=None,
               dynamic_range=None, crop=None):
    if dynamic_range is None:
        dynamic_range = 1
    if dtype is None:
        dtype = np.float32
    res = resample_to_delta_t(strain, 1 / sample_rate) * dynamic_range
    if low_freq_cutoff is not None:
        ret = highpass(res, low_freq_cutoff).astype(dtype)
    else:
        ret = res.astype(dtype)
    if crop is None:
        return ret
    else:
        return ret.time_slice(float(ret.start_time) + crop,
                              float(ret.end_time) - crop)

def get_raw_segments(dets):
    start_time = 1238166018 #Start O3a
    end_time = 1253977218 #End O3a
    ret = {}
    if not isinstance(dets, (list, tuple)):
        dets = [dets]
    for det in dets:
        seglist = dq.query_flag(det, 'DATA', start_time, end_time)
        seglist -= dq.query_flag(det, 'CBC_CAT1_VETO', start_time, end_time)
        seglist -= dq.query_flag(det, 'CBC_CAT2_VETO', start_time, end_time)
        seglist -= dq.query_flag(det, 'CBC_HW_INJ', start_time, end_time)
        seglist -= dq.query_flag(det, 'BURST_HW_INJ', start_time, end_time)
        ret[det] = seglist
    return ret

def get_segments(dets, exclude_known_detections=True,
                 merger_exclude_time=10, minimum_duration=None, crop=0):
    segs = get_raw_segments(dets)
    tmp = None
    for det in dets:
        if tmp is None:
            tmp = segs[det]
        else:
            tmp = tmp & segs[det]
    segs = tmp
    
    if exclude_known_detections:
        exclude_segs = segmentlist([])
        logging.info('Finding known detection times')
        catalog = Catalog(source='gwtc-2')
        for merger in catalog.mergers.values():
            start = int(merger.time - merger_exclude_time)
            end = int(np.ceil(merger.time + merger_exclude_time))
            exclude_segs.append(segment(start, end))
        exclude_segs.coalesce()
        segs -= exclude_segs
    
    if minimum_duration is not None:
        tmp = segmentlist([])
        for seg in segs:
            if seg[1] - seg[0] < minimum_duration + 2 * crop:
                continue
            tmp.append(seg)
        tmp.coalesce()
        segs = tmp
    return segs

def main():
    parser = ArgumentParser()
    
    parser.add_argument('--output', type=str, required=True,
                        help="The path at which to store the output.")
    parser.add_argument('--low-frequency-cutoff', type=float, default=15,
                        help="The low frequency cutoff to apply to the data.")
    parser.add_argument('--detectors', type=str, nargs='+', default=['H1', 'L1'],
                        help="The detectors to downsample data from.")
    parser.add_argument('--exclude-known-detections', action='store_true',
                        help="Exclude segments that contain known events.")
    parser.add_argument('--minimum-duration', type=int, default=2*60*60,
                        help="The minimum duration of any segment.")
    parser.add_argument('--merger-exclude-time', type=float, default=10,
                        help="The symmetric minimum duration in seconds around known detections to exclude. Default: 10")
    parser.add_argument('--crop', type=int, default=4,
                        help="The amount of time in seconds to crop from the beginning and end of every downsampled segment to remove filter artifacts.")
    parser.add_argument('--verbose', action='store_true',
                        help="Print status updates.")
    parser.add_argument('--force', action='store_true',
                        help="Overwrite existing files.")
    
    args = parser.parse_args()
    
    log_level = logging.INFO if args.verbose else logging.WARN
    
    logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s',
                        level=log_level, datefmt='%d-%m-%Y %H:%M:%S')
    
    if os.path.isfile(args.output) and not args.force:
        raise IOError(f'File {args.output} already exists. Set the flag --force to overwrite it.')
    
    #Get list of segments that have sufficient data quality and duration
    logging.info(f'Grabbing segments for detectors {args.detectors}')
    # segs = get_segments(args.detectors)
    # tmp = None
    # for det in args.detectors:
    #     if tmp is None:
    #         tmp = segs[det]
    #     else:
    #         tmp = tmp & segs[det]
    # segs = tmp
    
    # if args.exclude_known_detections:
    #     exclude_segs = segmentlist([])
    #     logging.info('Finding known detection times')
    #     catalog = Catalog(source='gwtc-2')
    #     for merger in catalog.mergers.values():
    #         start = int(merger.time - args.merger_exclude_time)
    #         end = int(np.ceil(merger.time + args.merger_exclude_time))
    #         exclude_segs.append(segment(start, end))
    #     exclude_segs.coalesce()
    #     segs -= exclude_segs
    
    # #Enforce minimum duration
    # tmp = segmentlist([])
    # for seg in segs:
    #     if seg[1] - seg[0] < args.minimum_duration + 2 * args.crop:
    #         continue
    #     tmp.append(seg)
    # tmp.coalesce()
    # segs = tmp
    segs = get_segments(args.detectors,
                        exclude_known_detections=args.exclude_known_detections,
                        merger_exclude_time=args.merger_exclude_time,
                        minimum_duration=args.minimum_duration,
                        crop=args.crop)
    
    #Access segments one by one
    strlen = len(str(len(segs)))
    if args.verbose:
        iterator = tqdm.tqdm(segs, desc='Downsampling segments')
    else:
        iterator = segs
    
    for seg in iterator:
        start, end = seg
        for det in args.detectors:
            logging.info(f'Trying segment {(start, end)}')
            ts = query_and_read_frame(f'{det}_GWOSC_O3a_4KHZ_R1',
                                      f'{det}:GWOSC-4KHZ_R1_STRAIN',
                                        start,
                                        end)
            ts = downsample(ts,
                            2048,
                            low_freq_cutoff=args.low_frequency_cutoff,
                            dynamic_range=DYN_RANGE_FAC,
                            crop=args.crop)
            start_time = int(float(ts.start_time))
            ts.save(args.output, group=f'{det}/{start_time}')
    return

if __name__ == "__main__":
    main()
