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

from segments import DqSegment, DqSegmentList

def downsample(strain, sample_rate, low_freq_cutoff=None, dtype=None):
    if dtype is None:
        dtype = np.float32
    res = resample_to_delta_t(strain, 1 / sample_rate)
    if low_freq_cutoff is not None:
        return highpass(res, low_freq_cutoff).astype(dtype)
    else:
        return res.astype(dtype)

def download_segment_data(run, det, cat, start, end):
    url = 'https://www.gw-openscience.org/timeline/segments/json/{}/{}_{}/{}/{}/'.format(run, det, cat, start, end)
    
    with urllib.request.urlopen(url) as response:
        data = response.read().decode('utf-8')
        data = json.loads(data)
    return data

def get_segments(dets, cats):
    ret = {}
    if not isinstance(dets, (list, tuple)):
        dets = [dets]
    if not isinstance(cats, (list, tuple)):
        cats = [cats]
    for det in dets:
        detsegs = None
        for cat in cats:
            data = download_segment_data('O3a_4KHZ_R1', det, cat,
                                         1238166018, 1253977218)
            if detsegs is None:
                detsegs = DqSegmentList.from_dict(data)
            else:
                detsegs = detsegs and DqSegmentList.from_dict(data)
        ret[det] = detsegs
    return ret

def main():
    parser = ArgumentParser()
    
    parser.add_argument('--output', type=str, required=True,
                        help="The path at which to store the output.")
    parser.add_argument('--low-frequency-cutoff', type=float, default=15,
                        help="The low frequency cutoff to apply to the data.")
    parser.add_argument('--detectors', type=str, nargs='+', default=['H1', 'L1'],
                        help="The detectors to downsample data from.")
    parser.add_argument('--category', type=str, nargs='+', default=['CBC_CAT1'],
                        help="The categorie(s) of segments to consider. Default: CBC_CAT1")
    parser.add_argument('--exclude-known-detections', action='store_true',
                        help="Exclude segments that contain known events.")
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
    logging.info(f'Grabbing segments for detectors {args.detectors} of categories {args.category}')
    segs = get_segments(args.detectors, args.category)
    tmp = None
    for det in args.detectors:
        if tmp is None:
            tmp = segs[det]
        else:
            tmp = tmp and segs[det]
    segs = tmp.min_duration(60 * 60 * 5) #5 hours minimum duration
    
    if args.exclude_known_detections:
        logging.info('Finding known detection times')
        catalog = Catalog(source='gwtc-2')
        exclude_times = [merger.time for merger in catalog.mergers.values()]
        tmp = []
        logging.info('Excluding known detection times')
        for seg in segs:
            found = False
            for t in exclude_times:
                if t in seg:
                    found = True
                    break;
            if not found:
                tmp.append(seg)
        tmp = sorted(tmp, key=lambda seg: seg.start)
        segs = DqSegmentList(tmp, flag=segs.flag)
    
    #Access segments one by one
    strlen = len(str(len(segs)))
    if args.verbose:
        iterator = tqdm.tqdm(segs, desc='Downsampling segments')
    else:
        iterator = segs
    
    dur = 0
    for seg in iterator:
        ts_cache = []
        try:
            for det in args.detectors:
                logging.info(f'Trying segment {(seg.start, seg.end)}')
                ts = query_and_read_frame(f'{det}_GWOSC_O3a_4KHZ_R1',
                                          f'{det}:GWOSC-4KHZ_R1_STRAIN',
                                            seg.start,
                                            seg.end)
                ts_cache.append((det, ts))
        except:
            logging.warn(f'Could not query data for segment {(seg.start, seg.end)}')
            continue
        for det, ts in ts_cache:
            ts = downsample(ts, 2048, low_freq_cutoff=args.low_frequency_cutoff)
            ts.save(args.output, group=f'{det}/{seg.start}')
    return

if __name__ == "__main__":
    main()
