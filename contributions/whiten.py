#! /usr/bin/env python

### Import modules
from argparse import ArgumentParser
import logging
import h5py
from tqdm import tqdm
import os.path
import numpy as np
from pycbc.psd import inverse_spectrum_truncation, interpolate
import pycbc.types

def copy_attrs(in_obj, out_obj):
    for key, attr in in_obj.attrs.items():
        out_obj.attrs[key] = attr
    return

class SafeCaster:
    def __init__(self, dtype_bits):
        if dtype_bits is None:
            self.dtype = None
        elif isinstance(dtype_bits, int):
            if dtype_bits%8==0:
                self.dtype = np.dtype('f%i' % (dtype_bits//8))
            else:
                raise ValueError("Data type in bits must"
                                "be divisible by 8.")
        else:
            raise ValueError
    def __call__(self, in_arr):
        if self.dtype is None:
            return in_arr
        else:
            return in_arr.astype(self.dtype)

def whiten(strain, delta_t=1./2048., segment_duration=0.5,
                max_filter_duration=0.25, trunc_method='hann',
                remove_corrupted=True, low_frequency_cutoff=None,
                psd=None, **kwargs):
    """Whiten a strain.

    Arguments
    ---------
    strain : numpy array
        The strain to be whitened. Can be one- or two-dimensional,
        in which case the whitening is performed along the second axis.
    delta_t : {float, 1./2048.}
        Sampling rate of the input strain.
    segment_duration : {float, 0.5}
        Duration in seconds to use for each sample of the spectrum.
    max_filter_duration : {float, 0.25}
        Maximum length of the time-domain filter in seconds.
    trunc_method : {None, 'hann'}
        Function used for truncating the time-domain filter.
        None produces a hard truncation at `max_filter_len`.
    remove_corrupted : {True, boolean}
        If True, the region of the time series corrupted by
        the whitening is excised before returning. If False,
        the corrupted regions are not excised and the full
        time series is returned.
    low_frequency_cutoff : {None, float}
        Low frequency cutoff to be passed to the inverse spectrum
        truncation. This should be matched to a known low frequency
        cutoff of the data if there is one.
    psd : {None, numpy array}
        PSD to be used for whitening. If not supplied,
        it is estimated from the time series.
    kwargs : keywords
        Additional keyword arguments are passed on to the
        `pycbc.psd.welch` method.

    Returns
    -------
    whitened_data : numpy array
        The whitened time series.

    """
    if strain.ndim==1:
        colored_ts = pycbc.types.TimeSeries(strain, delta_t=delta_t)
        if psd is None:
            psd = colored_ts.psd(segment_duration, **kwargs)
        elif isinstance(psd, np.ndarray):
            assert psd.ndim==1
            logging.warning("WARNING: Assuming PSD delta_f based "
                "on delta_t, length of PSD and EVEN length "
                "of original time series. Tread carefully!")
            assumed_duration = delta_t*(2*len(psd)-2)
            psd = pycbc.types.FrequencySeries(psd,
                    delta_f=1./assumed_duration)
        elif isinstance(psd, pycbc.types.FrequencySeries):
            pass
        else:
            raise ValueError("Unknown format of PSD.")
        psd = interpolate(psd, colored_ts.delta_f)
        max_filter_len = int(max_filter_duration*colored_ts.sample_rate)

        psd = inverse_spectrum_truncation(psd,
                    max_filter_len=max_filter_len,
                    low_frequency_cutoff=low_frequency_cutoff,
                    trunc_method=trunc_method)

        # inv_psd = np.array([0. if num==0. else 1./num for num in psd])
        inv_psd = 1./psd
        white_ts = (colored_ts.to_frequencyseries()*inv_psd**0.5
                ).to_timeseries()

        if remove_corrupted:
            white_ts = white_ts[max_filter_len//2:(len(colored_ts)-max_filter_len//2)]

        return white_ts.numpy()

    elif strain.ndim==2:
        psds_1d = None
        if isinstance(psd, np.ndarray):
            if psd.ndim==1:
                psds_1d = [psd for _ in strain]
        if psds_1d is None:
            if (psd is None) or isinstance(psd,
                                pycbc.types.FrequencySeries):
                psds_1d = [psd for _ in strain]
            else:
                assert len(psd)==len(strain)
                psds_1d = psd

        white_segments = [whiten(sd_strain, delta_t=delta_t,
                segment_duration=segment_duration,
                max_filter_duration=max_filter_duration,
                trunc_method=trunc_method,
                remove_corrupted=remove_corrupted,
                low_frequency_cutoff=low_frequency_cutoff, psd=psd_1d,
                **kwargs) for sd_strain, psd_1d in zip(strain, psds_1d)]
        return np.stack(white_segments, axis=0)

    else:
        raise ValueError("Strain numpy array dimension must be 1 or 2.")


def main():
    parser = ArgumentParser(description="Strain whitening script "
        "to be used for development of submission to the MLGWSC-1 "
        "mock data challenge. Written by Ondřej Zelenka.")

    parser.add_argument('--verbose', action='store_true',
            help="Print update messages.")
    parser.add_argument('--debug', action='store_true',
            help="Show debug messages.")
    parser.add_argument('--force', action='store_true',
            help="Overwrite existing output file.")

    parser.add_argument('inputfile', type=str, help="The path to the "
        "input data file. It should be of the format generated "
        "by the generate_data.py script.")
    parser.add_argument('outputfile', type=str, help="The path where "
        "to store the whitened data. The file must not exist.")
    parser.add_argument('-s', '--segment-duration', type=float,
        default=0.5, help="Duration in seconds to use for each sample "
        "of the spectrum for PSD estimation. Default: 0.5.")
    parser.add_argument('-d', '--max-filter-duration', type=float,
        default=0.25, help="Maximum length of the time-domain filter "
        " in seconds. Default: 0.25.")
    parser.add_argument('--hard-trunc', action='store_true',
        help="Use a hard truncation of the time-domain filter. "
        "Otherwise, a Hann window is used.")
    parser.add_argument('--keep-corrupted', action='store_true',
        help="Keep the region of the time series corrupted "
        "by the whitening. Otherwise, it is excised.")
    parser.add_argument('-f', '--low-frequency-cutoff', type=float,
        default=20., help="Low frequency cutoff to be passed to the "
        "inverse spectrum truncation. This should be matched "
        "to a known low frequency cutoff of the data if there is one. "
        "Default: 20.")
    parser.add_argument('--dtype', type=int, default=None,
        help="Data type specified by number of bits. If supplied, "
        "the input data will be cast to the corresponding "
        "floating point data type before whitening.")
    parser.add_argument('--compress', action='store_true',
        help="Compress the output file.")

    args = parser.parse_args()

    ### Set up logging
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARN
    logging.basicConfig(format="%(levelname)s | %(asctime)s: "
            "%(message)s", level=log_level, datefmt='%d-%m-%Y %H:%M:%S')

    ### Check existence of output file
    if os.path.isfile(args.outputfile) and not args.force:
        raise RuntimeError("Output file exists.")
    else:
        pass

    if args.hard_trunc:
        trunc_method = None
    else:
        trunc_method = 'hann'

    logging.debug(("Initializing caster to data type %s"
        % str(args.dtype)))
    caster = SafeCaster(args.dtype)
    if args.compress:
        ds_write_kwargs = {'compression': 'gzip', 'compression_opts': 9,
                            'shuffle': True}
    else:
        ds_write_kwargs = {}

    with h5py.File(args.inputfile, 'r') as infile, h5py.File(
                            args.outputfile, 'w') as outfile:
        total = sum([len(grp) for grp in infile.values()])
        copy_attrs(infile, outfile)
        with tqdm(desc='Whitening individual datasets',
            disable=not args.verbose, ascii=True, total=total) as pbar:
            for detector_group_name, in_detector_group in infile.items():
                out_detector_group = outfile.create_group(detector_group_name)
                copy_attrs(in_detector_group, out_detector_group)
                for segment_name, in_segment in in_detector_group.items():
                    segment_colored_data = in_segment[()]
                    segment_white_data = whiten(segment_colored_data,
                        delta_t=1./infile.attrs['sample_rate'],
                        segment_duration=args.segment_duration,
                        max_filter_duration=args.max_filter_duration,
                        trunc_method=trunc_method,
                        remove_corrupted=(not args.keep_corrupted),
                        low_frequency_cutoff=args.low_frequency_cutoff)
                    out_segment = out_detector_group.create_dataset(
                        segment_name, data=caster(segment_white_data),
                        **ds_write_kwargs)
                    copy_attrs(in_segment, out_segment)
                    if not args.keep_corrupted:
                        out_segment.attrs['start_time'] += args.max_filter_duration/2
                    pbar.update(1)

if __name__=='__main__':
    main()
