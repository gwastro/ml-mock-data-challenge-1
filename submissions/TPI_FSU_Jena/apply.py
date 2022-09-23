#! /usr/bin/env python

### Import modules
from argparse import ArgumentParser
import logging
import numpy as np
import h5py
import pycbc.waveform, pycbc.noise, pycbc.psd, pycbc.distributions, pycbc.detector
import os, os.path
from tqdm import tqdm
import multiprocessing as mp

import torch

#####################
# General functions #
#####################

### Set default device and data type
default_device = 'cpu'
dtype = torch.float32

class SegmentSlicer(object):
    def __init__(self, infile, key, step_size=0.1, peak_offset=0.6,
                            slice_length=2048, detectors=None,
                            white=False, whitened_file=None,
                            save_psd=False, low_frequency_cutoff=None,
                            segment_duration=0.5, max_filter_duration=0.25):
        self.step_size = step_size
        self.peak_offset = peak_offset
        self.slice_length = slice_length
        self.detectors = detectors
        self.white = white
        # self.delta_t = 1./infile.attrs['sample_rate']
        self.delta_t = 1./(1./infile[self.detectors[0]][key].attrs['delta_t'])
        self.index_step_size = int(self.step_size/self.delta_t)
        # Exact step size
        self.time_step_size = self.delta_t*self.index_step_size
        self.segment_duration = segment_duration
        self.max_filter_duration = max_filter_duration
        # For debugging of whitening
        self.whitened_file = whitened_file
        self.low_frequency_cutoff = low_frequency_cutoff
        self.key = key

        self.dss = [infile[det_key][self.key] for det_key in self.detectors]
        self.start_time = self.dss[0].attrs['start_time']
        for ds in self.dss:
            assert (ds.attrs['start_time']==self.start_time)
        logging.debug("SegmentSlicer initialized with index_step_size=%i, "
            "time_step_size=%f and segment key %s at data type %s"
            % (self.index_step_size, self.time_step_size, self.key, self.dss[0].dtype))
        self.process(save_psd)
        return

    def process(self, save_psd):
        whitened_dss = []
        self.psds = []
        for ds, key in zip(self.dss, self.detectors):
            if self.white:
                new_ds = ds[()]
            else:
                new_ds = whiten(ds, delta_t=self.delta_t, low_frequency_cutoff=self.low_frequency_cutoff,
                    segment_duration=self.segment_duration, max_filter_duration=self.max_filter_duration, return_psd=save_psd)
                if save_psd:
                    new_ds, psd = new_ds
                    self.psds.append(psd)
            #self.current_dss.append(new_ds.astype(np.float32))
            whitened_dss.append(new_ds)
            # For debugging of whitening
            if not self.whitened_file is None:
                with h5py.File(self.whitened_file, 'a') as wfile:
                    wfile[key].create_dataset(self.key, data=new_ds)
        self.dss = np.stack(whitened_dss, axis=0)
        if not self.white:
            self.start_time += 0.125
        self.white = True
        return
 
    def __len__(self):
        if self.white:
            full_slice_length = self.slice_length
        else:
            full_slice_length = 512 + self.slice_length
        index_step_size = int(self.step_size/self.delta_t)
        return 1+((self.dss.shape[1]-full_slice_length)//index_step_size)
    
    def __iter__(self):
        self.current_index = 0
        self.current_time = self.start_time
        return self
    
    def get_next_slice(self):
        if self.current_index+self.slice_length>self.dss.shape[1]:
            raise StopIteration
        else:
            this_slice = self.dss[:, self.current_index:self.current_index+self.slice_length]
            this_time = self.current_time + self.peak_offset
            self.current_index += self.index_step_size
            self.current_time += self.time_step_size
            return this_slice, this_time
    
    def __next__(self):
        return self.get_next_slice()

    def split_and_pop(self, ds_dict, size):
        data = self.dss
        self.dss = None
        max_split_index = int(np.floor(data.shape[1]/size))*size
        split_indices = list(range(size, max_split_index+1, size))
        self.ds_dict_keys = []
        for i, ds in enumerate(np.split(data, split_indices, axis=1)):
            new_key = '%s_%i' % (self.key, i)
            self.ds_dict_keys.append(new_key)
            logging.debug(("Saving data %s to shared dictionary" % new_key))
            ds_dict[new_key] = ds
        logging.debug(("Worker %s finished!" % self.key))
        return

    def stack_and_load(self, ds_dict):
        dss = []
        for key in self.ds_dict_keys:
            dss.append(ds_dict.pop(key))
        self.dss = np.concatenate(dss, axis=1)


class TorchSegmentSlicer(SegmentSlicer, torch.utils.data.IterableDataset):
    def __init__(self, *args, **kwargs):
        torch.utils.data.IterableDataset.__init__(self)
        SegmentSlicer.__init__(self, *args, **kwargs)
    
    def __next__(self):
        next_slice, next_time = self.get_next_slice()
        return torch.from_numpy(next_slice), torch.tensor(next_time)


### RETURNS NEW PSD WITH INFINITIES BELOW A LOWER FREQUENCY CUTOFF
### ONLY WORKS WITH pycbc.types.FrequencySeries!!! TODO: ADAPT TO NUMPY ARRAYS AS WELL
def regularize_psd(psd, low_frequency_cutoff, inf_value=np.inf):
    new_psd = np.copy(psd.numpy())
    for i, freq in enumerate(psd.get_sample_frequencies()):
        if freq<=low_frequency_cutoff:
            new_psd[i] = inf_value
    return pycbc.types.FrequencySeries(new_psd, delta_f=psd.delta_f, copy=False)


def whiten(strain, delta_t=1./2048., segment_duration=0.5,
                max_filter_duration=0.25, trunc_method='hann',
                remove_corrupted=True, low_frequency_cutoff=None,
                psd=None, return_psd=False, **kwargs):
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
    return_psd : {bool, False}
        Whether or not to return the PSD of the time series
        (whether supplied or estimated) and 
    kwargs : keywords
        Additional keyword arguments are passed on to the
        `pycbc.psd.welch` method.

    Returns
    -------
    whitened_data : numpy array
        The whitened time series.

    """
    if strain.ndim==1:
        from pycbc.psd import inverse_spectrum_truncation, interpolate
        colored_ts = pycbc.types.TimeSeries(strain, delta_t=delta_t)
        if psd is None:
            psd = colored_ts.psd(segment_duration, **kwargs)
        elif isinstance(psd, np.ndarray):
            assert psd.ndim==1
            logging.warning('Assuming PSD delta_f based on delta_t, length of PSD and EVEN length of original time series. Tread carefully!')
            assumed_duration = delta_t*(2*len(psd)-2)
            psd = pycbc.types.FrequencySeries(psd, delta_f=1./assumed_duration)
        elif isinstance(psd, pycbc.types.FrequencySeries):
            pass
        else:
            raise ValueError("Unknown format of PSD.")
        unprocessed_psd = psd
        psd = interpolate(psd, colored_ts.delta_f)
        max_filter_len = int(max_filter_duration*colored_ts.sample_rate)

        psd = inverse_spectrum_truncation(psd,
                    max_filter_len=max_filter_len,
                    low_frequency_cutoff=low_frequency_cutoff,
                    trunc_method=trunc_method)

        # inv_psd = np.array([0. if num==0. else 1./num for num in psd])
        inv_psd = 1./psd
        white_ts = (colored_ts.to_frequencyseries()*inv_psd**0.5).to_timeseries().numpy()

        if remove_corrupted:
            white_ts = white_ts[max_filter_len//2:(len(colored_ts)-max_filter_len//2)]

        if return_psd:
            return white_ts, unprocessed_psd
        else:
            return white_ts

    elif strain.ndim==2:
        psds_1d = None
        if isinstance(psd, np.ndarray):
            if psd.ndim==1:
                psds_1d = [psd for _ in strain]
        if psds_1d is None:
            if (psd is None) or isinstance(psd, pycbc.types.FrequencySeries):
                psds_1d = [psd for _ in strain]
            else:
                assert len(psd)==len(strain)
                psds_1d = psd

        white_segments = [whiten(sd_strain, delta_t=delta_t,
                segment_duration=segment_duration,
                max_filter_duration=max_filter_duration,
                trunc_method=trunc_method, remove_corrupted=remove_corrupted,
                low_frequency_cutoff=low_frequency_cutoff,
                psd=psd_1d, return_psd=return_psd, **kwargs) for sd_strain, psd_1d in zip(strain, psds_1d)]
        if return_psd:
            psds = [elem[1] for elem in white_segments]
            white_segments = np.stack([elem[0] for elem in white_segments], axis=0)
            return white_segments, psds
        else:
            return np.stack(white_segments, axis=0)

    else:
        raise ValueError("Strain numpy array dimension must be 1 or 2.")


def get_clusters(triggers, cluster_threshold=0.35):
    """Cluster a set of triggers into candidate detections.
    
    Arguments
    ---------
    triggers : list of triggers
        A list of triggers.  A trigger is a list of length two, where
        the first entry represents the trigger time and the second value
        represents the accompanying output value from the network.
    cluster_threshold : {float, 0.35}
        Cluster triggers together which are no more than this amount of
        time away from the boundaries of the corresponding cluster.
    
    Returns
    cluster_times :
        A numpy array containing the single times associated to each
        cluster.
    cluster_values :
        A numpy array containing the trigger values at the corresponing
        cluster_times.
    cluster_timevars :
        The timing certainty for each cluster. Injections must be within
        the given value for the cluster to be counted as true positive.
    """
    all_clusters = []
    for trigger_list in triggers.values():
        clusters = []
        for trigger in trigger_list:
            new_trigger_time = trigger[0]
            if len(clusters)==0:
                start_new_cluster = True
            else:
                last_cluster = clusters[-1]
                last_trigger_time = last_cluster[-1][0]
                start_new_cluster = (new_trigger_time - last_trigger_time)>cluster_threshold
            if start_new_cluster:
                clusters.append([trigger])
            else:
                last_cluster.append(trigger)
        all_clusters.append(clusters)

    all_clusters = [item for sublist in all_clusters for item in sublist]
    logging.info("Clustering has resulted in %i independent triggers. Centering triggers at their maxima." % len(all_clusters))

    cluster_times = []
    cluster_values = []
    cluster_timevars = []

    ### Determine maxima of clusters and the corresponding times and append them to the cluster_* lists
    for cluster in all_clusters:
        times = [trig[0] for trig in cluster]
        values = np.array([trig[1] for trig in cluster])
        max_index = np.argmax(values)
        cluster_times.append(times[max_index])
        cluster_values.append(values[max_index])
        cluster_timevars.append(0.2)

    cluster_times = np.array(cluster_times)
    cluster_values = np.array(cluster_values)
    cluster_timevars = np.array(cluster_timevars)

    return cluster_times, cluster_values, cluster_timevars


##############################
# PyTorch specific functions #
##############################

def get_base_network(path=None, device=default_device, detectors=1):
    """Return an instance of a coherent detection network.
    
    Arguments
    ---------
    path : {None or str, None}
        Path to the network (weights) that should be loaded. If None
        a new network will be initialized.
    device : {str, 'cpu'}
        The device on which the network is located.
    
    Returns
    -------
    network
    """

    network = torch.nn.Sequential(                          # Shapes
        torch.nn.BatchNorm1d(detectors),                    # ( 1*detectors)x2048
        torch.nn.Conv1d(detectors, 8*detectors, 33),        # ( 8*detectors)x2016
        torch.nn.ELU(),                                     # ( 8*detectors)x2016
        torch.nn.Conv1d(8*detectors, 8*detectors, 32),      # ( 8*detectors)x1985
        torch.nn.ELU(),                                     # ( 8*detectors)x1985
        torch.nn.Conv1d(8*detectors, 8*detectors, 17),      # ( 8*detectors)x1969
        torch.nn.ELU(),                                     # ( 8*detectors)x1969
        torch.nn.Conv1d(8*detectors, 8*detectors, 16),      # ( 8*detectors)x1954
        torch.nn.MaxPool1d(4),                              # ( 8*detectors)x 488
        torch.nn.ELU(),                                     # ( 8*detectors)x 488
        torch.nn.Conv1d(8*detectors, 8*detectors, 17),      # (16*detectors)x 472
        torch.nn.ELU(),                                     # (16*detectors)x 472
        torch.nn.Conv1d(8*detectors, 16*detectors, 16),     # (16*detectors)x 457
        torch.nn.ELU(),                                     # (16*detectors)x 457
        torch.nn.Conv1d(16*detectors, 16*detectors, 9),     # (16*detectors)x 449
        torch.nn.ELU(),                                     # (16*detectors)x 449
        torch.nn.Conv1d(16*detectors, 16*detectors, 8),     # (16*detectors)x 442
        torch.nn.MaxPool1d(3),                              # (16*detectors)x 147
        torch.nn.ELU(),                                     # (16*detectors)x 147
        torch.nn.Conv1d(16*detectors, 16*detectors, 9),     # (32*detectors)x 139
        torch.nn.ELU(),                                     # (32*detectors)x 139
        torch.nn.Conv1d(16*detectors, 32*detectors, 8),     # (32*detectors)x 132
        torch.nn.ELU(),                                     # (32*detectors)x 132
        torch.nn.Conv1d(32*detectors, 32*detectors, 9),     # (32*detectors)x 124
        torch.nn.ELU(),                                     # (32*detectors)x 124
        torch.nn.Conv1d(32*detectors, 32*detectors, 8),     # (32*detectors)x 117
        torch.nn.MaxPool1d(2),                              # (32*detectors)x  58
        torch.nn.ELU(),                                     # (32*detectors)x  58
        torch.nn.Flatten(),                                 #      1856*detectors
        torch.nn.Linear(1856*detectors, 64*detectors),      #        64*detectors
        torch.nn.Dropout(p=0.5),                            #        64*detectors
        torch.nn.ELU(),                                     #        64*detectors
        torch.nn.Linear(64*detectors, 64*detectors),        #        64*detectors
        torch.nn.Dropout(p=0.5),                            #        64*detectors
        torch.nn.ELU(),                                     #        64*detectors
        torch.nn.Linear(64*detectors, 2)                    #         2*detectors
        )
    if path is not None:
        network.load_state_dict(torch.load(path, map_location=device))
    network.to(dtype=dtype, device=device)
    return network

class CoherentNetwork(torch.nn.Module):
    def __init__(self, base_network, regularize=False):
        super().__init__()
#        self.add_module('base_network', base_network)
        self.base_network = base_network
#        self.add_module('softmax', torch.nn.Softmax(dim=1))
        self.softmax = torch.nn.Softmax(dim=1)
        self.regularize = regularize
    def forward(self, inputs):
        x = self.base_network(inputs)
        if self.regularize:
            return torch.unsqueeze(x[:, :1] - x[:, 1:], dim=1)
            # raise NotImplementedError("Regularization unsolved as of now")
        else:
            return self.softmax(x)

class ReducedStateDictCoherentNetwork(CoherentNetwork):
    def state_dict(self, *args, **kwargs):
        return self.base_network.state_dict(*args, **kwargs)
    def load_state_dict(self, *args, **kwargs):
        return self.base_network.load_state_dict(*args, **kwargs)

def get_coherent_network(path=None, device=default_device, detectors=1, regularize=False, reduced_state_dict=True):
    """Return an instance of a coherent detection network.

    Arguments
    ---------
    path : {None or str, None}
        Path to the network (weights) that should be loaded. If None
        a new network will be initialized.
    device : {str, 'cpu'}
        The device on which the network is located.
    detectors : {int, 1}
        Number of detectors to be used by the search.
    regularize : {bool, False}
        Whether to apply the USR regularization method.

    Returns
    -------
    network
    """

    base_network = get_base_network(path=path, device=device, detectors=detectors)
    if reduced_state_dict:
        network = ReducedStateDictCoherentNetwork(base_network, regularize=regularize)
    else:
        network = CoherentNetwork(base_network, regularize=regularize)
    return network

def product(elements):
    out = elements[0]
    if len(elements)>1:
        for elem in elements[1:]:
            out *= elem
    return out

def tensor_product(elements):
    out = torch.ones_like(elements[0])
    for elem in elements:
        out *= elem
    return out

class CoincidentNetwork(torch.nn.Module):
    def __init__(self, base_networks, regularize=False):
        super().__init__()
        self.num_detectors = len(base_networks)
#        for i, base_net in enumerate(base_networks):
#            self.add_module('base_network_%i' % i, base_net)
#        self.base_networks = [getattr(self, 'base_network_%i' % i) for i in range(self.num_detectors)]
        self.base_networks = torch.nn.ModuleList(base_networks)
        self.regularize = regularize
    def forward(self, inputs):
        split_inputs = torch.split(inputs, 1, dim=1)
        split_outputs = [base_net(inp)[:, 1:] for inp, base_net in zip(split_inputs, self.base_networks)]
        reduced_outputs = tensor_product(split_outputs)
        return torch.cat((torch.ones_like(reduced_outputs)-reduced_outputs, reduced_outputs), dim=1)
        # split_outputs = [base_net(inp)[:, :1] for inp, base_net in zip(split_inputs, self.base_networks)]
        # reduced_outputs = tensor_product(split_outputs)
        # return torch.cat((reduced_outputs, torch.ones_like(reduced_outputs)-reduced_outputs), dim=1)
    # def state_dict(self, *args, **kwargs):
    #     return self.base_network.state_dict(*args, **kwargs)
    # def load_state_dict(self, *args, **kwargs):
    #     return self.base_network.load_state_dict(*args, **kwargs)

def get_coincident_network(path=None, device=default_device, detectors=1, regularize=False):
    """Return an instance of a coincident detection network.
    
    Arguments
    ---------
    path : {None or str, None}
        Path to the network (weights) that should be loaded. If None
        a new network will be initialized.
    device : {str, 'cpu'}
        The device on which the network is located.
    detectors : {int, 1}
        Number of detectors to be used by the search.
    regularize : {bool, False}
        Whether to apply the USR regularization method.
    
    Returns
    -------
    network
    """

    single_det_networks = [get_coherent_network(path=None, device=device, detectors=1, reduced_state_dict=False) for _ in range(detectors)]
    network = CoincidentNetwork(single_det_networks)
    if not path is None:
        network.load_state_dict(torch.load(path))
    return network

def worker(inp):
    fpath = inp.pop('fpath')
    key = inp.pop('key')
    # pbar = inp.pop('pbar')
    # Network = inp.pop('Network')
    wdata_dict = None
    if 'wdata_dict' in inp.keys():
        wdata_dict = inp.pop('wdata_dict')

    with h5py.File(fpath, 'r') as infile:
        slicer = TorchSegmentSlicer(infile, key, **inp)
    if wdata_dict is None:
        logging.debug("Worker key %s finished processing, returning slicer" % key)
    else:
        logging.debug("Worker key %s finished processing, splitting and saving" % key)
        slicer.split_and_pop(wdata_dict, 10**6)
    return slicer

def evaluate_slices(slicer, Network, device='cpu', trigger_threshold=0.2):
    new_triggers = []
    data_loader = torch.utils.data.DataLoader(slicer,
                  batch_size=512, shuffle=False)
    with torch.no_grad():
        for slice_batch, slice_times in data_loader:
            output_values = Network(slice_batch.to(dtype=dtype, device=device))[:, 0]
            trigger_bools = torch.gt(output_values, trigger_threshold)
            for slice_time, trigger_bool, output_value in zip(slice_times, trigger_bools, output_values):
                if trigger_bool.clone().cpu().item():
                    new_triggers.append([slice_time.clone().cpu().item(), output_value.clone().cpu().item()])
    return new_triggers


def get_triggers(weights_path, coincident, usr, inputfile, step_size=0.1,
                 trigger_threshold=0.2, device='cpu',
                 verbose=False, white=False,
                 whitened_file=None, low_frequency_cutoff=20.,
                 num_workers=-1):
    """Use a network to generate a list of triggers, where the network
    outputs a value above a given threshold.
    
    Arguments
    ---------
    Network : network as returned by get_network
        The network to use during the evaluation.
    inputfile : str
        Path to the input data file.
    step_size : {float, 0.1}
        The step size (in seconds) to use for slicing the data.
    trigger_threshold : {float, 0.2}
        The value to use as a threshold on the network output to create
        triggers.
    device : {str, `cpu`}
        The device on which the calculations are carried out.
    verbose : {bool, False}
        Print update messages.
    white : {bool, False}
        Indicates that the data in inputfile is already whitened.
    whitened_file : {h5py File object, None}
        H5py file where data fed to the network will be stored
        (before slicing). If used with whiten_slices, will raise
        ValueError.
    num_workers : number of processes for parallelization. Set to a
        negative number to use as many processes as there are CPUs
        available. Set to 0 to run sequentially. Default: -1
    
    Returns
    -------
    triggers:
        A list of of triggers. A trigger is a list of length two, where
        the first entry represents the trigger time and the second value
        represents the accompanying output value from the network.
    """

    if num_workers < 0:
        num_workers = mp.cpu_count()

    detectors = ['H1', 'L1']
    if not whitened_file is None:
        with h5py.File(whitened_file, 'w') as wfile:
            for key in detectors:
                wfile.create_group(key)

    logging.debug("Initializing network.")
    if coincident:
        Network = get_coincident_network(path=weights_path, device=device, detectors=2, regularize=usr)
    else:
        Network = get_coherent_network(path=weights_path, device=device, detectors=2, regularize=usr)
    Network.to(dtype=dtype, device=device)
    Network.eval()
    triggers = {}

    arguments = []
    with h5py.File(inputfile, 'r') as infile:
        det_grp = next(iter(infile.values()))
        for key in list(det_grp.keys()):
            tmp = {}
            tmp['fpath'] = inputfile
            tmp['key'] = key
            # tmp['pbar'] = pbar
            # tmp['Network'] = Network
            tmp['step_size'] = step_size
            tmp['low_frequency_cutoff'] = low_frequency_cutoff
            tmp['white'] = white
            tmp['whitened_file'] = whitened_file
            tmp['detectors'] = ['H1', 'L1']
            arguments.append(tmp)

        arguments.sort(key=(lambda x: len(infile[x['detectors'][0]][x['key']])), reverse=True)

    if num_workers > 0:
        mp.set_start_method('forkserver')
        m = mp.Manager()
        wdata_dict = m.dict()
        for tmp in arguments:
            tmp['wdata_dict'] = wdata_dict
        with mp.Pool(num_workers) as pool:
            for slicer in tqdm(pool.imap_unordered(worker, arguments),
                    disable=not verbose, ascii=True,
                    total=len(arguments)):
                ### Gradually apply network to all samples and if output exceeds the trigger threshold, save the time and the output value
                slicer.stack_and_load(wdata_dict)
                triggers[slicer.key] = evaluate_slices(slicer, Network, trigger_threshold=trigger_threshold, device=device)
    else:
        for kwargs in tqdm(arguments, disable=not verbose, ascii=True):
            slicer = worker(kwargs)
            triggers[slicer.key] = evaluate_slices(slicer, Network, trigger_threshold=trigger_threshold, device=device)

    triggers = sorted(triggers.items(), key=lambda x: x[0])
    return dict(triggers)



def main():
    parser = ArgumentParser(description="CNN GW search script for submission to the MLGWSC-1 mock data challenge. Written by Ondřej Zelenka.")

    parser.add_argument('--verbose', action='store_true', help="Print update messages.")
    parser.add_argument('--debug', action='store_true', help="Show debug messages.")
    parser.add_argument('--force', action='store_true', help="Overwrite existing output file.")

    parser.add_argument('inputfile', type=str, help="The path to the input data file.")
    parser.add_argument('outputfile', type=str, help="The path where to store the triggers. The file must not exist.")
    parser.add_argument('--white', action='store_true', help="Use if the data in inputfile is already whitened. Otherwise the samples are whitened during evaluation.")
    parser.add_argument('--softmax', action='store_true', help="Indicates that a Softmax layer should be used at the end of the network. Otherwise, the USR is used as a default.")
    parser.add_argument('--coincident', action='store_true', help="Use a coincident search instead of a coherent search. Default: False")
    parser.add_argument('-w', '--weights', type=str, default='best_state_dict.pt', help="The path to the file containing the network weights. Default: best_state_dict.pt")
    parser.add_argument('-t', '--trigger-threshold', type=float, default=-8.0, help="The threshold to mark triggers. Default: -8.0")
    parser.add_argument('--step-size', type=float, default=0.1, help="The sliding window step size between analyzed samples. Default: 0.1")
    parser.add_argument('--cluster-threshold', type=float, default=0.35, help="The farthest in time that two slices can be to form a cluster. Default: 0.35")
    parser.add_argument('--device', type=str, default='cuda', help="Device to be used for analysis. Use 'cuda' for the GPU. Also, 'cpu:0', 'cuda:1', etc. (zero-indexed). Default: cuda")
    parser.add_argument('--debug-triggers-file', type=str, default=None, help="Path to a file where all triggers before clustering will be stored. Optional.")
    parser.add_argument('--debug-whitened-file', type=str, default=None, help="Path to a file where the whitened input data will be stored. Optional.")
    parser.add_argument('--num-workers', type=int, default=8, help="Number of processes to use. Set to a negative number to use as many processes as there are CPUs available. Set to 0 to run sequentially. Default: 8")

    args = parser.parse_args()

    ### Set up logging
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARN
    logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s', level=log_level, datefmt='%d-%m-%Y %H:%M:%S')

    ### Check existence of output file
    if os.path.isfile(args.outputfile) and not args.force:
        raise RuntimeError("Output file exists.")

    if not args.debug_triggers_file is None and not args.force:
        if os.path.isfile(args.debug_triggers_file):
            raise RuntimeError("Triggers file exists.")

    if not args.debug_whitened_file is None and not args.force:
        if os.path.isfile(args.debug_whitened_file):
            raise RuntimeError("Whitened file exists.")

    triggers = get_triggers(args.weights,
                        args.coincident,
                        not args.softmax,
                        args.inputfile,
                        step_size=args.step_size,
                        trigger_threshold=args.trigger_threshold,
                        device=args.device,
                        verbose=args.verbose,
                        white=args.white,
                        whitened_file=args.debug_whitened_file,
                        num_workers=args.num_workers,
                        low_frequency_cutoff=20.)
    logging.info("A total of %i samples exceeded the threshold of %f" % (sum([len(trigger_list) for trigger_list in triggers.values()]), args.trigger_threshold))

    if not args.debug_triggers_file is None:
        with h5py.File(args.debug_triggers_file, 'w') as triggers_file:
            for key, trigger_list in triggers.items():
                triggers_file.create_dataset(key, data=trigger_list)

    time, stat, var = get_clusters(triggers, args.cluster_threshold)

    with h5py.File(args.outputfile, 'w') as outfile:
        ### Save clustered values to the output file and close it
        logging.debug("Saving clustered triggers into %s." % args.outputfile)

        outfile.create_dataset('time', data=time)
        outfile.create_dataset('stat', data=stat)
        outfile.create_dataset('var', data=var)

        logging.debug("Triggers saved, closing file.")

if __name__=='__main__':
    main()
