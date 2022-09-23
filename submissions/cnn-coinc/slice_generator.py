import multiprocessing as mp
import h5py
import queue
import time
import warnings

import numpy as np
from pycbc.types import TimeSeries


class Slicer(object):
    """Generator to access slices of an input file as output by generate_data.py
    for the MLGWSC-1.
    
    Arguments
    ---------
    filepath : str
        Path to the file that should be sliced.
    step_size : {int, 204}
        The stride of the slicer in samples.
    window_size : {int, 2048}
        The size of the window in samples for which processing is done.
    workers : {None or int > 0, None}
        How many processes to start for data reading and processing.
    prefetch : {int, 0}
        How many samples to pre-calculate. Can improve performance at
        at the cost of memory efficiency.
    timeout : {float, 0.01}
        How long to wait when trying to read from or write to a prallel
        queue.
    batch_size : {int, 1}
        The number of samples to accumulate for each call to __next__.
    
    Notes
    -----
    +To apply processing to the data, sub-class this class and overwrite
     the methods
     -process_slice
     -format_return
     The process_slice method runs in parallel, if multiple workers were
     requested. Any heavy processing should be put into this method.
     The format_return method runs sequentially and should only do light
     re-formatting of the output.
    +Usage:
     >>> gen = Slicer(filepath, workers=2, prefetch=4)
     >>> with gen:
     >>>    results = list(gen)
    """
    def __init__(self, filepath, step_size=204, window_size=2048,
                 workers=None, prefetch=0, timeout=0.01, batch_size=1):
        self.filepath = filepath
        self.step_size = int(step_size)
        self.window_size = int(window_size)
        self.workers = workers
        self.prefetch = prefetch
        self.timeout = timeout
        self.batch_size = batch_size
        self.entered = False
        self._init_file_vars()
        self.determine_n_slices()
        self.reset()
    
    def _init_file_vars(self):
        with h5py.File(self.filepath, 'r') as fp:
            self.detectors = ['H1', 'L1']
            self.sample_rate = 2048
            self.flow = 15
            self.keys = sorted(fp[self.detectors[0]].keys(),
                               key=lambda inp: int(inp))
    
    def determine_n_slices(self):
        self.n_slices = {}
        start = 0
        with h5py.File(self.filepath, 'r') as fp:
            for ds_key in self.keys:
                ds = fp[self.detectors[0]][ds_key]
                
                nsteps = int((len(ds) - self.window_size) // self.step_size)
                
                self.n_slices[ds_key] = {'start': start,
                                         'stop': start + nsteps,
                                         'len': nsteps}
                start += nsteps
    
    @property
    def n_samples(self):
        if not hasattr(self, 'n_slices'):
            self.determine_n_slices()
        return sum([val['len'] for val in self.n_slices.values()])
    
    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))
    
    def empty_queues(self):
        while True:
            try:
                self.fetched.get(timeout=0.01)
            except (queue.Empty, AttributeError):
                break
        while True:
            try:
                self.index_queue.get(timeout=0.01)
            except (queue.Empty, AttributeError):
                break
    
    def reset(self):
        self.index = 0
        self.empty_queues()
        self.last_index_put = -1
        if hasattr(self, 'last_fetched'):
            self.last_fetched.value = -1
    
    def _access_index(self, index):
        for ds, dic in self.n_slices.items():
            if dic['start'] <= index and index < dic['stop']:
                return (ds, index - dic['start'])
        else:
            raise IndexError('Index not found')
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self):
            raise StopIteration
        if self.workers is None \
           or self.prefetch < 1 \
           or not self.entered:  # Single process
            if self.workers is not None \
               and self.workers > 0 \
               and self.prefetch > 0:
                warnings.warn(("Multiple workers were requested but the "
                               "generator was not entered. Remember to use "
                               "the generator as a context manager. Running "
                               "sequentially."), RuntimeWarning)
            batch_idxs = list(range(self.index * self.batch_size,
                                    min(self.n_samples,
                                        (self.index + 1) * self.batch_size)))
            ret = [[] for _ in self.detectors]
            for idx in batch_idxs:
                ds, dsidx = self._access_index(idx)
                start = dsidx * self.step_size
                stop = start + self.window_size
                with h5py.File(self.filepath, 'r') as fp:
                    for i, det in enumerate(self.detectors):
                        data = fp[det][ds][start:stop]
                        ret[i].append(self.process_slice(data, det))
        else:  # Multiprocessing
            upper = min(self.index + self.prefetch, len(self))
            if upper > self.last_index_put:
                for i in range(self.last_index_put+1, upper):
                    self.index_queue.put(i)
                    self.last_index_put = i
                if len(self) <= upper:
                    self.last_index_put = len(self)
            while True:
                try:
                    ret = self.fetched.get(timeout=self.timeout)
                    break
                except queue.Empty:
                    continue
        
        self.index += 1
        ret = [np.stack(pt) for pt in ret]
        return self.format_return(ret)
    
    def _fetch_func(self, pidx, index_pipe, output_pipe, event):
        ret = None
        index = None
        with h5py.File(self.filepath, 'r') as fp:
            while not event.is_set():
                if ret is None:
                    try:
                        index = index_pipe.get(timeout=self.timeout)
                        batch_idxs = list(range(index * self.batch_size,
                                          min(self.n_samples,
                                              (index + 1) * self.batch_size)))
                        ret = [[] for _ in self.detectors]
                        for idx in batch_idxs:
                            ds, dsidx = self._access_index(idx)
                            start = dsidx * self.step_size
                            stop = start + self.window_size
                            for i, det in enumerate(self.detectors):
                                data = fp[det][ds][start:stop]
                                ret[i].append(self.process_slice(data, det))
                    except queue.Empty:
                        continue
                try:
                    if self.last_fetched.value + 1 != index:
                        time.sleep(self.timeout)
                    else:
                        output_pipe.put(ret, timeout=self.timeout)
                        self.last_fetched.value = index
                        ret = None
                except queue.Full:
                    continue
    
    def __enter__(self):
        if self.workers is not None and self.workers > 0 and self.prefetch > 0:
            self.event = mp.Event()
            self.processes = []
            self.fetched = mp.Queue(maxsize=2*self.prefetch)
            self.index_queue = mp.Queue(maxsize=2*self.prefetch)
            self.last_fetched = mp.Value('i', -1)
            for i in range(self.workers):
                process = mp.Process(target=self._fetch_func,
                                     args=(i,
                                           self.index_queue,
                                           self.fetched,
                                           self.event))
                self.processes.append(process)
                process.start()
            self.entered = True
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        if hasattr(self, 'event'):
            self.event.set()
            self.empty_queues()
            if hasattr(self, 'processes'):
                self.empty_queues()
                while len(self.processes) > 0:
                    process = self.processes.pop(0)
                    process.join()
            self.event = None
            self.entered = False
    
    def process_slice(self, data, detector):
        """Applies processing to the raw data from one detector as read
        from the file.
        
        Arguments
        ---------
        data : numpy.array
            1 dimensional array of length self.window_size with dtype
            specified by the read file (usually numpy.float32).
        detector : str
            The string specifying the detector.
        
        Returns
        -------
        data
            Processed data that can be pickled.
        """
        return data
    
    def format_return(self, data):
        """Formats the return value of the generator for a single slice
        and both detectors.
        
        Arguments
        ---------
        data : list of self.process_slice return values
            A list containing the processed single detector slices.
        
        Returns
        -------
        data
            The formatted data that can be understood by the consumer.
        """
        return data


class SliceGenerator(Slicer):
    def __init__(self, *args, pre_whitened=False, time_offset=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_whitened = pre_whitened
        self.time_offset = time_offset
    
    def process_slice(self, data, detector):
        if not self.pre_whitened:
            data = TimeSeries(data, delta_t=1 / self.sample_rate)
            data = data.whiten(1, 1, low_frequency_cutoff=18)
            data = data[:int(self.sample_rate)].numpy()
        data = np.expand_dims(data, axis=-1)
        return data
    
    def sample_times(self):
        times = np.zeros(self.n_samples)
        time_offset = self.time_offset
        if not self.pre_whitened:
            time_offset += 1
        with h5py.File(self.filepath, 'r') as fp:
            for i in range(self.n_samples):
                ds_key, dsidx = self._access_index(i)
                ds = fp[self.detectors[0]][ds_key]
                
                start_time = ds.attrs['start_time']
                time = start_time + dsidx * self.step_size / self.sample_rate
                
                times[i] = time
        return times + time_offset
