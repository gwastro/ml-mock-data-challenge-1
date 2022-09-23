#! /usr/bin/env python3
import argparse
import logging
from pycbc.types import TimeSeries
from tqdm import tqdm
import numpy as np

np.random.seed(0)
import h5py
import torch

torch.manual_seed(0)
from torch import nn
from torch.nn import functional as F
from torch.fft import rfft, rfftfreq, irfft


class Slicer(object):
    """Class that is used to slice and iterate over a single input data
    file.

    Arguments
    ---------
    infile : open file object
        The open HDF5 file from which the data should be read.
    step_size : {float, 0.1}
        The step size (in seconds) for slicing the data.
    peak_offset : {float, 0.6}
        The time (in seconds) from the start of each window where the
        peak is expected to be on average.
    slice_length : {int, 2048}
        The length of the output slice in samples.
    detectors : {None or list of datasets}
        The datasets that should be read from the infile. If set to None
        all datasets listed in the attribute `detectors` will be read.
    """

    def __init__(self, infile, step_size=0.1, peak_offset=0.6,
                 slice_length=2048, detectors=None, whiten=True):
        self.infile = infile
        self.step_size = step_size  # this is the approximate one passed as an argument, the exact one is defined in the __next__ method
        self.peak_offset = peak_offset
        self.slice_length = slice_length
        self.detectors = detectors
        self.whiten = whiten
        # self.detectors = ['H1', 'L1']
        # if self.detectors is None:
        #     self.detectors = [self.infile[key] for key in list(self.infile.attrs['detectors'])]
        self.detector_names = list(self.infile.keys())
        self.detectors = [self.infile[key] for key in self.detector_names]
        self.keys = sorted(list(self.detectors[0].keys()),
                           key=lambda inp: int(inp))
        self.determine_n_slices()
        return

    def determine_n_slices(self):
        self.n_slices = {}
        start = 0
        for ds_key in self.keys:
            ds = self.detectors[0][ds_key]
            dt = ds.attrs['delta_t']
            index_step_size = int(self.step_size / dt)

            if self.whiten:
                nsteps = int((len(ds) - self.slice_length - 512) // \
                             index_step_size)
            else:
                nsteps = int((len(ds) - self.slice_length) // \
                             index_step_size)

            self.n_slices[ds_key] = {'start': start,
                                     'stop': start + nsteps,
                                     'len': nsteps}
            start += nsteps

    def __len__(self):
        return sum([val['len'] for val in self.n_slices.values()])

    def _generate_access_indices(self, index):
        assert index.step is None or index.step == 1, 'Slice with step is not supported'
        ret = {}
        start = index.start
        stop = index.stop
        for key in self.keys:
            cstart = self.n_slices[key]['start']
            cstop = self.n_slices[key]['stop']
            if cstart <= start and start < cstop:
                ret[key] = slice(start, min(stop, cstop))
                start = ret[key].stop
        return ret

    def generate_data(self, key, index):
        # Ideally set dt = self.detectors[0][key].attrs['delta_t']
        # Due to numerical limitations this may be off by a single sample
        dt = 1. / 2048  # This definition limits the scope of this object
        index_step_size = int(self.step_size / dt)
        sidx = (index.start - self.n_slices[key]['start']) * index_step_size
        eidx = (index.stop - self.n_slices[key]['start']) * index_step_size + self.slice_length + 512
        rawdata = [det[key][sidx:eidx] for det in self.detectors]
        times = (self.detectors[0][key].attrs['start_time'] + sidx * dt) + index_step_size * dt * np.arange(
            index.stop - index.start) + self.peak_offset

        data = np.zeros((index.stop - index.start, len(rawdata), self.slice_length))
        for detnum, rawdat in enumerate(rawdata):
            for i in range(index.stop - index.start):
                sidx = i * index_step_size
                if self.whiten:
                    eidx = sidx + self.slice_length + 512
                    ts = TimeSeries(rawdat[sidx:eidx], delta_t=dt)
                    ts = ts.whiten(0.5, 0.25, low_frequency_cutoff=18.)
                else:
                    eidx = sidx + self.slice_length
                    ts = TimeSeries(rawdat[sidx:eidx], delta_t=dt)
                data[i, detnum, :] = ts.numpy()
        return data, times

    def __getitem__(self, index):
        is_single = False
        if isinstance(index, int):
            is_single = True
            if index < 0:
                index = len(self) + index
            index = slice(index, index + 1)
        access_slices = self._generate_access_indices(index)

        data = []
        times = []
        for key, idxs in access_slices.items():
            dat, t = self.generate_data(key, idxs)
            data.append(dat)
            times.append(t)
        data = np.concatenate(data)
        times = np.concatenate(times)

        if is_single:
            return data[0], times[0]
        else:
            return data, times


class TorchSlicer(Slicer, torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        torch.utils.data.Dataset.__init__(self)
        Slicer.__init__(self, *args, **kwargs)

    def __getitem__(self, index):
        next_slice, next_time = Slicer.__getitem__(self, index)
        return torch.from_numpy(next_slice), torch.tensor(next_time)


def get_clusters(triggers, cluster_threshold=0.35, var=0.2):
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
    clusters = []
    for trigger in triggers:
        new_trigger_time = trigger[0]
        if len(clusters) == 0:
            start_new_cluster = True
        else:
            last_cluster = clusters[-1]
            last_trigger_time = last_cluster[-1][0]
            start_new_cluster = (new_trigger_time - last_trigger_time) > cluster_threshold
        if start_new_cluster:
            clusters.append([trigger])
        else:
            last_cluster.append(trigger)

    logging.info(
        "Clustering has resulted in %i independent triggers. Centering triggers at their maxima." % len(clusters))

    cluster_times = []
    cluster_values = []
    cluster_timevars = []

    ### Determine maxima of clusters and the corresponding times and append them to the cluster_* lists
    for cluster in clusters:
        times = [trig[0] for trig in cluster]
        values = np.array([trig[1] for trig in cluster])
        max_index = np.argmax(values)
        cluster_times.append(times[max_index])
        cluster_values.append(values[max_index])
        cluster_timevars.append(var)

    cluster_times = np.array(cluster_times)
    cluster_values = np.array(cluster_values)
    cluster_timevars = np.array(cluster_timevars)

    return cluster_times, cluster_values, cluster_timevars


def get_triggers(Network, inputfile, step_size=0.1,
                 trigger_threshold=0.2, device='cpu',
                 verbose=False, dtype=torch.float32,
                 batch_size=512, slicer_cls=TorchSlicer,
                 num_workers=8, whiten=True, slice_length=2048):
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

    Returns
    -------
    triggers:
        A list of of triggers. A trigger is a list of length two, where
        the first entry represents the trigger time and the second value
        represents the accompanying output value from the network.
    """
    Network.to(dtype=dtype, device=device)
    with h5py.File(inputfile, 'r') as infile:
        slicer = slicer_cls(infile, step_size=step_size, whiten=whiten, slice_length=slice_length)
        triggers = []
        data_loader = torch.utils.data.DataLoader(slicer,
                                                  batch_size=batch_size,
                                                  shuffle=False, num_workers=num_workers,
                                                  pin_memory=True if 'cuda' in device else False)
        ### Gradually apply network to all samples and if output exceeds the trigger threshold, save the time and the output value
        iterable = tqdm(data_loader, desc="Iterating over dataset") if verbose else data_loader
        for slice_batch, slice_times in iterable:
            with torch.no_grad():
                output_values = Network(slice_batch.to(device=device))
                if isinstance(output_values, tuple):
                    output_added_times = output_values[1]
                    output_values = output_values[0]
                else:
                    output_added_times = None
                output_values = output_values[:, 0]
                trigger_bools = torch.gt(output_values, trigger_threshold)
                if output_added_times is not None:
                    n_slices = output_added_times.size(0) // slice_batch.size(0)
                    slice_times = slice_times.repeat_interleave(n_slices)
                    slice_times += output_added_times
                    for slice_time, trigger_bool, output_value in zip(slice_times, trigger_bools, output_values):
                        if trigger_bool.clone().cpu().item():
                            triggers.append(
                                [slice_time.clone().cpu().item() + 0.125, output_value.clone().cpu().item()])
                else:
                    for slice_time, trigger_bool, output_value in zip(slice_times, trigger_bools, output_values):
                        if trigger_bool.clone().cpu().item():
                            triggers.append(
                                [slice_time.clone().cpu().item() + 0.125, output_value.clone().cpu().item()])
        logging.info("A total of %i slices have exceeded the threshold of %f." % (len(triggers), trigger_threshold))
    return triggers


class DAIN_Layer(nn.Module):
    def __init__(self, mode='full', mean_lr=0.00001, gate_lr=0.001, scale_lr=0.00001, input_dim=144):
        super(DAIN_Layer, self).__init__()
        self.mode = mode
        self.mean_lr = mean_lr
        self.gate_lr = gate_lr
        self.scale_lr = scale_lr

        # Parameters for adaptive average
        self.mean_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.mean_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))

        # Parameters for adaptive std
        self.scaling_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.scaling_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))

        # Parameters for adaptive scaling
        self.gating_layer = nn.Linear(input_dim, input_dim)

        self.eps = 1e-8

    def forward(self, x):
        # Expecting  (n_samples, dim,  n_feature_vectors)
        # Step 1:
        avg = torch.mean(x, 2)
        adaptive_avg = self.mean_layer(avg)
        adaptive_avg = adaptive_avg.resize(adaptive_avg.size(0), adaptive_avg.size(1), 1)
        x = x - adaptive_avg

        # # Step 2:
        std = torch.mean(x ** 2, 2)
        std = torch.sqrt(std + self.eps)
        adaptive_std = self.scaling_layer(std)
        adaptive_std[adaptive_std <= self.eps] = 1

        adaptive_std = adaptive_std.resize(adaptive_std.size(0), adaptive_std.size(1), 1)
        x = x / adaptive_std

        # Step 3:
        avg = torch.mean(x, 2)
        gate = F.sigmoid(self.gating_layer(avg))
        gate = gate.resize(gate.size(0), gate.size(1), 1)
        x = x * gate
        return x


def torch_inverse_spectrum_truncation(psd, max_filter_len, low_frequency_cutoff=9., delta_f=1.,
                                      trunc_method='hann'):
    N = (psd.size(1) - 1) * 2
    inv_asd = torch.zeros_like(psd)
    kmin = int(low_frequency_cutoff / delta_f)
    inv_asd[0, kmin:N // 2] = (1. / psd[0, kmin:N // 2]) ** 0.5
    q = irfft(inv_asd, n=N, norm='forward')
    trunc_start = max_filter_len // 2
    trunc_end = N - max_filter_len // 2
    if trunc_method == 'hann':
        trunc_window = torch.hann_window(max_filter_len, dtype=torch.float64).to(psd.device)
        q[0, 0:trunc_start] *= trunc_window[-trunc_start:]
        q[0, trunc_end:] *= trunc_window[:max_filter_len // 2]
    if trunc_start < trunc_end:
        q[0, trunc_start:trunc_end] = 0
    psd_trunc = rfft(q, n=N, norm='forward')
    psd_trunc *= psd_trunc.conj()

    psd = 1 / torch.abs(psd_trunc)
    return psd / 2


class Whiten(nn.Module):
    def __init__(self, delta_t, low_frequency_cutoff=15., m=1.25, max_filter_len=1.):
        super().__init__()
        # store psd estimate
        self.max_filter_len = max_filter_len
        self.delta_t = delta_t
        self.delta_f = 1 / m
        m /= delta_t
        self.m = int(m)
        self.d = int(m / 2)
        self.psd_est = None
        self.norm = nn.Parameter(torch.zeros(2, 1281), requires_grad=False)
        self.frequencies = None
        self.low_frequency_cutoff = low_frequency_cutoff
        self.frequencies = rfftfreq(self.m, d=self.delta_t)

    def initialize(self, noise_t):
        if noise_t.dim() == 2:
            noise_t = noise_t.unsqueeze(0).unsqueeze(2)
        n_channels = noise_t.size(1)
        psds = []
        for c in range(n_channels):
            psd = self.estimate_psd(noise_t[:, c, :, :].unsqueeze(1))
            psds.append(psd)
        self.psd_est = torch.cat(psds, dim=0)
        self.norm.data = self.psd_est ** 0.5

    def estimate_psd(self, noise_t):
        """
        noise in (1, C, 1, D) Tensor format
        """
        # step 1: split signal into L segments of M length with D overlap
        m = self.m
        d = self.d
        segments = F.unfold(noise_t, kernel_size=(1, m), stride=(1, d)).double()

        # step 2: apply hann window over all segments
        w_hann = torch.hann_window(segments.size(1), periodic=True, dtype=torch.float64).to(segments.device)
        segments_w = segments * w_hann.unsqueeze_(1)

        # step 3: compute FFT for all segments
        segments_fft = rfft(segments_w, dim=1, norm="forward")

        segments_sq_mag = torch.abs(segments_fft * segments_fft.conj())
        segments_sq_mag[0, 0, :] /= 2
        segments_sq_mag[0, -1, :] /= 2

        # step 4: aggregate (we use the mean, but pycbc uses median by default)
        t_psd = torch.mean(segments_sq_mag, dim=2)
        t_psd *= 2 * self.delta_f * m / (w_hann * w_hann).sum()

        # final step: interpolate if needed and inverse spectrum truncation
        if t_psd.size(1) != 1281:
            t_psd = F.interpolate(t_psd.unsqueeze(1), 1281).squeeze(1)
            self.frequencies = rfftfreq(int(1.25 * 2048), d=self.delta_t)
        t_psd = torch_inverse_spectrum_truncation(t_psd, int(self.max_filter_len / self.delta_t),
                                                  low_frequency_cutoff=self.low_frequency_cutoff,
                                                  delta_f=self.delta_f)
        return t_psd

    def forward(self, signal):
        return self.whiten(signal)
        # then whiten signal with self.psd_est and return the whitened version

    def whiten(self, signal):
        signal_f = rfft(signal.double(), dim=2, norm='forward')
        signal_t = irfft(signal_f / self.norm, norm='forward', n=signal.size(2))
        return signal_t.float()


class CropWhitenNet(nn.Module):
    def __init__(self, net=None, norm=None, deploy=False, m=0.625, mfl=0.5, f=15.):
        super(CropWhitenNet, self).__init__()
        self.net = net
        self.norm = norm
        self.whiten = Whiten(1 / 2048, low_frequency_cutoff=f, m=m, max_filter_len=mfl)
        self.deploy = deploy
        self.step = 0.1

    def forward(self, x):
        n_batch = x.size(0)
        slice_len = x.size(2)
        # crop for eval
        segments_wh = []
        with torch.no_grad():
            c = x.size(1)
            for i, sample in enumerate(x):
                self.whiten.initialize(sample)
                sample = sample.unsqueeze(0).unsqueeze(2)
                x_segments = F.unfold(sample, kernel_size=(1, 2560), stride=(1, 204)).contiguous()
                n = sample.size(0)
                l = x_segments.size(2)
                x_segments = x_segments.view(n, c, -1, l).permute(3, 1, 2, 0).squeeze_(3)
                segments_wh.append(self.whiten(x_segments)[:, :, 256:-256])
            segments_wh = torch.cat(segments_wh)
        if self.norm is not None:
            segments_wh = self.norm(segments_wh)
        if self.deploy:
            added_time = torch.arange(0, slice_len / 2048. - 1.25 + self.step, self.step).repeat(n_batch)
            return self.net(segments_wh), added_time
        else:
            return self.net(segments_wh)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        if out_channels != in_channels or stride > 1:
            self.x_transform = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.x_transform = nn.Identity()

        self.body = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        x = F.relu(self.body(x) + self.x_transform(x))
        return x


class ResNet52(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            ResBlock(2, 8),
            ResBlock(8, 8),
            ResBlock(8, 8),
            ResBlock(8, 8),
            ResBlock(8, 16, stride=2),
            ResBlock(16, 16),
            ResBlock(16, 16),
            ResBlock(16, 32, stride=2),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 64, stride=2),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64, stride=2),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64, stride=2),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 32),
            ResBlock(32, 16),
            ResBlock(16, 16),
            ResBlock(16, 16),
        )
        self.cls_head = nn.Sequential(
            nn.Conv1d(16, 32, 64), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 2, 1), nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.cls_head(x).squeeze(2)


# Set default weights filename
weights_path = 'trained_models/d4_model/weights.pt'

# Set data type to be used
dtype = torch.float32

sample_rate = 2048
delta_t = 1. / sample_rate
delta_f = 1 / 1.25


def main(args):
    device = 'cuda:0' if torch.cuda.device_count() > 0 else 'cpu'

    base_model = ResNet52().to(device)
    norm = DAIN_Layer(input_dim=2).to(device)

    net = CropWhitenNet(base_model, norm).to(device)
    net.whiten.max_filter_len = 0.5
    net.whiten.legacy = False
    net.deploy = True
    net.load_state_dict(torch.load(weights_path, map_location=device))

    net.eval()

    # run on foreground
    inputfile = args.inputfile
    outputfile = args.outputfile
    step_size = 3.1
    slice_dur = 4.25
    trigger_threshold = 0.4
    cluster_threshold = 0.35
    var = 0.3

    test_batch_size = 32

    with torch.no_grad():
        triggers = get_triggers(net,
                                inputfile,
                                step_size=step_size,
                                trigger_threshold=trigger_threshold,
                                device=device,
                                verbose=True,
                                batch_size=test_batch_size,
                                whiten=False,
                                slice_length=int(slice_dur * 2048))

    time, stat, var = get_clusters(triggers, cluster_threshold, var=var)

    with h5py.File(outputfile, 'w') as outfile:
        print("Saving clustered triggers into %s." % outputfile)

        outfile.create_dataset('time', data=time)
        outfile.create_dataset('stat', data=stat)
        outfile.create_dataset('var', data=var)

        print("Triggers saved, closing file.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('inputfile', type=str, help="The path to the input data file.")
    parser.add_argument('outputfile', type=str, help="The path where to store the triggers.")

    args = parser.parse_args()

    main(args)
