### Import modules
from argparse import ArgumentParser
import logging
import numpy as np
import h5py
import pycbc.waveform, pycbc.noise, pycbc.psd, pycbc.distributions, pycbc.detector
import os, os.path
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
K = keras.backend

#####################
# General functions #
#####################

### Set default weights filename
default_weights_fname = 'weights'

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
                 slice_length=2048, detectors=None):
        self.infile = infile
        self.step_size = step_size        # this is the approximate one passed as an argument, the exact one is defined in the __next__ method
        self.peak_offset = peak_offset
        self.slice_length = slice_length
        self.detectors = detectors
        if self.detectors is None:
            self.detectors = [self.infile[key] for key in list(self.infile.attrs['detectors'])]
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
            
            nsteps = int((len(ds) - self.slice_length - 512) // \
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
        dt = self.detectors[0][key].attrs['delta_t']
        index_step_size = int(self.step_size / dt)
        sidx = (index.start - self.n_slices[key]['start']) * index_step_size
        eidx = (index.stop - self.n_slices[key]['start']) * index_step_size + self.slice_length + 512
        rawdata = [det[key][sidx:eidx] for det in self.detectors]
        times = (self.detectors[0][key].attrs['start_time'] + sidx * dt) + index_step_size * dt * np.arange(index.stop - index.start) + self.peak_offset
        
        data = np.zeros((index.stop - index.start, len(rawdata), self.slice_length))
        for detnum, rawdat in enumerate(rawdata):
            for i in range(index.stop - index.start):
                sidx = i * index_step_size
                eidx = sidx + self.slice_length + 512
                ts = pycbc.types.TimeSeries(rawdat[sidx:eidx], delta_t=dt)
                ts = ts.whiten(0.5, 0.25, low_frequency_cutoff=18.)
                data[i, detnum, :] = ts.numpy()
        return data, times
    
    def __getitem__(self, index):
        is_single = False
        if isinstance(index, int):
            is_single = True
            if index < 0:
                index = len(self) + index
            index = slice(index, index+1)
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


def generate_dataset(samples, verbose=False):
    """Generate a dataset that can be used for training and/or
    validation purposes.
    
    Arguments
    ---------
    samples : int
        The number of training samples to generate.
    verbose : {bool, False}
        Print update messages.
    """
    ### Create the detectors
    detectors_abbr = ('H1', 'L1')
    detectors = []
    for det_abbr in detectors_abbr:
        detectors.append(pycbc.detector.Detector(det_abbr))

    ### Create the power spectral densities of the respective detectors
    psd_fun = pycbc.psd.analytical.aLIGOZeroDetHighPower
    psds = [psd_fun(1281, 4./5., 18.) for _ in range(len(detectors))]

    ### Initialize the random distributions
    skylocation_dist = pycbc.distributions.sky_location.UniformSky()
    np_gen = np.random.default_rng()

    ### Create labels
    label_wave = np.array([1., 0.])
    label_noise = np.array([0., 1.])

    ### Generate data
    datasets = []
    num_waveforms, num_noises = samples
    logging.info(("Generating dataset with %i injections and %i pure "
                "noise samples") % (num_waveforms, num_noises))
    samples = []
    labels = []
    iterable = range(num_waveforms+num_noises)
    iterable = tqdm(iterable) if verbose else iterable
    for i in iterable:
        is_waveform = i<num_waveforms
        # Generate noise
        noise_fun = pycbc.noise.gaussian.frequency_noise_from_psd
        noise = [noise_fun(psd).to_timeseries().numpy() for psd in psds]
        noise = np.stack(noise, axis=0)
        # If in the first part of the dataset, generate waveform
        if is_waveform:
            # Generate source parameters
            waveform_kwargs = {'delta_t': 1./2048., 'f_lower': 18.}
            waveform_kwargs['approximant'] = 'IMRPhenomD'
            masses = np_gen.uniform(10., 50., 2)
            waveform_kwargs['mass1'] = max(masses)
            waveform_kwargs['mass2'] = min(masses)
            angles = np_gen.uniform(0., 2*np.pi, 3)
            waveform_kwargs['coa_phase'] = angles[0]
            waveform_kwargs['inclination'] = angles[1]
            declination, right_ascension = skylocation_dist.rvs()[0]
            pol_angle = angles[2]
            # Take the injection time randomly in the LIGO O3a era
            injection_time = np_gen.uniform(1238166018, 1253977218)
            # Generate the full waveform
            waveform = pycbc.waveform.get_td_waveform(**waveform_kwargs)
            h_plus, h_cross = waveform
            # Properly time and project the waveform
            start_time = injection_time + h_plus.get_sample_times()[0]
            h_plus.start_time = start_time
            h_cross.start_time = start_time
            h_plus.append_zeros(2560)
            h_cross.append_zeros(2560)
            strains = [det.project_wave(h_plus, h_cross, right_ascension, declination, pol_angle) for det in detectors]
            # Place merger randomly within the window between 0.5 s and 0.7 s of the time series and form the PyTorch sample
            time_placement = np_gen.uniform(0.5, 0.7)+0.125
            time_interval = injection_time-time_placement
            time_interval = (time_interval, time_interval+1.249)    # 1.499 to not get a too long strain
            strains = [strain.time_slice(*time_interval) for strain in strains]
            for strain in strains:
                to_append = 2560 - len(strain)
                if to_append>0:
                    strain.append_zeros(to_append)
            # Compute network SNR, rescale to generated target network SNR and inject into noise
            network_snr = np.sqrt(sum([pycbc.filter.matchedfilter.sigmasq(strain, psd=psd, low_frequency_cutoff=18.) for strain, psd in zip(strains, psds)]))
            target_snr = np_gen.uniform(5., 15.)
            sample = noise + np.stack([strain.numpy() for strain in strains], axis=0)*target_snr/network_snr
        # If in the second part of the dataset, merely use pure noise as the full sample
        else:
            sample = noise
        # Whiten
        sample = [pycbc.types.TimeSeries(strain, delta_t=1./2048.) for strain in sample]
        sample = [strain.whiten(0.5, 0.25, remove_corrupted=True, low_frequency_cutoff=18.) for strain in sample]
        sample = np.stack([strain.numpy() for strain in sample], axis=0)
        # Append to list of samples, as well as the corresponding label
        samples.append(sample)
        if is_waveform:
            labels.append(label_wave)
        else:
            labels.append(label_noise)
    # Merge samples and labels into just two tensors (more memory efficient) and initialize dataset
    samples = np.stack(samples, axis=0)
    labels = np.stack(labels, axis=0)
    return samples, labels

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
    clusters = []
    for trigger in triggers:
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

    logging.info("Clustering has resulted in %i independent triggers. Centering triggers at their maxima." % len(clusters))

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
        cluster_timevars.append(0.2)

    cluster_times = np.array(cluster_times)
    cluster_values = np.array(cluster_values)
    cluster_timevars = np.array(cluster_timevars)

    return cluster_times, cluster_values, cluster_timevars

   
############################
# Keras specific functions #
############################

class KerasSlicer(keras.utils.Sequence):
    def __init__(self, *args, **kwargs):
        keras.utils.Sequence.__init__(self)
        self.batch_size = kwargs.pop('batch_size', 32)
        self.slicer = Slicer(*args, **kwargs)
        self.times = {}
    
    def __len__(self):
        return int(len(self.slicer) / self.batch_size)
    
    def __getitem__(self, index):
        start = self.batch_size * index
        stop = min(start + self.batch_size, len(self.slicer))
        data, times = self.slicer[start:stop]
        if index not in self.times:
            self.times[index] = times
        return data.transpose(0, 2, 1)


def reg_loss(y_true, y_pred, eps=1e-6):
    x = eps + (1 - 2 * eps) * y_pred
    y = y_true
    
    loss = K.sum(y * K.log(x), axis=-1)
    return -K.mean(loss)


def get_network(path=None):
    """Return an instance of a network.
    
    Arguments
    ---------
    path : {None or str, None}
        Path to the network (weights) that should be loaded. If None
        a new network will be initialized.
    
    Returns
    -------
    network
    """
    if path is None:
        inp = keras.layers.Input(shape=(2048, 2))
        x = keras.layers.BatchNormalization()(inp)
        x = keras.layers.Conv1D(4, 64, activation='elu')(x)
        x = keras.layers.Conv1D(4, 32)(x)
        x = keras.layers.MaxPooling1D(4)(x)
        x = keras.layers.Activation('elu')(x)
        x = keras.layers.Conv1D(8, 32, activation='elu')(x)
        x = keras.layers.Conv1D(8, 16)(x)
        x = keras.layers.MaxPooling1D(3)(x)
        x = keras.layers.Activation('elu')(x)
        x = keras.layers.Conv1D(16, 16)(x)
        x = keras.layers.MaxPooling1D(4)(x)
        x = keras.layers.Activation('elu')(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(32)(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Activation('elu')(x)
        x = keras.layers.Dense(16)(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Activation('elu')(x)
        x = keras.layers.Dense(2, activation='softmax')(x)
        network = keras.models.Model(inputs=[inp], outputs=[x])
        
        network.compile(loss=reg_loss)
    else:
        network = keras.models.load_model(path,
                                          custom_objects={'reg_loss': reg_loss})
    
    return network


def train(Network, training_dataset, validation_dataset, output_training,
          weights_path, batch_size=32, learning_rate=5e-5,
          epochs=100, clip_norm=100, verbose=False):
    """Train a network on given data.
    
    Arguments
    ---------
    Network : network as returned by get_network
        The network to train.
    training_dataset : (np.array, np.array)
        The data to use for training. The first entry has to contain the
        input data, whereas the second entry has to contain the target
        labels.
    validation_dataset : (np.array, np.array)
        The data to use for validation. The first entry has to contain
        the input data, whereas the second entry has to contain the
        target labels.
    output_training : str
        Path to a directory in which the loss history and the best
        network weights will be stored.
    batch_size : {int, 32}
        The mini-batch size used for training the network.
    learning_rate : {float, 5e-5}
        The learning rate to use with the optimizer.
    epochs : {int, 100}
        The number of full passes over the training data.
    clip_norm : {float, 100}
        The value at which to clip the gradient to prevent exploding
        gradients.
    verbose : {bool, False}
        Print update messages.
    
    Returns
    -------
    network
    """
    opti = keras.optimizers.Adam(lr=learning_rate, epsilon=1e-8,
                                 clipnorm=clip_norm)
    Network.compile(optimizer=opti, loss=reg_loss)
    
    check = keras.callbacks.ModelCheckpoint(weights_path,
                                            save_best_only=True)
    logger = keras.callbacks.CSVLogger(os.path.join(output_training,
                                                    'losses.csv'))
    v = 1 if verbose else 0
    
    val_dat, val_lab = validation_dataset
    val_dat = val_dat.transpose(0, 2, 1)
    
    Network.fit(x=training_dataset[0].transpose(0, 2, 1),
                y=training_dataset[1],
                validation_data=(val_dat, val_lab),
                batch_size=batch_size,
                epochs=epochs,
                verbose=v,
                callbacks=[logger, check])
    Network = get_network(path=weights_path)
    
    return Network


def get_triggers(Network, inputfile, step_size=0.1,
                 trigger_threshold=0.2, verbose=0):
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
    verbose : {bool, False}
        Print update messages..
    
    Returns
    -------
    triggers:
        A list of of triggers. A trigger is a list of length two, where
        the first entry represents the trigger time and the second value
        represents the accompanying output value from the network.
    """
    v = 1 if verbose else 0
    with h5py.File(inputfile, 'r') as infile:
        slicer = KerasSlicer(infile,
                             step_size=step_size,
                             batch_size=512)
        triggers = []
        logging.info('Processing data')
        ret = Network.predict(slicer, verbose=v).T[0]
        times = np.concatenate([slicer.times[i] for i in range(len(slicer))])
    
    logging.info('Finding triggers')
    idxs = np.where(ret > trigger_threshold)[0]
    if len(idxs) == 0:
        return []
    triggers = [list(pt) for pt in zip(times[idxs], ret[idxs])]
    return triggers


def main():
    parser = ArgumentParser(description=("Basic example CNN training and "
                                         "corresponding search script "
                                         "supplied for the MLGWSC-1."))

    testing_group = parser.add_argument_group('testing')
    training_group = parser.add_argument_group('training')

    parser.add_argument('--verbose', action='store_true',
                        help="Print update messages.")
    parser.add_argument('--debug', action='store_true',
                        help="Show debug messages.")
    parser.add_argument('--train', action='store_true',
                        help="Train the network before applying.")

    testing_group.add_argument('inputfile', type=str,
                               help="The path to the input data file.")
    testing_group.add_argument('outputfile', type=str,
                               help=("The path where to store the "
                                     "triggers. The file must not "
                                     "exist."))
    testing_group.add_argument('-w', '--weights', type=str,
                               help=("The path to the file containing "
                                     "the network weights. If the "
                                     "--train option is present, the "
                                     "trained weights are used instead. "
                                     "Default: %s." % default_weights_fname))
    testing_group.add_argument('-t', '--trigger-threshold', type=float,
                               default=0.2,
                               help=("The threshold to mark triggers. "
                                     "Default: 0.2"))
    testing_group.add_argument('--step-size', type=float, default=0.1,
                               help=("The sliding window step size "
                                     "between analyzed samples. "
                                     "Default: 0.1"))
    testing_group.add_argument('--cluster-threshold', type=float,
                               default=0.35,
                               help=("The farthest in time that two "
                                     "slices can be to form a cluster. "
                                     "Default: 0.35"))

    training_group.add_argument('-o', '--output-training', type=str,
                                help=("Path to the directory where the "
                                      "outputs will be stored. The "
                                      "directory must exist."))
    training_group.add_argument('--training-samples', type=int, nargs=2,
                                default=[10000, 10000],
                                help=("Numbers of training samples as "
                                      "`injections` `pure noise "
                                      "samples`. Default: 10000 10000"))
    training_group.add_argument('--validation-samples', type=int,
                                nargs=2, default=[2000, 2000],
                                help=("Numbers of validation samples as "
                                      "`injections` `pure noise samples`. "
                                      "Default: 2000 2000"))
    training_group.add_argument('--learning-rate', type=float,
                                default=5e-5,
                                help=("Learning rate of the optimizer. "
                                      "Default: 0.00005"))
    training_group.add_argument('--epochs', type=int, default=100,
                                help=("Number of training epochs. "
                                      "Default: 100"))
    training_group.add_argument('--batch-size', type=int, default=32,
                                help=("Batch size of the training "
                                      "algorithm. Default: 32"))
    training_group.add_argument('--clip-norm', type=float, default=100.,
                                help=("Gradient clipping norm to "
                                      "stabilize the training. Default: 100."))

    args = parser.parse_args()

    ### Set up logging
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARN
    logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s',
                        level=log_level, datefmt='%d-%m-%Y %H:%M:%S')

    ### Check existence of output file
    if os.path.isfile(args.outputfile):
        raise RuntimeError("Output file exists.")
    else:
        pass

    ### Initialize network
    logging.debug("Initializing network.")
    Network = get_network(path=args.weights)
    
    if args.train:
        TrainDS = generate_dataset(args.training_samples, args.verbose)
        ValidDS = generate_dataset(args.validation_samples, args.verbose)
        weights_path = os.path.join(args.output_training, default_weights_fname)
        Network = train(Network,
                        TrainDS,
                        ValidDS,
                        args.output_training,
                        weights_path,
                        verbose=args.verbose,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        epochs=args.epochs,
                        clip_norm=args.clip_norm)
        
    triggers = get_triggers(Network,
                            args.inputfile,
                            step_size=args.step_size,
                            trigger_threshold=args.trigger_threshold,
                            verbose=args.verbose)

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