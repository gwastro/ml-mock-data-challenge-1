### Import modules
from argparse import ArgumentParser
import logging
import numpy as np
import h5py
import pycbc.waveform, pycbc.noise, pycbc.psd
import pycbc.distributions, pycbc.detector
import os, os.path
from tqdm import tqdm

import torch

### Set default weights filename
default_weights_fname = 'weights.pt'

### Set data type to be used
dtype = torch.float32


### Basic dataset class for easy PyTorch loading
class Dataset(torch.utils.data.Dataset):
	def __init__(self, samples, labels,
				store_device='cpu', train_device='cpu'):
		torch.utils.data.Dataset.__init__(self)
		self.samples = torch.from_numpy(samples)
		self.labels = torch.from_numpy(labels)
		self.samples = self.samples.to(dtype=dtype,device=store_device)
		self.labels = self.labels.to(dtype=dtype,device=store_device)
		self.train_device = train_device
		assert len(self.samples)==len(self.labels)
		return
	
	def __len__(self):
		return len(self.samples)
	
	def __getitem__(self, i):
		sample = self.samples[i].to(device=self.train_device)
		label = self.labels[i].to(device=self.train_device)
		return sample, label

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
		self.step_size = step_size		# approximate step
		self.peak_offset = peak_offset
		self.slice_length = slice_length
		self.detectors = detectors
		if self.detectors is None:
			self.detectors = []
			for key in list(self.infile.attrs['detectors']):
				self.detectors.append(self.infile[key])
		return

	def __len__(self):
		slices = 0
		for key in self.detectors[0].keys():
			ds = self.detectors[0][key]
			full_slice_length = 512 + self.slice_length
			index_step_size = int(self.step_size/ds.attrs['delta_t'])
			sli_num = 1+((len(ds)-full_slice_length)//index_step_size)
			slices += max((0, sli_num))
		return slices
	
	def __iter__(self):
		self.ds_key_iter = iter(self.detectors[0].keys())
		self.current_ds_key = None
		return self
	
	def start_next_ds(self):
		self.current_ds_key = next(self.ds_key_iter)
		self.current_dss = []
		for det in self.detectors:
			self.current_dss.append(det[self.current_ds_key])
		self.current_index = 0
		self.current_time = self.current_dss[0].attrs['start_time']
		self.delta_t = self.current_dss[0].attrs['delta_t']
		for ds in self.current_dss:
			assert (ds.attrs['start_time']==self.current_time)
			assert (ds.attrs['delta_t']==self.delta_t)
		# Exact integer step size
		self.index_step_size = int(self.step_size/self.delta_t)
		# Exact step size
		self.time_step_size = self.delta_t*self.index_step_size
		return
	
	def get_next_slice_in_ds(self):
		if self.current_index+self.slice_length+512>len(self.current_dss[0]):
			raise StopIteration
		else:
			this_slice = [pycbc.types.TimeSeries(ds[self.current_index:self.current_index+self.slice_length+512], delta_t=self.delta_t) for ds in self.current_dss]
			this_slice = [strain.whiten(0.5, 0.25, remove_corrupted=True, low_frequency_cutoff=18.) for strain in this_slice]
			this_slice = np.stack([strain.numpy() for strain in this_slice], axis=0)
			this_time = self.current_time + self.peak_offset + 0.125
			self.current_index += self.index_step_size
			self.current_time += self.time_step_size
			return this_slice, this_time
	
	def get_next_slice(self):
		if self.current_ds_key is None:
			self.start_next_ds()
		try:
			return self.get_next_slice_in_ds()
		except StopIteration:
			logging.debug(("Iterator over dataset starting at inttime "
				"%i raised StopIteration." % int(self.current_ds_key)))
			self.start_next_ds()
			return self.get_next_slice()
	
	def __next__(self):
		return self.get_next_slice()

class TorchSlicer(Slicer, torch.utils.data.IterableDataset):
	def __init__(self, *args, **kwargs):
		torch.utils.data.IterableDataset.__init__(self)
		Slicer.__init__(self, *args, **kwargs)
	
	def __next__(self):
		next_slice, next_time = self.get_next_slice()
		return torch.from_numpy(next_slice), torch.tensor(next_time)

def get_network(path=None, device='cpu'):
	"""Return an instance of a network.
	
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

	network = torch.nn.Sequential(		# Shapes
		torch.nn.BatchNorm1d(2),		#  2x2048
		torch.nn.Conv1d(2, 4, 64),		#  4x1985
		torch.nn.ELU(),					#  4x1985
		torch.nn.Conv1d(4, 4, 32),		#  4x1954
		torch.nn.MaxPool1d(4),			#  4x 489
		torch.nn.ELU(),					#  4x 489
		torch.nn.Conv1d(4, 8, 32),		#  8x 458
		torch.nn.ELU(),					#  8x 458
		torch.nn.Conv1d(8, 8, 16),		#  8x 443
		torch.nn.MaxPool1d(3),			#  8x 147
		torch.nn.ELU(),					#  8x 147
		torch.nn.Conv1d(8, 16, 16),		# 16x 132
		torch.nn.ELU(),					# 16x 132
		torch.nn.Conv1d(16, 16, 16),	# 16x 117
		torch.nn.MaxPool1d(4),			# 16x  29
		torch.nn.ELU(),					# 16x  29
		torch.nn.Flatten(),				#     464
		torch.nn.Linear(464, 32),		#      32
		torch.nn.Dropout(p=0.5),		#      32
		torch.nn.ELU(),					#      32
		torch.nn.Linear(32, 16),		#      16
		torch.nn.Dropout(p=0.5),		#      16
		torch.nn.ELU(),					#      16
		torch.nn.Linear(16, 2),			#       2
		torch.nn.Softmax(dim=1)			#       2
		)
	if path is not None:
		network.load_state_dict(torch.load(path))
	network.to(dtype=dtype, device=device)
	return network

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
			time_interval = (time_interval, time_interval+1.249)	# 1.499 to not get a too long strain
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


def train(Network, training_dataset, validation_dataset, output_training,
          weights_path, store_device='cpu', train_device='cpu',
          batch_size=32, learning_rate=5e-5, epochs=100, clip_norm=100,
          verbose=False):
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
	weights_path: str
		Path where the trained network weights will be stored.
	store_device : {str, `cpu`}
		The device on which the data sets should be stored.
	train_device : {str, `cpu`}
		The device on which the network should be trained.
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
	### Set up data loaders as a PyTorch convenience
	logging.debug("Setting up datasets and data loaders.")
	TrainDS = Dataset(*training_dataset, store_device=store_device, train_device=train_device)
	ValidDS = Dataset(*validation_dataset, store_device=store_device, train_device=train_device)
	TrainDL = torch.utils.data.DataLoader(TrainDS, batch_size=batch_size, shuffle=True)
	ValidDL = torch.utils.data.DataLoader(ValidDS, batch_size=500, shuffle=True)

	### Initialize loss function, optimizer and output file
	logging.debug("Initializing loss function, optimizer and output file.")
	loss = torch.nn.BCELoss()
	opt = torch.optim.Adam(Network.parameters(), lr=learning_rate)
	with open(os.path.join(output_training, 'losses.txt'), 'w') as outfile:

		### Training loop
		best_loss = 1.e10 # impossibly bad value
		iterable1 = range(1, epochs+1)
		iterable1 = tqdm(iterable1, desc="Optimizing network") if verbose else iterable1
		for epoch in iterable1:
			# Training epoch
			Network.train()
			training_running_loss = 0.
			training_batches = 0
			iterable2 = TrainDL
			iterable2 = tqdm(iterable2, desc="Iterating over training dataset", leave=False) if verbose else iterable2
			for training_samples, training_labels in iterable2:
				# Optimizer step on a single batch of training data
				opt.zero_grad()
				training_output = Network(training_samples)
				training_loss = loss(training_output, training_labels)
				training_loss.backward()
				# Clip gradients to make convergence somewhat easier
				torch.nn.utils.clip_grad_norm_(Network.parameters(), max_norm=clip_norm)
				# Make the actual optimizer step and save the batch loss
				opt.step()
				training_running_loss += training_loss.clone().cpu().item()
				training_batches += 1
			# Evaluation on the validation dataset
			Network.eval()
			with torch.no_grad():
				validation_running_loss = 0.
				validation_batches = 0
				iterable2 = ValidDL
				iterable2 = tqdm(iterable2, desc="Computing validation loss", leave=False) if verbose else iterable2
				for validation_samples, validation_labels in iterable2:
					# Evaluation of a single validation batch
					validation_output = Network(validation_samples)
					validation_loss = loss(validation_output, validation_labels)
					validation_running_loss += validation_loss.clone().cpu().item()
					validation_batches += 1
			# Print information on the training and validation loss in the current epoch and save current network state
			validation_loss = validation_running_loss/validation_batches
			output_string = '%04i    %f    %f' % (epoch, training_running_loss/training_batches, validation_loss)
			outfile.write(output_string + '\n')
			# Save 
			if validation_loss<best_loss:
				torch.save(Network.state_dict(), weights_path)
				best_loss = validation_loss

		logging.debug(("Training complete with best validation loss "
						"%f, closing losses output file." % best_loss))
	
	Network.load_state_dict(torch.load(weights_path))
	return Network

def get_triggers(Network, inputfile, step_size=0.1,
                 trigger_threshold=0.2, device='cpu', verbose=False):
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
		slicer = TorchSlicer(infile, step_size=step_size)
		triggers = []
		data_loader = torch.utils.data.DataLoader(slicer,
		                                          batch_size=512,
		                                          shuffle=False)
		### Gradually apply network to all samples and if output exceeds the trigger threshold, save the time and the output value
		iterable = tqdm(data_loader, desc="Iterating over dataset") if verbose else data_loader
		for slice_batch, slice_times in iterable:
			with torch.no_grad():
				output_values = Network(slice_batch.to(dtype=dtype, device=device))[:, 0]
				trigger_bools = torch.gt(output_values, trigger_threshold)
				for slice_time, trigger_bool, output_value in zip(slice_times, trigger_bools, output_values):
					if trigger_bool.clone().cpu().item():
						triggers.append([slice_time.clone().cpu().item(), output_value.clone().cpu().item()])
		logging.info("A total of %i slices have exceeded the threshold of %f." % (len(triggers), trigger_threshold))
	return triggers

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


def main():
	parser = ArgumentParser(description="Basic example CNN training and corresponding search script supplied for the MLGWSC-1.")

	testing_group = parser.add_argument_group('testing')
	training_group = parser.add_argument_group('training')

	parser.add_argument('--verbose', action='store_true', help="Print update messages.")
	parser.add_argument('--debug', action='store_true', help="Show debug messages.")
	parser.add_argument('--train', action='store_true', help="Train the network before applying.")

	testing_group.add_argument('inputfile', type=str, help="The path to the input data file.")
	testing_group.add_argument('outputfile', type=str, help="The path where to store the triggers. The file must not exist.")
	testing_group.add_argument('-w', '--weights', type=str, help="The path to the file containing the network weights. If the --train option is present, the trained weights are used instead. Default: %s." % default_weights_fname)
	testing_group.add_argument('-t', '--trigger-threshold', type=float, default=0.2, help="The threshold to mark triggers. Default: 0.2")
	testing_group.add_argument('--step-size', type=float, default=0.1, help="The sliding window step size between analyzed samples. Default: 0.1")
	testing_group.add_argument('--cluster-threshold', type=float, default=0.35, help="The farthest in time that two slices can be to form a cluster. Default: 0.35")
	testing_group.add_argument('--device', type=str, default='cpu', help="Device to be used for analysis. Use 'cuda' for the GPU. Also, 'cpu:0', 'cuda:1', etc. (zero-indexed). Default: cpu")
	# testing_group.add_argument('--batch-size', type=int, default=512, help="Size of batches in which the network is evaluated. Default: 512")

	training_group.add_argument('-o', '--output-training', type=str, help="Path to the directory where the outputs will be stored. The directory must exist.")
	training_group.add_argument('--training-samples', type=int, nargs=2, default=[10000, 10000], help="Numbers of training samples as 'injections' 'pure noise samples'. Default: 10000 10000")
	training_group.add_argument('--validation-samples', type=int, nargs=2, default=[2000, 2000], help="Numbers of validation samples as 'injections' 'pure noise samples'. Default: 2000 2000")
	training_group.add_argument('--learning-rate', type=float, default=5e-5, help="Learning rate of the optimizer. Default: 0.00005")
	training_group.add_argument('--epochs', type=int, default=100, help="Number of training epochs. Default: 100")
	training_group.add_argument('--batch-size', type=int, default=32, help="Batch size of the training algorithm. Default: 32")
	training_group.add_argument('--clip-norm', type=float, default=100., help="Gradient clipping norm to stabilize the training. Default: 100.")
	training_group.add_argument('--train-device', type=str, default='cpu', help="Device to train the network. Use 'cuda' for the GPU. Also, 'cpu:0', 'cuda:1', etc. (zero-indexed). Default: cpu")
	training_group.add_argument('--store-device', type=str, default='cpu', help="Device to store the datasets. Use 'cuda' for the GPU. Also, 'cpu:0', 'cuda:1', etc. (zero-indexed). Default: cpu")

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
	if os.path.isfile(args.outputfile):
		raise RuntimeError("Output file exists.")
	else:
		pass

	### Initialize network
	logging.debug("Initializing network.")
	Network = get_network(path=args.weights, device=args.train_device)

	if args.train:
		TrainDS = generate_dataset(args.training_samples, args.verbose)
		ValidDS = generate_dataset(args.validation_samples, args.verbose)
		weights_path = os.path.join(args.output_training, default_weights_fname)
		Network = train(Network, TrainDS, ValidDS, args.output_training, weights_path,
                        store_device=args.store_device, train_device=args.train_device,
                        batch_size=args.batch_size, learning_rate=args.learning_rate,
                        epochs=args.epochs, clip_norm=args.clip_norm, verbose=args.verbose)
		
	triggers = get_triggers(Network,
	                        args.inputfile,
	                        step_size=args.step_size,
	                        trigger_threshold=args.trigger_threshold,
	                        device=args.device,
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
