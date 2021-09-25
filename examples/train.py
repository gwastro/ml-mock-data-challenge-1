### Import modules
from argparse import ArgumentParser
from math import pi, sqrt
from scipy.signal import tukey
import pycbc.waveform, pycbc.noise, pycbc.psd, pycbc.distributions, pycbc.detector
import os, os.path
import torch
import logging

### Basic dataset class for easy PyTorch loading
class Dataset(torch.utils.data.Dataset):
	def __init__(self, samples, labels, device=torch.device('cpu')):
		torch.utils.data.Dataset.__init__(self)
		self.samples = samples
		self.labels = labels
		self.device = device
		assert len(self.samples)==len(self.labels)
		return
	def __len__(self):
		return len(self.samples)
	def __getitem__(self, i):
		return self.samples[i].to(device=self.device), self.labels[i].to(device=self.device)

### Function to compute product of elements of an iterable argument
def product(iterable):
	prod = 1
	for num in iterable:
		prod *= num
	return prod

if __name__=='__main__':
	### Create the argument parser and define the arguments
	parser = ArgumentParser(description="Basic example training script supplied for the MLGWSC-1. Tested with Python 3.9.1.")
	parser.add_argument('-o', '--output', type=str, required=True, help="Path to the directory where the outputs will be stored. The directory must exist and be empty.")
	parser.add_argument('--train-device', type=str, default='cpu', help="Device to train the network. Use 'cuda' for the GPU. Also, 'cpu:0', 'cuda:1' (zero-indexed). Default: cpu")
	parser.add_argument('--store-device', type=str, default='cpu', help="Device to store the datasets. Use 'cuda' for the GPU. Also, 'cpu:0', 'cuda:1' (zero-indexed). Default: cpu")
	parser.add_argument('--training-samples', type=int, nargs=2, default=[10000, 10000], help="Numbers of training samples as 'injections' 'pure noise samples'. Default: 10000 10000")
	parser.add_argument('--validation-samples', type=int, nargs=2, default=[2000, 2000], help="Numbers of validation samples as 'injections' 'pure noise samples'. Default: 2000 2000")
	parser.add_argument('--learning-rate', type=float, default=0.00005, help="Learning rate of the optimizer. Default: 0.00005")
	parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs. Default: 100")
	parser.add_argument('--batch-size', type=int, default=32, help="Batch size of the training algorithm. Default: 32")
	parser.add_argument('--clip-norm', type=float, default=100., help="Gradient clipping norm to stabilize the training. Default: 100.")
	parser.add_argument('--verbose', '-v', action='count', default=0, help="Desired verbosity level.")
	parser.add_argument('--debug', action='store_true', help="Show debug messages.")

	### Parse arguments
	args = parser.parse_args()

	### Set up logging
	if args.debug:
		log_level = logging.DEBUG
	else:
		log_level = logging.INFO if args.verbose>0 else logging.WARN
	logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s', level=log_level, datefmt='%d-%m-%Y %H:%M:%S')

	### Create output directory; if it exists, check that it's empty
	if os.path.isdir(args.output):
		if len(os.listdir(args.output))==0:
			pass
		else:
			raise RuntimeError("Output directory is not empty.")
	else:
		raise RuntimeError("Output directory does not exist.")

	### Choose the data type used
	dtype = torch.float32

	### Detectors
	detectors_abbr = ('H1', 'L1')
	logging.debug("Initializing detectors.")
	detectors = [pycbc.detector.Detector(det_abbr) for det_abbr in detectors_abbr]

	### Create the power spectral density and get the inverse ASD with the proper cutoffs, as well as the properly stacked Tukey window
	logging.debug("Creating Tukey window, PSDs and corresponding whitening frequency filter.")
	psds = [pycbc.psd.analytical.aLIGOZeroDetHighPower(1025, 1., 18.) for _ in range(len(detectors))]
	window = torch.from_numpy(tukey(2048, alpha=1./12.))
	window = torch.stack([window for _ in range(len(psds))], dim=0).to(device=args.store_device, dtype=dtype)
	freq_filter = torch.stack([torch.tensor([0. if elem==0. else elem**(-.5) for elem in psd], dtype=dtype, device=args.store_device) for psd in psds], dim=0)

	### Save the Tukey window as well as the frequency filter for use in the evaluation
	whitening_filter_path = os.path.join(args.output, 'whitening_filter.pt')
	logging.debug("Saving window and frequency filter to %s." % whitening_filter_path)
	torch.save({'window': window, 'filter': freq_filter}, whitening_filter_path)

	### Initialize the sky location random distribution
	skylocation_dist = pycbc.distributions.sky_location.UniformSky()

	### Create labels
	label_wave = torch.tensor([1., 0.]).to(device=args.store_device, dtype=dtype)
	label_noise = torch.tensor([0., 1.]).to(device=args.store_device, dtype=dtype)

	### Generate data
	if args.verbose>1:
		print_step = 1
	else:
		print_step = 1000
	datasets = []
	for num_waveforms, num_noises in (args.training_samples, args.validation_samples):
		logging.info("Generating dataset with %i injections and %i pure noise samples" % (num_waveforms, num_noises))
		samples = []
		labels = []
		for i in range(num_waveforms+num_noises):
			is_waveform = i<num_waveforms
			# Generate noise
			noise = torch.stack([torch.from_numpy(pycbc.noise.gaussian.frequency_noise_from_psd(psd).to_timeseries().numpy()) for psd in psds], dim=0)
			# If in the first part of the dataset, generate waveform
			if is_waveform:
				# Generate source parameters
				waveform_kwargs = {'approximant': 'IMRPhenomD', 'delta_t': 1./2048., 'f_lower': 18.}
				masses = 40.*torch.rand(2) + 10.
				waveform_kwargs['mass1'], waveform_kwargs['mass2'] = max(masses).item(), min(masses).item()
				angles = 2*pi*torch.rand(3)
				waveform_kwargs['coa_phase'] = angles[0].item()
				waveform_kwargs['inclination'] = angles[1].item()
				declination, right_ascension = skylocation_dist.rvs()[0]
				pol_angle = angles[2].item()
				injection_time = 1238166018 + (1253977218 - 1238166018)*torch.rand(1).item()	# Take the injection time randomly in the LIGO O3a era
				# Generate the full waveform
				h_plus, h_cross = pycbc.waveform.get_td_waveform(**waveform_kwargs)
				# Properly time and project the waveform
				h_plus.start_time = h_cross.start_time = injection_time + h_plus.get_sample_times()[0]
				h_plus.append_zeros(2048)
				h_cross.append_zeros(2048)
				strains = [det.project_wave(h_plus, h_cross, right_ascension, declination, pol_angle) for det in detectors]
				# Place merger randomly within the window between 0.5 s and 0.7 s of the time series and form the PyTorch sample
				time_placement = 0.5 + 0.2*torch.rand(1).item()
				time_interval = injection_time-time_placement
				time_interval = (time_interval, time_interval+0.999)	# 0.999 to not get a too long strain
				strains = [strain.time_slice(*time_interval) for strain in strains]
				for strain in strains:
					to_append = 2048 - len(strain)
					if to_append>0:
						strain.append_zeros(to_append)
				# Compute network SNR, rescale to generated target network SNR and inject into noise
				network_snr = sqrt(sum([pycbc.filter.matchedfilter.sigmasq(strain, psd=psd, low_frequency_cutoff=18.) for strain, psd in zip(strains, psds)]))
				target_snr = 5. + torch.rand(1).item()*10.
				sample = noise + torch.stack([torch.from_numpy(strain.numpy()) for strain in strains], dim=0)*target_snr/network_snr
			# If in the second part of the dataset, merely use pure noise as the full sample
			else:
				sample = noise
			# Convert sample to desired data type and storage device
			sample = sample.to(device=args.store_device, dtype=dtype)
			# Whiten 
			sample = torch.fft.irfft(freq_filter*torch.fft.rfft(sample*window))
			# Append to list of samples, as well as the corresponding label
			samples.append(sample)
			if is_waveform:
				labels.append(label_wave)
			else:
				labels.append(label_noise)
			if (i+1)%print_step==0:
				logging.info("%i samples complete" % (i+1))
		# Merge samples and labels into just two tensors (more memory efficient) and initialize dataset
		samples = torch.stack(samples, dim=0)
		labels = torch.stack(labels, dim=0)
		datasets.append(Dataset(samples, labels, device=args.train_device))

	### Set up data loaders as a PyTorch convenience
	logging.debug("Setting up data loaders.")
	TrainDL = torch.utils.data.DataLoader(datasets[0], batch_size=args.batch_size, shuffle=True)
	ValidDL = torch.utils.data.DataLoader(datasets[1], batch_size=100, shuffle=True)

	# Simple CNN
	logging.info("Initializing network")
	Network = torch.nn.Sequential(		#  2x2048
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
	Network.to(device=args.train_device, dtype=dtype)

	### Print network overview and test that the network processes tensor sizes as desired
	logging.debug("Following network initialized:")
	logging.debug(str(Network))
	logging.info("Network has %i free parameters." % sum([product(par.shape) for par in Network.parameters()]))

	test_input = torch.randn(32, 2, 2048).to(device=args.train_device, dtype=dtype)
	logging.debug("Testing on simple random batch, input shape %s returns %s output shape." % (str(tuple(test_input.shape)), str(tuple(Network(test_input).shape))))

	### Initialize loss function, optimizer and output file
	logging.debug("Initializing loss function, optimizer and output file.")
	loss = torch.nn.BCELoss()
	opt = torch.optim.Adam(Network.parameters(), lr=args.learning_rate)
	outfile = open(os.path.join(args.output, 'losses.txt'), 'w', buffering=1)

	### Training loop
	logging.info("Starting optimization loop:")
	logging.info("epoch   training    validation")
	for epoch in range(1, args.epochs+1):
		# Training epoch
		Network.train()
		training_running_loss = 0.
		training_batches = 0
		for training_samples, training_labels in TrainDL:
			# Optimizer step on a single batch of training data
			opt.zero_grad()
			training_output = Network(training_samples)
			training_loss = loss(training_output, training_labels)
			training_loss.backward()
			# Clip gradients to make convergence somewhat easier
			torch.nn.utils.clip_grad_norm_(Network.parameters(), max_norm=args.clip_norm)
			# Make the actual optimizer step and save the batch loss
			opt.step()
			training_running_loss += training_loss.clone().cpu().item()
			training_batches += 1
		# Evaluation on the validation dataset
		Network.eval()
		with torch.no_grad():
			validation_running_loss = 0.
			validation_batches = 0
			for validation_samples, validation_labels in ValidDL:
				# Evaluation of a single validation batch
				validation_output = Network(validation_samples)
				validation_loss = loss(validation_output, validation_labels)
				validation_running_loss += validation_loss.clone().cpu().item()
				validation_batches += 1
		# Print information on the training and validation loss in the current epoch and save current network state
		output_string = '%04i    %f    %f' % (epoch, training_running_loss/training_batches, validation_running_loss/validation_batches)
		logging.info(output_string)
		outfile.write(output_string + '\n')
		torch.save(Network.state_dict(), os.path.join(args.output, 'state_dict_%04i.pt' % epoch))

	logging.debug("Training complete, closing file.")
	outfile.close()
