### Import modules
from argparse import ArgumentParser
import torch
import h5py
import os.path
import numpy as np
import logging

### Set data type to be used
dtype = torch.float32

### Implement a dataset slicer
class Slicer(torch.utils.data.IterableDataset):
	def __init__(self, infile, step_size=0.1, peak_offset=0.6, slice_length=2048, detectors=None):
		torch.utils.data.IterableDataset.__init__(self)
		self.infile = infile
		self.step_size = step_size		# this is the approximate one passed as an argument, the exact one is defined in the __next__ method
		self.peak_offset = peak_offset
		self.slice_length = slice_length
		self.detectors = detectors
		if self.detectors is None:
			self.detectors = [self.infile[key] for key in list(self.infile.attrs['detectors'])]
		return
	def __iter__(self):
		self.ds_key_iter = iter(self.detectors[0].keys())
		self.current_ds_key = None
		return self
	def start_next_ds(self):
		self.current_ds_key = next(self.ds_key_iter)
		self.current_dss = [det[self.current_ds_key] for det in self.detectors]
		self.current_index = 0
		self.current_time = self.current_dss[0].attrs['start_time']
		self.delta_t = self.current_dss[0].attrs['delta_t']
		for ds in self.current_dss:
			assert (ds.attrs['start_time']==self.current_time) and (ds.attrs['delta_t']==self.delta_t)
		self.index_step_size = int(self.step_size/self.delta_t)		# this is the integer step size
		self.time_step_size = self.delta_t*self.index_step_size		# this is the exact step size used in the algorithm
		return
	def get_next_slice(self):
		if self.current_index+self.slice_length>len(self.current_dss[0]):
			raise StopIteration
		else:
			this_slice = torch.stack([torch.from_numpy(ds[self.current_index:self.current_index+self.slice_length]) for ds in self.current_dss], dim=0)
			this_time = torch.tensor(self.current_time + self.peak_offset)
			self.current_index += self.index_step_size
			self.current_time += self.time_step_size
			return this_slice, this_time
	def __next__(self):
		if self.current_ds_key is None:
			self.start_next_ds()
		try:
			return self.get_next_slice()
		except StopIteration:
			logging.debug("Iterator over dataset starting at integer time %i raised StopIteration." % int(self.current_ds_key))
			self.start_next_ds()
			return next(self)

if __name__=='__main__':
	### Create the argument parser and define the arguments
	parser = ArgumentParser(description="Evaluation script for the example training script supplied for the MLGWSC-1. Tested with Python 3.9.1.")
	parser.add_argument('-i', '--input-file', type=str, required=True, help="The path to the input data file.")
	parser.add_argument('-s', '--state-dict', type=str, required=True, help="The path to the state dictionary containing the network weights.")
	parser.add_argument('-w', '--whitening', type=str, required=True, help="The path to the file containing the Tukey window and the whitening frequency filter.")
	parser.add_argument('-o', '--output-file', type=str, required=True, help="The path where to store the triggers. The file must not exist.")
	parser.add_argument('-t', '--trigger-threshold', type=float, default=0.2, help="The threshold to mark triggers. Default: 0.2")
	parser.add_argument('--step-size', type=float, default=0.1, help="The sliding window step size between analyzed samples. Default: 0.1")
	parser.add_argument('--cluster-threshold', type=float, default=0.35, help="The farthest in time that two slices can be to form a cluster. Default: 0.35")
	parser.add_argument('--device', type=str, default='cpu', help="Device to be used. Use 'cuda' for the GPU. Also, 'cpu:0', 'cuda:1' (zero-indexed). Default: cpu")
	parser.add_argument('--batch-size', type=int, default=512, help="Size of batches in which the network is evaluated. Default: 512")
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

	### Check existence of output file
	if os.path.isfile(args.output_file):
		raise RuntimeError("Output file exists.")
	else:
		pass

	### Load the Tukey window and whitening frequency filter, also unsqueeze to add batch dimension
	logging.debug("Loading whitening filters from %s." % args.whitening)
	filters = torch.load(args.whitening)
	window = torch.unsqueeze(filters['window'].to(device=args.device, dtype=dtype), 0)
	freq_filter = torch.unsqueeze(filters['filter'].to(device=args.device, dtype=dtype), 0)

	### Simple CNN
	logging.info("Initializing network.")
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
	logging.debug("Loading weights from %s." % args.state_dict)
	Network.load_state_dict(torch.load(args.state_dict))
	Network.to(device=args.device, dtype=dtype)
	Network.eval()

	### Open both the input and output file
	logging.debug("Opening input and output files.")
	infile = h5py.File(args.input_file, 'r')
	outfile = h5py.File(args.output_file, 'w')

	triggers = []

	### Initialize the data loading process
	logging.debug("Initializing data loading process.")
	slicer = Slicer(infile)
	data_loader = torch.utils.data.DataLoader(slicer, batch_size=args.batch_size, shuffle=False)

	### Gradually apply network to all samples and if output exceeds the trigger threshold, save the time and the output value
	logging.info("Starting iteration over dataset with batch size %i." % args.batch_size)
	if args.verbose>1:
		print_step = 1
	else:
		print_step = 100000
	to_print = print_step
	processed = 0
	for slice_batch, slice_times in data_loader:
		whitened_slices = torch.fft.irfft(freq_filter*torch.fft.rfft(slice_batch.to(device=args.device, dtype=dtype)*window))
		with torch.no_grad():
			output_values = Network(whitened_slices)[:, 0]
			trigger_bools = torch.gt(output_values, args.trigger_threshold)
			for slice_time, trigger_bool, output_value in zip(slice_times, trigger_bools, output_values):
				if trigger_bool.clone().cpu().item():
					triggers.append([slice_time.clone().cpu().item(), output_value.clone().cpu().item()])
		processed += len(slice_times)
		if processed>=to_print:
			logging.info("Processed %i slices." % processed)
			to_print += print_step
	logging.info("Network evaluation finished with %i slices." % processed)
	logging.info("A total of %i slices have exceeded the threshold of %f." % (len(triggers), args.trigger_threshold))

	### Close input file
	infile.close()

	### Sort triggers by time
	logging.debug("Sorting triggers by time to be sure about correct clustering.")
	triggers.sort(key=lambda inp: inp[0])

	### Cluster the triggers into candidate detections
	clusters = []
	for trigger in triggers:
		if len(clusters)==0:
			start_new_cluster = True
		else:
			start_new_cluster = ((trigger[0] - clusters[-1][-1][0])>args.cluster_threshold)
		if start_new_cluster:
			clusters.append([trigger])
		else:
			clusters[-1].append(trigger)

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

	### Save clustered values to the output file and close it
	logging.debug("Saving clustered triggers into %s." % args.output_file)
	cluster_times = np.array(cluster_times)
	cluster_values = np.array(cluster_values)
	cluster_timevars = np.array(cluster_timevars)

	outfile.create_dataset('time', data=cluster_times)
	outfile.create_dataset('stat', data=cluster_values)
	outfile.create_dataset('var', data=cluster_timevars)

	logging.debug("Triggers saved, closing file.")
	outfile.close()
