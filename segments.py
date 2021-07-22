import numpy as np
from pycbc.types import TimeSeries
from pycbc.inject import InjectionSet

# Notes:
#  -Right now the initial time series have to overlap. Why is that
#   required?
#  -Right now the initial time series overlap must be >= the duration.
#   What if both of them are long enough and we could slide them in such
#   a way that they would overlap?

class OverlapSegment(object):
    def __init__(self, *timeseries, duration=None):
        self.duration = duration
        self.timeseries = None
        self.detectors = None
        for ts in timeseries:
            self.add_timeseries(ts)
        pass
    
    @property
    def start_time(self):
        if self.timeseries is None:
            return None
        return max([ts.start_time for ts in self.timeseries])
    
    @property
    def end_time(self):
        if self.timeseries is None:
            return None
        return max([ts.end_time for ts in self.timeseries])
    
    @property
    def delta_t(self):
        if self.timeseries is None:
            return None
        return self.timeseries[0].delta_t
    
    @property
    def sample_rate(self):
        if self.timeseries is None:
            return None
        return self.timeseries[0].sample_rate
    
    def add_timeseries(self, timeseries):
        det, ts = timeseries
        if self.timeseries is None:
            if self.duration is not None:
                if ts.duration < self.duration:
                    raise ValueError(f'Cannot add time series.')
            self.timeseries = [ts]
            self.detectors = [det]
            return
        
        if ts.delta_t != self.delta_t:
            raise ValueError(f'Delta t not matching.')
        
        new_start = float(max(self.start_time, ts.start_time))
        new_end = float(min(self.end_time, ts.end_time))
        new_max_dur = new_end - new_start
        
        if self.duration is None:
            if new_max_dur < 0:
                raise ValueError(f'Cannot add time series.')
            self.timeseries.append(ts)
            self.detectors.append(det)
            return
        
        if new_max_dur < self.duration:
            raise ValueError(f'Cannot add time series.')
        self.timeseries.append(ts)
        self.detectors.append(det)
        return
    
    def get(self, seed=None, shift=True, random_start_time=False):
        return self.get2(seed=seed, shift=shift,
                         random_start_time=random_start_time)
     
    def get1(self, seed=None, shift=True, random_start_time=False):
    # This function only shifts the first time series and the last n - 1
    # time series. The last n - 1 time series will be shifted all by the same amount
        if self.timeseries is None:
            return
        
        rs = np.random.RandomState(seed=seed)
        
        if random_start_time and self.duration is not None:
            mintime = float(self.start_time)
            maxtime = float(self.end_time) - float(self.duration)
            if maxtime > mintime:
                start_time = rs.uniform(mintime, maxtime)
            else:
                start_time = mintime
        else:
            start_time = float(self.start_time)
        
        indices = []
        for ts in self.timeseries:
            start = int((float(self.start_time) - float(ts.start_time))
                        / ts.delta_t)
            if self.duration is None:
                end = int((float(self.end_time) - float(ts.start_time))
                          / ts.delta_t)
            else:
                end = start + int(self.duration / ts.delta_t)
            if len(indices) > 0:
                assert (end - start) == (indices[-1][1] - indices[-1][0])
            indices.append((start, end))
        
        ret = []
        if not shift:
            for i, ts in enumerate(self.timeseries):
                ret.append(TimeSeries(ts.data[indices[i][0]:indices[i][1]],
                                          epoch=start_time,
                                          delta_t=ts.delta_t))
            return ret
        
        #Handle upper
        rsmin = -indices[0][0]
        rsmax = len(self.timeseries[0]) - indices[0][1]
        if rsmin >= rsmax:
            shiftidx = 0
        else:
            shiftidx = rs.randint(rsmin, rsmax)
        sidx = indices[0][0] + shiftidx
        eidx = indices[0][1] + shiftidx
        dat = self.timeseries[0].data[sidx:eidx]
        ret.append(TimeSeries(dat, epoch=start_time,
                              delta_t=self.delta_t))
        
        #Only one segment
        if len(self.timeseries) < 2:
            return ret
        
        #Handle lower
        tss = self.timeseries[1:]
        rsmin = -min([pt[0] for pt in indices[1:]])
        rsmax = min([len(ts) - ind[1] for (ts, ind) in zip(tss, indices[1:])])
        if rsmin >= rsmax:
            shiftidx = 0
        else:
            shiftidx = rs.randint(rsmin, rsmax)
        for ind, ts in zip(indices[1:], tss):
            sidx = ind[0] + shiftidx
            eidx = ind[1] + shiftidx
            dat = ts.data[sidx:eidx]
            ret.append(TimeSeries(dat, epoch=start_time,
                                  delta_t=self.delta_t))
        return ret
    
    def get2(self, seed=None, shift=True, random_start_time=False):
        if self.timeseries is None:
            return
        
        rs = np.random.RandomState(seed=seed)
        
        if random_start_time and self.duration is not None:
            mintime = float(self.start_time)
            maxtime = float(self.end_time) - float(self.duration)
            if maxtime > mintime:
                start_time = rs.uniform(mintime, maxtime)
            else:
                start_time = mintime
        else:
            start_time = float(self.start_time)
        
        indices = []
        for ts in self.timeseries:
            start = int((float(self.start_time) - float(ts.start_time))
                        / ts.delta_t)
            if self.duration is None:
                end = int((float(self.end_time) - float(ts.start_time))
                          / ts.delta_t)
            else:
                end = start + int(self.duration / ts.delta_t)
            if len(indices) > 0:
                assert (end - start) == (indices[-1][1] - indices[-1][0])
            indices.append((start, end))
        
        ret = []
        if not shift:
            for i, ts in enumerate(self.timeseries):
                ret.append(TimeSeries(ts.data[indices[i][0]:indices[i][1]],
                                      epoch=start_time,
                                      delta_t=ts.delta_t))
            return ret
        
        #Handle upper
        for ind, ts in zip(indices, self.timeseries):
            rsmin = -ind[0]
            rsmax = len(ts) - ind[1]
            if rsmin >= rsmax:
                shiftidx = 0
            else:
                shiftidx = rs.randint(rsmin, rsmax)
            sidx = ind[0] + shiftidx
            eidx = ind[1] + shiftidx
            dat = ts.data[sidx:eidx]
            ret.append(TimeSeries(dat, epoch=start_time,
                                  delta_t=self.delta_t))
        
        return ret
    
    @classmethod
    def from_hdf_datasets(cls, *datasets, **kwargs):
        """Constructor from an opened HDF5-file dataset.
        
        Arguments
        ---------
        datasets : One or multiple dataset objects of an open HDF5 file
            The datasets to read from. They need to contain the
            attributes `delta_t` and `start_time`.
        kwargs :
            All keyword arguments are passed to the usual constructor.
        """
        timeseries = []
        for ds in datasets:
            timseries.append(TimeSeries(ds[()],
                                        epoch=ds.attrs['start_time'],
                                        delta_t=ds.attrs['delta_t']))
        return cls(timeseries, **kwargs)
    
    @classmethod
    def from_hdf_group(cls, group, **kwargs):
        """Constructor from an opened HDF5-file group.
        
        Arguments
        ---------
        group : group object of an open HDF5 file
            The group to read from. It is expected to contain only
            datasets each of which is expected to have the two
            attributes `delta_t` and `start_time`. The keys of the group
            are sorted by the inbuilt `sorted` function before they are
            loaded.
        kwargs :
            All keyword arguments are passed to the usual constructor.
        """
        timeseries = []
        for key in sorted(group.keys()):
            ds = group[key]
            timseries.append(TimeSeries(ds[()],
                                        epoch=ds.attrs['start_time'],
                                        delta_t=ds.attrs['delta_t']))
        return cls(timeseries, **kwargs)

class SegmentList(object):
    def __init__(self, *segments):
        self.segments = None
        for segment in segments:
            self.add_segment(segment)
    
    def add_segment(self, segment):
        if self.segments is None:
            self.segments = [segment]
            return
        
        idx = np.searchsorted([float(seg.start_time) for seg in self.segments],
                              float(segment.start_time), side='right')
        if idx == len(self.segments):
            self.segments.append(segment)
            return
        
        #TODO: Consider random start times from Segment.get
        if float(segment.start_time) + segment.duration > float(self.segments[idx].start_time):
            raise ValueError(f'Segment is too long to be added without overlap.')
        
        self.segments.insert(idx, segment)
    
    def apply_injections(self, injection_file, shift_injections=True,
                         seed=None, shift_data=True,
                         random_start_time=False):
        """Apply injections from an injection file to a set of segments.
        
        Arguments
        ---------
        injection_file : str
            Path to a file containing injections.
        shift_injections : {bool, True}
            If this option is set the injections are believed to start
            tc == 0. The times are then shifted in such a way that they
            line up with the disjoint segments. This means the remaining
            injections are shifted such that they line up with the start
            of the next segment. For an example see the notes below.
        seed : {None or int, None}
            The seed to use for time-shifting the strain data. See the
            documentation of OverlapSegment.get for more information.
        shift_data : {bool, True}
            Whether or not to shift the strain data of the segments.
        random_start_time : {bool, False}
            Randomize the start time of the segments within the
            applicable limits.
        
        Returns
        -------
        list:
            Returns a list of dictionaries. The key of the dictionaries
            are the detectors available from the corresponding segment.
            The values are the TimeSeries containing injections.
        
        Notes
        -----
        shift_injections:
            Consider two segments with with start and end times
            [1, 6], [8, 15] and two injections [1, 10]. Since the
            injections are believed to have a start-time of 0, the first
            injection with t = 1 will be injected into the first segment
            [1, 6] at a GPS-time of t = 2 = 1 + start_time = 1 + 1. The
            second injection would, therefore, have a GPS injection time
            of t = 11, which is outside the segment.
            The first segment covered 5 seconds in duration. Hence only
            injections with t > 5 from the injection set will be
            considered for the remaining segments.
            To align the injection times for the second segment the
            previously considered duration will be subtracted and the
            start time of the segment will be added to the injection
            times to obtain the correct GPS times. The injection GPS
            times for the second segment would thus be [4, 13]. The
            injection GPS time with t = 4 lies outside the the
            boundaries of the second segment and would thus be ignored.
            The second injection GPS time on the other hand lies within
            the second segment and is applied.
        """
        if self.segments is None:
            return None
        injector = InjectionSet(injection_file)
        injtable = injector.table
        passed_dur = 0
        ret = []
        for segment in self.segments:
            timeseries = segment.get(seed=seed, shift=shift_data,
                                     random_start_time=random_start_time)
            detectors = segment.detectors
            tmp = {}
            for det, ts in zip(detectors, timeseries):
                #TODO: Add padding here
                if shift_injections:
                    addition = float(ts.start_time) - passed_dur
                    injtable['tc'] += addition
                injector.apply(ts, det)
                if shift_injections:
                    injtable['tc'] -= addition
                tmp[det] = ts
            passed_dur += segment.duration
            ret.append(tmp)
        
        return ret
    
    def get(self, seed=None, shift=True, random_start_time=False):
        if self.segments is None:
            return None
        ret = []
        for segment in self.segments:
            timeseries = segment.get(seed=seed, shift=shift_data,
                                     random_start_time=random_start_time)
            detectors = segment.detectors
            ret.append({det: ts for (det, ts) in zip(detectors, timeseries)})
        return ret
