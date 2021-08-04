import numpy as np
import json
import h5py
import time
from pycbc.types import TimeSeries
from pycbc.inject import InjectionSet

# Notes:
#  -Right now the initial time series have to overlap. Why is that
#   required?
#  -Right now the initial time series overlap must be >= the duration.
#   What if both of them are long enough and we could slide them in such
#   a way that they would overlap?

class OverlapSegment(object):
    """A class to handle time series of different detectors which are
    overlapping in time.
    
    It provides functionality to shift those time series with respect to
    one another and thereby create unique combinations.
    
    Arguments
    ---------
    timeseries : tuple of (str, TimeSeries)
        One or multiple tuples where the first entry specifies the
        detector and the second argument is the actual time series.
    duration : {None or float, None}
        The required duration of the overlapping region.
    """
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
                         seed=None, shift_data=True, padding_start=0,
                         padding_end=0, random_start_time=False,
                         f_lower=None, return_times=False,
                         return_indices=False):
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
        padding_start : {float, 0}
            The amount of time at the beginning of each segment to not
            put injections into.
        padding_end : {float, 0}
            The amount of time at the end of each segment to not put
            injections into.
        random_start_time : {bool, False}
            Randomize the start time of the segments within the
            applicable limits.
        f_lower : {float or None, None}
            The lower frequency cutoff of injections.
        return_times : {bool, False}
            Return the shifted injection times.
        return_indices : {bool, False}
            Return injection-file indices of injected signals.
        
        Returns
        -------
        list:
            Returns a list of dictionaries. The key of the dictionaries
            are the detectors available from the corresponding segment.
            The values are the TimeSeries containing injections.
        np.array, optional:
            The injection times as shifted by the function.
        np.array, optional:
            The indices in the injection file of the injected signals.
        
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
        indices = []
        injtimes = []
        for segment in self.segments:
            timeseries = segment.get(seed=seed, shift=shift_data,
                                     random_start_time=random_start_time)
            detectors = segment.detectors
            
            #All time series should have same start time, end time, and delta_t
            ts = timeseries[0]
            
            #Shift injections to appropriate start time
            if shift_injections:
                addition = float(ts.start_time) - passed_dur + padding_start
                injtable['tc'] += addition
            
            idxs = np.where(np.logical_and(float(ts.start_time) + padding_start <= injtable['tc'],
                                           injtable['tc'] <= float(ts.end_time) - padding_end))[0]

            indices.append(idxs)
            injtimes.append(injtable['tc'][idxs])
            
            tmp = {}
            for det, ts in zip(detectors, timeseries):
                tscopy = ts.copy()
                
                #Apply injections
                injector.apply(tscopy, det, f_lower=f_lower,
                               simulation_ids=list(idxs))
                
                tmp[det] = tscopy
            
            #Shift injections back to original
            if shift_injections:
                injtable['tc'] -= addition
            
            if segment.duration is None:
                passed_dur += float(segment.end_time) - float(segment.start_time)
            else:
                passed_dur += segment.duration
            ret.append(tmp)
        
        indices = np.concatenate(indices)
        injtimes = np.concatenate(injtimes)
        
        if return_times:
            if return_indices:
                return ret, injtimes, indices
            else:
                return ret, injtimes
        else:
            if return_indices:
                return ret, indices
            else:
                return ret
    
    def get(self, seed=None, shift=True, random_start_time=False):
        if self.segments is None:
            return None
        ret = []
        for segment in self.segments:
            timeseries = segment.get(seed=seed, shift=shift,
                                     random_start_time=random_start_time)
            detectors = segment.detectors
            ret.append({det: ts for (det, ts) in zip(detectors, timeseries)})
        return ret
    
    def get_full(self, **kwargs):
        gotten = self.get(**kwargs)
        dets = self.detectors
        ret = {det: [] for det in dets}
        for dic in gotten:
            seglen = max([len(ts) for ts in dic.values()])
            dt = list(dic.values())[0].delta_t
            epoch = float(list(dic.values())[0].start_time)
            for det in dets:
                if det in dic:
                    ret[det].append(dic[det])
                else:
                    ret[det].append(TimeSeries(np.zeros(seglen),
                                               delta_t=dt,
                                               epoch=epoch))
        return ret
    
    def get_full_seglist(self, **kwargs):
        gotten = self.get(**kwargs)
        dets = self.detectors
        ret = SegmentList()
        for dic in gotten:
            seglen = len(list(dic.values())[0])
            dt = list(dic.values())[0].delta_t
            epoch = float(list(dic.values())[0].start_time)
            seg = OverlapSegment()
            for det in dets:
                if det in dic:
                    ts = dic[det]
                else:
                    ts = TimeSeries(np.zeros(seglen),
                                    delta_t=dt,
                                    epoch=epoch)
                seg.add_timeseries((det, ts))
            ret.add_segment(seg)
        return ret
    
    def min_duration(self, duration):
        segs = []
        for seg in self.segments:
            if seg.duration is None:
                dur = float(seg.end_time) - float(seg.start_time)
            else:
                dur = seg.duration
            if dur < duration:
                continue
            segs.append(seg)
        return SegmentList(*segs)
    
    @property
    def detectors(self):
        detectors = set([])
        for seg in self.segments:
            detectors = detectors.union(set(seg.detectors))
        return list(detectors)
    
    @property
    def start_time(self):
        if self.segments is None:
            return
        return min([seg.start_time for seg in self.segments])
    
    @property
    def end_time(self):
        if self.segments is None:
            return
        return max([seg.end_time for seg in self.segments])
    
    @property
    def duration(self):
        ret = 0
        for seg in self.segments:
            if seg.duration is None:
                ret += seg.end_time - seg.start_time
            else:
                ret += seg.duration
        return ret

class DqSegment(object):
    def __init__(self, start, end, flag):
        assert start <= end
        self.start = start
        self.end = end
        self.flag = flag
    
    @property
    def duration(self):
        return self.end - self.start
    
    @classmethod
    def from_dict(cls, dic, flag, start_key='start', end_key='end'):
        return cls(dic[start_key], dic['end_key'], flag)
    
    def overlap(self, other):
        if not isinstance(other, DqSegment):
            raise TypeError
        if self.start > other.end or self.end < other.start:
            return 0
        else:
            nstart = max(self.start, other.start)
            nend = min(self.end, other.end)
            return nend - nstart
    
    def __contains__(self, item):
        return self.start <= item <= self.end
    
    def __and__(self, other):
        if not isinstance(other, DqSegment):
            raise TypeError
        if self.flag == other.flag:
            nflag = self.flag
        else:
            nflag = self.flag + '&' + other.flag
        nstart = max(self.start, other.start)
        nend = min(self.end, other.end)
        if nstart > nend:
            return
        return DqSegment(nstart, nend, nflag)
    
    def __or__(self, other):
        if not isinstance(other, DqSegment):
            raise TypeError
        if self.start > other.end or self.end < other.start:
            raise ValueError
        if self.flag == other.flag:
            nflag = self.flag
        else:
            nflag = self.flag + '|' + other.flag
        nstart = min(self.start, other.start)
        nend = max(self.end, other.end)
        if nstart > nend:
            return
        return DqSegment(nstart, nend, nflag)
    
class DqSegmentList(object):
    def __init__(self, dq_segments, flag=None):
        self.dq_segments = dq_segments
        self.flag = flag
    
    def min_duration(self, duration):
        tmp = []
        for seg in self.dq_segments:
            if seg.duration > duration:
                tmp.append(seg)
        return DqSegmentList(tmp, flag=self.flag)
    
    @classmethod
    def from_json(cls, fpath, flag, start_key='start', end_key='end'):
        with open(fpath, 'r') as fp:
            data = json.load(fp)
        keys = sorted(list(data.keys()))
        dq_segments = []
        for key in keys:
            dq_segments.append(DqSegment.from_dict(data[key], flag,
                                                   start_key=start_key,
                                                   end_key=end_key))
        return cls(dq_segments, flag)
    
    @classmethod
    def from_dict(cls, dic):
        flag = dic['id'][3:]
        dq_segments = []
        for start, end in dic['segments']:
            dq_segments.append(DqSegment(start, end, flag))
        return cls(dq_segments, flag=flag)
    
    def __len__(self):
        return len(self.dq_segments)
    
    @property
    def duration(self):
        return sum([seg.duration for seg in self.dq_segments])
    
    def __iter__(self):
        return iter(self.dq_segments)
    
    def __and__(self, other):
        if isinstance(other, DqSegment):
            ret = []
            for seg in self.dq_segments:
                tmp = seg and other
                if tmp is not None:
                    ret.append(tmp)
            flag = None
            if self.flag is not None:
                if self.flag == other.flag:
                    flag = self.flag
            return DqSegmentList(ret, flag=flag)
        elif isinstance(other, DqSegmentList):
            ret = []
            for seg in other.dq_segments:
                tmp = self and seg
                ret.extend(tmp.dq_segments)
            if self.flag is not None and other.flag is not None:
                if self.flag == other.flag:
                    flag = self.flag
            return DqSegmentList(tmp, flag=flag)
        else:
            raise TypeError
        
        
