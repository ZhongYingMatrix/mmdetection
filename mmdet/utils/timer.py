import time
from collections import defaultdict
 
class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
 
    def __enter__(self):
        self.start = time.time()
        return self
 
    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print('elapsed time: %f ms' % self.msecs)

class TimeStamp(object):
    def __init__(self):
        self.last_time = time.time()
        self.works_dict = defaultdict(float)

    def __call__(self, work):
        new_time = time.time()
        msecs = (new_time - self.last_time)*1000
        print(work,'takes %s ms' % msecs)
        self.last_time = new_time

    def accumulate(self, work):
        new_time = time.time()
        msecs = (new_time - self.last_time)*1000
        self.works_dict[work] += msecs
        self.last_time = new_time

    def over(self):
        for work in self.works_dict:
            print(work,'takes %s ms' % self.works_dict[work])
        self.works_dict = defaultdict(float)
        self.last_time = time.time()