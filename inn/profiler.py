import time

# this will eventually handle TensorFlow profiling as well.
class Profiler(object):
    def __enter__(self):
        self.start_time = time.time()
        self.end_time = None
        self.elapsed = None

        return self

    def __exit__(self, type, value, tb):
        if not (self.end_time is None):
            raise ValueError("profiler was already ended")
        
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
