class LazyRAM():
    """
    Lazily loads items into a map
    Assumes that items in the list don't change
    For machines that have too much RAM to play with this might speed up training

    Examples
    """

    def __init__(self, iterable=None):
        self._cache = {}
        self.iterable = iterable
    
    def __getitem__(self, k):
        v = self.check_cache(k)
        if v is None:
            v = self._cache[k] = self.iterable.__getitem__(k)
        return v 

    def __setitem__(self, k, v):
        self.iterable.__setitem__(k,v)
            
    def check_cache(self, k):
        v = None
        if k in self._cache:
            v = self._cache[k]
            print("From cache")
        return v

    def __call__(self, cls):
        old = cls.__getitem__
        cache = self._cache

        def __getitem__(_self, k):
            v = self.check_cache(k)
            if v is None: 
                v = cache[k] = old(_self, k)
            return v 

        cls.__getitem__ = __getitem__
        return cls

if __name__ == '__main__':
    import os

    @LazyRAM()
    class Test(list):

        def __getitem__(self, k):
            return super().__getitem__(k)

    photo_directory = '/home/harry/data/vggface2/test_cropped'
    people = os.listdir(photo_directory)
    
    # Test with decorator
    test = Test([1,2,3,4])
    test[0]
    test[0]
    
    # Test with init list
    lram = LazyRAM([1,2,3,4])
    lram[0]
    lram[0]
