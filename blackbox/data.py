class Data:
    def __init__(self, data=None):
        # TODO
        # self.data should be the right data structure
        # for calculating log prob's in any of the model wrappers but
        # be generic enough to enable our subsampling routines to work
        self.data = data

    def sample(self):
        return None

class IID(Data):
    def __init__(self, *args, **kwargs):
        Data.__init__(self, *args, **kwargs)

    def sample(self):
        # TODO
        # subsample data uniformly if size of data > threshold
        return self.data
