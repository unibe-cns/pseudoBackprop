import numpy as np
from torch.utils.data.dataset import Dataset
        
class ParityDataset(Dataset):
    def __init__(self, inputs=2, samples=100, seed=42):
        """
            Returns a dataset and classes sampled from the parity function
            (-1)**sum(inputs)
            (for inputs = 2, this is XOR)
            inputs: number of inputs
        """
        super(ParityDataset, self).__init__()
        # using a numpy RNG to allow compatibility to other deep learning frameworks
        self.rng = np.random.RandomState(seed)
        self.inputs = inputs
        self.samples = samples
        self.__vals = []
        self.__cs = []
        self.class_names = ['zero', 'one']
        for i in range(samples):
            goal_class = self.rng.randint(0,2)
            x_array, c = self.get_sample(goal=goal_class)
            self.__vals.append(x_array)
            self.__cs.append(c)

    def get_sample(self, goal=None):
        # sample until goal is satisfied
        found_sample_yet = False
        while not found_sample_yet:
            # sample x
            x_array = self.rng.randint(0,2,self.inputs)
            # check if the array has the correct parity as required by the goal
            c = self.which_class(x_array)
            if goal is None or c == goal:
                found_sample_yet = True
                break
        return x_array, c

    def which_class(self, x_array):
        # calculate the parity
        return np.sum(x_array) % 2

    def __getitem__(self, index):
        sample = (self.__vals[index].copy(), self.__cs[index])
        return sample

    def __len__(self):
        return len(self.__cs)
