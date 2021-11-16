import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt



def map_to_device(device, t):
    return tuple(map(lambda x: x.to(device), t))

#--------------------------------------------------------------------
def entropy_loss(logits):
    p_softmax = F.softmax(logits, dim=1)
    mask = p_softmax.ge(0.000001)  # greater or equal to #mask is for prevention of log(0) err.
    mask_out = torch.masked_select(p_softmax, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(p_softmax.size(0))

#--------------------------------------------------------------------
class OptimizerManager:
    def __init__(self, optims):
        self.optims = optims  

    def __enter__(self):
        for op in self.optims:
            op.zero_grad()

    def __exit__(self, exceptionType, exception, exceptionTraceback):
        for op in self.optims:
            op.step()
        self.optims = None
        if exceptionTraceback:
            print(exceptionTraceback)
            return False
        return True
    
#--------------------------------------------------------------------
class IteratorWrapper:
    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(loader)

    def __iter__(self):
        self.iterator = iter(self.loader)

    def get_next(self):
        try:
            items = self.iterator.next()
        except:
#             print("we are doomed!")
            self.__iter__()
            items = self.iterator.next()
        return items

#--------------------------------------------------------------------
def plotter(x, y, title=None, y_label="Accuracy"):
    fig, ax = plt.subplots(figsize=(8,6))

    ax.plot(x, y)

    x_ticks = x
    plt.xticks(x_ticks)

    ax.grid(True)

    plt.title(title)

    plt.xlabel("Epoch")

    if y_label=="Accuracy":
        plt.ylim(0, 1)
        y_ticks = np.arange(0, 1.1, 0.1)
        plt.yticks(y_ticks)
        plt.ylabel(y_label)

    if y_label=="Loss":
        y_ticks = np.arange(0, 2.5, 0.5)
        plt.yticks(y_ticks)
        plt.ylabel(y_label)

    plt.show()