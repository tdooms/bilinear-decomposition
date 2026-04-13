from torch.utils.data import Dataset
from torchvision import datasets

class MNIST(Dataset):
    """Wrapper class that loads MNIST onto the GPU for speed reasons."""
    def __init__(self, train=True, download=True, device="cuda"):
        dataset = datasets.MNIST(root='./data', train=train, download=download)
        self.x = dataset.data.float().to(device).unsqueeze(1) / 255.0
        self.y = dataset.targets.to(device)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.x.size(0)

class FMNIST(Dataset):
    """Wrapper class that loads F-MNIST onto the GPU for speed reasons."""
    def __init__(self, train=True, download=True, device="cuda"):
        dataset = datasets.FashionMNIST(root='./data', train=train, download=download)
        self.x = dataset.data.float().to(device).unsqueeze(1) / 255.0
        self.y = dataset.targets.to(device)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.x.size(0)