from main import Model
from dataset import LowLightDataset
from torch.utils.data import DataLoader

dataset = LowLightDataset()

data_loader = DataLoader(dataset, batch_size=3)

model = Model()

input = next(iter(data_loader))

model(input)