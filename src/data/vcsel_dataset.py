import torch
from torch.utils.data import Dataset


class VCSELDataset(Dataset):
    def __init__(self, data):
        self.inputs = [torch.tensor(row, dtype=torch.float) for row in data['input_data']]
        self.outputs = [torch.tensor(row, dtype=torch.float) for row in data['output_data']]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input_tokens": self.inputs[idx],
            "output_tokens": self.outputs[idx]
        }
