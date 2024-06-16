import torch
import torch.nn as nn
from torchsummary import summary

import matplotlib.pyplot as plt

class TransientRegressor(nn.Module):

    def __init__(self):
        super().__init__()

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.lin1 = nn.Linear(64, 32)
        self.lin2 = nn.Linear(32, 1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.gmp(x)
        x = x.squeeze(-1)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        return x

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    trnn = TransientRegressor().to(device)

    input_data = torch.randn(4, 1, 44100).to(device)
    output = trnn(input_data)
    print("Output shape: ", output.shape)

    loss_fn = nn.MSELoss()
    optimiser = torch.optim.Adam(trnn.parameters(),
                                 lr=0.01)
    summary(trnn.cuda(), (1, 44100 * 16))

    time_shift = output[0].item()

    max_shift = 50

    time_shift = min(max(time_shift, -max_shift), max_shift) * -1

    print("Before cat: ", output)
    # output = torch.cat([output[time_shift:], output[:time_shift]], dim=0)
    print("Aftret cat: ", output)
    output = output.unsqueeze(0)

    print(output)

    output = input_data.detach().cpu()

    plt.plot(output[0], 'g')
    plt.show()