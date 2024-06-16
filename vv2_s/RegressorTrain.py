import torch
import torchaudio
import torchsummary
from torch import nn
from torch.utils.data import DataLoader

from transAlignDataset import NUM_SAMPLES, SAMPLE_RATE, TARGET_DIR, AUDIO_DIR

from transAlignDataset import TransientDataset
from TransientRegressor import TransientRegressor

BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.01

# Создание DataLoader
def create_data_loader(train_data, batch_size):
    train_data_loader = DataLoader(train_data, batch_size)
    return train_data_loader

def train(model, data_loader, loss_fn, optim, device, epochs):
    best_loss = float("inf")
    for i in range(epochs):
        print(f"\nEpoch {i + 1}")
        epoch_loss = 0.0
        for input, target in data_loader:
            input, target = input.to(device), target.to(device)
            #input, target = input.unsqueeze(1), target.unsqueeze(1)

            optim.zero_grad()
            print("Input shape: ", input.shape)
            pred = model(input)

            loss = loss_fn(pred, target)
            loss.backward()
            optim.step()

            epoch_loss += loss.item()
        print(f"Epoch loss: {epoch_loss / len(data_loader)}")
        if i > 0 and epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), "best_regressor-50eps.pth")
        print("-----------------------")
    print("Finish training")

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=2048,
        hop_length=512,
        n_mels=64
    ).to(device)

    tds = TransientDataset(AUDIO_DIR,
                           TARGET_DIR,
                           mel_spectrogram,
                           SAMPLE_RATE,
                           NUM_SAMPLES,
                           device)

    train_data_loader = create_data_loader(tds, BATCH_SIZE)

    dummy_input = tds[0][0]
    num_samples = dummy_input.size(1)
    duration = num_samples / 44100

    print("dummy_duration = ", duration)

    print("dummy_input.shape = ", dummy_input.shape)

    dummy_input = dummy_input.to(device)#.unsqueeze_(1)

    regressor = TransientRegressor().to(device)
    print("dummy_input.shape = ", dummy_input.shape)
    torchsummary.summary(regressor, dummy_input.shape)

    loss_fn = nn.MSELoss()
    optimiser = torch.optim.Adam(regressor.parameters(),
                                lr=LEARNING_RATE)

    train(regressor, train_data_loader, loss_fn, optimiser, device, EPOCHS)

    torch.save(regressor.state_dict(), "transientRegressor-s-50epochs.pth")
    print("Trained model saved at transientRegressor-s-50epochs.pth")
    print("_________________________________________________\n\n")