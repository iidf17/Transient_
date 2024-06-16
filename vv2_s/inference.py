import time

import torch
import torchaudio
from torchsummary import summary
import torch.nn.functional as F

from transAlignDataset import TransientDataset
from transAlignDataset import AUDIO_DIR, TARGET_DIR, SAMPLE_RATE, NUM_SAMPLES
from TransientRegressor import TransientRegressor


def load_audio_file(file_path, sr=44100):
    y, sr = torchaudio.load(file_path)
    y = y.unsqueeze(0)
    return y, sr


def get_duration(sound, sr=44100) -> int:
    num_samples = sound.shape[-1]
    duration = num_samples / sr
    return int(duration)


def preprocess_inp_data(inp_data):
    dim = inp_data.dim()
    dur = get_duration(inp_data)
    if dim > 3:
        print(f"Inp_data dims overload = {dim}, {inp_data.shape}")
        while dim != 3:
            inp_data.squeeze_(1)
            print(f"Inp_data squeezed: {inp_data.shape}")
            dim = inp_data.dim()
        #inp_data = inp_data.expand(dur, -1, -1, -1)
    elif dim < 3:
        print(f"Inp_data dims not enouf = {dim}, {inp_data.shape}")
        while dim != 4:
            inp_data.unsqueeze_(1)
            print(f"Inp_data squeezed: {inp_data.shape}")
            dim = inp_data.dim()
        #inp_data = inp_data.expand(dur, -1, -1, -1)
    else:
        pass
        #inp_data = inp_data.expand(dur, -1, -1, -1)
    print(f"Returning: {inp_data.shape} and {dur} sec")
    return inp_data, dur


def shift_transients(waveform, shift_value):
    num_samples = waveform.size(1)
    grid = torch.linspace(-1, 1, num_samples, device=device).unsqueeze(0)
    grid = grid - shift_value

    grid = grid.unsqueeze(2)

    zeros = torch.zeros_like(grid)

    grid = torch.stack((grid, zeros), dim=-1)

    waveform = waveform.unsqueeze(0).unsqueeze(1)

    print("Waveform shape = ", waveform.shape)
    print("Grid shape = ", grid.shape)

    shifted_waveform = F.grid_sample(waveform, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

    shifted_waveform = shifted_waveform.squeeze()

    return shifted_waveform


def predict(regressor, input_data, max_shift=50):
    regressor.eval()
    with torch.no_grad():
        shift_value = regressor(input_data)
        shift_value = shift_value.item()

        print("time shift = ", shift_value)
        print('input_data before: ', input_data)

        shifted_waveform = shift_transients(input_data.squeeze(0), shift_value)
        print('input_data returning: ', shifted_waveform)

        return shifted_waveform


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    regressorNN = TransientRegressor().to(device)

    regressor_state_dict = torch.load("transientRegressor-s-50epochs.pth")
    regressorNN.load_state_dict(regressor_state_dict)

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

    input_data, sr = load_audio_file("TransientDataset\\Input\\full.wav")
    input_data = input_data.to(device)
    # summary(regressorNN, input_data.shape)

    input_data, dur = preprocess_inp_data(input_data)

    inp_data = input_data

    aligned_data = predict(regressorNN, input_data)

    print("Aligned shape: ", aligned_data.shape)

    t = time.time()

    aligned_data = aligned_data.unsqueeze(0).detach().cpu()
    print("Detached ", aligned_data.shape)

    t_el = time.time() - t
    print(f"Spended time = {t_el} sec")

    saved_file_name = 'aligned_Full21.wav'
    torchaudio.save(saved_file_name, aligned_data, 44100)
    print("Aligned file saved in ", saved_file_name)
