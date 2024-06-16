import torch
import torchaudio

from transAlignDataset import TransientDataset
from transAlignDataset import AUDIO_DIR, TARGET_DIR, SAMPLE_RATE, NUM_SAMPLES
from TransientRegressor import TransientRegressor

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    regr = TransientRegressor().to(device)
    example_inp = torch.rand(1, 1, NUM_SAMPLES).to(device)

    regr.load_state_dict(torch.load("transientRegressor-s-50epochs.pth"))
    regr.eval()

    traced_model = torch.jit.trace(regr, example_inp)
    traced_model.save("traced_Regressor-s-50ep.pth")