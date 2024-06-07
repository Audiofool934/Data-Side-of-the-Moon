import torch
from torch import nn
import numpy as np
from torchsummary import summary


class Encoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super(Encoder, self).__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        
        self.flatten = nn.Flatten(start_dim=1)
        
        self.encoder_lin = nn.Sequential(
            # nn.Linear(128 * 16 * 41, encoded_space_dim),
            
            nn.Linear(128 * 16 * 41, 256),
            nn.ReLU(True),
            nn.Linear(256, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

def load_encoder(model_path, encoded_space_dim):
    model = Encoder(encoded_space_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")))
    model.eval()
    return model

def encode_data(encoder, np_array):
    
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    encoder = encoder.to(device)
    
    spectrogram = np_array[np.newaxis, :, :]
    spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        encoded_spectrogram = encoder(spectrogram_tensor.unsqueeze(0))
    
    return encoded_spectrogram.flatten().cpu().detach().numpy()

def encoder_summary(encoder,input_size=(1, 256, 646)):
    print("Encoder summary:")
    summary(encoder, input_size=input_size)

if __name__=="__main__":

    model_path="models/Echoes_128/encoder.pth"
    file_path = ""
    save_path = ""

    encoded_space_dim = 128
    encoder = load_encoder(model_path, encoded_space_dim)
    
    encoder_summary(encoder, input_size=(1, 256, 646))

    # np_array = np.load(file_path)
    # encoded_result = encode_data(encoder, np_array)
    
    # np.save(save_path, encoded_result)
    