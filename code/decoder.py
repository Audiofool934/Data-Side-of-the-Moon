import torch
from torch import nn
import numpy as np
from torchsummary import summary

class Decoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super(Decoder, self).__init__()
        
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 512),
            nn.LeakyReLU(0.05, inplace = True),
            nn.Linear(512, 256 * 16 * 41),
            nn.LeakyReLU(0.05, inplace = True),
            nn.Dropout(0.3)
        )

        self.unflatten = nn.Unflatten(dim = 1, unflattened_size = (256, 16, 41))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=(1, 0)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.05, inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.05, inplace=True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=(1, 0)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.05, inplace=True),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=(1, 1))
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x

def load_decoder(model_path, encoded_space_dim):

    model = Decoder(encoded_space_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")))
    model.eval()  # Set the model to evaluation mode
    return model

def decode_data(decoder, vector):
        
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = torch.device("cpu")
     
    if isinstance(vector,str):
        new_np_array = np.load(vector)
        new_tensor = torch.from_numpy(new_np_array).float()
    else:
        new_tensor=torch.from_numpy(vector).float()

    if len(new_tensor.shape) == 1:
        new_tensor = new_tensor.unsqueeze(0)

    decoder = decoder.to(device)
    new_tensor = new_tensor.to(device)

    decoded_result = decoder(new_tensor)

    return  decoded_result.squeeze().cpu().detach().numpy()

def encoder_summary(decoder, input_size = 128):
    print("\nDecoder summary:")
    summary(decoder, (input_size,))

if __name__=="__main__":
    
    model_path = "models/Echoes/decoder.pth"
    file_path = 'Whole Lotta Love.npy'
    save_path = "Whole Lotta Love_Recon.npy"
    
    encoded_space_dim = 128
    
    # decoder = load_decoder(model_path, encoded_space_dim)
    decoder = Decoder(encoded_space_dim = encoded_space_dim)
    encoder_summary(decoder = decoder, input_size = encoded_space_dim)
    
    # decoded_result=decode_data(decoder, file_path)
    # np.save(save_path, decoded_result)