import numpy as np
import torch
import torch.nn as nn

# KL divergence function
def kl_divergence(mu, log_var):
    # KL Divergence between the learned Gaussian distribution and the standard Gaussian N(0,1)
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

### Training function
def train_epoch(encoder, decoder, device, dataloader, optimizer, beta):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device
        image_batch = image_batch.to(device)
        # Encode data
        mu, log_var = encoder(image_batch)
        # Decode data
        encoded_data = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        
        reconstructed = decoder(encoded_data)
        reconstruction_loss = nn.MSELoss()(reconstructed, image_batch)

        kl_loss = kl_divergence(mu, log_var)

        loss = reconstruction_loss + beta * kl_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

### Testing function
def test_epoch(encoder, decoder, device, dataloader, beta):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    val_loss = []
    
    with torch.no_grad():  # No need to track gradients in validation/testing
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            
            # Encode data (mu and log_var)
            mu, log_var = encoder(image_batch)
            # Reparameterization trick to sample latent vector z
            encoded_data = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)

            # Decode data
            decoded_data = decoder(encoded_data)
            
            # Calculate reconstruction loss
            reconstruction_loss = nn.MSELoss()(decoded_data, image_batch)

            # Calculate KL divergence loss
            kl_loss = kl_divergence(mu, log_var)
            
            # Total validation loss
            loss = reconstruction_loss + beta * kl_loss
            
            # Append loss for tracking
            val_loss.append(loss.detach().cpu().numpy())

    return np.mean(val_loss)