import numpy as np
import torch


def kl_divergence(mu, log_var):
    # KL Divergence between the learned Gaussian distribution and the standard Gaussian N(0,1)
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())


def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer, beta):
    encoder.train()
    decoder.train()
    train_loss = []

    for image_batch, _ in dataloader:

        image_batch = image_batch.to(device)
        
        mu, log_var = encoder(image_batch)
        kl_loss = kl_divergence(mu, log_var)
        
        encoded_data = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        recon = decoder(encoded_data)
        recon_loss = loss_fn(recon, image_batch)
        # print("encoded_data: ", encoded_data[0][:10])
        
        loss = recon_loss + beta * kl_loss
        
        # print(f"recon_loss: {recon_loss} \n kl_loss: {kl_loss} \n")
        # print(f"mu: {mu[0][:10]} \n log_var: {log_var[0][:10]}")
        
        # bp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


def test_epoch(encoder, decoder, device, dataloader, loss_fn, beta):
    encoder.eval()
    decoder.eval()
    val_loss = []
    
    with torch.no_grad():
        
        for image_batch, _ in dataloader:
            image_batch = image_batch.to(device)
            
            mu, log_var = encoder(image_batch)
            kl_loss = kl_divergence(mu, log_var)
            
            encoded_data = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
            recon = decoder(encoded_data)
            recon_loss = loss_fn(recon, image_batch)    
            
            loss = recon_loss + beta * kl_loss
            val_loss.append(loss.cpu().numpy())
            
        return np.mean(val_loss)