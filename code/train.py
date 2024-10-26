import numpy as np
import torch

def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    encoder.train()
    decoder.train()
    train_loss = []

    for image_batch, _ in dataloader:

        image_batch = image_batch.to(device)
        
        encoded_data = encoder(image_batch)
        decoded_data = decoder(encoded_data)

        loss = loss_fn(decoded_data, image_batch)
        # print(loss, decoded_data.shape, image_batch.shape)
        
        # bp
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print('\t train loss: %f' % (loss.data))
        train_loss.append(loss.cpu().numpy())

    return np.mean(train_loss)


def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        orig_images = []
        recon_images = []
        
        for image_batch, _ in dataloader:
            image_batch = image_batch.to(device)
            encoded_data = encoder(image_batch)
            decoded_data = decoder(encoded_data)

            recon_images.append(decoded_data.cpu())
            orig_images.append(image_batch.cpu())

        recon_images = torch.cat(recon_images)
        orig_images = torch.cat(orig_images) 

        val_loss = loss_fn(recon_images, orig_images)
    return val_loss.data