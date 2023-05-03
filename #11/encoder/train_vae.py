import model

"""
Training the network for a given number of epochs
The loss after every epoch is printed
"""
for epoch in range(model.num_epochs):
    for idx, data in enumerate(model.train_loader, 0):
        imgs, _ = data
        imgs = imgs.to(model.device)

        # Feeding a batch of images into the network to obtain the output image, mu, and logVar
        out, mu, logVar = model.net(imgs)

        # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
        kl_divergence = -0.5 * model.torch.sum(1 + logVar - mu.pow(2) - logVar.exp())
        loss = F.binary_cross_entropy(out, imgs, size_average=False) + kl_divergence

        # Backpropagation based on the loss
        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()

    print('Epoch {}: Loss {}'.format(epoch, loss))

