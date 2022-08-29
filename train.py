import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn.functional as F
from network import BatchDiscriminator, Generator
import random
from real_sample import GenRealSamples2


device = 'cuda'
batch = 2
feature_dim = 128
latent_dim = 16

dataset = GenRealSamples2(100, feature_dim).to(device)
def batch_from_samples(samples, batch_size):
    index = random.sample(range(samples.shape[0]), batch_size)
    return samples[index]


generator = Generator(latent_dim, feature_dim).to(device)
discriminator = BatchDiscriminator(feature_dim, latent_dim).to(device)
G_optim = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0001)
D_optim = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=0.0001)

real_positive_sum = 0
fake_negtive_sum = 0
fake_rate = 0.1
for step in range(1000000):
    generator.train(), discriminator.train()
    
    gauss_sample = torch.randn(batch, latent_dim).to(device)
    fake_sample = generator(gauss_sample)
    real_sample_ori = batch_from_samples(dataset, batch)
    # blend real and fake as real
    gauss_sample_4real = torch.randn(batch, latent_dim).to(device)
    fake_sample_4real = generator(gauss_sample_4real)
    real_index = (torch.rand(batch, 1).to(device)-fake_rate).ceil()
    real_sample = (real_sample_ori*real_index + fake_sample_4real*(1-real_index)).detach()

    real_logit = discriminator(real_sample)
    fake_logit = discriminator(fake_sample.detach())

    # update D 
    # d_loss = F.mse_loss(real_logit - fake_logit, torch.ones_like(real_logit).to(device))
    # d_loss = 1.0 * (F.softplus(-real_logit).mean() + F.softplus(fake_logit).mean())
    d_loss = F.binary_cross_entropy_with_logits(real_logit - fake_logit, torch.ones_like(real_logit).to(device))
    D_optim.zero_grad()
    d_loss.backward()
    D_optim.step()

    #update G
    real_logit = discriminator(real_sample)
    fake_logit = discriminator(fake_sample)
    # g_loss = F.mse_loss(fake_logit - real_logit.detach(), torch.ones_like(real_logit).to(device))
    # g_loss = 1.0 * F.softplus(-fake_logit).mean()
    g_loss = F.binary_cross_entropy_with_logits(fake_logit - real_logit.detach(), 1.0*torch.ones_like(real_logit).to(device))
    G_optim.zero_grad()
    g_loss.backward()
    G_optim.step()
    
    
    real_logit_ori = discriminator(real_sample_ori)
    real_positive_sum = real_positive_sum + (real_logit_ori-fake_logit).sign().sum().detach()
    fake_negtive_sum = fake_negtive_sum + (fake_logit-real_logit).sign().sum().detach()
    if step % 500 == 0:
        rrrr = 0.5*real_positive_sum/(500*batch)
        ffff = 0.5*fake_negtive_sum/(500*batch)
        real_positive_sum = 0
        fake_negtive_sum = 0
        if (rrrr-ffff)/2.0 > 0.6:
            fake_rate = fake_rate + 0.01
            # fake_rate = fake_rate + (1.-fake_rate)*0.1
        else:
            fake_rate = fake_rate - 0.01
        #     fake_rate = fake_rate + (1.-fake_rate)*0.1*(rrrr-ffff)/2.0
        print("step: %d, d_loss: %.4f, g_loss:%.4f" % (step, d_loss, g_loss))
        print("real_logit:", real_logit)
        print("fake_logit:", fake_logit)
        print("rrrr:", rrrr)
        print("ffff:", ffff)
        print("fake_rate:", fake_rate)


    if step % 500 == 0:
        generator.eval(), discriminator.eval()
        sample_num = 10
        plt.figure("figure")
        plt.subplot(2,1,1), plt.title('real')
        # plt.gca().set_aspect(1)
        kkkk = batch_from_samples(dataset, sample_num).cpu().numpy()
        for i in range(sample_num):
            plt.scatter(np.arange(0, feature_dim)/10.,kkkk[i],alpha=0.2)

        plt.subplot(2,1,2), plt.title('fake')
        # plt.gca().set_aspect(1)
        gauss = torch.randn(sample_num, latent_dim).to(device)
        kkkk = generator(gauss).detach().cpu().numpy()
        for i in range(sample_num):
            plt.scatter(np.arange(0, feature_dim)/10.,kkkk[i],alpha=0.2)

        # save img
        os.makedirs('result', exist_ok=True)
        plt.savefig("./result/step_%07d.png" % step)
        plt.clf()