import matplotlib.pyplot as plt
import numpy as np
import math
import os
import torch
import torch.nn.functional as F
from network import BatchDiscriminator, Generator, Discriminator
import random
from real_sample import GenRealSamples2
from loss import FocalLoss


device = 'cuda'
batch = 2
feature_dim = 128
latent_dim = 16

dataset = GenRealSamples2(100, feature_dim).to(device)
def batch_from_samples(samples, batch_size):
    index = random.sample(range(samples.shape[0]), batch_size)
    return samples[index]


generator = Generator(latent_dim, feature_dim).to(device)
# discriminator = BatchDiscriminator(feature_dim, latent_dim).to(device)
discriminator = Discriminator(feature_dim, latent_dim).to(device)

G_optim = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
D_optim = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

real_positive_sum = 0
fake_negtive_sum = 0
fake_rate = 0.1
feature_rate = 0.01

real_slide_window = None
fake_slide_window = None
last_aver = 0
last_aver2 = 0


for step in range(1000000):
    generator.train(), discriminator.train()
    
    gauss_sample = torch.randn(batch, latent_dim).to(device)
    fake_sample = generator(gauss_sample)
    real_sample_ori = batch_from_samples(dataset, batch)
    # blend real and fake as real
    gauss_sample_4real = torch.randn(batch, latent_dim).to(device)
    fake_sample_4real = generator(gauss_sample_4real)
    real_index = (torch.rand(batch, 1).to(device)-fake_rate).ceil()
    real_sample = (real_sample_ori*real_index + fake_sample_4real*(1-real_index)).detach_()
    real_sample.requires_grad = True

    real_logit, rlatent = discriminator(real_sample)
    fake_logit, flatent = discriminator(fake_sample.detach())


    hhh = torch.exp(-0.5*(gauss_sample**2).mean(1, keepdim=True))
    # print("kkkkkkkkkkkkkkkkkkk", hhh)
    kld = feature_rate * ((flatent*flatent).mean() + ((flatent-rlatent)**2).mean())

    # update D 
    # d_loss = F.mse_loss(real_logit - fake_logit, torch.ones_like(real_logit).to(device))
    # d_loss = 1.0 * (F.softplus(-real_logit).mean() + F.softplus(fake_logit).mean())
    d_loss = F.binary_cross_entropy_with_logits(real_logit - fake_logit, torch.ones_like(real_logit).to(device)) + kld
    # d_loss = FocalLoss()(real_logit - fake_logit, torch.ones_like(real_logit).to(device))
    # d_loss = F.binary_cross_entropy_with_logits(real_logit - fake_logit, 0.5+0.5*(torch.ones_like(real_logit).to(device)-hhh))

    # if step > 0 and step % 2 == 0:
    #     grad_real, = torch.autograd.grad(outputs=real_logit.sum(), inputs=real_sample, create_graph=True)
    #     grad_real = grad_real.view(batch,-1)
    #     r1_loss = (grad_real*grad_real).sum(1).mean()
    #     D_r1_loss = 5*(5. * r1_loss * 4)
    #     d_loss = d_loss + D_r1_loss

    D_optim.zero_grad()
    d_loss.backward()
    D_optim.step()

    #update G
    real_logit, _ = discriminator(real_sample)
    fake_logit, _ = discriminator(fake_sample)
    # g_loss = F.mse_loss(fake_logit - real_logit.detach(), torch.ones_like(real_logit).to(device))
    # g_loss = 1.0 * F.softplus(-fake_logit).mean()
    g_loss = F.binary_cross_entropy_with_logits(fake_logit - real_logit.detach(), torch.ones_like(real_logit).to(device))
    # g_loss = FocalLoss()(fake_logit - real_logit, torch.ones_like(real_logit).to(device))
    # g_loss = F.binary_cross_entropy_with_logits(fake_logit - real_logit.detach(), 0.5 + 0.5*hhh)
    G_optim.zero_grad()
    g_loss.backward()
    G_optim.step()
    
    
    real_logit_ori, _ = discriminator(real_sample_ori)
    real_positive_sum = real_positive_sum + (real_logit_ori-fake_logit).sign().sum().detach()
    fake_negtive_sum = fake_negtive_sum + (fake_logit-real_logit).sign().sum().detach()

    real_sign = (real_logit_ori-fake_logit).sign().view(-1).detach()
    fake_sign = (fake_logit-real_logit).sign().view(-1).detach()
    if (real_slide_window != None):
        if (real_slide_window.shape[0] < 1000):
            real_slide_window = torch.cat([real_slide_window, real_sign])
            fake_slide_window = torch.cat([fake_slide_window, fake_sign])
        else:
            real_slide_window = torch.cat([real_slide_window[real_sign.shape[0]:], real_sign])
            fake_slide_window = torch.cat([fake_slide_window[fake_sign.shape[0]:], fake_sign])
    else:
        real_slide_window = real_sign
        fake_slide_window = fake_sign
    
    if real_slide_window.shape[0] == 1000:
        rrrr = real_slide_window.sum()/1000.
        ffff = fake_slide_window.sum()/1000.
        # if step%200==0 and (rrrr-ffff)/2.0 > 0.6:
        #     fake_rate = fake_rate + 0.01
        # if step%200==0 and (rrrr-ffff)/2.0 < 0.6:
        #     fake_rate = fake_rate - 0.01
        # if fake_rate < 0.01:
        #     fake_rate = 0.01

        if  (rrrr+ffff) > 0.07 and (rrrr+ffff) >= last_aver2:
            feature_rate = feature_rate+0.01
        if  (rrrr+ffff) < 0.07:
            feature_rate = feature_rate-0.01
        if feature_rate<0.01:
            feature_rate = 0.01

        if step % 1 == 0:
            print("rrrr:", rrrr)
            print("ffff:", ffff)
            print("last_aver:", last_aver)
            print("aver:", (rrrr-ffff)/2.0)
            print("fake_rate:", fake_rate)
            print("feature_rate:", feature_rate)
            print("----------------------------------------------")
        if step % 5 == 0:
            last_aver = (rrrr-ffff)/2.0
            last_aver2 = rrrr+ffff
    

    # if step % 500 == 0:
    #     # rrrr = 0.5*real_positive_sum/(500*batch)
    #     # ffff = 0.5*fake_negtive_sum/(500*batch)
    #     rrrr = real_positive_sum/(500*batch)
    #     ffff = fake_negtive_sum/(500*batch)
    #     real_positive_sum = 0
    #     fake_negtive_sum = 0
    #     if (rrrr-ffff)/2.0 > 0.6:
    #         fake_rate = fake_rate + 0.01
    #         # fake_rate = fake_rate + (1.-fake_rate)*0.1
    #     else:
    #         fake_rate = fake_rate - 0.01
    #     #     fake_rate = fake_rate + (1.-fake_rate)*0.1*(rrrr-ffff)/2.0
    #     print("step: %d, d_loss: %.4f, g_loss:%.4f" % (step, d_loss, g_loss))
    #     print("real_logit:", real_logit)
    #     print("fake_logit:", fake_logit)
    #     print("rrrr:", rrrr)
    #     print("ffff:", ffff)
    #     print("fake_rate:", fake_rate)


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