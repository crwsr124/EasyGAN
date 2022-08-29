import torch
import matplotlib.pyplot as plt
import numpy as np

def GenRealSamples1(nums, feature_dim):
    real_samples = torch.randn(nums, feature_dim) + 4.*torch.range(1,nums).view(nums, 1)
    return real_samples


def GenRealSamples2(nums, feature_dim):
    one_class_num = nums//7
    real_samples1 = torch.randn(one_class_num, feature_dim)
    real_samples2 = torch.randn(one_class_num, feature_dim) + 20.0
    real_samples3 = torch.randn(one_class_num, feature_dim) + torch.range(1,feature_dim).view(1, feature_dim)/feature_dim
    real_samples4 = torch.randn(one_class_num, feature_dim) + 8.*torch.range(1,feature_dim).view(1, feature_dim)/feature_dim
    real_samples5 = torch.randn(one_class_num, feature_dim) + 16.*torch.range(1,feature_dim).view(1, feature_dim)/feature_dim
    real_samples6 = torch.flip(real_samples5, [1])
    real_samples7 = torch.randn(one_class_num, feature_dim) + 10.

    return torch.concat((real_samples1, real_samples2, real_samples3, real_samples4, real_samples5, real_samples6, real_samples7))

def test():
    samples = GenRealSamples2(7, 128)
    plt.figure("figure")
    for i in range(samples.shape[0]):
        plt.scatter(np.arange(0, 128)/10.,samples[i],alpha=0.2)
    plt.show()

# test()