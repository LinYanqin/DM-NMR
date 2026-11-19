import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import nmrglue as ng
import os


def create_mask_from_txt(file_path, y_axis):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    index = []
    for line in lines:
        values = list(map(int, line.split()))
        values = values[::-1]
        index.append(values)
    num = len(index)
    mask = np.zeros(y_axis)
    for i in range(num):
        mask[index[i][0]] = 1
    rate = num/y_axis
    print(f"The sampling rate is {rate * 100:.3f}%")
    return mask


def saveUnderSampledSpectrumToTXT(FID):
    F = np.fft.fft(FID)
    temp = np.real(F.reshape(-1))
    factor = np.max(temp)
    F = F / factor
    Size = F.shape
    N = Size[0]
    M = np.zeros((N, 2))
    M[:, 0] = np.real(F)
    M[:, 1] = np.imag(F)
    M = M.astype(np.float32)
    return factor, M


def saveSpectrumToTXT(FID, factor):
    F = np.fft.fft(FID)
    F = F / factor
    Size = F.shape
    N = Size[0]
    M = np.zeros((N, 2))
    M[:, 0] = np.real(F)
    M[:, 1] = np.imag(F)
    M = M.astype(np.float32)
    return M

def preprocess_BMRB(data_path, target_path, mask_path, M=300, N=256, noise_factor=0):

    dic, data = ng.pipe.read(target_path)
    label_f = np.fliplr(data)
    label_f = label_f / np.max(np.real(label_f.reshape(-1)))
    mask = create_mask_from_txt(mask_path, N)

    langda = noise_factor
    noise = np.sqrt(langda) * np.random.randn(M, N) + 1j * np.sqrt(langda) * np.random.randn(M, N)
    t_noise = np.fft.ifft2(label_f) + noise
    label_impure = np.fft.fft2(t_noise)

    label_impure /= np.max(np.real(label_impure.reshape(-1)))
    area = label_impure[0:40, 0:25]
    SNR = 1 / np.std(area.reshape(-1))

    t_label = np.fft.ifft(label_impure)

    input_data = np.zeros((M, N, 2))
    label_data = np.zeros((M, N, 2))
    factor = np.zeros((1, M))
    mask_data = np.zeros((M, 1, N))
    for i in range(M):
        FID1 = t_label[i, :]
        FID1_under = FID1*mask
        f, under = saveUnderSampledSpectrumToTXT(FID1_under)
        label = saveSpectrumToTXT(FID1, f)
        label_data[i, :, :] = label
        factor[0, i] = f
        input_data[i, :, :] = under
        mask_data[i, 0, :] = mask
    input_name = str(data_path / 'input_data.mat')
    factor_name = str(data_path / 'factor.mat')
    mask_name = str(data_path / 'mask.mat')
    label_name = str(data_path / 'label_data.mat')
    sio.savemat(input_name, {'input_data': input_data})
    sio.savemat(factor_name, {'factor': factor})
    sio.savemat(mask_name, {'mask_data': mask_data})
    sio.savemat(label_name, {'label_data': label_data})
    return SNR

