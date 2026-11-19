import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import nmrglue as ng
import os


def create_mask_from_txt(file_path,y_axis=64, x_axis=64):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    index = []
    for line in lines:
        values = list(map(int, line.split()))
        values = values[::-1]
        index.append(values)
    num = len(index)
    mask = np.zeros((y_axis, x_axis))
    for i in range(num):
        mask[index[i][1], index[i][0]] = 1
    rate = num/(y_axis*x_axis)
    print(f"The sampling rate is {rate * 100:.3f}%")
    return mask


def saveUnderSampledSpectrumToTXT(FID):
    F = np.fft.fftshift(np.fft.fft2(FID))
    temp = np.real(F.reshape(-1))
    factor = np.max(temp)
    F = F / factor
    Size = F.shape
    N1, N2 = Size[0], Size[1]
    M = np.zeros((N1, N2, 2))
    M[:, :, 0] = np.real(F)
    M[:, :, 1] = np.imag(F)
    M = M.astype(np.float32)
    return factor, M


def preprocess_BMRB(data_file_path, target_path, mask_path, y_axis=64, x_axis=64, z_axis=280, noise_factor=0):
    # data = sio.loadmat(target_path)
    # # data = data['FID_zf']
    # data = data['data']
    # data = np.transpose(data,[1,0,2])
    # fid = data
    data_path = './' + str(data_file_path) + '/'
    dic, fid = ng.pipe.read(target_path)
    fid = np.transpose(fid,[1,0,2])
    mask = create_mask_from_txt(mask_path, y_axis, x_axis)
    # mask = np.reshape(mask, [y_axis, x_axis])
    # mask = create_mask_from_txt(mask_path,y_axis, x_axis)
    R1R2 = fid[::2, ::2, :]
    R1I2 = fid[::2, 1::2, :]
    I1R2 = fid[1::2, ::2, :]
    I1I2 = fid[1::2, 1::2, :]

    langda = noise_factor
    noise = np.sqrt(langda) * np.random.randn(y_axis, x_axis, z_axis) + 1j * np.sqrt(langda) * np.random.randn(y_axis, x_axis, z_axis)
    FID_real1 = R1R2 + 1j * R1I2 + noise
    FID_real2 = I1R2 + 1j * I1I2 + noise
    #
    # FID_real1 = R1R2 + 1j * R1I2
    # FID_real2 = I1R2 + 1j * I1I2
    # a = FID_real1[:,0,0]
    # plt.plot(np.real(a))
    # plt.show()


    temp = np.fft.fft2(FID_real1, s=(y_axis, x_axis), axes=(0, 1))
    temp /= np.max(np.real(temp.reshape(-1)))
    area = temp[1:5, 1:5, :5]
    SNR = 1 / np.std(area.reshape(-1))

    input = np.zeros((z_axis, y_axis, x_axis, 4))
    factor = np.zeros((2, z_axis))
    for i in range(z_axis):
        FID1 = FID_real1[:, :, i]
        FID1_under = FID1*mask
        FID2 = FID_real2[:, :, i]
        FID2_under = FID2*mask
        f, under = saveUnderSampledSpectrumToTXT(FID1_under)
        factor[0, i] = f
        input[i, :, :, 0:2] = under
        f, under = saveUnderSampledSpectrumToTXT(FID2_under)
        factor[1, i] = f
        input[i, :, :, 2:4] = under
    input_name = data_path + 'input.mat'
    factor_name = data_path + 'factor.mat'
    mask_name = data_path + 'mask.mat'
    label_name = data_path + 'label.mat'
    fid = np.transpose(fid,[1,0,2])
    sio.savemat(input_name, {'input': input})
    sio.savemat(factor_name, {'factor': factor})
    sio.savemat(mask_name, {'mask': mask})
    sio.savemat(label_name, {'fid': fid})
    return SNR


