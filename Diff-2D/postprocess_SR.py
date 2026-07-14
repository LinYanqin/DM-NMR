import nmrglue as ng
#import mat4py as mt
import pylab
import numpy as np
import scipy.io as sio
import os
import subprocess


def postprocess_data(test_path, y_axis=64, x_axis=64, z_axis=732):
    name_fac = test_path + 'factor.mat'
    fac = sio.loadmat(name_fac)['factor']
    name_res = test_path + 'res1.mat'
    res_data = sio.loadmat(name_res)['res']
    res_data = np.transpose(res_data, (0, 2, 3, 1))
    rec_data = np.zeros([z_axis, y_axis, x_axis, 2], dtype=np.complex)
    rec_data[:,:,:, 0] = res_data[:,:,:, 0] + 1j * res_data[:,:,:, 1]
    rec_data[:,:,:, 1] = res_data[:,:,:, 2] + 1j * res_data[:,:,:, 3]

    for k in range(z_axis):
        rec_data[k, :, :, 0] = rec_data[k, :, :, 0] * fac[0, k]
        rec_data[k, :, :, 1] = rec_data[k, :, :, 1] * fac[1, k]
    rec_data = np.transpose(rec_data, (1, 2, 0, 3))
    res_3D = np.zeros((2 * y_axis, 2 * x_axis, z_axis))
    for k in range(z_axis):
        temp = np.fft.ifft2(np.fft.ifftshift(rec_data[:, :, k, 0]))
        res_3D[::2, ::2, k] = np.real(temp)
        res_3D[::2, 1::2, k] = np.imag(temp)
        temp = np.fft.ifft2(np.fft.ifftshift(rec_data[:, :, k, 1]))
        res_3D[1::2, ::2, k] = np.real(temp)
        res_3D[1::2, 1::2, k] = np.imag(temp)
    name_res = test_path + 'res_full.mat'
    res_3D = np.transpose(res_3D, (1, 0, 2))
    sio.savemat(name_res, {'res_3D': res_3D})


def fid_with_nmrpipe(data_path_idx, nmrpipe_dat_path, com_path):

    dic, data = ng.pipe.read(f'{nmrpipe_dat_path}')

    print(data.ndim)
    print(data.shape)
    print(data.dtype)

    M = sio.loadmat(os.path.join('./Process_data/', data_path_idx, 'res_full.mat'))
    Data = M['res_3D']
    Data = np.array(Data, dtype='float32')
    ng.pipe.write('./nmrpipe_data/res_full%d.dat' % int(0 + 1), dic, Data, overwrite=True)

    M = sio.loadmat(os.path.join('./Process_data/', data_path_idx, 'label.mat'))
    Data = M['fid']
    Data = np.array(Data, dtype='float32')
    ng.pipe.write('./nmrpipe_data/label_3D.dat', dic, Data, overwrite=True)

    print("[*] FID to nmrpipe Job finished!")

    subprocess.run(["csh",f'{com_path}'])
    print("[*] nmrpipe shell Job finished!")

    dic, data = ng.pipe.read('./nmrpipe_data/resCN%d.dat' % int(0 + 1))
    dic1, data1 = ng.pipe.read('./nmrpipe_data/resCN3D%d.dat' % int(0 + 1))
    sio.savemat(os.path.join('./Res_data/', data_path_idx, 'resCN3D%d.mat' % int(0 + 1)), {'resCN3D': data1})
    sio.savemat(os.path.join('./Res_data/', data_path_idx, 'resCN%d.mat' % int(0 + 1)), {'resCN': data})
    dic, data = ng.pipe.read('./nmrpipe_data/label3D.dat')
    dic1, data1 = ng.pipe.read('./nmrpipe_data/labelCN.dat')
    sio.savemat(os.path.join('./Res_data/', data_path_idx, 'label_3D.mat'), {'label_3D': data})
    sio.savemat(os.path.join('./Res_data/', data_path_idx, 'labelCN.mat'), {'labelCN': data1})

    print("[*] nmrpipe to FID Job finished!")


if __name__ == "__main__":

    SR = '8'
    nmrpipe_dat_path = '/data2/cwh/BMRB/3D/16101/AR3445A_HNCO_B800/fid_temp_jtf_zf.dat'
    com_path = '/data4/cwh/ADM-NMR-2D/nmrpipe_script/SR.com'
    for sample_n in range(1, 11):
        data_path_idx = f'nmr_jtf/Exp_SR_{SR}_Type_16101/SR/{sample_n}/'
        test_path = './Process_data/' + data_path_idx
        postprocess_data(test_path,y_axis=32, x_axis=64, z_axis=280)
        fid_with_nmrpipe(data_path_idx, nmrpipe_dat_path, com_path)
        print(f"[*] {sample_n} has been finished")


