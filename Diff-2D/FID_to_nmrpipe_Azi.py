import nmrglue as ng
#import mat4py as mt
import pylab
import numpy as np
import scipy.io as sio
import os

data_path_idx = 'nmr_dc/Pois_SR_8_Type_Azi/'

# dic, data = ng.pipe.read(os.path.join('./Processed_Data', data_path_idx, 'fid_temp.dat'))
dic, data = ng.pipe.read('/data2/cwh/BMRB/3D/Aruzin/fid_temp_ZF.dat')
# dic, data = ng.pipe.read("/data2/cwh/BMRB/3D/6585/at1g24000_hnco.fid/fid_temp_ZF.dat")
# dic, data = ng.pipe.read("/data2/cwh/BMRB/3D/15822/CmR9.003_3D_V600_HNCO.fid/fid_temp_ZF.dat")
# dic, data = ng.pipe.read("/data2/cwh/BMRB/3D/15100/er317_hncoca_5_9_06.fid/fid_temp_ZF.dat")
# dic, data = ng.pipe.read("/data2/cwh/BMRB/3D/7175/sr399_cbcaconh_12_15_05.fid/fid_temp_ZF.dat")
# dic, data = ng.pipe.read("/data2/cwh/BMRB/3D/15063/hncacb.fid/fid_temp_ZF.dat")
# dic, data = ng.pipe.read("/data2/cwh/BMRB/3D/15217/SR358_B600_3D_HBHAcoNH.fid/fid_temp_ZF.dat")
# dic, data = ng.pipe.read("/data2/cwh/BMRB/3D/16101/AR3445A_HNCO_B800/fid_temp_ZF.dat")

print(data.ndim)
print(data.shape)
print(data.dtype)

M = sio.loadmat(os.path.join('./Process_data/', data_path_idx, 'res_full.mat'))
Data = M['res_3D']
Data = np.array(Data, dtype='float32')
ng.pipe.write('./nmrpipe_data/res_full%d.dat'%int(0+1), dic, Data, overwrite=True)

M = sio.loadmat(os.path.join('./Process_data/', data_path_idx, 'label.mat'))
Data = M['fid']
Data = np.array(Data, dtype='float32')
ng.pipe.write('./nmrpipe_data/label_3D.dat', dic, Data, overwrite=True)

# M = sio.loadmat(os.path.join('./Processed_Data/', data_path_idx, 'noise_label.mat'))
# Data = M['noise_label']
# Data = np.array(Data, dtype='float32')
# ng.pipe.write("./nmrpipe_data/noise_label_3D.dat", dic, Data, overwrite=True)
print("[*] Job finished!")
