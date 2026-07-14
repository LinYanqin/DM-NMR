import nmrglue as ng
import scipy.io as sio
import os
#
data_path_idx = 'nmr_dc/Pois_SR_8_Type_Azi/'

dic, data = ng.pipe.read('./nmrpipe_data/resCN%d.dat' % int(0+1))
dic1, data1 = ng.pipe.read('./nmrpipe_data/resCN3D%d.dat' % int(0 + 1))
sio.savemat(os.path.join('./Res_data/', data_path_idx, 'resCN3D%d.mat' % int(0 + 1)), {'resCN3D': data1})
sio.savemat(os.path.join('./Res_data/', data_path_idx, 'resCN%d.mat' % int(0+1)), {'resCN': data})
dic, data = ng.pipe.read('./nmrpipe_data/label3D.dat')
dic1, data1 = ng.pipe.read('./nmrpipe_data/labelCN.dat')
sio.savemat(os.path.join('./Res_data/', data_path_idx, 'label_3D.mat'), {'label_3D': data})
sio.savemat(os.path.join('./Res_data/', data_path_idx, 'labelCN.mat'), {'labelCN': data1})
# dic, data = ng.pipe.read("./nmrpipe_data/noise_label3D.dat")
# dic1, data1 = ng.pipe.read("./nmrpipe_data/noise_labelCN.dat")
# sio.savemat(os.path.join('./Res_data/', data_path_idx, 'noise_label_3D.mat'), {'noise_label_3D': data})
# sio.savemat(os.path.join('./Res_data/', data_path_idx, 'noise_labelCN.mat'), {'noise_labelCN': data1})
print("[*] Job finished!")
#


