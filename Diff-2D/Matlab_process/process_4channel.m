clear;
% filepath = '/data4/cwh/ADM-NMR-2D/Process_data/nmr_dc/Pois_SR_8_Type_Azi/';
filepath = '/data4/cwh/DISCO_NMR/Process_data/Azi_udno/';

fac = load(strcat(filepath,'factor.mat'));
fac = fac.factor;
res = load(strcat(filepath,'res1.mat'));
res = res.res;
res = permute(res,[1,3,4,2]);

[z_axis,y_axis,x_axis,c_axis] = size(res);
rec_data(:,:,:,1) = res(:,:,:,1) + 1j*res(:,:,:,2);
rec_data(:,:,:,2) = res(:,:,:,3) + 1j*res(:,:,:,4);


for k = 1:z_axis
    rec_data(k,:,:,1) = rec_data(k,:,:,1) * fac(1,k);
    rec_data(k,:,:,2) = rec_data(k,:,:,2) * fac(2,k);
end
rec_data = permute(rec_data,[2,3,1,4]);
res_3D = zeros(2*y_axis,2*x_axis,z_axis);
for k = 1:z_axis
    temp = ifft2(ifftshift(rec_data(:,:,k,1))); 
    res_3D(1:2:end,1:2:end,k) = real(temp);
    res_3D(1:2:end,2:2:end,k) = imag(temp);
    temp = ifft2(ifftshift(rec_data(:,:,k,2))); 
    res_3D(2:2:end,1:2:end,k) = real(temp);
    res_3D(2:2:end,2:2:end,k) = imag(temp);
end
name_res = strcat(filepath,'res_full.mat');
res_3D = permute(res_3D,[2,1,3]);    % used by BMRB
save(name_res, 'res_3D');

% load(strcat(filepath,'label.mat'))
% R1R2 = fid(1:2:end,1:2:end,:);
% R1I2 = fid(1:2:end,2:2:end,:);
% I1R2 = fid(2:2:end,1:2:end,:);
% I1I2 = fid(2:2:end,2:2:end,:);
% FID_real1 = R1R2+1j*R1I2;
% FID_real2 = I1R2+1j*I1I2;
% label1 = zeros(y_axis,x_axis,z_axis);
% label2 = zeros(y_axis,x_axis,z_axis);
% for k = 1:z_axis
%     label1(:,:,k) = fftshift(fft2(FID_real1(:,:,k)));
%     label2(:,:,k) = fftshift(fft2(FID_real2(:,:,k)));
% end
% figure,subplot(1,2,1);contour(abs(rec_data(:,:,1,1)),10);subplot(1,2,2);contour(abs(label1(:,:,1)),10);