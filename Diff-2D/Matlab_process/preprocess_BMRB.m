clear;
y_axis = 40;   %32  %40  %64  %32
x_axis = 40;   %40  %40  %40  %64
z_axis = 333;  %257 %333 %422 %280
filepath='/data4/cwh/ADM-NMR-2D/Process_data/nmr_jtf/Pois_SR_8_Type_15100/';
% filename = '/data2/cwh/BMRB/3D/Aruzin/';
% filename = '/data2/cwh/BMRB/3D/6585/at1g24000_hnco.fid/';
% filename = "/data2/cwh/BMRB/3D/15822/CmR9.003_3D_V600_HNCO.fid/";
filename = "/data2/cwh/BMRB/3D/15100/er317_hncoca_5_9_06.fid/";
% filename = "/data2/cwh/BMRB/3D/7175/sr399_cbcaconh_12_15_05.fid/";
% filename = "/data2/cwh/BMRB/3D/15063/hncacb.fid/";
% filename = "/data2/cwh/BMRB/3D/15217/SR358_B600_3D_HBHAcoNH.fid/";
% filename = "/data2/cwh/BMRB/3D/16101/AR3445A_HNCO_B800/";
% filename = "/data2/cwh/BMRB/3D/15063/hnco.fid/";
% filename = "/data2/cwh/BMRB/3D/16101/AR3445A_CN_NOESY_B800/";
index = load(strcat(filename,'mask/mask11.txt'));
num = length(index);
mask= zeros(y_axis,x_axis);
for i = 1:num
    mask((index(i,1)+1),(index(i,2)+1)) = 1; %Generate mask
end
name = strcat(filename,'test_jtf.mat');
data = load(name);
data = data.data;
data = permute(data,[2,1,3]);
fid = data;
R1R2 = fid(1:2:end,1:2:end,:);
R1I2 = fid(1:2:end,2:2:end,:);
I1R2 = fid(2:2:end,1:2:end,:);
I1I2 = fid(2:2:end,2:2:end,:);
FID_real1 = R1R2+j*R1I2;
FID_real2 = I1R2+j*I1I2; %Recombine data
langda=max(FID_real1(:))*0.0;
noise=sqrt(langda)* randn(y_axis,x_axis,z_axis)+1i*sqrt(langda)*randn(y_axis,x_axis,z_axis);
FID_real1 = FID_real1 + noise;
FID_real2 = FID_real2 + noise;
temp = fft(FID_real1,y_axis,1);
temp = fft(temp,x_axis,2);
temp = temp/max(real(temp(:)));
area = temp(10:20,10:20,1:20);
SNR = 1/std(area(:))
input = zeros(z_axis,y_axis,x_axis,4);
noise_label = zeros(y_axis*2,x_axis*2,z_axis);
factor = zeros(2,z_axis);
noise_label(1:2:end,1:2:end,:) = real(FID_real1);
noise_label(1:2:end,2:2:end,:) = imag(FID_real1);
noise_label(2:2:end,1:2:end,:) = real(FID_real2);
noise_label(2:2:end,2:2:end,:) = imag(FID_real2);
for i = 1:z_axis
    FID1 = FID_real1(:,:,i);
    FID1_under = FID1.*mask;
    FID2 = FID_real2(:,:,i);
    FID2_under = FID2.*mask;
    [f,under] = saveUnderSampledSpectrumToTXT(FID1_under,i,1);
    factor(1,i) = f;
    input(i,:,:,1:2) = under;
    [f,under] = saveUnderSampledSpectrumToTXT(FID2_under,i,1);
    factor(2,i) = f;
    input(i,:,:,3:4) = under;
end
noiselabel_name = strcat(filepath,'noise_label.mat');
label_name = strcat(filepath,'label.mat');
input_name = strcat(filepath,'input.mat');
factor_name = strcat(filepath,'factor.mat');
mask_name = strcat(filepath,'mask.mat');

fid = permute(fid,[2,1,3]);

save(input_name,'input');
save(factor_name,'factor');
save(mask_name,'mask');
save(noiselabel_name,'noise_label');
save(label_name,'fid');