clear;
% filename = '/data2/cwh/BMRB/2D/18180/TROSY/';
% filename = '/data2/cwh/BMRB/2D/15610/bnr55_nhsqc_600_9_28_07.fid/';
filename = '/data2/cwh/BMRB/2D/15430/N15hsqc.fid/';
% filename = '/data2/cwh/BMRB/2D/17008/3_NH_TROSY/';
label = load(strcat(filename,'test.mat'));
label = label.data;
label = fliplr(label);
[M,N] = size(label);
label_t = ifft2(label);
filepath = '/data4/cwh/ADM-NMR-1D/Process_data/nmr_dc/Rand_SR_25_Freq_15430/';
maskpath = strcat(filename,'mask/mask23.txt');
langda=max(label_t(:))*0.;
noise1=sqrt(langda)*randn(M,N)+1i*sqrt(langda)*randn(M,N);
label_t = label_t + noise1;
label_f = fft2(label_t);
label_SNR = label_f/max(real(label_f(:)));
area = label_SNR(M-30:M,1:10);
SNR = 1/std(area(:))
t_label = ifft(label_f,N,2);
% load('/data2/cwh/Sample/mask_pos25_128_20000.mat');
% mask_index = mask_pos(14,:)';
mask_index = textread(maskpath);
mask = zeros(1,N);
l = length(mask_index);
for k = 1:1:l
    mask(1,mask_index(k,1)+1) = 1;
end
% load(strcat(filepath, 'DPSmask.mat'));
% mask=DPSmask;
under = t_label.*mask;
input_data = zeros(M,N,2);
label_data = zeros(M,N,2);
factor = zeros(1,M);
mask_data = zeros(M,1,N);
for i = 1:M
    FID1 = t_label(i,:);
    FID2 = under(i,:);
    [f,input_gb1] = saveUnderSampledSpectrumToTXT(FID2,1,1);
    label_gb1 = saveSpectrumToTXT(FID1,1,1,f);
    input_data(i,:,1) = input_gb1(:,1);
    input_data(i,:,2) = input_gb1(:,2);
    label_data(i,:,1) = label_gb1(:,1);
    label_data(i,:,2) = label_gb1(:,2);
    factor(1,i) = f;
    mask_data(i,1,:) = mask;
end
name_input = strcat(filepath,'input_data.mat');
name_label = strcat(filepath,'label_data.mat');
name_mask = strcat(filepath,'mask.mat');
name_factor = strcat(filepath,'factor.mat');
save(name_input,'input_data');
save(name_label,'label_data');
save(name_mask,'mask_data');
save(name_factor,'factor');