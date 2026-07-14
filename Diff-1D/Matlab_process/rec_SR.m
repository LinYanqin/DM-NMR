SR = 25;
for n = 1:10
    filename = '/data2/cwh/BMRB/2D/15430/N15hsqc.fid/';
    label = load(strcat(filename,'test.mat'));
    label = label.data;
    label = fliplr(label);
    [M,N] = size(label);
    label_t = ifft2(label);
    filepath = strcat('/data4/cwh/ADM-NMR-1D/Process_data/nmr_dc/Rand_SR_',num2str(SR),'_Freq_15430/SR/',num2str(n),'/');
%     maskpath = strcat(filename,'mask/',num2str(SR),'/mask',num2str(n),'.txt');
    maskpath = strcat(filename,'mask/Rand/mask',num2str(n),'.txt');
    langda=max(label_t(:))*0.;
    noise1=sqrt(langda)*randn(M,N)+1i*sqrt(langda)*randn(M,N);
    label_t = label_t + noise1;
    label_f = fft2(label_t);
    label_SNR = label_f/max(real(label_f(:)));
    area = label_SNR(M-30:M,1:10);
    SNR = 1/std(area(:));
    t_label = ifft(label_f,N,2);
    mask_index = textread(maskpath);
    mask = zeros(1,N);
    l = length(mask_index);
    for k = 1:1:l
        mask(1,mask_index(k,1)+1) = 1;
    end
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
end