clear;clc;
filename = '/data2/cwh/BMRB/2D/15430/N15hsqc.fid/';
label = load(strcat(filename,'test.mat'));
label = label.data;
label = fliplr(label);
[M,N]=size(label);
label_f = label/max(real(label(:)));
label_t = ifft2(label_f);
RLNE_all = ones(4,14);
SR = 25;
for n=1:10
    load(strcat('/data4/cwh/ADM-NMR-1D/Process_data/nmr_dc/Exp_SR_',num2str(SR),'_Freq_15430/SR/',num2str(n),'/factor.mat'));
    load(strcat('/data4/cwh/ADM-NMR-1D/Process_data/nmr_dc/Exp_SR_',num2str(SR),'_Freq_15430/SR/',num2str(n),'/res1.mat'));
    rec1_real=reshape(res(:,1,:),[M,N]);
    rec1_imag=reshape(res(:,2,:),[M,N]);
    rec1_complex=rec1_real+1i*rec1_imag;
    rec1_out = factor'.*rec1_complex;
    res1 = rec1_out/max(real(rec1_out(:)));
    RLNE_diffusion = norm(real(res1(:))-real(label_f(:)))/norm(real(label_f(:)));
    RLNE_all(4,n) = RLNE_diffusion;

%     hmsIST=load(strcat(filename,'hmsIST/SR/',num2str(SR),'/',num2str(n),'/hmsIST.mat'));
    hmsIST=load(strcat(filename,'hmsIST/SR/Exp/',num2str(n),'/hmsIST.mat'));
    hmsIST=hmsIST.data;
    hmsIST=fliplr(hmsIST);
    hmsIST=hmsIST/max(real(hmsIST(:)));
    RLNE_hmsIST = norm(real(hmsIST(:))-real(label_f(:)))/norm(real(label_f(:)));
    RLNE_all(1,n) = RLNE_hmsIST;
    
%     SMILE=load(strcat(filename,'SMILE/SR/',num2str(SR),'/',num2str(n),'/SMILE.mat'));
    SMILE=load(strcat(filename,'SMILE/SR/Exp/',num2str(n),'/SMILE.mat'));
    SMILE=SMILE.data;
    SMILE=fliplr(SMILE);
    SMILE=SMILE/max(real(SMILE(:)));
    RLNE_SMILE = norm(real(SMILE(:))-real(label_f(:)))/norm(real(label_f(:)));
    RLNE_all(2,n) = RLNE_SMILE;
    
%     FIDnet=load(strcat(filename,'FIDnet/SR/',num2str(SR),'/',num2str(n),'/FIDnet.mat'));
    FIDnet=load(strcat(filename,'FIDnet/SR/Exp/',num2str(n),'/FIDnet.mat'));
    FIDnet=FIDnet.data;
    FIDnet=fliplr(FIDnet);
    FIDnet=FIDnet/max(real(FIDnet(:)));
    RLNE_FIDnet = norm(real(FIDnet(:))-real(label_f(:)))/norm(real(label_f(:)));
    RLNE_all(3,n) = RLNE_FIDnet;
end
for j=1:4
    RLNE_all(j,11) = max(RLNE_all(j,1:10));
    RLNE_all(j,12) = min(RLNE_all(j,1:10));
    RLNE_all(j,13) = mean(RLNE_all(j,1:10));
    RLNE_all(j,14) = std(RLNE_all(j,1:10));
end
disp('The SR test has completation');