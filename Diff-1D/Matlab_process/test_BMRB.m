% label = load('/data2/cwh/BMRB/2D/15610/bnr55_nhsqc_600_9_28_07.fid/test.mat');
% label = load('/data2/cwh/BMRB/2D/18180/TROSY/test.mat');
label = load('/data2/cwh/BMRB/2D/15430/N15hsqc.fid/test.mat');
% label = load('/data2/cwh/BMRB/2D/17008/3_NH_TROSY/test.mat');
label = label.data;
label = fliplr(label);
[M,N]=size(label);
label_f = label/max(real(label(:)));
label_t = ifft2(label_f);
load('/data4/cwh/ADM-NMR-1D/Process_data/nmr_dc/Pois_SR_25_Freq_15430/factor.mat');
load('/data4/cwh/ADM-NMR-1D/Process_data/nmr_dc/Pois_SR_25_Freq_15430/res1.mat');
rec1_real=reshape(res(:,1,:),[M,N]);
rec1_imag=reshape(res(:,2,:),[M,N]);
rec1_complex=rec1_real+1i*rec1_imag;
rec1_out = factor'.*rec1_complex;
res1 = rec1_out/max(real(rec1_out(:)));

level = 9;
figure,contour(abs(label_f),level);title('label')
figure,contour(abs(res1),level);title('diffusion')
RLNE = norm(real(res1(:))-real(label_f(:)))/norm(real(label_f(:)))
% fileID=fopen(strcat(filename,'RLNE.txt'),'w');
% fprintf(fileID,'RLNE=%6.4f\n',RLNE);
% fclose(fileID);
% close all;
% figure,subplot(2,1,1),plot(real(label_f(100,:)));
% subplot(2,1,2),plot(real(res1(100,:)));