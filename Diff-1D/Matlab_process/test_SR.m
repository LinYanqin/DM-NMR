clear;clc;
filename = '../nmrpipe_data/fully_sampled/';
label = load(strcat(filename,'test.mat'));
label = label.data;
label = fliplr(label);
[M,N]=size(label);
label_f = label/max(real(label(:)));
label_t = ifft2(label_f);
RLNE_all = ones(1,14);

SR = 10;

for n=1:10
    load(strcat('../Process_data/nmr/Pois_SR_',num2str(SR),'_15430/SR/',num2str(n),'/factor.mat'));
    load(strcat('../Process_data/nmr/Pois_SR_',num2str(SR),'_15430/SR/',num2str(n),'/res1.mat'));
    rec1_real=reshape(res(:,1,:),[M,N]);
    rec1_imag=reshape(res(:,2,:),[M,N]);
    rec1_complex=rec1_real+1i*rec1_imag;
    rec1_out = factor'.*rec1_complex;
    res1 = rec1_out/max(real(rec1_out(:)));
    RLNE_diffusion = norm(real(res1(:))-real(label_f(:)))/norm(real(label_f(:)));
    RLNE_all(1,n) = RLNE_diffusion;
end

RLNE_all(1,11) = max(RLNE_all(1,1:10));
RLNE_all(1,12) = min(RLNE_all(1,1:10));
RLNE_all(1,13) = mean(RLNE_all(1,1:10));
RLNE_all(1,14) = std(RLNE_all(1,1:10));

disp('The SR test has completation');