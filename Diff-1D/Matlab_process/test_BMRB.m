filepath = '../Process_data/nmr/Pois_SR_15_15430/';

label = load(strcat(filepath, 'label_data.mat'));
load(strcat(filepath, 'factor.mat'));
label = label.label_data;
[M,N,~] = size(label);
label_real=reshape(label(:,:,1),[M,N]);
label_imag=reshape(label(:,:,2),[M,N]);
label_complex=label_real+1i*label_imag;
label_out = factor'.*label_complex;
max_label = max(real(label_out(:)));
label_f=label_out/max_label;
load(strcat(filepath, 'factor.mat'));
load(strcat(filepath, 'res1.mat'));
rec1_real=reshape(res(:,1,:),[M,N]);
rec1_imag=reshape(res(:,2,:),[M,N]);
rec1_complex=rec1_real+1i*rec1_imag;
rec1_out = factor'.*rec1_complex;
res1 = rec1_out/max(real(rec1_out(:)));

level = 10;
figure,contour(abs(label_f),level);title('label')
figure,contour(abs(res1),level);title('DM-NMR')
RLNE = norm(real(res1(:))-real(label_f(:)))/norm(real(label_f(:)))