clear;
filename = "/data2/cwh/BMRB/3D/16101/AR3445A_HNCO_B800/";

y_axis = 64;   %64  %32  %40  %64  %32
x_axis = 64;   %64  %40  %40  %40  %64
z_axis = 280;  %732 %257 %333 %422 %280

label3D=load(strcat(filename,'label3D_jtf.mat'));
label3D=label3D.data;
label3D=label3D/max(label3D(:));

RLNE_all = ones(3,14);
SR = '8';
for n=1:10
%     res3D=load(strcat('/data4/cwh/ADM-NMR-2D/Res_data/nmr_jtf/Pois_SR_',SR,'_Type_16101/SR/',num2str(n),'/resCN3D1.mat'));
    res3D=load(strcat('/data4/cwh/ADM-NMR-2D/Res_data/nmr_jtf/Rand_SR_8_Type_16101/SR/',num2str(n),'/resCN3D1.mat'));
    res3D=res3D.resCN3D;
    res3D=res3D/max(res3D(:));
    RLNE_diffusion = norm(res3D(:)-label3D(:))/norm(label3D(:));
    RLNE_all(3,n) = RLNE_diffusion;

%     hmsIST3D=load(strcat(filename,'hmsIST/SR/',SR,'/',num2str(n),'/hmsIST3D.mat'));
    hmsIST3D=load(strcat(filename,'hmsIST/SR/Rand/',num2str(n),'/hmsIST3D.mat'));
    hmsIST3D=hmsIST3D.data;
    hmsIST3D=hmsIST3D/max(hmsIST3D(:));
    RLNE_hmsIST = norm(hmsIST3D(:)-label3D(:))/norm(label3D(:));
    RLNE_all(1,n) = RLNE_hmsIST;
    
%     SMILE3D=load(strcat(filename,'SMILE/SR/',SR,'/',num2str(n),'/SMILE3D.mat'));
    SMILE3D=load(strcat(filename,'SMILE/SR/Rand/',num2str(n),'/SMILE3D.mat'));
    SMILE3D=SMILE3D.data;
    SMILE3D=SMILE3D/max(SMILE3D(:));
    RLNE_SMILE = norm(SMILE3D(:)-label3D(:))/norm(label3D(:));
    RLNE_all(2,n) = RLNE_SMILE;
    
end

for j=1:3
    RLNE_all(j,11) = max(RLNE_all(j,1:10));
    RLNE_all(j,12) = min(RLNE_all(j,1:10));
    RLNE_all(j,13) = mean(RLNE_all(j,1:10));
    RLNE_all(j,14) = std(RLNE_all(j,1:10));
end
disp('The SR test has completation');
