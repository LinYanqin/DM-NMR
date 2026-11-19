clear;
filename = "../nmrpipe_data/fully_sampled/";

y_axis = 64;  
x_axis = 64; 
z_axis = 280;

label3D=load(strcat(filename,'label3D.mat'));
label3D=label3D.data;
label3D=label3D/max(label3D(:));

RLNE_all = ones(1,14);
SR = '8';
for n=1:10
    res3D=load(strcat('../Res_data/nmr/Pois_SR_',SR,'_Type_16101/SR/',num2str(n),'/resCN3D1.mat'));
    res3D=res3D.resCN3D;
    res3D=res3D/max(res3D(:));
    RLNE_diffusion = norm(res3D(:)-label3D(:))/norm(label3D(:));
    RLNE_all(1,n) = RLNE_diffusion;    
end

RLNE_all(1,11) = max(RLNE_all(1,1:10));
RLNE_all(1,12) = min(RLNE_all(1,1:10));
RLNE_all(1,13) = mean(RLNE_all(1,1:10));
RLNE_all(1,14) = std(RLNE_all(1,1:10));

disp('The SR test has completation');
