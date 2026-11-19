N1_set=[63, 95, 127, 159, 191, 223, 255];
N2=255;
c = 1;
max_peak = 10;
datapath='./Traindata/Label/';
if ~exist(datapath, 'dir')
    mkdir(datapath);
end
for p = 1:max_peak
    for iter=1:2000
        input_data = zeros(N2+1,2);
        label_data = zeros(N2+1,2);
        N1 = N1_set(randperm(7,1));
        mask = getmask(N1+1);
        FID1 = 0;
        fs = 4000 + (1300-1000)*rand(1);
        for i = 1:p
            Amplitude=0.001 + (1-0.001)*rand(1); 
            Omega = (fs-300)*rand(1);
            Tao1=0.01+(0.19)*rand(1);  
            FID1 = FID1 + generate1DFID_multiplepeaks(Amplitude,Omega,Tao1,N1,fs); 
        end
        % zerofill
        FID11=zeros(1,N2+1);
        FID11(1,1:N1+1)=FID1;

        FID2=FID11.*mask;
        [f,input] = saveUnderSampledSpectrumToTXT(FID2,p,iter);
        label = saveSpectrumToTXT(FID11,p,iter,f);
        
        input_data(:,1) = input(:,1);
        input_data(:,2) = input(:,2);
        label_data(:,1) = label(:,1);
        label_data(:,2) = label(:,2); 

        FileName=['1D_frequencydomain_',num2str(p),'peaks_',num2str(iter),'_y.txt'];
        dlmwrite(strcat(datapath,FileName), label_data,'delimiter' , ' ', 'newline', 'unix');
        c = c + 1
    end
end
