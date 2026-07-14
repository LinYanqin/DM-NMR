function [mask] = getmask(N1)
    filename = strcat('/data2/cwh/Sample/randsize/',num2str(N1),'.txt');
    index = load(filename);
    mask= zeros(1,256);
    num = length(index);
    for i = 1:num
        mask(1,(index(i,1)+1)) = 1; %Generate mask
    end
end