function [mask] = getmask(N1,N2)
    filename = strcat('./mask/',num2str(N1),'-',num2str(N2),'.txt');
    index = load(filename);
    mask= zeros(64,64);
    num = length(index);
    for i = 1:num
        mask((index(i,1)+1),(index(i,2)+1)) = 1; %Generate mask
    end
end
