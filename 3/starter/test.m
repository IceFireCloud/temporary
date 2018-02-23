%set random state
rng('shuffle');
%set dimension of the sub(sample) patches
subdimension=3;
%set the available range of the original matrix that to be sampled
AvailableIndex=(length(test)-subdimension+1);
%set the number of samples
numbers=10;
%initial the samples
submat = zeros(3,3,numbers);
%sample procedure
for k=1:numbers
    i=floor(AvailableIndex*rand(1)+1);
    j=floor(AvailableIndex*rand(1)+1);
    submat(1:3,1:3,k)= test(i:i+2,j:j+2);  
end
    
    