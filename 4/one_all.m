% one_all.m
% Code for one vs the rest scheme
% Xiao Zhou
% zhouxiao@bu.edu
% Discussed with classmates
% Referenced from internet
% 04/2017

function[] = one_all()
    twoS_squared = 2;
    data = load('MNIST_data');
    train_y = data.train_samples_labels;
    len_train = length(train_y);
    tgt = ones(len_train,10);
    a = zeros(len_train,10);
    b = zeros(10,1);
    for i = 0:9
        in = train_y ~= i;    
        tgt(in,i+1) = -1;
        in = train_y == i;
        tgt(in,i+1) = 1;
    end
    for i  =  1:10
        [a(:,i),b(i)] = svm(tgt(:,i),tgt(:,i),1);
    end
    
    train_X = data.train_samples;
    test_X = data.test_samples;
    tt = train_X*test_X';
    self_train = sum(train_X.*train_X,2);
    self_test = sum(test_X.*test_X,2);
    
    %disp(target)
    %disp(train_samples{i})
    %disp(target(:,i))

    err = 0;
    test_y = data.test_samples_labels;
    len_test = length(test_y);
    for i = 1:len_test
        [~,index] = max(learn_func_all(i));
        if((index-1) ~= test_y(i))
            err = err+1;
        end
    end
    
    fprintf('Overall Error Rate: %d\n',err/len_test);

    function[s] = learn_func_all(k)
        temp = zeros(len_train,10);
        temp1 = get_kernel(k);
        for i1 = 1:10
           temp(:,i1) = temp1; 
        end
        s = sum(a.*tgt.*temp)'+b;
    end

    function[re] = get_kernel(i)
        s = tt(:,i); 
        s = s*(-2);
        temp = self_test(i).*ones(len_train,1);
        s = s+self_train+temp;
        re = exp(-s/twoS_squared);
    end

end
