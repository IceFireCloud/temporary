% one_one.m
% Code for one vs one scheme
% Xiao Zhou
% zhouxiao@bu.edu
% Discussed with classmates
% Referenced from internet
% 04/2017

function[] = one_one()
    data = load('MNIST_data');
    b = zeros(45,1);
    a = cell(45,1);
    tar = cell(45,1);
    idx = cell(45,1);
    map = zeros(2,45);
    i = 1;

    for x = 0:8
        for y = (x+1):9
            map(1,i) = x;
            map(2,i) = y;        
            i = i+1;
        end
    end
    
    train_X = data.train_samples;
    train_y = data.train_samples_labels;
    test_X = data.test_samples;
    self_train = sum(train_X.*train_X,2);
    self_test = sum(test_X.*test_X,2);
    for i = 1:45
        idx{i} = find(train_y == map(1,i) | train_y == map(2,i));
        tar{i} = train_y(idx{i});        
        tar{i}(tar{i} == map(2,i)) = -1;
        tar{i}(tar{i} == map(1,i)) = 1;
        fprintf('x:%d ,y:%d\n',map(1,i),map(2,i));
        %disp(idx{i})
        [a{i},b(i)] = svm(idx{i},tar{i},0);
    end
    
    save('alpha.mat','a');
    save('b.mat','b');
    save('index.mat','idx');
    save('tar.mat','tar');

    
    err = 0;
    test_y = data.test_samples_labels;
    for i = 1:length(test_y)
        vote = zeros(10,1);
        for j = 1:45
            if learn_func(i,j)>0
                vote(map(1,j)+1) = vote(map(1,j)+1)+1;
            else
                vote(map(2,j)+1) = vote(map(2,j)+1)+1;
            end
        end
        [~,ind] = max(vote);
        if (ind-1) ~= test_y(i)
            err = err+1;
        end
    end
    fprintf('Overall Error Rate: %d\n',err/length(test_y));
    
    function[s] = learn_func(k,i)
        s = sum(a{i}.*tar{i}.*get_kernel(k,i))+b(i);
    end

    function[re] = get_kernel(k,i)
        s = train_X(idx{i},:)*data.test_samples(k,:)';
        s = s*(-2);
        temp = self_test(k).*ones(length(idx{i}),1);
        s = s+self_train(idx{i})+temp;
        re = exp(-s/2);
    end

end
