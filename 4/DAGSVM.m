% DAGSVM.m
% Code for training and testing an DAGSVM
% Xiao Zhou
% zhouxiao@bu.edu
% Discussed with classmates
% Referenced from internet
% 04/2017

function[] = DAGSVM()
    data = load('MNIST_data');
    train_X = data.train_samples;
    test_X = data.test_samples;
    self_train = sum(train_X.*train_X,2);
    self_test = sum(test_X.*test_X,2);
    map = zeros(2,45);
    mapping = zeros(10,10);
    i = 1;
    for x = 0:8
        for y = (x+1):9
            map(1,i) = x;
            map(2,i) = y;
            mapping(x+1,y+1) = i;
            i = i+1;
        end
    end
    fprintf('3,4: %d\n',mapping(4,5));
    a = load('alpha');
    b = load('b');
    idx = load('index');
    tar = load('tar');
    a = a.a;
    b = b.b;
    idx = idx.idx;
    tar = tar.tar;
    err = 0;
    test_y = data.test_samples_labels;
    for i = 1:length(test_X)
        duel = ones(10,1);
        old = 0;
        new = 9;
        %fprintf('%d\n\n\n',i);
        for j = 1:9
            %fprintf('value: %d ,%d, x:%d, y:%d\n',learn_func(i,mapping(old+1,new+1)),j,old,new);
            if learn_func(i,mapping(old+1,new+1)) > 0
                duel(new+1) = 0;
            else
                duel(old+1) = 0;
                old = new;
            end
            if j == 9
                break
            end
            temp1 = find(duel == 1);
            new = temp1(1)-1;
            if new == old
                new = temp1(2)-1;
            end        
            if new < old
                temp11 = new;
                new = old;
                old = temp11;
            end
        end
        if old ~= test_y(i)
             err = err+1;
        end
    end
    
    fprintf('Overall Error Rate: %d\n',err/length(test_y));
    
    function[s] = learn_func(k,i)
        s = sum(a{i}.*tar{i}.*get_kernel(k,i))+b(i);
    end

    function[re] = get_kernel(k,i)
        s = train_X(idx{i},:)*test_X(k,:)';
        s = s*(-2);
        temp = self_test(k).*ones(length(idx{i}),1);
        s = s+self_train(idx{i})+temp;
        re = exp(-s/2);
    end
end
