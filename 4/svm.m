% svm.m
% Code for training and testing an SVM classifier with nonlinear kernel.
% Xiao Zhou
% zhouxiao@bu.edu
% Discussed with classmates
% Referenced from internet
% 04/2017

function[a,b] = svm(idx,tgt,to_all)
    data = load('MNIST_data.mat');
    if to_all
        train_X = data.train_samples;
        len_i = length(train_X);
    else
        train_X = data.train_samples(idx,:);
        len_i = length(idx);
    end
    dot = sum(train_X.*train_X,2);
    a = zeros(len_i,1);
    error = zeros(len_i,1);
    c = 0.2;
    t = 0.001;
    eps = 0.005;
    b = 0;
    n_c = 0;
    examine = 1;
    count = 0;
    while n_c > 0 || examine
        n_c = 0;
        if examine
            for j = 1:len_i
                n_c = n_c + examineExample(j,0);
            end
        else
            for j = 1:len_i
                if a(j) ~= 0 && a(j) ~= c
                    n_c = n_c + examineExample(j,0);
                end
            end
        end
        count = count + 1;
        if examine
            examine = 0;
        else
            examine = 1;
        end
        
        if count > 25
            break
        end
    end
    
    fprintf('Number of Loops:%d\n',count);
    err = 0;
    for l = 1:len_i
        if examineExample(-1,l)*tgt(l) < 0
            err = err + 1;
        end
    end

    fprintf('Training Error:%d\n',err/len_i);

    function[result] = examineExample(i1,i91)
        if i91 ~= 0
            result = takeStep(-1,-1,i91);
            return
        end
        y1 = tgt(i1);
        alpha1 = a(i1);
        if alpha1 > 0 && alpha1 < c
            E1 = error(i1);
        else
            E1 = takeStep(-1,-1,i1) - y1;
        end
        r1 = y1*E1;
        if (r1<-t && alpha1<c) || (r1>t && alpha1>0)
            tmax = 0;
            i2 = 0;
            for k1 = 1:len_i
                if a(k1)>0 && a(k1)<c
                    temp = abs(error(k1)-E1);
                    if temp > tmax
                        tmax = temp;
                        i2 = k1;
                    end
                end
            end
            if i2 > 0 && takeStep(i1,i2,-1)
                result = 1;
                return
            end
            k0 = floor(rand*len_i);
            for k1 = k0:len_i+k0-1
                i2 = mod(k1,len_i);
                if ~i2
                    i2 = len_i;
                end
                if a(i2)>0 && a(i2)<c
                    if takeStep(i1,i2,-1)
                        result = 1;
                        return
                    end
                end
            end
            k0 = floor(rand*len_i);
            for k1 = k0:len_i+k0-1
                i2 = mod(k1,len_i);
                if i2 == 0
                    i2 = len_i;
                end
                if takeStep(i1,i2,-1) == 1
                    result = 1;
                    return
                end        
            end
        end
        result = 0;

        function[result] = takeStep(i1,i2,i90)
            if i90 ~= -1
                result = learned_func(i90);
                return
            end
            if i1 == i2
                result = 0;
                return
            end

            alph2 = a(i2);
            y2 = tgt(i2);
            if alph2>0 && alph2<c
                E2 = error(i2);
            else
                E2 = learned_func(i2)-y2;
            end
            s = y1*y2;

            [L,H] = compute_LH();
            if L == H
                result=0;
                return;
            end

            eta = compute_eta();
            if eta < 0
                a2 = alph2+y2*(E2-E1)/eta;
                if a2 < L
                    a2 = L;
                elseif a2 > H
                    a2 = H;
                end
            else
                [l_o, h_o] = computeLHo();
                if l_o > h_o+eps
                    a2 = L;
                elseif l_o < h_o-eps
                    a2 = H;
                else
                    a2 = alph2;
                end
            end

            if a2 < 1e-7
                a2 = 0;
            elseif a2 > c-(1e-7)
                a2 = c;
            end       
            if abs(a2-alph2) < eps*(a2+alph2+eps)
                result=0;
                return
            end
            a1 = alpha1-s*(a2-alph2);
            if a1 < 0
                a2 = a2+s*a1;
                a1 = 0;
            elseif a1 > c
                t = a1-c;
                a2 = a2+a2+s*t;
                a1 = c;
            end
            db = thresholdU();
            error_cacheU(db);
            a(i1) = a1;
            a(i2) = a2;
            result = 1;
            
            function[l_o,h_o] = computeLHo()
                c1 = eta/2;
                c2 = y2*(E1-E2)-eta*alph2;
                l_o = c1*L*L+c2*L;
                h_o = c1*H*H+c2*H;
            end
            function[eta] = compute_eta()
                k11 = kernel_func(i1,i1);
                k12 = kernel_func(i1,i2);
                k22 = kernel_func(i2,i2);
                eta = 2*k12-k11-k22;
            end
            function[L,H] = compute_LH()
                if y1 == y2
                    gamma = alpha1+alph2;
                    if gamma > c
                        L = gamma-c;
                        H = c;
                    else
                        L = 0;
                        H = gamma;
                    end
                else
                    gamma = alpha1-alph2;
                    if gamma > 0
                        L = 0;
                        H = c-gamma;
                    else
                        L = -gamma;
                        H = c;
                    end
                end
            end
            function[delta_b] = thresholdU()
                k11 = kernel_func(i1,i1);
                k12 = kernel_func(i1,i2);
                k22 = kernel_func(i2,i2);
                if a1>0 && a1<c
                    bnew = b-E1-y1*(a1-alpha1)*k11-y2*(a2-alph2)*k12;
                elseif a2>0 && a2<c
                    bnew = b-E2-y1*(a1-alpha1)*k12-y2*(a2-alph2)*k22;
                else
                    b1 = b-E1-y1*(a1-alpha1)*k11-y2*(a2-alph2)*k12;
                    b2 = b-E2-y1*(a1-alpha1)*k12-y2*(a2-alph2)*k22;
                    bnew = (b1+b2)/2;
                end
                delta_b = bnew-b;
                b = bnew;
            end
            function[] = error_cacheU(delta_b)
                t1 = y1*(a1-alpha1);
                t2 = y2*(a2-alph2);
                for i = 1:len_i
                    if 0<a(i) && a(i)<c
                        error(i) = error(i)+t1*kernel_func(i1,i)+t2*kernel_func(i2,i)+delta_b;%puls,not minus
                    end
                end
                error(i1) = 0;
                error(i2) = 0;
            end
            function[s] = learned_func(k)
                s = sum(a.*tgt.*get_kernel(k))+b;
            end
            function[re] = get_kernel(i)

                s = train_X*train_X(i,:)';
                s = s*(-2);
                temp = dot(i).*ones(len_i,1);
                s = s+dot+temp;
                re = exp(-s/2);

            end
            function[re] = kernel_func(i,k)
                s = train_X(k,:)*train_X(i,:)';
                s = s*(-2);
                s = s+dot(i)+dot(k);
                re = exp(-s/2);
            end
        end
    end
end
