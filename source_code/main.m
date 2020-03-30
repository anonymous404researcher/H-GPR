%function [result, MSLL] = demo_toy(criterion, q)
% demo of aggregation GPs for a 1D toy example
%function result = demo_toy(sr, sparseparam)


clear all
rng(100);  %100

% lr_group = [];
% sr_group = [];
% sparseparam_group = [];



% n = 10000; nt = 30000;
% [x,y,xt,yt] = load_data('data/kin40k','kin40k');




% n = 9000; ns = 1000; nt = 5000;
% [x1,y1,xt,yt] = load_data('data/pol','pol');
% x = x1(1:9000,:);
% y = y1(1:9000);
% xs = x1(9001:10000,:);
% ys = y1(9001:10000);



%lr = 0.004;



% generate data
% n = 500 ; sn = 0.5; nt = 500;
% f = @(x) 5*x.^2.*sin(12*x) + (x.^3-0.5).*sin(3*x-0.5) + 4*cos(2*x) ;
% x = linspace(0,1,n)';  y = f(x)+sn*randn(n,1);          % training data
% xt = linspace(-0.2,1.2,nt)'; yt = f(xt)+sn*randn(nt,1); % test data
%






% load('sarcos_inv.mat');
% load('sarcos_inv_test.mat');
% n = 44484; nt = 4449;
% x = sarcos_inv(:,1:21); y = sarcos_inv(:,22);
% xt = sarcos_inv_test(:,1:21); yt = sarcos_inv_test(:,22);

% n = 10000; nt = 5000;
% [x,y,xt,yt] = load_data('data/pol','pol');


% n = 7168; nt = 1024;
% [x,y,xt,yt] = load_data('data/pumadyn32nm','pumadyn32nm');


% load ('bike_data.mat');
% [ndata, ~] = size(M);
% X = M(:,1:14);
% Y = M(:,15);
% R = randperm(ndata);        
% xt = X(R(1:2379),:);
% yt = Y(R(1:2379));
% R(1:2379) = [];
% x = X(R,:);          
% y = Y(R);
% n = 15000; nt = 2379;   %Ms = 15


% load ('energy_data.mat');
% [ndata, ~] = size(M);
% X = M(:,[1:25,27:28]);
% Y = M(:,26);
% R = randperm(ndata);        
% xt = X(R(1:1735),:);
% yt = Y(R(1:1735));
% R(1:1735) = [];
% x = X(R,:);          
% y = Y(R);
% % x = x1(1:18000,:);
% % y = y1(1:18000,:);
% % xs = x1(17001:18000,:);
% % ys = y1(17001:18000);
% n = 18000; nt = 1735;   %Ms = 15
% 



% load ('protein.mat');
% [ndata, ~] = size(M);
% X = M(:,2:9);
% Y = M(:,1);
% R = randperm(ndata);        
% xt = X(R(1:5730),:);
% yt = Y(R(1:5730));
% R(1:5730) = [];
% x = X(R,:);          
% y = Y(R);
% % x = x1(1:35000,:);
% % y = y1(1:35000);
% % xs = x1(35001:40000,:);
% % ys = y1(35001:40000);
% n = 40000; nt = 5730;   %Ms = 50




% load ('electrical_data.mat');
% [ndata, ~] = size(M);
% X = M(:,1:9);
% Y = M(:,10);
% R = randperm(ndata);        
% xt = X(R(1:2000),:);
% yt = Y(R(1:2000));
% R(1:2000) = [];
% x1 = X(R,:);          
% y1 = Y(R);
% x = x1(1:7000,:);
% y = y1(1:7000);
% xs = x1(7001:8000,:);
% ys = y1(7001:8000,:);
% n = 7000; ns = 1000; nt = 2000;   %Ms = 10





load ('video_data.mat');
[ndata, ~] = size(video_data);
X = video_data(:,[1:7,9:11,13:18]);
Y = video_data(:,19);
R = randperm(ndata);        
xt = X(R(1:18784),:);
yt = Y(R(1:18784));
R(1:18784) = [];
x = X(R,:);          
y = Y(R);
% x = x1(1:45000,:);
% y = y1(1:45000);
% xs = x1(45001:50000,:);
% ys = y1(45001:50000);
n = 50000; nt = 18784;   %Ms = 80


% M = 800 ;
%    A = unidrnd(n, M, 800) ;
%         for i = 1:M
%             xs{i} = x(A(i,:),:) ; ys{i} = y(A(i,:)) ;
%          %   Xs{i} = X(A(i,:),:) ; Ys{i} = Y(A(i,:)) ;
%         end
% 














% load ('spatial_network.mat');
% [ndata, ~] = size(A);
% X = A(:,1:3);
% Y = A(:,4);
% R = randperm(ndata);        
% xt = X(R(1:34874),:);
% yt = Y(R(1:34874));
% R(1:34874) = [];
% x = X(R,:);          
% y = Y(R);
% % x = x1(1:45000,:);
% % y = y1(1:45000);
% % xs = x1(45001:50000,:);
% % ys = y1(45001:50000);
% n = 400000; nt = 34874;   %Ms = 500








sf2 = 1 ; ell = 1 ; sn2 = 0.1 ; 



d = size(x,2);
hyp.cov = log([ones(d,1)*ell;sf2]); hyp.lik = log(sn2); hyp.mean = [];
opts.Xnorm = 'Y' ; opts.Ynorm = 'Y' ;
opts.ell = ell ; opts.sf2 = sf2 ; opts.sn2 = sn2 ;
opts.meanfunc = []; opts.covfunc = @covSEard; opts.likfunc = @likGauss; opts.inffunc = @infGaussLik ;
opts.numOptFC = 25 ;

%-----------------------------------------------------
%---------------------Aggregation GP------------------
%-----------------------------------------------------
% model parameters

% train           
% partitionCriterion = 'kmeans' ; % 'random', 'kmeans', 'knkmeans'
% opts.Ms = 50 ; opts.partitionCriterion = partitionCriterion ;        
% [models,t_dGP_train] = aggregation_train(x,y,opts) ;
% 
% % predict
% criterion = 'BCM'; % PoE, GPoE, BCM, RBCM, GRBCM, KLRBCM, KLGRBCM, TERBCM
% 
% q = 1.05;
% [mu_dGP,s2_dGP,t_dGP_predict] = aggregation_predict(xt,models,criterion,q) ;
% [SMSE,MSLL,NLPD] = evaluate(mu_dGP,s2_dGP,x,xt,y,yt);
% fprintf('%s SMSE %6.4f, MSLL %6.4f, NLPD %6.4f\r', criterion, SMSE,MSLL,NLPD);

%-----------------------------------------------------
%---------------------Bootstrap GP--------------------
%-----------------------------------------------------
% train
opts.Ms = 1000;
opts.Msize = 500;
opts.partitionCriterion = 'dirtyKmeans'; % random, dirtyKmeans
[models,t_dGP_train] = aggregation_train_bootstrap(x,y,opts) ;

% predict
criterion = 'PoE'; % PoE, GPoE, BCM, RBCM, GRBCM, KLRBCM, KLGRBCM, TERBCM

q = 1.05;
[mu_dGP,s2_dGP,t_dGP_predict] = aggregation_predict(xt,models,criterion,q) ;
[SMSE,MSLL,NLPD] = evaluate(mu_dGP,s2_dGP,x,xt,y,yt);
fprintf('%s SMSE %6.4f, MSLL %6.4f, NLPD %6.4f\r', criterion, SMSE,MSLL,NLPD);