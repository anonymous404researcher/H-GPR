function [mu,s2,t_predict] = aggregation_predict(Xt,models,criterion,q,opts)
% Aggregation GP for prediction
% Inputs:
%        Xt: a nt*d matrix containing nt d-dimensional test points
%        models: a cell structure that contains the sub-models built on subsets, where models{i} is the i-th model
%                 fitted to {Xi and Yi} for 1 <= i <= M
%        criterion: an aggregation criterion to combine the predictions from M sub-models
%                   'PoE': product of GP experts
%                   'GPoE': generalized product of GP experts
%                   'BCM': Bayesian committee machine
%                   'RBCM': robust Bayesian committee machine
%                   'GRBCM': generalized robust Bayesian committee machine
%                   'NPAE': nested pointwise aggregation of experts
% Outputs:
%         mu: a nt*1 vector that represents the prediction mean at nt test points
%         s2: a nt*1 vector that represents the prediction variance at nt test points
%         t_predict: computing time for predictions
%
% H.T. Liu 2018/06/01 (htliu@ntu.edu.sg)

nt = size(Xt,1) ;  % number of test points
M = models{1}.Ms ; % number of experts


% predictions of each submodel
t1 = clock ;
if ~strcmp(criterion,'TEGRBCM') % no use for GRBCM
    for i = 1:M           
        [mu_experts{i},s2_experts{i}] = gp(opts.hyp,opts.inffunc,opts.meanfunc, ...
                                   opts.covfunc,opts.likfunc,models{i}.X,models{i}.Y,Xt);
    end
end

% use an aggregation criterion to combine predictions from submodels
mu = zeros(nt,1) ; s2 = zeros(nt,1) ;
switch criterion 
    case 'PoE' % product of GP experts
        for i = 1:M
            s2 = s2 + 1./s2_experts{i} ; 
        end
        s2 = 1./s2 ;

        for i = 1:M 
            mu = mu + s2.*(mu_experts{i}./s2_experts{i}) ;
        end
    case 'GPoE' % generalized product of GP experts using beta_i = 1/M  Information Entropy
        kss = feval(opts.covfunc,opts.hyp.cov,Xt,'diag') + exp(2*opts.hyp.lik);
        for i = 1:M
            %beta{i} = 1/M*ones(length(s2_experts{i}),1) ;
            beta{i} = 0.5*(log(kss) - log(s2_experts{i})) ;
            s2 = s2 + beta{i}./s2_experts{i} ;
        end
        s2 = 1./s2 ;

        for i = 1:M 
            mu = mu + s2.*(beta{i}.*mu_experts{i}./s2_experts{i}) ;
        end
    case 'TEGPoE' % generalized product of GP experts using beta_i = 1/M  Tsallis Entropy
        kss = feval(opts.covfunc,opts.hyp.cov,Xt,'diag') + exp(2*opts.hyp.lik);
        for i = 1:M
            %beta{i} = 1/M*ones(length(s2_experts{i}),1) ;
            %beta{i} = 0.5*(log(kss) - log(s2_experts{i})) ;
            beta{i} = sqrt(2*pi) / (sqrt(q)*(q-1)) * (sqrt(s2_experts{i}) ./ (sqrt(2*pi)*sqrt(s2_experts{i})).^q - ...
                           sqrt(kss) ./ (sqrt(2*pi)*sqrt(kss)).^q) ;
            s2 = s2 + beta{i}./s2_experts{i} ; 
        end
        s2 = 1./s2 ;

        for i = 1:M 
            mu = mu + s2.*(beta{i}.*mu_experts{i}./s2_experts{i}) ;
        end
    case 'BCM' % Bayesian committee machine        
        kss = feval(opts.covfunc,opts.hyp.cov,Xt,'diag') + exp(2*opts.hyp.lik); % because s2_experts consider noise
        
        for i = 1:M
            s2 = s2 + 1./s2_experts{i} ; 
        end
        s2 = 1./(s2 + (1-M)./kss) ;

        for i = 1:M 
            mu = mu + s2.*(mu_experts{i}./s2_experts{i}) ;
        end
    case 'RBCM' % robust Bayesian committee machine
        kss = feval(opts.covfunc,opts.hyp.cov,Xt,'diag') + exp(2*opts.hyp.lik); % because s2_experts consider noise

        beta_total = zeros(nt,1) ;
%         for i = 1:M
%             beta{i} = exp(0.5*(log(kss) - log(s2_experts{i}))) ;
%             beta_total = beta_total + beta{i} ;
%         end
%         for i = 1:M
%             beta{i} = beta{i} ./ beta_total ;
%         end
        for i = 1:M
            %beta{i} = 0.5*(log(kss) - log(s2_experts{i}) - 1 + s2_experts{i}./kss...
            %          + (mu_experts{i}.^2)./kss);
            beta{i} = 0.5*(log(kss) - log(s2_experts{i})) ;
            %beta{i} = 
            beta_total = beta_total + beta{i} ;

            s2 = s2 + beta{i}./s2_experts{i} ; 
        end
        s2 = 1./(s2 + (1-beta_total)./kss) ;

        for i = 1:M 
            mu = mu + s2.*(beta{i}.*mu_experts{i}./s2_experts{i}) ;
        end
    case 'TERBCM' % robust Bayesian committee machine
        kss = feval(opts.covfunc,opts.hyp.cov,Xt,'diag') + exp(2*opts.hyp.lik); % because s2_experts consider noise

        
%         for i = 1:M
%             beta{i} = exp(0.5*(log(kss) - log(s2_experts{i}))) ;
%             beta_total = beta_total + beta{i} ;
%         end
%         for i = 1:M
%             beta{i} = beta{i} ./ beta_total ;
%         end
        result = [];
        MSLL1 = [];
        for q = 1.05:0.05:5
            beta_total = zeros(nt,1) ;
            mu = zeros(nt,1) ; s2 = zeros(nt,1) ;
        for i = 1:M
            %beta{i} = 0.5*(log(kss) - log(s2_experts{i}) - 1 + s2_experts{i}./kss...
            %          + (mu_experts{i}.^2)./kss);
            beta{i} = sqrt(2*pi) / (sqrt(q)*(q-1)) * (sqrt(s2_experts{i}) ./ (sqrt(2*pi)*sqrt(s2_experts{i})).^q - ...
                           sqrt(kss) ./ (sqrt(2*pi)*sqrt(kss)).^q) ;
            %beta{i} = 
            beta_total = beta_total + beta{i} ;

            s2 = s2 + beta{i}./s2_experts{i} ; 
        end
        s2 = 1./(s2 + (1-beta_total)./kss) ;

        for i = 1:M 
            mu = mu + s2.*(beta{i}.*mu_experts{i}./s2_experts{i}) ;
        end
        result(end+1) = mse(mu, yt);
        s2p   = 0.5*mean(log(2*pi*s2));
        s2y    = 0.5*log(2*pi*std(y)^2);
        lossp = (yt-mu).^2;
        mup   = 0.5*mean(lossp./s2);
        dt    = size(Xt,1);
        muv   = ones(dt,1)*mean(y);
        loss  = (yt-muv).^2;
        muy    = 0.5*mean(loss/(std(y)^2));
        MSLL1(end+1)  = s2p-s2y+mup-muy;
        
        
        
        end
    case 'KLRBCM' % robust Bayesian committee machine
        kss = feval(opts.covfunc,opts.hyp.cov,Xt,'diag') + exp(2*opts.hyp.lik); % because s2_experts consider noise

        beta_total = zeros(nt,1) ;
%         for i = 1:M
%             beta{i} = exp(0.5*(log(kss) - log(s2_experts{i}))) ;
%             beta_total = beta_total + beta{i} ;
%         end
%         for i = 1:M
%             beta{i} = beta{i} ./ beta_total ;
%         end
        for i = 1:M
            beta{i} = 0.5*(log(kss) - log(s2_experts{i}) - 1 + s2_experts{i}./kss...
                      + (mu_experts{i}.^2)./kss);
            %beta{i} = 0.5*(log(kss) - log(s2_experts{i})) ;
            %beta{i} = 
            beta_total = beta_total + beta{i} ;

            s2 = s2 + beta{i}./s2_experts{i} ; 
        end
        s2 = 1./(s2 + (1-beta_total)./kss) ;

        for i = 1:M 
            mu = mu + s2.*(beta{i}.*mu_experts{i}./s2_experts{i}) ;
        end
    case 'GRBCM'
        % build M submodels based on cross data {X1,Xi} (1<=i<=M)
        kss = feval(opts.covfunc,opts.hyp.cov,Xt,'diag') + exp(2*opts.hyp.lik); % because s2_experts consider noise

        for i = 1:M
            if i == 1
                models_cross{i} = models{i} ;
            else
                model = models{i} ;
                model.X = [models{1}.X;models{i}.X] ; model.Y = [models{1}.Y;models{i}.Y] ; % X1 + Xi % Y1 + Yi
                model.X_norm = model.X ; model.Y_norm = model.Y;
            
                models_cross{i} = model ;
            end
        end

        for i = 1:M                            
            [mu_crossExperts{i},s2_crossExperts{i}] = gp(opts.hyp,opts.inffunc,opts.meanfunc, ...
                                    opts.covfunc,opts.likfunc,models_cross{i}.X,models_cross{i}.Y,Xt);
        end
        
        % combine predictions from GP experts
        beta_total = zeros(nt,1) ;
        %zero_num = zeros(M,1);
        for i = 1:M
            if i > 2
                beta{i} = 0.5*(log(s2_crossExperts{1}) - log(s2_crossExperts{i})) ;
            else 
                beta{i} = ones(nt,1) ; % beta_1 = beat_2 = 1 ;
            end
            beta_total = beta_total + beta{i} ;
            s2 = s2 + beta{i}./s2_crossExperts{i} ; 
        end

        s2 = 1./(s2 + (1-beta_total)./s2_crossExperts{1}) ;

        for i = 1:M 
            mu = mu + beta{i}.*mu_crossExperts{i}./s2_crossExperts{i} ;
        end
        mu = s2.*(mu + (1-beta_total).*mu_crossExperts{1}./s2_crossExperts{1})  ;
    case 'KLGRBCM'
        % build M submodels based on cross data {X1,Xi} (1<=i<=M)
        kss = feval(opts.covfunc,opts.hyp.cov,Xt,'diag') + exp(2*opts.hyp.lik); % because s2_experts consider noise

        for i = 1:M
            if i == 1
                models_cross{i} = models{i} ;
            else
                model = models{i} ;
                model.X = [models{1}.X;models{i}.X] ; model.Y = [models{1}.Y;models{i}.Y] ; % X1 + Xi % Y1 + Yi
                model.X_norm = [models{1}.X_norm;models{i}.X_norm] ; model.Y_norm = [models{1}.Y_norm;models{i}.Y_norm] ;
            
                models_cross{i} = model ;
            end
        end

        for i = 1:M                            
            [mu_crossExperts{i},s2_crossExperts{i}] = gp(models_cross{i}.hyp,models_cross{i}.inffunc,models_cross{i}.meanfunc, ...
                                    models_cross{i}.covfunc,models_cross{i}.likfunc,models_cross{i}.X_norm,models_cross{i}.Y_norm,Xt);
        end
        
        % combine predictions from GP experts
        beta_total = zeros(nt,1) ;
        %zero_num = zeros(M,1);
        for i = 1:M
            if i > 2
                beta{i} = 0.5*(log(s2_crossExperts{1}) - log(s2_crossExperts{i}) - 1 + s2_crossExperts{i}./kss...
                      + (mu_crossExperts{i}.^2)./s2_crossExperts{1});
                %beta{i} = 0.5*(log(s2_crossExperts{1}) - log(s2_crossExperts{i})) ;
            else 
                beta{i} = ones(nt,1) ; % beta_1 = beat_2 = 1 ;
            end
            beta_total = beta_total + beta{i} ;
            s2 = s2 + beta{i}./s2_crossExperts{i} ; 
        end

        s2 = 1./(s2 + (1-beta_total)./s2_crossExperts{1}) ;

        for i = 1:M 
            mu = mu + beta{i}.*mu_crossExperts{i}./s2_crossExperts{i} ;
        end
        mu = s2.*(mu + (1-beta_total).*mu_crossExperts{1}./s2_crossExperts{1})  ;
        
    case 'TEGRBCM'
        % build M submodels based on cross data {X1,Xi} (1<=i<=M)
        kss = feval(opts.covfunc,opts.hyp.cov,Xt,'diag') + exp(2*opts.hyp.lik); % because s2_experts consider noise

        for i = 1:M
            if i == 1
                models_cross{i} = models{i} ;
            else
                model = models{i} ;
                model.X = [models{1}.X;models{i}.X] ; model.Y = [models{1}.Y;models{i}.Y] ; % X1 + Xi % Y1 + Yi
                model.X_norm = [models{1}.X_norm;models{i}.X_norm] ; model.Y_norm = [models{1}.Y_norm;models{i}.Y_norm] ;
            
                models_cross{i} = model ;
            end
        end

        for i = 1:M                            
            [mu_crossExperts{i},s2_crossExperts{i}] = gp(models_cross{i}.hyp,models_cross{i}.inffunc,models_cross{i}.meanfunc, ...
                                    models_cross{i}.covfunc,models_cross{i}.likfunc,models_cross{i}.X_norm,models_cross{i}.Y_norm,Xt);
        end
        
        % combine predictions from GP experts
        beta_total = zeros(nt,1) ;
        %zero_num = zeros(M,1);
        %q = 0.5;
        for i = 1:M
            if i > 2
                %beta{i} = 0.5*(log(s2_crossExperts{1}) - log(s2_crossExperts{i})) ;
                beta{i} = sqrt(2*pi) / (sqrt(q)*(q-1)) * (sqrt(s2_crossExperts{i}) ./ (sqrt(2*pi)*sqrt(s2_crossExperts{i})).^q - ...
                           sqrt(s2_crossExperts{1}) ./ (sqrt(2*pi)*sqrt(s2_crossExperts{1})).^q) ;
            else 
                beta{i} = ones(nt,1) ; % beta_1 = beat_2 = 1 ;
            end
            beta_total = beta_total + beta{i} ;
            s2 = s2 + beta{i}./s2_crossExperts{i} ; 
        end

        s2 = 1./(s2 + (1-beta_total)./s2_crossExperts{1}) ;

        for i = 1:M 
            mu = mu + beta{i}.*mu_crossExperts{i}./s2_crossExperts{i} ;
        end
        mu = s2.*(mu + (1-beta_total).*mu_crossExperts{1}./s2_crossExperts{1})  ;
    case 'NPAE'
        kss = feval(opts.covfunc,opts.hyp.cov,Xt,'diag'); % no noise
        K_invs = inverseKernelMarix_submodels(models) ;
        K_cross = crossKernelMatrix_nestedKG(models) ;
        hyp_lik = opts.hyp.lik ; % noise para

        mu_all = zeros(nt,M) ;
        for i = 1:M, mu_all(:,i) = mu_experts{i} ; end

        for i = 1:nt
            M_x = mu_all(i,:)' ;

            x = Xt(i,:) ;
            k_M_x = kernelVector_nestedKG(x,models,K_invs) ;
            [K_M_x,K_M_x_inv] = kernelMatrix_nestedKG(x,models,K_invs,K_cross) ;

            mu(i) = k_M_x'*K_M_x_inv*M_x ;
            s2(i) = kss(i) - k_M_x'*K_M_x_inv*k_M_x + exp(2*hyp_lik) ;
        end
    otherwise
        error('No such aggregation model.') ;
end

muf = mu; s2f = s2 - exp(2*opts.hyp.lik);

t2 = clock ;
t_predict = etime(t2,t1) ;

end


%%%%%%%%%%%%%%%
function [K_invs] = inverseKernelMarix_submodels(models)
% calculate the covariance matrics Ks, the inverse matrics K_invs and the det of matrics K_dets of submodels
% used for the nestedKG criterion
M = length(models) ;

covfunc = models{1}.covfunc ; hyp_cov = models{1}.hyp.cov ; hyp_lik = models{1}.hyp.lik ;
for i = 1:M
    K_Xi = feval(covfunc,hyp_cov,models{i}.X_norm) + exp(2*hyp_lik)*eye(size(models{i}.X_norm,1)) ;

    K_invs{i} = eye(size(models{i}.X_norm,1))/K_Xi ;
end

end % end function


function K_cross = crossKernelMatrix_nestedKG(models)
% construct the covariance of training points
% used for the nestedKG criterion 
M = length(models) ;

covfunc = models{1}.covfunc ; hyp_cov = models{1}.hyp.cov ; hyp_lik = models{1}.hyp.lik ;
for i = 1:M 
    for j = 1:M 
        if i == j % self-covariance, should consider noise term
            K_cross{i}{j} = feval(covfunc,hyp_cov,models{i}.X_norm,models{j}.X_norm) + exp(2*hyp_lik)*eye(size(models{i}.X_norm,1)) ;
        else % cross-covariance
            K_cross{i}{j} = feval(covfunc,hyp_cov,models{i}.X_norm,models{j}.X_norm) ;
        end
    end
end

end % end function


function k_M = kernelVector_nestedKG(x,models,K_invs)
% construct the covariance between test points and training points
% used for the nestedKG criterion 
M = length(models) ;

covfunc = models{1}.covfunc ; hyp_cov = models{1}.hyp.cov ;
k_M = zeros(M,1) ;
for i = 1:M 
    k_x_Xi = feval(covfunc,hyp_cov,x,models{i}.X_norm);
    k_M(i) = k_x_Xi*K_invs{i}*k_x_Xi' ;
end

end % end function


function [K_M,K_M_inv] = kernelMatrix_nestedKG(x,models,K_invs,K_cross)
% construct the covariance of training points
% used for the nestedKG criterion 
M = length(models) ;

covfunc = models{1}.covfunc ; hyp_cov = models{1}.hyp.cov ; hyp_lik = models{1}.hyp.lik ;
K_M = zeros(M,M) ;
% obtain an upper triangular matrix to save compting time
for i = 1:M 
    for j = i:M 
        k_x_Xi = feval(covfunc,hyp_cov,x,models{i}.X_norm);
        k_Xj_x = feval(covfunc,hyp_cov,models{j}.X_norm,x);
        K_Xi_Xj = K_cross{i}{j} ;
        if i == j
            K_M(i,j) = 0.5*k_x_Xi*K_invs{i}*K_Xi_Xj*K_invs{j}*k_Xj_x ; % the coef 0.5 is used to ensure K_M = K_M + K_M' along diagonal line
        else
            K_M(i,j) = k_x_Xi*K_invs{i}*K_Xi_Xj*K_invs{j}*k_Xj_x ;
        end
    end
end
% obtain whole K_M
K_M = K_M + K_M' ;

jitter = 1e-10 ;
K_M_inv = eye(size(K_M,1))/(K_M + jitter*eye(M)) ;

end % end function


