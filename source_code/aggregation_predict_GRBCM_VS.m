function [mu,s2] = aggregation_predict_GRBCM_VS(Xt,models,opts)
[nt, d] = size(Xt);
mu = zeros(nt,1) ; s2 = zeros(nt,1) ;
M = opts.Ms;

for i = 1:M
    if i == 1
        model = models{i} ;
        models_cross{i} = model ;
    else
        model = models{i} ;
        model.X = [models{1}.X;models{i}.X] ; model.Y = [models{1}.Y;models{i}.Y] ;
        models_cross{i} = model ;
    end
end

for i = 1:M 
%     if strcmp(opts.induce_type,'VFE_opt')
    if opts.grbcm_baseline==1
        [mu_crossExperts{i},s2_crossExperts{i}] = gp(opts.hyp,opts.inffunc,opts.meanfunc, ...
                                    opts.covfunc,opts.likfunc,models_cross{i}.X,models_cross{i}.Y,Xt);
    else
        opts.hyp.xu = models_cross{i}.X;
        [mu_crossExperts{i},s2_crossExperts{i}] = gp(opts.hyp, opts.inffunc, ...
            opts.meanfunc, opts.covfuncF, opts.likfunc, ...
            opts.xvec, opts.yvec, Xt);
    end
%     elseif strcmp(opts.induce_type,'SPGP_opt')        
%         [mu_crossExperts{i},s2_crossExperts{i}] = spgp_pred(opts.yvec,opts.xvec, ...
%             models_cross{i}.X,Xt,opts.hyp);
%     end
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
end
