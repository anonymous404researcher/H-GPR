function models = aggregation_train_GRBCM_VS_apx(x,y,cidx,opts)
[n,d] = size(x) ;

M = opts.Ms;
[x_trains, y_trains] = partitionData(x,y,M,cidx,opts) ;
if opts.compute_hyp==1
meanfunc = opts.meanfunc ; covfunc  = opts.covfunc ; likfunc  = opts.likfunc ; inffunc  = opts.inffunc ;
ell = opts.ell ; sf2 = opts.sf2 ; sn2 = opts.sn2 ;
hyp = struct('mean', [], 'cov', [ones(d,1)*log(ell);log(sqrt(sf2))], 'lik', log(sqrt(sn2)));
numOptFC = opts.numOptFC ;
disp('Optimizing hyps in training...');
hyp_opt = minimize(hyp, @gp_factorise, numOptFC, inffunc, meanfunc, covfunc, likfunc, x_trains, y_trains); 
else 
    hyp_opt = opts.hyp;
end

% Export models
for i = 1:M
    model.X = x_trains{i}; model.Y = y_trains{i}; 
    model.hyp = hyp_opt ;
    model.Ms = M ;
  
    models{i} = model ;
end

end

function [xs, ys] = partitionData(x,y,M,cidx,opts)

[n,d] = size(x) ;
if M > n; warning('The partition number M exceeds the number of training points.'); end

if opts.grbcm_baseline == 0
    xs{1} = opts.xu ; ys{1} = opts.yu ; 
    tmp = 2;
    for k=1:(M-1)
        xs{tmp} = x(cidx==k & opts.global_index==1, :);
        ys{tmp} = y(cidx==k & opts.global_index==1, :);
        tmp = tmp+1;
    end
else
    xs{1} = x(opts.I_com,:) ; ys{1} = y(opts.I_com) ; 
    global_index = opts.global_index;
    global_index(opts.I_com) = 0;
    tmp = 2;
    for k=1:(M-1)
        xs{tmp} = x(cidx==k & global_index==1, :);
        ys{tmp} = y(cidx==k & global_index==1, :);
        tmp = tmp+1;
    end
end

end 
