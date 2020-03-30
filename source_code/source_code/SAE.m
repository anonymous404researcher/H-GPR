[train_x, train_y] = digitTrainCellArrayData;

n = 100;
idx = randi([1, size(train_x, 2)], n);
for i=1:n
    subplot(10, 10, i), imshow(train_x{idx(i)});
end
rng('default');
num_hid1 = 100;
    
ae1 = trainAutoencoder(train_x, num_hid1, ...
    'MaxEpochs', 400, ...
    'L2WeightRegularization', .004, ...
    'SparsityRegularization', 4, ...
    'SparsityProportion', .15, ...
    'ScaleData', false);

feat1 = encode(ae1, train_x);
num_hid2 = 50;
ae2 = trainAutoencoder(feat1, num_hid2, ...
        'MaxEpochs', 100, ...
        'L2WeightRegularization', .002, ...
        'SparsityRegularization', 4, ...
        'SparsityProportion', .1, ...
        'ScaleData', false);
view(ae2);

feat2 = encode(ae2, feat1);

softnet = trainSoftmaxLayer(feat2, train_y, 'MaxEpochs', 400);
view(softnet)

deepnet = stack(ae1, ae2, softnet);
view(deepnet)






















