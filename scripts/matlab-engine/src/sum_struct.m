function s = sum_struct(input)

s = 0;
abn = fieldnames(input);
for iv = 1:length(abn)
    
    % field name corresponding to a specific algorithm
    name = abn{iv};
    
    % reconstructed wave elevation for t_rec
    s = input.(name) + s;
end