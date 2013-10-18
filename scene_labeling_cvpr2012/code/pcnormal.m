function normal = pcnormal(pcloud,threshold,win) 
%
% function normal = pcnormal(pcloud,threshold,win) 
%
%

if nargin<2, threshold = 0.01; end;
if nargin<3, win=5; end;

[imh, imw, ddim] = size(pcloud);
normal = zeros(size(pcloud));
for i = 1:imh
    for j = 1:imw
        minh = max(i - win,1);
        maxh = min(i + win,size(pcloud,1));
        minw = max(j - win,1);
        maxw = min(j + win,size(pcloud,2));
        index = abs(pcloud(minh:maxh,minw:maxw,3) - pcloud(i,j,3)) < pcloud(i,j,3)*threshold;
        if sum(index(:)) > 3 & pcloud(i,j,3) > 0 % the minimum number of points required
            wpc = reshape(pcloud(minh:maxh,minw:maxw,:), (maxh-minh+1)*(maxw-minw+1),3);
            subwpc = wpc(index(:),:);
            subwpc = subwpc - ones(size(subwpc,1),1)*(sum(subwpc)/size(subwpc,1));
            [coeff,dummy] = eig(subwpc'*subwpc); 
            normal(i,j,:) = coeff(:,1)';
        end
    end
end


