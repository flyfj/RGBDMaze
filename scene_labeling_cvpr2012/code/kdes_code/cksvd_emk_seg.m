function [segfea,n] = cksvd_emk(feaSet, words, G, seg, ktype, kparam)
% extract image features using constrained kernel SVD
% mainly written when Liefeng Bo was in toyota technological institute at Chicago (TTI-C), working with Cristian Sminchisescu 
% modified by Xiaofeng for segmentation

% set the parameters 
wordsnum = size(words,2);
nseg=max(seg(:))+1;   % start with 0
segfea = zeros(wordsnum,nseg);
n = zeros(nseg,1);

patchsize = length(feaSet.feaArr);
for pp = 1:patchsize
    kdes = double(feaSet.feaArr{pp});
    kz{pp} = eval_kernel(kdes',words',ktype,kparam);
end

% spatial pyramid match with the learned low dimensional kernel
    xgrid = ceil(feaSet.x);
    ygrid = ceil(feaSet.y);
    allgrid = sub2ind( size(seg), ygrid, xgrid );

    for s=0:nseg-1,

        % find kdes localized in the corresponding segment
        ind = find( seg(allgrid) == s);
        n(s+1)=length(ind);
        if length(ind)
           kzind = [];
           for pp = 1:patchsize
               kzind = [kzind; kz{pp}(ind,:)];
           end

           % suppress similar kernel descriptors using max
           [valueaaa, indaaa] = max(kzind,[],2);
           [valuebbb, indbbb] = sort(valueaaa,'descend');
           [valueccc, indccc] = unique(indaaa(indbbb),'first');
           indgrid = indbbb(indccc);
           mkzind = mean(kzind(indgrid,:),1);
           segfea(:,s+1)= G*(mkzind');
        else
           segfea(:,s+1)= 0;
        end
    end

% could try to guess whether seg starts with 0 or 1
%if sum(seg==0)==0,
%  segfea=segfea(:,2:end);
%end

