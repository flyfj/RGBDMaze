
function [segfea2,wseg2,map] = features_merge_segments( segfea, seg, wseg, seg2 )
%
% function segfea2 = features_merge_segments( segfea, seg, wseg, seg2 )
%
%    assume seg is a strict oversegmentation of seg2
%    does not include region_extra features
%

nseg=max(seg(:))+1;
nseg2=max(seg2(:))+1;
nfeature=size(segfea,2);

assert( size(segfea,1)==nseg );
assert( length(wseg)==nseg );

segfea2=zeros(nseg2,nfeature);
wseg2=zeros(nseg2,1);

map=zeros(nseg,1);

for s=0:nseg2-1,
  ind=find(seg2==s);
  ss=unique( seg(ind) );
  if isempty(ss),
    warning(['segment ' num2str(s) ' found no correspondence.']);
    continue;
  end
  map(ss+1)=s;
  wseg2(s+1)=sum(wseg(ss+1));
  if wseg2(s+1)>0,
    if length(ss)==1,
      segfea2(s+1,:)=segfea(ss+1,:);
      wseg2(s+1)=wseg(ss+1);
    else
      for t=ss',
        segfea2(s+1,:)=segfea2(s+1,:)+wseg(t+1)*segfea(t+1,:);
      end
      segfea2(s+1,:)=segfea2(s+1,:)/wseg2(s+1);
    end
  end
end


