
function l = get_segment_label(  label, seg )
%
% function l = get_segment_label(  label, seg )
%
%

[h,w]=size(label);
assert( h==size(seg,1) ); assert( w==size(seg,2) );

nseg=max(seg(:))+1;
nlabel=max(label(:))+1;

n=accumarray( [seg(:)+1 label(:)+1], ones(size(seg(:))),  [nseg nlabel] );
[dummy,ind]=max(n,[],2);

l=ind-1;

