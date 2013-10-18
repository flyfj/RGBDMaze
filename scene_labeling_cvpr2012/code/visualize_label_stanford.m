
function [img_vis,err]=visualize_label( seg, label, pred )
%
% function [img_vis,err]=visualize_label( seg, label, pred )
%
%

color_stanford=[128 128 128
128 128 0
128 64 128
0 128 0
0 0 128
128 0 0
128 80 0
255 128 0]/256;

nclass=size(color_stanford,1);


[h,w]=size(seg);
assert(h==size(label,1));
assert(w==size(label,2));

img_1=zeros(h,w,3);
for c=1:nclass,
  ind=find(label==c);
  img_1(ind)=color_stanford(c,1);
  img_1(ind+h*w)=color_stanford(c,2);
  img_1(ind+h*w*2)=color_stanford(c,3);
end

img_2=zeros(h,w,3);
for s=0:max(seg(:)),
  c=pred(s+1);
  if c==0, continue; end;
  ind=find(seg==s);
  img_2(ind)=color_stanford(c,1);
  img_2(ind+h*w)=color_stanford(c,2);
  img_2(ind+h*w*2)=color_stanford(c,3);
end

% visualize error
err=0;

img_3=ones(h,w,3);
for s=0:max(seg(:)),
  c=pred(s+1);
  ind=find(seg==s & label>0 & label~=c);
  if isempty(ind), continue; end;
  err=err+length(ind);
  img_3(ind)=0;
  img_3(ind+h*w)=0;
  img_3(ind+h*w*2)=0;
end

img_vis=[img_1 img_2 img_3];


