function [pclouds, distance] = DepthtoClouds(depth, topleft, center, focal)

if nargin < 2
    topleft = [1 1];
end

% Primesense constants
if nargin < 3,
  center = [320 240];
end
if nargin < 4,
  focal = 570.3;
  %focal = 525;
end

[imh, imw] = size(depth);
%constant = 570.3;
constant = focal;

% convert depth image to 3d point clouds
pclouds = zeros(imh,imw,3);
xgrid = ones(imh,1)*(1:imw) + (topleft(1)-1) - center(1);
ygrid = (1:imh)'*ones(1,imw) + (topleft(2)-1) - center(2);
pclouds(:,:,1) = xgrid.*depth/constant;
pclouds(:,:,2) = ygrid.*depth/constant;
pclouds(:,:,3) = depth;
distance = sqrt(sum(pclouds.^2,3));

