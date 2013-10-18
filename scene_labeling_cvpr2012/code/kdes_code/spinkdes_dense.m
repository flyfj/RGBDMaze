function feaSet = SpinKDESDense(pcloud,normal, spinkdes_params, grid_space, patch_size, radius)

%% dense color kernel descriptors over uniform grids sampled from images

% default setting
if nargin < 2 | isempty(normal),
   normal=pcnormal(pcloud);
   normal=fix_normal_orientation( normal, pcloud );
end

if nargin < 3
   grid_space = 8;
end
if nargin < 4
   patch_size = 40;
end
if nargin < 5
   radius = 0.05;
end

maxsample = 256;
minsample = 10;

% densely and uniformly sample interest points
mpatch_size = max(patch_size); % maximum patch spin
[cloud_h, cloud_w, cdim] = size(pcloud);
rem_x = mod(cloud_w-mpatch_size, grid_space);
offset_x = floor(rem_x/2)+1;
rem_y = mod(cloud_h-mpatch_size, grid_space);
offset_y = floor(rem_y/2)+1;
[grid_x, grid_y] = meshgrid(offset_x:grid_space:cloud_w-mpatch_size+1, offset_y:grid_space:cloud_h-mpatch_size+1);
num_patches = numel(grid_x);

% color basis vector for all pixels
pcloudvector = reshape(pcloud, size(pcloud,1)*size(pcloud,2), size(pcloud,3));
spinkdes_arr = zeros(size(spinkdes_params.normvectors,1), num_patches);
% for all patches
it = 0;
keep = [];
for i = 1:num_patches
        
    % randomly sample point cloud
    rpoint(1,1) = pcloud(grid_y(i), grid_x(i), 1);
    rpoint(1,2) = pcloud(grid_y(i), grid_x(i), 2);
    rpoint(1,3) = pcloud(grid_y(i), grid_x(i), 3);
    cnormal(1,1) = normal(grid_y(i), grid_x(i), 1);
    cnormal(1,2) = normal(grid_y(i), grid_x(i), 2);
    cnormal(1,3) = normal(grid_y(i), grid_x(i), 3);
    if rpoint(3) > 0

       % reduce search region
       minh = max(grid_y(i) - patch_size,1);
       maxh = min(grid_y(i) + patch_size, size(pcloud,1));
       minw = max(grid_x(i) - patch_size,1);
       maxw = min(grid_x(i) + patch_size, size(pcloud,2));

       % sample nearest points and their normals
       subpcloud = pcloud(minh:maxh, minw:maxw,:);
       subnormal = normal(minh:maxh, minw:maxw,:);
       subpcloud = reshape(subpcloud, size(subpcloud,1)*size(subpcloud,2), 3);
       subnormal = reshape(subnormal, size(subnormal,1)*size(subnormal,2), 3);
       diff = subpcloud - repmat(rpoint, size(subpcloud,1), 1);
       dist = sqrt(sum(diff.^2, 2));
       index = find(dist < radius & subpcloud(:,3) > 0);

       if length(index) > minsample
          it = it + 1;
          keep = [keep i];
          perm = randsample(length(index),min(maxsample,length(index)),0);
          subpcloud = subpcloud(index(perm),:);
          subnormal = subnormal(index(perm),:);
          subnormal = subnormal./(repmat(sqrt(sum(subnormal.^2,2)),1,size(subnormal,2))+eps);

          % compute spin-type distance and angles of normals
          cnormalvector = repmat(cnormal,size(subpcloud,1),1);
          spin_x = sum(cnormalvector.*diff(index(perm),:), 2);
          spin_y_square = dist(index(perm)).^2 - spin_x.^2;
          spin_y_square(spin_y_square < 0) = 0;
          spin_y = sqrt(spin_y_square);
          nndot = sum(cnormalvector.*subnormal,2);
          nndot(nndot > 1) = 1-eps;
          nndot(nndot < -1) = -1+eps;
          angle = acos(nndot);
 
          % compute spin image kernel descriptor
          nkv = eval_kernel(spinkdes_params.npoints', [sin(angle) cos(angle)], spinkdes_params.ktype, spinkdes_params.kparam(1:2));
          spinkv = eval_kernel(spinkdes_params.spoints', [spin_x(:) spin_y(:)], spinkdes_params.ktype, spinkdes_params.kparam(3:4));
          mwkv = (nkv*spinkv')/size(nkv,2);
          spinkdes_arr(:,it) = spinkdes_params.normvectors*mwkv(:);
      end
   end
end

% output feature information
spinkdes_arr(:,it+1:end) = [];
feaSet.feaArr{1} = spinkdes_arr;
feaSet.x = grid_x(keep) + mpatch_size/2 - 0.5;
feaSet.y = grid_y(keep) + mpatch_size/2 - 0.5;
feaSet.width = cloud_w;
feaSet.height = cloud_h;


