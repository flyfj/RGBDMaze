function feaSet = rgbkdes_dense(im, rgbkdes_params, grid_space, patch_size, mask)
% dense color kernel descriptors over uniform grids sampled from images

% inputs
% im             -RGB image
% rgbkdes_params -parameters of color kernel descriptors
% grid_space     -dense grid size
% patch_size     -patch size for extracting gradient kernel descriptor
%
% outputs
% feaSet         -color kernel descriptors and their locations
% written by Liefeng Bo on July 13, 2010
%==========================================================================

% default setting
if nargin < 2
   load('rgbkdes_params.mat');
end
if nargin < 3
   grid_space = 8;
end
if nargin < 4
   patch_size = 16;
end
if nargin < 5
   mask = ones(size(im,1),size(im,2));
end

im = im2double(im); % normalize pixel values to [0 1]
if size(im,3) == 1
    im = color(im);
end

% densely and uniformly sample interest points
mpatch_size = max(patch_size); % maximum patch size
[im_h, im_w, rgb] = size(im);
rem_x = mod(im_w-mpatch_size, grid_space);
offset_x = floor(rem_x/2)+1;
rem_y = mod(im_h-mpatch_size, grid_space);
offset_y = floor(rem_y/2)+1;
[grid_x, grid_y] = meshgrid(offset_x:grid_space:im_w-mpatch_size+1, offset_y:grid_space:im_h-mpatch_size+1);
num_patches = numel(grid_x);

im_k = kernel_trans(im, rgbkdes_params);
% color basis vector for all pixels
for j = 1:length(patch_size)
    xx = repmat((0:patch_size(j)-1)/patch_size(j), [patch_size(j) 1]); % horizontal spatial position
    yy = repmat(((0:patch_size(j)-1)')/patch_size(j),[1 patch_size(j)]); % vertical spatial position
    
    % spatial basis vectors
    skv = eval_kernel(rgbkdes_params.spoints', [yy(:) xx(:)],rgbkdes_params.ktype,rgbkdes_params.kparam(end-1:end));
    mwkvs = zeros(size(rgbkdes_params.eigvectors,2), num_patches);

    keep = [];
    it = 0;
    % for all patches
    for i = 1:num_patches
        
        % find image patches around interst points
        x_lo = grid_x(i);
        x_hi = grid_x(i) + patch_size(j) - 1;
        y_lo = grid_y(i);
        y_hi = grid_y(i) + patch_size(j) - 1;  
        
        submask = mask(y_lo:y_hi,x_lo:x_hi);
        if sum(submask(:)) > patch_size(j)*sqrt(patch_size(j))
           keep = [keep i];
           it = it + 1;
           % compute kernel descriptor
           mwkv = (skv*reshape( im_k(y_lo:y_hi,x_lo:x_hi,:),patch_size(j)^2, rgbkdes_params.rgbsize))'/size(skv,2);
           mwkvs(:,i) = mwkv(:);
        end
    end
    rgbkdes_arr = rgbkdes_params.eigvectors*mwkvs;
    rgbkdes_arr(:,it+1:end) = [];
    feaSet.feaArr{j} = rgbkdes_arr;
end

% output feature information
feaSet.x = grid_x(keep) + mpatch_size/2 - 0.5;
feaSet.y = grid_y(keep) + mpatch_size/2 - 0.5;
feaSet.width = im_w;
feaSet.height = im_h;

function cim = color(im)

cim(:,:,1) = im;
cim(:,:,2) = im;
cim(:,:,3) = im;

function I_k = kernel_trans(I, rgbkdes_params)

[asize,bsize,rgbsize] = size(I);
I_rgb = reshape(I,asize*bsize,rgbsize);
I_k = eval_kernel(I_rgb, rgbkdes_params.rgbpoints', rgbkdes_params.ktype, rgbkdes_params.kparam(1:end-2));
I_k = reshape(I_k,asize,bsize,rgbkdes_params.rgbsize);


