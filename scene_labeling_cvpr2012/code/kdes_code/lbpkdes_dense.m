function feaSet = lbpkdes_dense(im, lbpkdes_params, grid_space, patch_size, low_contrast)
% dense shape kernel descriptors over uniform grids sampled from images

% inputs
% im             -RGB image
% lbpkdes_params -parameters of shape kernel descriptors
% grid_space     -dense grid size
% patch_size     -patch size for extracting gradient kernel descriptor
% low_contrast   -low contrast threshold
%
% outputs
% feaSet         -shape kernel descriptors and their locations
% written by Liefeng Bo on July 13, 2010
%==========================================================================

% default setting
if nargin < 2
    load('lbpkdes_params.mat');
end
if nargin < 3
    grid_space = 8;
end
if nargin < 4
    patch_size = 16;
end
if nargin < 5
    low_contrast = 0.2;
end

if size(im,3) == 1
    im = im2double(im); % normalize pixel values to [0 1]
else
    im = im2double(rgb2gray(im)); % convert color image to gray image and then normalize pixel values to [0 1]
end

% densely and uniformly sample interest points
mpatch_size = max(patch_size); % maximum patch size
[im_h, im_w] = size(im);
rem_x = mod(im_w-mpatch_size, grid_space);
offset_x = floor(rem_x/2)+1;
rem_y = mod(im_h-mpatch_size, grid_space);
offset_y = floor(rem_y/2)+1;
[grid_x, grid_y] = meshgrid(offset_x:grid_space:im_w-mpatch_size+1, offset_y:grid_space:im_h-mpatch_size+1);
num_patches = numel(grid_x);

% local binary pattern
im_c = lbp_trans(im);
% local binary pattern basis vectors for all pixels
im_k = kernel_trans(im_c, lbpkdes_params);

% for all patches
for j = 1:length(patch_size)
    xx = repmat((0:patch_size(j)-1)/patch_size(j), [patch_size(j) 1]); % horizontal spatial position
    yy = repmat(((0:patch_size(j)-1)')/patch_size(j),[1 patch_size(j)]); % vertical spatial position
    
    % spatial basis vectors
    skv = eval_kernel(lbpkdes_params.spoints', [yy(:) xx(:)],lbpkdes_params.ktype,lbpkdes_params.kparam(end-1:end));
    mwkvs = zeros( size(lbpkdes_params.eigvectors,2), num_patches);
    
    for i = 1:num_patches
 
        % find image patches around interst points
        x_lo = grid_x(i);
        x_hi = grid_x(i) + patch_size(j) - 1;
        y_lo = grid_y(i);
        y_hi = grid_y(i) + patch_size(j) - 1;
        
        % normalize standard deviations
        weight = im_c(y_lo:y_hi,x_lo:x_hi,end);
        weight = weight(:);
        if norm(weight) > low_contrast
            weight = weight/norm(weight);
        else
            weight = weight/low_contrast;
        end
        
        % compute kernel descriptor
        mwkv = ( (skv.*(ones(size(skv,1),1)*(weight')))*reshape(im_k(y_lo:y_hi,x_lo:x_hi,:),patch_size(j)^2,lbpkdes_params.lbpsize) )';
        mwkvs(:,i) = mwkv(:);
    end
    feaSet.feaArr{j} = (lbpkdes_params.eigvectors*mwkvs);
end

% output feature information
feaSet.x = grid_x(:) + mpatch_size/2 - 0.5;
feaSet.y = grid_y(:) + mpatch_size/2 - 0.5;
feaSet.width = im_w;
feaSet.height = im_h;

function I_k = kernel_trans(I_c, lbpkdes_params)

[asize,bsize,csize] = size(I_c);
I_cc = reshape(I_c,asize*bsize,csize);
I_k = eval_kernel(I_cc(:,1:end-1), lbpkdes_params.lbppoints', lbpkdes_params.ktype, lbpkdes_params.kparam(1:end-2));
I_k = reshape(I_k,asize,bsize,lbpkdes_params.lbpsize);


