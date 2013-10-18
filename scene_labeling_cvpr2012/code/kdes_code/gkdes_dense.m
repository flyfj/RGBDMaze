function feaSet = gkdes_dense(im, gkdes_params, grid_space, patch_size, low_contrast, sigma_edge)
% dense gradient kernel descriptors over uniform grids sampled from images

% inputs
% im             -RGB image
% gkdes_params   -parameters of gradient kernel descriptors
% grid_space     -dense grid size
% patch_size     -patch size for extracting gradient kernel descriptor
% low_contrast   -low contrast threshold
% sigma_edge     -parameter in gradient computation
%
% outputs
% feaSet         -gradient kernel descriptors and their locations
% written by Liefeng Bo on July 13, 2010
%==========================================================================

% default setting
if nargin < 2
    load('gkdes_params.mat');
end
if nargin < 3
    grid_space = 8;
end
if nargin < 4
    patch_size = 16;
end
if nargin < 5
    low_contrast = 0.8;
end
if nargin < 6
    sigma_edge = 0.8;
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

% HOG gradient filters
% G_X = [-1 0 1];
% G_Y = [-1; 0; 1];

[G_X,G_Y] = gen_dgauss(sigma_edge); % SIFT gradient filters
im_X = filter2(G_X, im, 'same'); % vertical edges
im_Y = filter2(G_Y, im, 'same'); % horizontal edges
im_mag = sqrt(im_X.^2 + im_Y.^2); % gradient magnitude
gvalue = 1e-5; % suppres threshold value
im_mag = max(im_mag, gvalue);
im_o(:,:,1) = im_X./im_mag; % normalized gradient vector
im_o(:,:,2) = im_Y./im_mag; % normalized gradient vector

% gradient basis vectors for all pixels
im_k = kernel_trans(im_o, gkdes_params);
for j = 1:length(patch_size)
    
    xx = repmat((0:patch_size(j)-1)/patch_size(j), [patch_size(j) 1]); % horizontal spatial position
    yy = repmat(((0:patch_size(j)-1)')/patch_size(j),[1 patch_size(j)]); % vertical spatial position

    % spatial basis vectors
    skv = eval_kernel(gkdes_params.spoints', [yy(:) xx(:)],gkdes_params.ktype,gkdes_params.kparam(end-1:end));
    mwkvs = zeros(size(gkdes_params.eigvectors,2), num_patches);

    % for all patches
    for i = 1:num_patches
        
        % find image patches around interst points
        x_lo = grid_x(i);
        x_hi = grid_x(i) + patch_size(j) - 1;
        y_lo = grid_y(i);
        y_hi = grid_y(i) + patch_size(j) - 1;
        
        % normalize gradient magnitudes
        weight = im_mag(y_lo:y_hi,x_lo:x_hi);
        weight = weight(:);
        if norm(weight) > low_contrast
            weight = weight/norm(weight);
        else
            weight = weight/low_contrast;
        end
        
        % compute kernel descriptor
        mwkv = ((skv.*(ones(size(skv,1),1)*(weight')))*reshape( im_k(y_lo:y_hi,x_lo:x_hi,:),patch_size(j)^2,gkdes_params.gsize))';
        mwkvs(:,i) = mwkv(:);
    end
    feaSet.feaArr{j} = gkdes_params.eigvectors*mwkvs;;
end

% output feature information
feaSet.x = grid_x(:) + mpatch_size/2 - 0.5;
feaSet.y = grid_y(:) + mpatch_size/2 - 0.5;
feaSet.width = im_w;
feaSet.height = im_h;

function I_k = kernel_trans(I_o, gkdes_params)
% compute basis vectors

[asize,bsize,gsize] = size(I_o);
I_oo = reshape(I_o,asize*bsize,gsize);
I_k = eval_kernel(I_oo, gkdes_params.gpoints', gkdes_params.ktype, gkdes_params.kparam(1:end-2));
I_k = reshape(I_k,asize,bsize,gkdes_params.gsize);

function [GX, GY] = gen_dgauss(sigma)

G = gen_gauss(sigma);
[GX,GY] = gradient(G);

GX = GX * 2 ./ sum(sum(abs(GX)));
GY = GY * 2 ./ sum(sum(abs(GY)));

function G = gen_gauss(sigma)

if all(size(sigma) == [1, 1])
    % isotropic gaussian
    f_im_w = 4 * ceil(sigma) + 1;
    G = fspecial('gaussian', f_im_w, sigma);
else
    % anisotropic gaussian
    f_im_w_x = 2 * ceil(sigma(1)) + 1;
    f_im_w_y = 2 * ceil(sigma(2)) + 1;
    G_x = normpdf(-f_im_w_x:f_im_w_x,0,sigma(1));
    G_y = normpdf(-f_im_w_y:f_im_w_y,0,sigma(2));
    G = G_y' * G_x;
end


