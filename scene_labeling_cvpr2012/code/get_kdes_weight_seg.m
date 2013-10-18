
function W=get_kdes_weight_seg(seg, grid_space, patch_size)
%
% function get_kdes_weight_seg(seg, grid_space, patch_size)
%

nseg=max(seg(:))+1;


mpatch_size = max(patch_size); % maximum patch size
[im_h, im_w] = size(seg);
rem_x = mod(im_w-mpatch_size, grid_space);
offset_x = floor(rem_x/2)+1;
rem_y = mod(im_h-mpatch_size, grid_space);
offset_y = floor(rem_y/2)+1;
[grid_x, grid_y] = meshgrid(offset_x:grid_space:im_w-mpatch_size+1, offset_y:grid_space:im_h-mpatch_size+1);

grid_x=round( grid_x + patch_size/2 - 0.5 );
grid_y=round( grid_y + patch_size/2 - 0.5 );

ind=sub2ind([im_h im_w],grid_y,grid_x);
ind=ind(:);

n=accumarray( seg(ind)+1, ones(size(ind)), [nseg 1] );
W=n;

