

%%

data_dir = 'F:\test\NewlyAlignedRGB_mug\NewlyAlignedRGB_mug\';

start_id = 100;
end_id = 100;

imgw = 640;
imgh = 480;

invF = [594.21 0 320/594.21; 0 591.04 240/594.21; 0 0 1];

model_data = [];

for i=start_id:end_id
    
    % load depth image
    cur_dfile = [data_dir num2str(i) '_depth.mat'];
    cur_dmap = load(cur_dfile);
    cur_dmap = cur_dmap.refinedDepth;
    
    % load segment image
    cur_segfile = [data_dir 'seg' num2str(i) '.txt'];
    cur_smap = load(cur_segfile);
    [rows, cols] = find(cur_smap>0);
    dvals = cur_dmap(sub2ind(size(cur_dmap), rows, cols));
    
    % homogeneous coordinates
    homo_coord = [cols'; rows'; dvals'];
    dvalmap = [dvals'; dvals'; dvals'];
    homo_coord = homo_coord ./ dvalmap;
    
    % get local coordinates
    local_coord = invF \ homo_coord;
    local_coord = local_coord .* dvalmap;
    
    % get global coordinates
    local_coord = [local_coord; ones(1, size(local_coord, 2))];
    
    cur_w2cfile = [data_dir num2str(i) '_w2c.mat'];
    cur_w2c = load(cur_w2cfile);
    cur_w2c = cur_w2c.m;
    
    global_coord = cur_w2c \ local_coord;
    % export point positions
    model_data = [model_data global_coord(1:3,:)];
    
    disp(['Finish ' num2str(i)]);
    
end

% output to ply file for rendering
fn = 'g:\\model.ply';
fp = fopen(fn, 'w');
fprintf(fp, 'ply\nformat ascii 1.0\nelement vertex %d\nproperty float32 x\nproperty float32 y\nproperty float32 z\n', size(model_data, 2));
fprintf(fp, 'end_header\n\n');
for i = 1: size(model_data, 2)
    fprintf(fp, '%f %f %f\n', model_data(1:3, i));
end
fclose(fp);