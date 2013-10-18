
if ~exist('nyu_depth_data_labeled.mat'),
  system('wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_data_labeled.mat');
end

% write images to individual files
load nyu_depth_data_labeled.mat images
nframe=size(images,4);
outdir='./image/';
  system(['mkdir -p ' outdir]);
for ii=1:nframe,
  id=num2str(ii,'%08d');
  img=images(:,:,:,ii);
  imwrite( img, [outdir '/' id '.jpg'], 'Quality', 90 );
end
clear images

% write depths to individual files
load nyu_depth_data_labeled.mat depths
nframe=size(depths,3);
outdir='./depth_uint16/';
  system(['mkdir -p ' outdir]);
for ii=1:nframe,
  id=num2str(ii,'%08d');
  depth=depths(:,:,ii);
  depth=uint16(depth*1000);
  save([outdir '/' id '.mat'],'depth');
end
clear depths

disp('DONE.');

