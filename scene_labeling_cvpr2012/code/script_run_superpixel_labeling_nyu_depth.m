
addpath ./kdes_code

run_id='01';

rootdir='../nyu_depth/';
imgdir=[rootdir '/images/'];
depthdir=[rootdir '/depth_uint16/'];
depthdir_raw=[rootdir '/depth_raw/'];
labeldir=[rootdir '/labels/'];
basesegdir=[rootdir '/segmentation/ucm_base/'];
ucm_dir=[rootdir '/segmentation/ucm_half_gpb'];

basefeatdir=[rootdir '/features/base/'];
  system(['mkdir -p ' basefeatdir]);

savedir=[rootdir '/save'];
  system(['mkdir -p ' savedir]);

nframe=2284;
use_nyu_depth=1;

load([rootdir '/nyu_data_depths_raw_mask250.mat'],'mask');

% compute kdes features on base-level superpixels
compute_base_features=1;
if compute_base_features,
  compute_features_baseseg_nyu_depth;
end

visualize_output=0;

% classify superpixels at level 0.08

run_superpixel=1;

if run_superpixel,
  k_ucm=0.08;
  generate_ucm_segmentation;
  collect_superpixel_features_nyu_depth;
  classify_superpixel;
  eval_superpixel_nyu_depth;
  disp(['accuracy=' num2str(accu) ' for k_ucm=' num2str(k_ucm)]);
end

% classify all levels of the segmentaion tree
run_segmentation_tree=1;
k_ucm_all=[0.04 0.06 0.08 0.12 0.17 0.21];

if run_segmentation_tree,
  nlevel=length(k_ucm_all);
  accus=zeros(nlevel,1);
  for ilevel=1:nlevel,
    k_ucm=k_ucm_all(ilevel);
    generate_ucm_segmentation;
    collect_superpixel_features_nyu_depth;
    classify_superpixel;
    eval_superpixel_nyu_depth;
    accus(ilevel)=accu;
  end
end
%accus:    0.69173 0.7140 0.7210 0.7161 0.6843 0.6353


% put multi-layer results together and classify again
if run_segmentation_tree,
  nlevel=length(k_ucm_all);
  compute_mapping_segmentation;
  classify_segmentation_tree; 
  eval_superpixel_nyu_depth;
  disp(['accuracy=' num2str(accu) ' for segmentation tree']);
  % accu=0.75837
end



