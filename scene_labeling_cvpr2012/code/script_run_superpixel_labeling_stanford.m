
addpath ./kdes_code

run_id='01';

rootdir='../stanford/';
imgdir=[rootdir '/images/'];
labeldir=[rootdir '/labels/'];
basesegdir=[rootdir '/segmentation/ucm_base/'];
ucm_dir=[rootdir '/segmentation/ucm_gpb'];

basefeatdir=[rootdir '/features/base/'];
  system(['mkdir -p ' basefeatdir]);

savedir=[rootdir '/save'];
  system(['mkdir -p ' savedir]);

nframe=715;
use_stanford=1;


% compute kdes features on base-level superpixels
compute_base_features=1;
if compute_base_features,
  compute_features_baseseg_stanford;
end

visualize_output=0;

% classify superpixels at level 0.08

run_superpixel=1;

if run_superpixel,
  k_ucm=0.08; % 0.08;
  generate_ucm_segmentation;
  collect_superpixel_features_stanford;
  classify_superpixel;
  eval_superpixel_stanford;
  disp(['accuracy=' num2str(accu) ' for k_ucm=' num2str(k_ucm)]);
end

% classify all levels of the segmentaion tree
run_segmentation_tree=1;

k_ucm_all=[0.04 0.06 0.08 0.10 0.15];
if run_segmentation_tree,
  nlevel=length(k_ucm_all);
  accus=zeros(nlevel,1);
  for ilevel=1:nlevel,
    k_ucm=k_ucm_all(ilevel);
    generate_ucm_segmentation;
    collect_superpixel_features_stanford;
    classify_superpixel;
    eval_superpixel_stanford;
    accus(ilevel)=accu;
  end
end
% accus=[0.8056 0.8211 0.8217 0.8132 0.7830];

% put multi-layer results together and classify again
if run_segmentation_tree,
  nlevel=length(k_ucm_all);
  compute_mapping_segmentation;
  classify_segmentation_tree; 
  eval_superpixel_stanford;
end



