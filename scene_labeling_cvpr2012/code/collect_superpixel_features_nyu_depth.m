
k_ucm=round(k_ucm*100)/100;
k_ucm_id=num2str(k_ucm*100,'%02d');

featdir=[rootdir '/features/ucm' k_ucm_id];
  system(['mkdir -p ' featdir]);

savefile=[featdir '/features_kucm' k_ucm_id '.mat'];

segdir=[rootdir '/segmentation/ucm' k_ucm_id];

if ~exist(savefile),

disp('computing and collecting kdes features for target superpixels...');


F_all=cell(nframe,1);
L_all=cell(nframe,1);
W_all=cell(nframe,1);

for ii=1:nframe,
  id=num2str(ii,'%08d');

  load([basesegdir '/' id '.mat'],'seg');
  seg0=seg;
  load([segdir '/' id '.mat'],'seg');

  label=imread([labeldir '/' id '.png']);
  label=double(label)/15;
  assert(all(label(:)==round(label(:))));

  l=get_segment_label( label, seg);
  L_all{ii}=l';

  load([basefeatdir '/' id '.mat']);

    f_g=gkdes';
    f_r=rgbkdes';
    f_gd=gkdes_depth';
    f_s=spinkdes';

  wseg0 = get_kdes_weight_seg(seg0, 2, 16 );
  f_1 = features_merge_segments( [f_g f_r f_gd], seg0, wseg0, seg );
  wseg0 = get_kdes_weight_seg(seg0, 2, 40 );
  f_2 = features_merge_segments( f_s, seg0, wseg0, seg );

  % recompute f_ex
  data=load([depthdir '/' id '.mat'],'depth');
  depth=data.depth;
  depth=double(depth)/1000;
  data=load([depthdir_raw '/' id '.mat'],'depth_raw');
  depth_raw=data.depth_raw;
  depth_raw=double(depth_raw)/1000;

  f_ex_2= region_features_extra_rgbd( depth, depth_raw+mask*10, seg );

  F_all{ii}=single([f_1 f_2 f_ex_2]');

  nseg=max(seg(:))+1;
  w=accumarray( seg(:)+1, ones(prod(size(seg)),1), [nseg 1] );
  W_all{ii}=w';

  if mod(ii,50)==0, disp(num2str([ii nframe])); end;
end
save(savefile,'-V7.3','F_all','L_all','W_all');

else

load(savefile);

end

