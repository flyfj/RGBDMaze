
k_ucm=round(k_ucm*100)/100;
k_ucm_id=num2str(k_ucm*100,'%02d');

featdir=[rootdir '/features/ucm' k_ucm_id];
  system(['mkdir -p ' featdir]);

segdir=[rootdir '/segmentation/ucm' k_ucm_id];

savefile=[featdir '/features_kucm' k_ucm_id '.mat'];

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
  label=load([labeldir '/' id '.txt'])+1;
  l=get_segment_label( label, seg);
  L_all{ii}=l';

  load([basefeatdir '/' id '.mat']);

  f_g=double(gkdes'); f_r=double(rgbkdes'); f_l=double(lbpkdes');
  wseg0 = get_kdes_weight_seg(seg0, 2, 16 );
  f_2 = features_merge_segments( [f_g f_r f_l], seg0, wseg0, seg );
  f_ex_2= region_features_extra_rgb( seg );
  F_all{ii}=single([f_2 f_ex_2]');

  nseg=max(seg(:))+1;
  w=accumarray( seg(:)+1, ones(prod(size(seg)),1), [nseg 1] );
  W_all{ii}=w';
end
save(savefile,'F_all','L_all','W_all');

else

load(savefile);

end

