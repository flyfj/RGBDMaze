
savefile=[rootdir '/segmentation/map_segmentation_ks.mat'];

if ~exist(savefile),

disp('computing mappings between segmentations...');

nsegs=zeros(nframe,nlevel);
maps=cell(nframe,nlevel);

for ii=1:nframe,
  id=num2str(ii,'%08d');
    k_ucm=k_ucm_all(1);
    ucm_id=num2str( round(k_ucm*100),'%02d' );
    segdir_1=[rootdir '/segmentation/ucm' ucm_id];
  load([segdir_1 '/' id '.mat'],'seg');
  seg0=seg;
  nseg0=max(seg(:))+1;
  nsegs(ii,1)=nseg0;
  for ilevel=2:nlevel,
    k_ucm=k_ucm_all(ilevel);
    ucm_id=num2str( round(k_ucm*100),'%02d' );
    segdir_k=[rootdir '/segmentation/ucm' ucm_id];
    load([segdir_k '/' id '.mat'],'seg');
    nsegs(ii,ilevel)=max(seg(:))+1;
    wseg0 = get_kdes_weight_seg(seg0, 2, 16 );
    [dummy,dummy,map] = features_merge_segments( zeros(nseg0,1), seg0, wseg0, seg );
    maps{ii,ilevel}=map;
  end
  %if mod(ii,50)==0, disp(id); end;
end
  save(savefile,'nsegs','maps');
else
  load(savefile,'nsegs','maps');
end




