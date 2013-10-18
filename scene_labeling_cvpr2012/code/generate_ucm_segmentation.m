

k_ucm=round(k_ucm*100)/100;
k_ucm_id=num2str(k_ucm*100,'%02d');

segdir=[rootdir '/segmentation/ucm' k_ucm_id];
if ~exist(segdir),

system(['mkdir -p ' segdir]);

halfsize=findstr(ucm_dir,'half');
for ii=1:nframe,
  id=num2str(ii,'%08d');
  load([ucm_dir '/' id '.mat'],'ucm2');
  seg=bwlabel( ucm2<=k_ucm );
  seg=seg(2:2:end,2:2:end)-1;
  if halfsize, seg=imresize(seg,2,'nearest'); end;
  save([segdir '/' id '.mat'],'seg');
end

end

