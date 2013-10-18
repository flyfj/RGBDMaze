
gkdes_words=load_kdes_words('gkdes',0.001);
rgbkdes_words=load_kdes_words('rgbkdes',0.01);
lbpkdes_words=load_kdes_words('lbpkdes',0.01);

disp('computing kdes features for base superpixels... (could take a while)');

% run matlabpool first
parfor ii=1:nframe,

id=num2str(ii,'%08d');
savefile=[basefeatdir '/' id '.mat'];

if exist(savefile), continue; end;

img=im2double(imread([imgdir '/' id '.jpg']));
data=load([basesegdir '/' id '.mat'],'seg');
seg=data.seg;

% compute kdes features on seg
feaSet=gkdes_dense( img, gkdes_words.params, gkdes_words.grid_space );
feaSet.feaArr{1}=single(feaSet.feaArr{1});
gkdes=cksvd_emk_seg( feaSet, gkdes_words.words, gkdes_words.G, seg, gkdes_words.ktype, gkdes_words.kparam );

feaSet=rgbkdes_dense( img, rgbkdes_words.params, rgbkdes_words.grid_space );
feaSet.feaArr{1}=single(feaSet.feaArr{1});
rgbkdes=cksvd_emk_seg( feaSet, rgbkdes_words.words, rgbkdes_words.G, seg, rgbkdes_words.ktype, rgbkdes_words.kparam );

feaSet=lbpkdes_dense( img, lbpkdes_words.params, lbpkdes_words.grid_space );
feaSet.feaArr{1}=single(feaSet.feaArr{1});
lbpkdes=cksvd_emk_seg( feaSet, lbpkdes_words.words, lbpkdes_words.G, seg, lbpkdes_words.ktype, lbpkdes_words.kparam );

f_ex=region_features_extra_rgb( seg );

gkdes=single(gkdes); rgbkdes=single(rgbkdes); lbpkdes=single(lbpkdes);

save_feature_rgb(savefile,gkdes,rgbkdes,lbpkdes,f_ex);

end    %  ii



