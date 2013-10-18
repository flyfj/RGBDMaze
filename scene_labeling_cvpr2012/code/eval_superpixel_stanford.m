

disp('evaluating pixel accuracy on test images...');

%visualize_output=0;
if visualize_output,
  visdir=[rootdir '/vis_results/ucm' k_ucm_id];
  system(['mkdir -p ' visdir]);
end

n=zeros(nclass,nclass);
count=0;
for i=1:length(test_list),
  ii=test_list(i);
  id=num2str(ii,'%08d');
  load([segdir '/' id '.mat'],'seg');
  nseg=max(seg(:))+1;
  label=load([labeldir '/' id '.txt'])+1;
  p=pred(count+1:count+nseg);
  for s=0:nseg-1,
    ind=find(seg==s & label>0);
    n=n+accumarray( [label(ind)+1 ones(size(ind))*p(s+1)], ones(size(ind)), [nclass nclass] );
  end
  count=count+nseg;
  img=im2double(imread([rootdir '/images/' num2str(ii,'%08d') '.jpg']));
  if visualize_output,
    [img_vis,err]=visualize_label_stanford( seg, label, p-1 );
    img_vis=[img img_vis];
    imwrite( img_vis, [visdir '/' num2str(err,'%08d') '_' num2str(ii,'%04d') '.jpg'], 'Quality', 90);
  end
end
c=n./repmat(sum(n,2),1,nclass);

n_pixel=n;
c_pixel=c;

n=n(2:end,2:end);
accu=sum(diag(n))/sum(n(:));
% accu=0.8217  for k_ucm=0.08



