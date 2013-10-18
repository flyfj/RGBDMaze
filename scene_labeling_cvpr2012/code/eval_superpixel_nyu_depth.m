
disp('evaluating pixel accuracy on test images...');

n=zeros(nclass,nclass);
count=0;
for i=1:length(test_list),
  ii=test_list(i);
  id=num2str(ii,'%08d');
  load([segdir '/' id '.mat'],'seg');
  nseg=max(seg(:))+1;
  label=imread([labeldir '/' id '.png']);
  label=double(label)/15;
  p=pred(count+1:count+nseg);
  for s=0:nseg-1,
    ind=find(seg==s & label>0);
    n=n+accumarray( [label(ind)+1 ones(size(ind))*p(s+1)], ones(size(ind)), [nclass nclass] );
  end
  count=count+nseg;
end
c=n./repmat(sum(n,2),1,nclass);

n_pixel=n;
c_pixel=c;

c=c(2:end,2:end);
accu=mean(diag(c));


