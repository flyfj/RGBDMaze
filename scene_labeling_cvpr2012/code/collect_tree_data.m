
%list=train_list;

clear -f nfeature ntotal count;

nlevel=length(k_ucm_all);

for ilevel=1:nlevel,
  k_ucm=k_ucm_all(ilevel);
  k_ucm_id=num2str(round(k_ucm*100),'%02d');
  featdir=[rootdir '/features/ucm' k_ucm_id];
  feature_file=[featdir '/features_kucm' k_ucm_id '.mat'];
    if ~exist(feature_file), error('feature file not found'); end;
  load(feature_file,'F_all','L_all','W_all');
  model_file=[savedir '/model_kucm' k_ucm_id '_ScaleEx_e4.mat'];
    if ~exist(model_file), error('model file not found'); end;
  load(model_file,'model','Fmax');

    F=horzcat(F_all{list})';
    L=horzcat(L_all{list})';
    W=horzcat(W_all{list})';
    if exist('use_stanford','var') & use_stanford,
      F(:,end-10:end)=F(:,end-10:end)./repmat(Fmax,size(F,1),1);
    end
    if exist('use_nyu_depth','var') & use_nyu_depth,
      F(:,1:end-15)=power(abs(F(:,1:end-15)),0.3).*sign(F(:,1:end-15));
      F(:,end-14:end)=F(:,end-14:end)./repmat(Fmax,size(F,1),1);
    end

  nclass=max(L)+1;
  np=length(L);
  ps=zeros(np,nclass-1);
  for c=2:nclass,
    ind=find(model.Label==(c-1));
    ps(:,c-1)=F*model.w(ind,:)';
  end

  if ilevel==1,
    ps_all{ilevel}=ps;
    L1=L;
    W1=W;
    nfeature=size(F,2);
    ntotal=size(F,1);
  else
    ps1=zeros(ntotal,nclass-1);
    count=zeros(2,1);
    for k=1:length(list),
      ii=list(k);
      f=ps(count(2)+1:count(2)+nsegs(ii,ilevel),:);
      ps1(count(1)+1:count(1)+nsegs(ii,1),:)=f(maps{ii,ilevel}+1,:);
      count(1)=count(1)+nsegs(ii,1);
      count(2)=count(2)+nsegs(ii,ilevel);
    end
    ps_all{ilevel}=ps1;
  end
end

F=horzcat(ps_all{:});
W=W1;
L=L1;

% return F1,W1,L1,ps_all

