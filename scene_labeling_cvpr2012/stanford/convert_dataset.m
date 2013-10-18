
if ~exist('iccv09Data.tar.gz'),
  system('wget http://dags.stanford.edu/data/iccv09Data.tar.gz');
end

system('tar xvf iccv09Data.tar.gz');

indir='./iccv09Data/images/';
outdir='./images/';
  system(['mkdir -p ' outdir]);
files=dir([indir '/*jpg']);
assert(length(files)==715);
for ii=1:length(files),
  id=num2str(ii,'%08d');
  system(['cp ' indir '/' files(ii).name ' ' outdir '/' id '.jpg']);
end

indir='./iccv09Data/labels/';
outdir='./labels/';
  system(['mkdir -p ' outdir]);
files=dir([indir '/*regions.txt']);
assert(length(files)==715);
for ii=1:length(files),
  id=num2str(ii,'%08d');
  system(['cp ' indir '/' files(ii).name ' ' outdir '/' id '.txt']);
end

disp('DONE.');




