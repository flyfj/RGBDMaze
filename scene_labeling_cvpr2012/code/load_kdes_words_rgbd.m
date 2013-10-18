
function kdes_words=load_kdes_words( kdesname, kparam )
%
% function kdes_words=load_kdes_words( kdesname, kparam )
%
%

kdes_words.ktype='rbf'; 
kdes_words.kparam=kparam;

load(['kdes_data/' kdesname '_params.mat'],[kdesname '_params']);
eval(['kdes_words.params=' kdesname '_params;']);

kdes_words.grid_space=2;

load(['kdes_data/' kdesname 'words_200_fergus.mat'],[kdesname '_words']);
  eval(['words=' kdesname '_words;']);
  K = eval_kernel(words',words',kdes_words.ktype,kdes_words.kparam);
  K = K + 1e-6*eye(size(K));
  G = chol(inv(K));
kdes_words.words=words;
kdes_words.G=G;


