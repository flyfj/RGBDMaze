
function normal2 = fix_normal_orientation( normal, pcloud )
%
% function normal2 = fix_normal_orientation( normal, pcloud )
%
%    make normal always point at origin
%

ndim=ndims(normal);
assert( ndim==ndims(pcloud) );
assert( size(normal,ndim)==3 );
assert( size(pcloud,ndim)==3 );

s=dot( pcloud, normal, ndim );
normal2=normal.*repmat( sign(s), [ones(1,ndim-1) 3] );


