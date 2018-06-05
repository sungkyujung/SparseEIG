function [v,d,lambda,q] = seig(A,B,k,sp,option)
% SEIG Sparse Generalized eigendecomposition via Penalized Orthogonal
% Iteration (POI)
%
% This is a wrapper for a simple use of POI algorithm to solve a GEP. 
% 
% [v,d,lambda] = seig(A,B); A and B are symmetric non-negative definite matrices
% of the size p x p, v is the first eigenvector, d is the corresponding
% largest eigenvalue, lambda is the tuning parameter used.
% If B is the identity matrix, eye(p), or is empty,[], then this corresponds
% to the penalized solution to the eigenvalue decomposition. 
%
% [v,d,lambda,q] = seig(A,B,k); k is the dimension of the subspace, or the number of
% eigenvectors. By default, k = 1. For k > 1, v is the eigenvector matrix
% estimate of size p x k, and d is the eigenvalue matrix, the k x k diagonal
% matrix consisting of decreasing eigenvalue estimates. Here, q is the 
% orthogonal basis that spans columns of v (Note that v is only B-orthogonal).
% 
% [v,d] = seig(A,B,k,sp); sp is a "scaled" tuning parameter used in the
% penalization of POI. (sp stands for a sparsity parameter.) 
% For sp = 0, the tuning parameter lambda = 0. For sp in (0,1], lambda = sp
% * lambda_max, where lambda_max is the maximum nontrivial value of lambda, 
% obtained by POIlim.m. By default, sp = 1 / 2. 
% If sp = [], then the default value is used. We strongly 
% recommend to use the cross-validation in POIcv.m to tune lambda. 
%
% [v,d] = seig(A,B,k,sp,option), where option is a string designating
% the type of POI algorithm. By default, option = 'POI-C'. 
%  option =  'POI-L', 'POI-C', 'FastPOI-L', or 'FastPOI-C';
%   A = Penalized Orthogonal Iteration with coefficient-wise L1 penalty (lasso)
%   C = Penalized Orthogonal Iteration with coordinate-wise L1 penalty (group lasso)
%   Da= Fast POI with coefficient-wise L1 penalty (lasso)
%   D =  Fast POI with coordinate-wise L1 penalty (group lasso)
%  
% See also POI, POIlim, POIv, POIcv. 
%
% Last updated May 2018
% Sungkyu Jung

default.option = 'POI-C';
default.k = 1; 
if nargin < 3; 
    k = default.k; 
    option = default.option;
    sp = 1/2 ; 
elseif nargin < 4; 
    option = default.option;
    sp = 1/2 ; 
elseif nargin < 5;
    option = default.option;
end
    
if isempty(sp)
    sp = 1/2 ; 
end

lambda = POIlim(A,option,k)*sp; 

if rank(B) < size(B,1)
    p = size(B,1); 
    B = B + log(p)/rank(B) * eye(p);
end

maxIterInner =500;
Q = POI(B, A, lambda, k, option, maxIterInner);
gepsolutions=POIv(B,A,Q);
v = gepsolutions.U;
d = gepsolutions.Lambda;
q = gepsolutions.Q;
