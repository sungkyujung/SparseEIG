function [GEPsolutions] = POIv(B, A, Q)
% POIv Approximate GEP solutions from eigenspace Q
% [GEPsolutions] = POIv(B, A, Q)
% input:
%  A: p-by-p symmetric matrix (need not be full rank),
%  B: p-by-p symmetric positive definite.
%  Q: p-by-k orthogonal matrix (output from POI) 
%
% output:
%  GEPsolutions: matlab structure array containing solutions of GEP.
%    1) GEPsolutions.U is the ordered eigenvectors 
%    2) GEPsolutions.Lambda is the k x k matrix of eigenvalues 
%                           (obtained by u'Au/u'Bu)
%    3) GEPsolutions.LambdaA is the k x k matrix of eigenvalues 
%                           (obtained by regression)
%    4) GEPsolutions.Lambdastar is the k x k matrix of eigenvalues 
%              (obsolete)
%
% See also POIcv, POIlim, POI
%
% Last updated May 2018
% Sungkyu Jung

[~,k] = size(Q);

% Handle the case where Q is not full rank 
%
zeroFlag = (sum(abs(Q)) == 0) ;
r_zero = sum(zeroFlag);
if r_zero == k        % Q is the zero matrix
    GEPsolutions.U =  Q ;
    GEPsolutions.Lambda = zeros(k); 
    GEPsolutions.Q = Q;
    GEPsolutions.LambdaA = zeros(k);
    GEPsolutions.Lambdastar = zeros(k);
    return;
elseif r_zero > 0   % Q has at least one zero column
    Q = Q(:,~zeroFlag);
    k = k-r_zero; 
end

% Obtain the ordered generalized eigenvectors from Q 
% (Proposition 1)
[T, Lambda] = eigsadj(Q' * A * Q , Q' * B * Q, k);
U = Q*T;
ll = diag(Lambda);
[llud, lludid]= sort(ll,'descend');
U = U(:,lludid);

% Now regression
% (eq 12) 
alpha = A * U;
beta = B * U;
LambdaA = zeros(k,k);
for i = 1:k
    LambdaA(i,i) = ( beta(:,i)' * beta(:,i) ) \ beta(:,i)' * alpha(:,i);
end

GEPsolutions.U =  U ;
GEPsolutions.Lambda = diag(llud);
GEPsolutions.LambdaA = LambdaA;

% Now get Q from U and LambdaStar; obsolete; eq. (21) in note_onSpGEP_20160513.pdf
[QQ,RR]=qr(U,0);
GEPsolutions.Q = QQ;
GEPsolutions.Lambdastar = RR * Lambda / RR; 




