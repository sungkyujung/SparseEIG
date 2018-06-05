function lmax = POIlim(A,option,k)
% POILIM returns the value of lambda for which GEPLassoSJ returns
% the trivial solution.
%
% lmax = POIlim(A) returns the upper bound of lambda, for one eigenvector
% estimation using POI-C.
% lmax = POIlim(A,option,k) Here, k (default 1) is the rank to be
% estimated, option is one of the following:
%  option =  'POI-L', 'POI-C' (default), 'FastPOI-L', or 'FastPOI-C'; or
%   'A' = Penalized Orthogonal Iteration with coefficient-wise L1 penalty (lasso)
%   'C' = Penalized Orthogonal Iteration with coordinate-wise L1 penalty (group lasso)
%   'Da' = Fast POI with coefficient-wise L1 penalty (lasso)
%   'D' =  Fast POI with coordinate-wise L1 penalty (group lasso)
%
%
% See also POIcv, POIlim, POIv
%
% Last updated May 2018
% Sungkyu Jung

if nargin <= 2
    k = 1;
end

if nargin == 1;
    option = 'C';
end

if strcmp(option,'POI-L'); option = 'A';end
if strcmp(option,'POI-C'); option = 'C';end
if strcmp(option,'FastPOI-L'); option = 'Da';end
if strcmp(option,'FastPOI-C'); option = 'D';end

 
if double(option(1)) < double('D')
    %  choice = 'AC';
    AnsortedSq = sort(abs(A),'descend').^2;
    AnsortedSqcumsum = cumsum(AnsortedSq);
    lmaxCvec = zeros(k,1); 
    for kk = 1:k
        [lmaxCvec(kk), ~] = max(sqrt( AnsortedSqcumsum(kk,:) ));
    end
else
    
    switch option
        case 'Da' %  options Da (FastPOI-L)
            [deltaMat_TR,~] = eigs(A,k);
            lmaxCvec = max(max(abs(deltaMat_TR)));
            
        case 'D' %  options D (FastPOI-C)
            [deltaMat_TR,~] = eigs(A,k);
            lmaxCvec = max(sqrt(sum(deltaMat_TR.^2,2)));
    end
end

if strcmp(option,'C')
    lmax = lmaxCvec(k);
else
    lmax = lmaxCvec(1);
end

% lmaxCvec = POIlim(A,'C',k), where lmaxCvec is a k-vector of upper bounds of
% lambdas. Here, k (default 1) is the rank to be estimated .
% For option 'A' in POI, lmaxA = lmaxCvec(1) is the smallest lambda
% for which the resulting Q is trivial (for ANY dimension k).
% For option 'C' in POI, lmaxC = lmaxCvec(k) provides an upper bound
% of lambda for which the resulting Q is non-trivial (for dimension k).
%
%
% lmaxCvec = POIlim(A,'D')
% lmaxCvec = POIlim(A,'Da',k)
% lmaxCvec = POIlim(A,'D',k)
% For option 'Da' in POI, lmaxA = lmaxCvec(1) is the smallest lambda
% for which the resulting Q is trivial (for ANY dimension k).
% For option 'D' in POI, lmaxC = lmaxCvec(1) provides an upper bound
% of lambda for which the resulting Q is non-trivial (for dimension k).
%
