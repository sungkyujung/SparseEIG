function [Q, cnt, cntVEC] = POI(B, A, lambda, r, option, maxIterInner,Q0)
% POI Penalized Orthogonal Iteration to solve GEV
% Q = POI(B, A, lambda, r)
% Q = POI(B, A, lambda, r, option)
% Q = POI(B, A, lambda, r, option, maxIterInner)
% Q = POI(B, A, lambda, r, option, maxIterInner,Q0)
% [Q, cnt, cntVEC] = POI(B, A, lambda, r, option, maxIterInner,Q0)
% input:
%  A: p-by-p symmetric matrix (need not be full rank),
%  B: p-by-p symmetric positive definite.
%  lambda: tuning parameters - a vector with length 1 or r (number of eigenvectors).
%  r: dimension of subspace (number of eigenvectors)
%  option =  POI-L, POI-C, FastPOI-L, or FastPOI-C (default);
%   A = Penalized Orthogonal Iteration with coefficient-wise L1 penalty (lasso)
%   C = Penalized Orthogonal Iteration with coordinate-wise L1 penalty (group lasso)
%   Da= Fast POI with coefficient-wise L1 penalty (lasso)
%   D =  Fast POI with coordinate-wise L1 penalty (group lasso)
%
%  maxIterInner: maximum number of iteration for cyclic descent
%               (default = 100)
%  Q0 : initial value (for warm start) (default = [])
%
% output
% Q : p-by-r orthogonal basis matrix
% cnt : numbers of numerical iterations, outer.
% cntVEC : numbers of numerical iterations, inner.
%
%
% See also POIcv, POIlim, POIv
%
% Last updated May 2018
% Sungkyu Jung


if nargin  < 5 
    maxIterInner = 100; 
    option = 'D'; 
end

if nargin  < 6 
    maxIterInner = 100; 
end

if strcmp(option,'POI-L'); option = 'A';end
if strcmp(option,'POI-C'); option = 'C';end
if strcmp(option,'FastPOI-L'); option = 'Da';end
if strcmp(option,'FastPOI-C'); option = 'D';end

% options for convergence
maxIter = 100;
optTol = 1e-9; % for convergence
cntVEC = [];
errorVEC = [];


if length(lambda) == 1
    lambda = repmat(lambda,r,1); % sequence needed only for POI with lasso
end

% the following shortcut is added for options A and C
if double(option(1)) < double('D')
    
    % if lambda == 0 then skip iterations and return the naive eigenvectors
    if norm(lambda) == 0 ;
        [Q, ~] = eigsadj(A, B, r);
        [Q,~] = qr(Q,0);
        cnt = 0;
        return;
    end
    
    % set initial value of Q. Only needed for Penalized Orthogonal Iterations.
    if nargin > 6 % then set the input Q0 as the initial Q
        Qold = Q0;
    else    %if isempty(Qold) % then the initial Q is set as the naive eigenvalue.
        [Q, ~] = eigs(A, B, r);
        [Qold,~] = qr(Q,0);
    end
end

% initialization for all options.
[p,~] = size(A);
Z = zeros(p,r);
Bii = diag(B);

switch option
    case 'A' % POI-L 
        % outer loop
        for cnt = 1:maxIter
            for j = 1:r       % This problem separates to r sub-problems.
                z = Qold(:,j);   % initial value for z := z_j
                aTq = A * z;     % a_i' q_j
                for cnt2 = 1:maxIterInner
                    zold = z;
                    for i = 1:p
                        Si = aTq(i) - B(i,:)*z  + Bii(i) * z(i) ;
                        z(i) = sign(Si) * max( abs(Si) - lambda(j) , 0) / Bii(i);
                    end
                    if norm(zold - z) < optTol
                        break;
                    end
                end
                Z(:,j) = z;
                cntVEC(j,cnt) = cnt2;
            end
            
            % Step 2: QR decomposition
            % intervention for zero column.
            zeroFlag = (sum(abs(Z)) == 0) ;
            r_zero = sum(zeroFlag);
            if r_zero > 0
                [Qnonzero,~]=qr(Z(:,zeroFlag~=1), 0);
                Q = zeros(p,r);
                Q(:,zeroFlag~=1) = Qnonzero;
            else
                [Q,~] = qr(Z,0);
            end
            
            % Check termination
            errorVEC(cnt) = norm( Q * Q' - Qold * Qold' );
            if errorVEC(cnt) < optTol;
                % disp(['Exit normally with error ' num2str(norm( Q * Q' - Qold * Qold' )) ])
                break;
            end
            % disp(['Next outer iteration with error ' num2str(norm( Q * Q' - Qold * Qold' )) ])
            Qold = Q;
        end
        
        
    case 'C' % POI-C
        % outer loop
        for cnt = 1:maxIter
            lambda = lambda(1); % we need only one tuning parameter
            Z = Qold; % initialize
            deltaMat_TR = zeros(p,r);
            for i = 1:p
                deltaMat_TR(i,:) = A(i,:)*Qold;
            end
            % delta i = deltaMat_TR(i,:)';
            for cnt2 = 1:maxIterInner
                Zold = Z;
                for g = 1:p
                    ag_tr = deltaMat_TR(g,:) - B(g,:) * Z + Bii(g) * Z(g,:);
                    Z(g,:) = ag_tr * max(1 - lambda / norm(ag_tr), 0) / Bii(g);
                end
                if norm(Zold - Z) < optTol
                    break;
                end
            end
            cntVEC(cnt) = cnt2;
            
            % Step 2: QR decomposition
            % intervention for zero column.
            zeroFlag = (sum(abs(Z)) == 0) ;
            r_zero = sum(zeroFlag);
            if r_zero > 0
                [Qnonzero,~]=qr(Z(:,zeroFlag~=1), 0);
                Q = zeros(p,r);
                Q(:,zeroFlag~=1) = Qnonzero;
            else
                [Q,~] = qr(Z,0);
            end
            
            % Check termination
            errorVEC(cnt) = norm( Q * Q' - Qold * Qold' );
            if errorVEC(cnt) < optTol;
                % disp(['Exit normally with error ' num2str(norm( Q * Q' - Qold * Qold' )) ])
                break;
            end
            % disp(['Next outer iteration with error ' num2str(norm( Q * Q' - Qold * Qold' )) ])
            Qold = Q;
        end
        
    case 'D' % FastPOI-C
        % no outer loop
        for cnt = 1:1           
        [deltaMat_TR,~] = eigs(A,r);
                    lambda = lambda(1); % we need only one tuning parameter
                    Z = deltaMat_TR; % initialize
                    for cnt2 = 1:maxIterInner
                        Zold = Z;
                        for g = 1:p
                            ag_tr = deltaMat_TR(g,:) - B(g,:) * Z + Bii(g) * Z(g,:);
                            Z(g,:) = ag_tr * max(1 - lambda / norm(ag_tr), 0) / Bii(g);
                        end
                        if norm(Zold - Z) < optTol
                            break;
                        end
                    end
                    cntVEC = cnt2;
        
            % Step 2: QR decomposition
            % intervention for zero column.
            zeroFlag = (sum(abs(Z)) == 0) ;
            r_zero = sum(zeroFlag);
            if r_zero > 0
                [Qnonzero,~]=qr(Z(:,zeroFlag~=1), 0);
                Q = zeros(p,r);
                Q(:,zeroFlag~=1) = Qnonzero;
            else
                [Q,~] = qr(Z,0);
            end
        end
       
    case 'Da' % FastPOI-L
        % no outer loop
        for cnt = 1:1           
        [deltaMat_TR,~] = eigs(A,r);                    
        for j = 1:r       % This problem separates to r sub-problems.                        
                        z = deltaMat_TR(:,j);   % initial value for z := z_j
                        aTq = z;     % a_i' q_j
                        for cnt2 = 1:maxIterInner
                            zold = z;
                            for i = 1:p
                                Si = aTq(i) - B(i,:)*z  + Bii(i) * z(i) ;
                                z(i) = sign(Si) * max( abs(Si) - lambda(j) , 0) / Bii(i);
                            end
                            if norm(zold - z) < optTol
                                break;
                            end
                            
                        end
                        Z(:,j) = z;
                        cntVEC(j) = cnt2;
            end
        
            % Step 2: QR decomposition
            % intervention for zero column.
            zeroFlag = (sum(abs(Z)) == 0) ;
            r_zero = sum(zeroFlag);
            if r_zero > 0
                [Qnonzero,~]=qr(Z(:,zeroFlag~=1), 0);
                Q = zeros(p,r);
                Q(:,zeroFlag~=1) = Qnonzero;
            else
                [Q,~] = qr(Z,0);
            end
        end                    
                    
end   
end
        
% adjusted generalized eigenvalue decomposition 

function [T1,D1] = eigsadj(A1,B1,k)

if nargin == 2
[T1, D1] = eigs(A1, B1);
else  
[T1, D1] = eigs(A1, B1,k);  
end

tmp = diag(T1'*B1*T1);
for kk = 1:k
    T1(:,kk) = T1(:,kk) ./ sqrt(tmp(kk));
end
end





