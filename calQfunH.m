function Q2 = calQfunH(ff,Mu2,varargin)
%==========================================
% Input
%==========================================
% f: [m,1]
% z: [m,1]
% S: [m,m]
% ff: Frequency, [nf,1]
% Mu2(:,:,k) = E[Yitak*Yitak']
%==========================================
% Output
%==========================================
% Q2: Q-function value related to f and z
%==========================================
if nargin == 5
    f = varargin{1};
    z = varargin{2};
    S = varargin{3};
else
    m = size(Mu2,1);
    fzS = varargin{1};
    f = fzS(1:m);
    z = fzS(m+1:2*m);
    diagS = fzS(2*m+1:3*m);
    ReSij = fzS(3*m+1:3*m+m*(m-1)/2);
    ImSij = fzS(3*m+m*(m-1)/2+1:end);
    vecS = vec2diag(m).'*diagS + dup_mat(m,'hh')*ReSij + 1i*dup_mat(m,'sk')*ImSij;
    S = reshape(vecS,m,m);
    S = (S+S')/2;
end
beta(:,1,:) = f./ff.';   % [m,1,nf]
h = 1./(1 - beta.^2 - 1i*2*beta.*z);   % [m,1,nf]
H = h.*S.*pagectranspose(h);   % [m,m,nf]
Q2 = sum(log(pagedet(H))) + trace(sum(pagemtimes(pageinv(H),Mu2),3));
Q2 = real(Q2);
end