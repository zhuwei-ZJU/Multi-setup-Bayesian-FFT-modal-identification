%% Verctorize the modal parameters in multiple setup model
%============================================================================
% 240225-Originally written by Wei at UIUC
%============================================================================
% Input
%============================================================================
% f: natural frequency, {ns,1} cell, each containing [m,1] vector
% z: damping ratio, {ns,1} cell, each containing [m,1] vector
% S: modal force PSD, {ns,1} cell, each containing [m,m] Hermitian matrix
% Se: prediction error PSD, {ns,1} cell, each containing a scalar
% PHI: mode shape, [n,m] matrix
%============================================================================
% Output
%============================================================================
% xx: [ns(m+1)^2+m*n,1] vector
%============================================================================
function xx = mpmat2vec_multi(f,z,S,Se,PHI)
ns = length(f);
m = size(PHI,2);
f = reshape(cell2mat(f),m,ns);   % [m,ns]
z = reshape(cell2mat(z),m,ns);   % [m,ns]
S = reshape(cell2mat(S)',m,m,ns); % [m,m,ns]
diagS = permute(real(pagediag(S)),[1,3,2]);   % [m,ns]
Sij = permute(pagevec(S,'hh'),[1,3,2]);   % [m*(m-1)/2,ns]
ReSij = real(Sij);   % [m*(m-1)/2,ns]
ImSij = imag(Sij);   % [m*(m-1)/2,ns]
Se = cell2mat(Se).'; % [1,ns]
xx = [vec([f;z;diagS;ReSij;ImSij;Se]);PHI(:)];
end