%% Reshape the modal parameters in multiple setup model
%============================================================================
% 240225-Originally written by Wei at UIUC
% 240314-Make sure S is Hermitian
%============================================================================
% Input
%============================================================================
% xx: [ns(m+1)^2+m*n,1] vector
% n: number of measured dofs
% m: number of modes
%============================================================================
% Output
%============================================================================
% f: natural frequency, {1,ns} cell, each containing [m,1] vector
% z: damping ratio, {1,ns} cell, each containing [m,1] vector
% S: modal force PSD, {1,ns} cell, each containing [m,m] Hermitian matrix
% Se: prediction error PSD, {1,ns} cell, each containing a scalar
% PHI: mode shape, [n,m] matrix
%============================================================================
function [f,z,S,Se,PHI] = mpvec2mat_mult(xx,n,m)
ntheta = length(xx);
ns = (ntheta-m*n)/(m+1)^2;
mpM = reshape(xx(1:end-n*m),(m+1)^2,ns);
PHI = reshape(xx(end-n*m+1:end),n,m);
[f,z,S,Se] = deal(cell(1,ns));
for r = 1:ns
    mpC = mat2cell(mpM(:,r),[m,m,m,m*(m-1)/2,m*(m-1)/2,1]);
    [f{r},z{r},diagS,ReSij,ImSij,Se{r}] = deal(mpC{:});
    vecS = vec2diag(m).'*diagS + dup_mat(m,'hh')*ReSij + 1i*dup_mat(m,'sk')*ImSij;
    S{r} = reshape(vecS,m,m);
    S{r} = (S{r}+S{r}')/2;
end
end