%% Compute the gradient and Hessian, consistent for the Schur Complement theory
%============================================================================
% 240614-Originally written by Wei at UIUC
% 240619-Allow optional output
%============================================================================
% Input
%============================================================================
% f: natural frequency {1,ns} cell, each contains: [m,1]
% z: damping ratio {1,ns} cell, each contains: [m,1]
% S: modal force PSD {1,ns} cell, each contains: [m,m]
% Se: prediction error {1,ns} cell, each contains: scalar
% PHI: global mode shape [n,m]
% ff: Frequency, {1,ns}, [nf,1]
% F: FFT of measured data, {1,ns}, [nf,nr]
% C: selection matrix, {1,ns} [nr,n]
%============================================================================
% Output
%============================================================================
% gX: gradient of other parameters, [1,ns*(m+1)^2]
% gPHI: gradient of mode shape, [1,m*n]
% HXX => (d/dvec(X))(dL/dvec(X).'), [ns(m+1)^2,ns(m+1)^2]
% HPHIX => (d/dvec(X))(dL/dvec(PHI).'), [mn,ns(m+1)^2]
% HPHIPHI => (d/dvec(PHI))(dL/dvec(PHI).'), [mn,mn]
%============================================================================
function [gX,gPHI,HXX,HPHIX,HPHIPHI] = calgradHess_multi(f,z,S,Se,PHI,ff,F,C)
ns = length(f);
[n,m] = size(PHI);
nXr = (m+1)^2;
gX = zeros(1,ns*nXr);
gPHI = zeros(1,m*n);
if nargout == 2
    for r = 1:ns
        phir = C{r}*PHI;
        % Index of X and PHI in single and multiple setup
        nr = size(phir,1);
        aa = repelem((0:n:(m-1)*n),nr).';
        IXr = [1:2*m,2*m+m*nr+1:(m+1)^2+m*nr];
        IPHIr = 2*m+1:2*m+m*nr;
        IIXr = (r-1)*nXr+1:r*nXr;
        % Index of new columns or rows in global g and H
        [II0,~] = find(C{r}.');   % [nr,1]
        gr = NLLFHess_scalarSe(f{r},z{r},phir,S{r},Se{r},ff{r},F{r});
        % Gradient
        gX(IIXr) = gr(IXr);
        gPHI(repmat(II0,m,1)+aa) = gPHI(repmat(II0,m,1)+aa) + gr(IPHIr);
    end
else
    HXX = sparse(ns*nXr,ns*nXr);
    HPHIX = sparse(m*n,ns*nXr);
    HPHIPHI = sparse(m*n,m*n);
    for r = 1:ns
        phir = C{r}*PHI;
        % Index of X and PHI in single and multiple setup
        nr = size(phir,1);
        aa = repelem((0:n:(m-1)*n),nr).';
        IXr = [1:2*m,2*m+m*nr+1:(m+1)^2+m*nr];
        IPHIr = 2*m+1:2*m+m*nr;
        IIXr = (r-1)*nXr+1:r*nXr;
        % Index of new columns or rows in global g and H
        [II0,~] = find(C{r}.');   % [nr,1]
        [gr,Hr] = NLLFHess_scalarSe(f{r},z{r},phir,S{r},Se{r},ff{r},F{r});
        % Gradient
        gX(IIXr) = gr(IXr);
        gPHI(repmat(II0,m,1)+aa) = gPHI(repmat(II0,m,1)+aa) + gr(IPHIr);
        % Hessian
        HXX = HXX + sparse(repmat(IIXr,1,nXr),repelem(IIXr,nXr),Hr(IXr,IXr),ns*nXr,ns*nXr);
        IIPHIX = repmat(II0,m,nXr)+aa;
        HPHIX = HPHIX + sparse(IIPHIX,repmat(IIXr,m*nr,1),Hr(IPHIr,IXr),m*n,ns*nXr);
        IIPHIPHI = repmat(II0,m,nr*m)+aa;
        HPHIPHI = HPHIPHI + sparse(IIPHIPHI,IIPHIPHI.',Hr(IPHIr,IPHIr),m*n,m*n);
    end
    HXX = (HXX+HXX.')/2;
    HPHIPHI = (HPHIPHI+HPHIPHI.')/2;
end
end