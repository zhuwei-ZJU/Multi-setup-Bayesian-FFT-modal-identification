%% Adopt the EM algorithm for one step in the multiple setup Bayesian model
%============================================================================
% 240521-Originally written by Wei at UIUC
%============================================================================
% Input
%============================================================================
% xx0: Current parameter set, [ntheta,1], ntheta = ns*(m+1)^2+m*n
% ff: Frequency, {1,ns}, [nf,1]
% F: FFT of measured data, {1,ns}, [nf,nr]
% C: selection matrix, {1,ns} [nr,n]
%============================================================================
% Output
%============================================================================
% xx1: Updated parameter set, [ntheta,1]
%============================================================================
function xx1 = onestepEM_multi(xx0,ff,F,C,f1f2)
% not only 1 step Newton
ns = length(C);     % number of setups
n = size(C{1},2);   % number of dofs
m = roots([ns,2*ns+n,ns-length(xx0)]); % number of modes
m = round(m(m>0));
Nf = cellfun(@(x) size(x,1),F);   % number of fft points
nnr = cellfun(@(x) size(x,1),C);  % number of dofs in each setup
ns = (length(xx0)-m*n)/(m+1)^2;   % number of setups
F3 = cellfun(@(x)reshape(x.',size(x,2),1,size(x,1)), ...
    F,'UniformOutput',false); % {1,ns},[nr,1,nf]
[f,z,S,Se,PHI] = mpvec2mat_mult(xx0,n,m);
fzt = cellfun(@vertcat,f,z,'UniformOutput',false);
[phi,fztp1,ftp1,ztp1,QfunHt,QfunHtp1,dQ2,ddQ2,beta,ih] = deal(cell(1,ns));
for r = 1:ns
    phi{r} = C{r}*PHI;
end
% E step
[Mu,~,Mu2,Nu2] = cellfun(@EYitak12,f,z,phi,S,Se,ff,F,'UniformOutput',false);
M = cellfun(@(x) real(sum(x,3)),Mu2,'UniformOutput',false); % {1,ns} cell, [m,m]
N = cellfun(@(x,y) real(sum(pagemtimes(conj(x),'none',y,'transpose'),3)), ...
    F3,Mu,'UniformOutput',false); % {1,ns} cell, [nr,m]
% M step

for r = 1:ns
    QfunHt{r} = calQfunH(ff{r},Mu2{r},f{r},z{r},S{r});
    tfz = 1;
    while tfz < 2
        %=====================================
        % Newton's method to optimize f and z
        %=====================================
        [dQ2{r},ddQ2{r}] = caldQfunH(ff{r},Mu2{r},Nu2{r},f{r},z{r},S{r});
        ddQ2{r} = (ddQ2{r}+ddQ2{r}.')/2;
%         dkfz = - ddQ2{r}\dQ2{r}.';
%         fun = @(alpha) calQfunH(fzt{r}+alpha*dkfz,phi{r},S{r},Se{r},ff{r},F{r});
%         dtfz = fminbnd(fun,0,1);
        fztp1{r} = fzt{r} - ddQ2{r}\dQ2{r}.';
        ftp1{r} = fztp1{r}(1:m);
        ztp1{r} = fztp1{r}(m+1:end);
%         ftp1{r} = f{r} - ddQ2{r}(1:m,1:m)\dQ2{r}(1:m).';
%         [dQ2{r},ddQ2{r}] = caldQfunH(ff{r},Mu2{r},Nu2{r},ftp1{r},z{r},S{r});
%         ztp1{r} = z{r} - ddQ2{r}(m+1:2*m,m+1:2*m)\dQ2{r}(m+1:2*m).';
        QfunHtp1{r} = calQfunH(ff{r},Mu2{r},ftp1{r},ztp1{r},S{r});
        if QfunHtp1{r}>QfunHt{r}||any(ftp1{r}<f1f2{r}(1))||...
                any(ftp1{r}>f1f2{r}(2))||any(ztp1{r}<0)||any(ztp1{r}>1)
%                 all(abs(fztp1{r}-fzt{r})./abs(fzt{r})<1e-3)
            break
        else
            fzt{r} = fztp1{r};
            f{r} = ftp1{r};
            z{r} = ztp1{r};
            QfunHt{r} = QfunHtp1{r};
            tfz = tfz+1;
        end
    end
    % update Se{r}
    Se{r} = 1/Nf(r)/nnr(r)*(sum(pagemtimes(F3{r},'ctranspose',F3{r},'none'),3)- ...
        2*trace(phi{r}*N{r}.') + trace(phi{r}*M{r}*phi{r}.'));
    % update S{r}
    beta{r} = f{r}./ff{r}.';   % [m,nf]
    ih{r}(:,1,:) = 1 - beta{r}.^2 - 1i*2*z{r}.*beta{r};
    S{r} = 1/Nf(r)*sum(ih{r}.*Mu2{r}.*pagectranspose(ih{r}),3);
    S{r} = (S{r}+S{r}')/2;
end
% {1,ns} cell, [m*n,m*n]
C1 = cellfun(@(a,b,c) 1./a*kron(b,c.'*c),Se,M,C,'UniformOutput',false);
% {1,ns} cell, [m*n,1]
C2 = cellfun(@(a,b,c) 1./a*reshape(b.'*c,[],1),Se,C,N,'UniformOutput',false);
vecPHI = sum(cell2mat(reshape(C1,1,1,[])),3)\sum(cell2mat(reshape(C2,1,1,[])),3);
PHI = reshape(vecPHI,n,m);
% renormalize the phi and S{r} to bypass the norm constraints
rnorm = sqrt(sum(PHI.^2));
PHI = PHI./rnorm;
for r = 1:ns
    S{r} = diag(rnorm)*S{r}*diag(rnorm);
    S{r} = (S{r}+S{r}')/2;
end
xx1 = mpmat2vec_multi(f.',z.',S.',Se.',PHI);
end