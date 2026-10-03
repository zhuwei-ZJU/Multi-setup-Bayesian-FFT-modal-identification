function out = bayoma_oneband_EMNewton_multi(in)
% update f{r},z{r},S{r},Se{r} PHI together
% Nondimensionalize the gradient and Hessian by EM convergence result
% remove the in.scale

% use sparse form of H 
%=========================================
% Data preparation
%=========================================
C = in.selmat;
ns = length(in.data);
f1f2 = cellfun(@(x) x.f1f2,in.mode,'UniformOutput',false);
f1f2mat = cell2mat(f1f2.'); % [1,2*ns]
f1 = f1f2mat(:,1);  % [ns,1]
f2 = f1f2mat(:,2);  % [ns,1]
f0 = cellfun(@(x) x.f0,in.mode,'UniformOutput',false);
fs = cell2mat(cellfun(@(x) x.fs,in.data,'UniformOutput',false));
% detrend and normalization
[F,ff] = deal(cell(1,ns));
ntSetup = zeros(1,ns);
for r = 1:ns
    in.data{r}.tdata = detrend(in.data{r}.tdata);
    nt = size(in.data{r}.tdata,1);
    ntSetup(r) = nt;
    nf0 = floor((nt+1)/2); % number of ffts up to Nyquist
    ff0 = (0:nf0-1).'*fs(r)/nt; % (nf0,1), ordinates of freq.
    F0 = sqrt(2/fs(r)/nt)*fft(in.data{r}.tdata);    % scaled FFT - two sided
    F{r} = F0(2:nf0,:);
    ff{r} = ff0(2:nf0); % (:,1)
    I1I2 = round(f1f2{r}*nt/fs(r));
    II = I1I2(1):I1I2(2);
    F{r} = F{r}(II,:); % (nf0,n)
    ff{r} = ff{r}(II); % (nf0,1)
    in.data{r} = rmfield(in.data{r},'tdata');
end
F3 = cellfun(@(x)reshape(x.',size(x,2),1,size(x,1)), ...
    F,'UniformOutput',false); % {1,ns},[nr,1,nf]
%=========================================
% Initial setting
%=========================================
n = size(C{1},2);   % number of dofs
nnr = cellfun(@(x) size(x,1),C);  % number of dofs in each setup
Nf = cellfun(@(x) size(x,1),F);   % number of fft points
z0 = 0.01;
m = length(f0{1}{1}); % number of modes
nXr = (m+1)^2;   % number of parameters other than mode shapes in each setup
[f,z,phi,S,Se] = deal(cell(1,ns));
% Obtain each setup's local spectral directions, then assemble the
% global initial shape using only the supplied selection matrices.
localPhi = cell(1,ns);
for r = 1:ns
    f{r} = f0{r}{1}.';
    z{r} = z0*ones(m,1);
    nr = nnr(r);
    [~, I0] = min(abs(ff{r} - f{r}(1)));
    dn0 = max(1,round(0.005*ntSetup(r)*f{r}(1)/fs(r)));
    II = max(1,I0-dn0):min(Nf(r),I0+dn0);
    DD = real(F{r}(II,:).' * conj(F{r}(II,:)))/numel(II);
    DD = (DD + DD.')/2;
    [vtmp,~] = svd(DD);
    localPhi{r} = vtmp(:,1:m);
end
PHI = initialModeShapeFromSelmat(C,F,localPhi);
for r = 1:ns
    phi{r} = C{r}*PHI;
    % initialize S
    beta00 = f{r}./ff{r}.';   % [m,nf]
    ih00 = reshape(1 - beta00.^2 - 1i*2*z{r}.*beta00,m,1,Nf(r));  % [m,1,nf]
    hPPF = pagemtimes(ih00.*((phi{r}.'*phi{r})\phi{r}.'),F3{r});
    S{r} = sum(pagemtimes(hPPF,'none',hPPF,'ctranspose'),3)/Nf(r);
    S{r} = (S{r}+S{r}')/2;
    % initialize Se
    if nr == m
        Se{r} = geomean(diag(S{r}));
    else
        A1 = sum(pagemtimes(F3{r},'ctranspose',F3{r},'none'));
        PPrr = phi{r}/(phi{r}.'*phi{r})*phi{r}.';
        A2 = sum(pagemtimes2(F3{r},'ctranspose',PPrr,'none',F3{r},'none'));
        Se{r} = real((A1-A2)/(nr-m))/Nf(r);
    end
end

% Optimization
%=========================================
% Part I: EM (f,z updated by Newton)
%=========================================
nIterEM = in.maxEMIter;
tEM = 1;
xxtEM = mpmat2vec_multi(f.',z.',S.',Se.',PHI);
nIterNewton = in.maxNewtonIter;
tNewton = 1;
ntheta = ns*((m+1)^2)+m*n;
XX = zeros(ntheta,nIterEM+nIterNewton+1);
XX(:,1) = xxtEM;
L = zeros(nIterEM+nIterNewton+1,1);
L1 = cellfun(@calNLLF,ff,F,f,z,phi,S,Se);   % [1,ns]
L(1) = sum(L1);
disp('EM algorithm begins...');
fprintf(['IterEM.\tfreq (Hz) ',repmat(' ',1,12*(m-1))]);
fprintf(['damping   ',repmat(' ',1,10*(m-1))]);
fprintf('Se (g^2/Hz)       ');
fprintf(['Sii (g^2/Hz)      ',repmat(' ',1,10*(m-1))]);
fprintf('\n');
tic;
while tEM <= nIterEM
    fprintf('%-3i\t',tEM-1);
    fprintf('%-6.3e ',[f{in.dispsetup}(:).',z{in.dispsetup}(:).', ...
        Se{in.dispsetup},diag(S{in.dispsetup}).']);
    fprintf('\n');
    xxtp1EM = onestepEM_multi(xxtEM,ff,F,C,f1f2);
    [f,z,S,Se,PHI] = mpvec2mat_mult(xxtp1EM,n,m);
    for r = 1:ns
        phi{r} = C{r}*PHI;
    end
    LtEM = cellfun(@calNLLF,ff,F,f,z,phi,S,Se);   % [1,ns]
    L(tEM+1) = sum(LtEM);
    XX(:,tEM+1) = xxtp1EM;
    % Covergence test
    if abs(L(tEM+1)-L(tEM))/abs(L(tEM)) < in.tol_cvg && tEM >= in.minEMIter
        fprintf(['EM part stops as the change of NLLF value is less than ' ...
            'a relative tolerance of %5.3e\n'],in.tol_cvg);
        tEM = tEM + 1;
        break;
    else
        xxtEM = xxtp1EM;
    end
    tEM = tEM + 1;
end
time_EM = toc;
if tEM > nIterEM
    fprintf('EM part stops as it meets a maximum iteration of %5.3e\n',in.maxEMIter);
end
disp(['Elapsed time is ',num2str(time_EM,8),' seconds.']);
%=========================================
% Part II: Newton's method
%=========================================
disp('Newton''s method begins...');
fprintf(['IterNewton.\tfreq (Hz) ',repmat(' ',1,10*(m-1))]);
fprintf(['damping   ',repmat(' ',1,10*(m-1))]);
fprintf('Se        ');
fprintf(['Sii       ',repmat(' ',1,10*(m-1))]);
fprintf('\n');
tic
IIPHI = ntheta-m*n+1:ntheta;
xx0 = XX(:,tEM);
xxtNewton = ones(ntheta,1);
GRAD = zeros(tEM+nIterNewton+1,ntheta);
steplen = zeros(nIterNewton,1);
Lt = L(tEM);
PHIt = PHI;

% Imc = arrayfun(@(x) eye(m),1:ns,'UniformOutput',false);
% K = cellfun(@kron,Imc,C,'UniformOutput',false);   % {ns,1} cell, [m*nr,m*n]
% g = zeros(1,ntheta);
% H = zeros(ntheta,ntheta);
while tNewton <= nIterNewton
    fprintf('%-3i\t',tNewton-1);
    fprintf('%-6.3e ',[f{in.dispsetup}(:).',z{in.dispsetup}(:).', ...
        Se{in.dispsetup},diag(S{in.dispsetup}).']);
    fprintf('\n');

    % for r = 1:ns
    %     % Index of X and PHI in single and multiple setup
    %     IXr = [1:2*m,2*m+m*nr+1:(m+1)^2+m*nr];
    %     IPHIr = 2*m+1:2*m+m*nr;
    %     IIXr = (r-1)*nXr+1:r*nXr;
    %     [gr,Hr] = NLLFHess_scalarSe(f{r},z{r},phi{r},S{r},Se{r},ff{r},F{r});
    %     % Gradient
    %     g(IIXr) = gr(IXr);
    %     g(IIPHI) = g(IIPHI) + gr(IPHIr)*K{r};
    %     % Hessian matrix
    %     H(IIXr,IIXr) = Hr(IXr,IXr);
    %     H(IIXr,IIPHI) = H(IIXr,IIPHI) + Hr(IXr,IPHIr)*K{r};
    %     H(IIPHI,IIXr) = H(IIXr,IIPHI).';
    %     H(IIPHI,IIPHI) = H(IIPHI,IIPHI) + K{r}.'*Hr(IPHIr,IPHIr)*K{r};
    % end
    % g = g.*xx0.';
    % H = xx0.*H.*xx0.';

    [gX,gPHI,HXX,HPHIX,HPHIPHI] = calgradHess_multi(f,z,S,Se,PHI,ff,F,C);
    g = [gX,gPHI].*xx0.';
    GRAD(tEM+tNewton,:) = g;
    H = xx0.*[HXX,HPHIX.';HPHIX,HPHIPHI].*xx0.';
    H = (H+H.')/2;
    Jmat = zeros(m,ntheta);
    Jmat(:,IIPHI) = vec(PHI).'.*repelem(eye(m),1,n);
    U = null(Jmat.*xx0.');
    Hp = U.'*H*U;   % Hessian account for the anorm constraints
    Hp = (Hp+Hp.')/2; % make sure symmetric

    % revised Cholesky decomposition to make sure positive definite
    if any(eig(Hp)<0)
        [LHp,DHp,PHp,~] = modchol_ldlt(Hp);
        Hp = PHp.'*LHp*DHp*LHp.'*PHp;
        Hp = (Hp+Hp.')/2;
    end

    % % absolute eigvalue to ensure positive definite (worse than revised Cholesky)
    % if any(eig(Hp)<0)
    %     [VHp,DHp] = eig(Hp);
    %     Hp = VHp*abs(DHp)*VHp.';
    %     Hp = (Hp+Hp.')/2; % make sure symmetric
    % end

    PCM = U/Hp*U.';
    dt = 1;
    dk = -PCM*g.';

%     [VHp,DHp] = eig(Hp);
%     DHp = diag(DHp);
%     if any(DHp<=0)
%         iDHp = 1./abs(DHp);
%         PCM = U*VHp*diag(iDHp)*VHp.'*U.';
%     else
%         PCM = U/Hp*U.';
%     end
%     dt = 1;
%     dk = -PCM*g.';
%     steplenfun = @(alpha) calNLLF_multi(xxtNewton + alpha*dk,C,ff,F);
%     dt = fminbnd(steplenfun,0,1);
    while dt > in.steplentol
        xxtp1Newton = xxtNewton+dt*dk;
        [f,z,S,Se,PHI] = mpvec2mat_mult(xxtp1Newton.*xx0,n,m);
        % renormalize the phi and S{r} to bypass the norm constraints
        rnorm = sqrt(sum(PHI.^2));
        PHI = PHI./rnorm;
        for r = 1:ns
            S{r} = diag(rnorm)*S{r}*diag(rnorm);
            S{r} = (S{r}+S{r}')/2;
            phi{r} = C{r}*PHI;
        end
        xx = mpmat2vec_multi(f.',z.',S.',Se.',PHI);
        Ltp1 = sum(cellfun(@calNLLF,ff,F,f,z,phi,S,Se));
        fmat = cell2mat(f).';   % [ns,m]
        zmat = cell2mat(z).';   % [ns,m]
        Semat = cell2mat(Se);   % [1,ns]
        eigS = cell2mat(cellfun(@eig,S,'UniformOutput',false));   % [m,ns]
        cohS = cell2mat(cellfun(@(S) vec(abs(S./sqrt(diag(S)*diag(S).'))), ...
            S,'UniformOutput',false)); % [m^2,ns]
%         if m>1
%             aa = abs(calMAC(PHI));   % [m(m-1)/2,1]
%         end
        if Ltp1>Lt || any(fmat<f1,'all')||any(fmat>f2,'all')...
        ||any(zmat<0,'all')||any(zmat>1,'all')||any(Semat<0)...
        ||any(eigS<0,'all')||any(cohS>1,'all')
            dt = dt/2;
        else
            L(tEM+tNewton) = Ltp1;
            break;
        end
    end
    if any(dt==in.steplentol)
        warning('The step length reaches the bound, results are not reliable!');
    end
    steplen(tNewton) = dt;
    XX(:,tEM+tNewton) = xx;
    % Covergence test
%     INewton = find(xxtNewton);
    INewton = find(xxtNewton(1:ns*nXr));
    macPHI = calMAC(PHIt,PHI);
    if all(abs(xxtp1Newton(INewton)-xxtNewton(INewton))./abs(xxtNewton(INewton)) < in.tol_cvg)... 
        && all(abs(abs(diag(macPHI))-1) < in.tol_cvg) && (tNewton >= in.minNewtonIter)
        fprintf(['Newton part stops as the parameter changes are less than' ...
            ' a relative tolerance of %5.3e\n'],in.tol_cvg);
        tNewton = tNewton+1;
        break;
    else
        xxtNewton = xxtp1Newton;
        Lt = Ltp1;
        PHIt = PHI;
    end
    tNewton = tNewton+1;
end
time_Newton = toc;
if tNewton > nIterNewton
    fprintf('Newton part stops as it meets a maximum iteration of %5.3e\n',in.maxNewtonIter);
end
disp(['Elapsed time is ',num2str(time_Newton,8),' seconds.']);
%=========================================
% Set output
%=========================================
XX(:,tEM+tNewton:end) = [];
L(tEM+tNewton:end) = [];
GRAD(tEM+tNewton:end,:) = [];
for tt = 1:tEM
    [ftt,ztt,Stt,Sett,PHItt] = mpvec2mat_mult(XX(:,tt),n,m);
    [gXtt,gPHItt] = calgradHess_multi(ftt,ztt,Stt,Sett,PHItt,ff,F,C);
    GRAD(tt,:) = [gXtt,gPHItt].*xx0.';
end
steplen(tNewton:end,:) = [];
% Optimum of the modal parameters
for r = 1:ns
    [out.f{r},IIf] = sort(f{r}.');
    out.z{r} = z{r}(IIf).';
    out.phi{r} = phi{r}(:,IIf);
    out.S{r}{1} = S{r}(IIf,IIf);
    out.Sii{r} = real(diag(out.S{r}{1})).';
    out.Se{r} = Se{r}*ones(1,m);
    out.rms{r} = sqrt(pi/4*diag(out.S{r}{1}).'.*out.f{r}./out.z{r});
    out.sn{r} = out.Sii{r}/4./out.z{r}.^2./out.Se{r};
end
% Global mode shape
out.PHI = PHI.';
% Posterior COV of the modal parameters
PCM = xx0.*PCM.*xx0.';
diagPCM = diag(PCM).';
for r = 1:ns
    II0 = (r-1)*nXr;
    out.coefv{r}.f = sqrt(diagPCM(II0+1:II0+m))./f{r}.';
    out.coefv{r}.z = sqrt(diagPCM(II0+m+1:II0+2*m))./z{r}.';
    out.coefv{r}.Sii = sqrt(diagPCM(II0+2*m+1:II0+3*m))./real(diag(S{r})).';
    out.coefv{r}.Se = sqrt(diagPCM(II0+(m+1)^2))./Se{r}*ones(1,m);
    out.coefv{r}.phi = sqrt(sum(reshape(diagPCM(end-m*n+1:end)',n,m)));
end
% resort according to the value of the natural frequency
for r = 1:ns
    [out.f{r},IIf] = sort(f{r}.');
    out.coefv{r}.f = out.coefv{r}.f(IIf);
    out.coefv{r}.z = out.coefv{r}.z(IIf);
    out.coefv{r}.Sii = out.coefv{r}.Sii(IIf);
    out.coefv{r}.phi = out.coefv{r}.phi(IIf);
end
% Iterated parameters, function values, gradients and covriance matrices
out.XX{1} = real(XX);
out.loglik{1} = -L;
out.grad{1} = GRAD;
out.normgrad{1} = vecnorm(GRAD.').';
out.PCM{1} = PCM;
% Iterated step length
out.steplen{1} = steplen;
% Number of iterations and consuming time
out.nEMIter = tEM-1;
out.nNewtonIter = tNewton-1;
out.time_EM = time_EM;
out.time_Newton = time_Newton;
out.time_req = time_EM+time_Newton;
end