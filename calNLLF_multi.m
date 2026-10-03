function L = calNLLF_multi(xx,C,ff,F)
ntheta = length(xx);
ns = length(C);
n = size(C{1},2);
m = roots([ns,2*ns+n,ns-ntheta]);
m = m(m>0);
[f,z,S,Se,PHI] = mpvec2mat_mult(xx,n,m);
phi = cell(1,ns);
for r = 1:ns
    phi{r} = C{r}*PHI;
end
L = sum(cellfun(@calNLLF,ff,F,f,z,phi,S,Se));
end
 