function out = bayoma_EMNewton_multi(in)
% If you use this program, please cite:
% Wei Zhu, Bin-bin Li, Dong-Hui Yang, Ting-Hua Yi. Multi-setup Bayesian 
% FFT modal identification with multiple modes leveraging on 
% second order information[J]. 
% Mechanical Systems and Signal Processing, 2026, 261: 115020.

% Multi-setup Bayesian FFT modal identification by EM and Newton updates.
% Required input fields:
%   data{r}.tdata : samples x channels; data{r}.fs : sampling rate (Hz)
%   mode{r}.f1f2 : nband x 2 frequency-band limits (Hz)
%   mode{r}.f0   : nband x 1 cell of initial modal frequencies (Hz)
%   selmat{r}    : channels x global DOFs binary selection matrix
% mode may contain one cell, in which case it is reused for every setup.
% No XLS, DOF numbering, or setup file is required.
% Output fields and band-wise aggregation follow the original implementation.

% copyright statement
disp('============================================');
disp('Multi-setup BAYesian Operational Modal Analysis by EM and Newton updates');
disp('-- Wei Zhu (weizhu321@hotmail.com) - Oct 12, 2026');
disp('-- Binbin Li (bbl@zju.edu.cn) - Oct 12, 2026');
disp('============================================');

if ~isfield(in, 'tol_cvg') || isempty(in.tol_cvg), in.tol_cvg = 1e-3; end
if ~isfield(in, 'minEMIter') || isempty(in.minEMIter), in.minEMIter = 10; end
if ~isfield(in, 'maxEMIter') || isempty(in.maxEMIter), in.maxEMIter = 100; end
if ~isfield(in, 'maxfzIter') || isempty(in.maxfzIter), in.maxfzIter = 20; end
if ~isfield(in, 'steplentol') || isempty(in.steplentol), in.steplentol = eps; end
if ~isfield(in, 'minNewtonIter') || isempty(in.minNewtonIter), in.minNewtonIter = 1; end
if ~isfield(in, 'maxNewtonIter') || isempty(in.maxNewtonIter), in.maxNewtonIter = 100; end
if ~isfield(in, 'dispsetup') || isempty(in.dispsetup), in.dispsetup = 1; end

if ~isfield(in, 'data') || ~iscell(in.data) || isempty(in.data)
    error('in.data must be a nonempty cell array.');
end
if ~isfield(in, 'mode') || ~iscell(in.mode) || isempty(in.mode)
    error('in.mode must be a nonempty cell array.');
end
if ~isfield(in, 'selmat') || ~iscell(in.selmat)
    error('in.selmat must be a cell array of selection matrices.');
end
ns = numel(in.data);
if numel(in.selmat) ~= ns
    error('in.selmat must contain one matrix per setup.');
end
in.data = reshape(in.data, 1, []);
in.selmat = reshape(in.selmat, 1, []);
if isscalar(in.mode)
    in.mode = repmat(in.mode, 1, ns);
elseif numel(in.mode) ~= ns
    error('in.mode must contain one entry or one entry per setup.');
end
in.mode = reshape(in.mode, 1, []);
if ~isscalar(in.dispsetup) || in.dispsetup < 1 || ...
        in.dispsetup > ns || in.dispsetup ~= floor(in.dispsetup)
    error('in.dispsetup must be a valid setup index.');
end

n = size(in.selmat{1}, 2);
coverage = false(1, n);
for r = 1:ns
    data = in.data{r};
    mode = in.mode{r};
    C = in.selmat{r};
    if ~isstruct(data) || ~isfield(data, 'tdata') || ~isfield(data, 'fs') || ...
            ~isnumeric(data.tdata) || ~ismatrix(data.tdata) || ...
            ~isnumeric(data.fs) || isempty(data.fs)
        error('Each in.data entry must contain numeric tdata and fs.');
    end
    if numel(data.fs) > 1
        warning('Only fs(1) will be used in setup %d.', r);
    end
    in.data{r}.fs = data.fs(1);
    if ~isfinite(in.data{r}.fs) || in.data{r}.fs <= 0
        error('Sampling rate must be positive and finite in setup %d.', r);
    end
    if ~isnumeric(C) && ~islogical(C)
        error('Selection matrix in setup %d must be numeric or logical.', r);
    end
    if size(C, 2) ~= n || size(C, 1) ~= size(data.tdata, 2) || ...
            any(~isfinite(C(:))) || any(C(:) ~= 0 & C(:) ~= 1) || ...
            any(sum(C, 2) ~= 1) || nnz(sum(C, 1)) ~= size(C, 1)
        error(['selmat{%d} must have one unit entry per channel, ', ...
            'unique DOFs within the setup, and match tdata columns.'], r);
    end
    coverage = coverage | any(C, 1);
    if ~isstruct(mode) || ~isfield(mode, 'f1f2') || ~isfield(mode, 'f0') || ...
            ~isnumeric(mode.f1f2) || size(mode.f1f2, 2) ~= 2 || ...
            ~iscell(mode.f0) || size(mode.f1f2, 1) ~= numel(mode.f0)
        error('Each in.mode entry needs nband-by-2 f1f2 and nband f0 cells.');
    end
    if any(~isfinite(mode.f1f2(:))) || any(mode.f1f2(:,1) <= 0) || ...
            any(mode.f1f2(:,2) <= mode.f1f2(:,1)) || ...
            any(mode.f1f2(:,2) >= in.data{r}.fs / 2)
        error('Frequency bands must lie strictly between zero and Nyquist.');
    end
    if r == 1
        nband = numel(mode.f0);
        nmodes = cellfun(@numel, mode.f0);
    elseif numel(mode.f0) ~= nband || any(cellfun(@numel, mode.f0) ~= nmodes)
        error('Every setup must have the same bands and mode counts.');
    end
    for ib = 1:numel(mode.f0)
        initial = mode.f0{ib};
        if ~isnumeric(initial) || ~isreal(initial) || ...
                any(~isfinite(initial(:))) || ...
                any(initial(:) < mode.f1f2(ib,1)) || ...
                any(initial(:) > mode.f1f2(ib,2))
            error('Initial modal frequencies must be inside their bands.');
        end
        if numel(initial) > size(C, 1)
            error('Setup %d has fewer channels than modes in band %d.', r, ib);
        end
    end
end
if n == 0 || ~all(coverage)
    error('Every global DOF column of selmat must be measured.');
end
if nband == 0 || any(nmodes == 0)
    error('At least one frequency band and one mode per band are required.');
end

if isfield(in, 'iband') && ~isempty(in.iband)
    if ~isnumeric(in.iband) || any(~isfinite(in.iband)) || ...
            any(in.iband < 1) || any(in.iband > nband) || ...
            any(in.iband ~= floor(in.iband))
        error('in.iband must contain valid positive band indices.');
    end
    for r = 1:ns
        in.mode{r}.f1f2 = in.mode{r}.f1f2(in.iband, :);
        in.mode{r}.f0 = in.mode{r}.f0(in.iband);
    end
    nb = numel(in.iband);
else
    nb = nband;
end

% Calculate MPV and PCM independently in every requested frequency band.
tempinmode = in.mode;
outib = cell(nb,1);
for ib = 1:nb
    for r = 1:ns
        in.mode{r}.f1f2 = tempinmode{r}.f1f2(ib,:);
        in.mode{r}.f0 = tempinmode{r}.f0(ib);
    end
    outib{ib} = bayoma_oneband_EMNewton_multi(in); 
end
%========================================================
% Set output
%========================================================
outfields = fieldnames(outib{1});
for ib = 1:nb
    for jj = 1:length(outfields)
        outib{ib}.(outfields{jj}) = outib{ib}.(outfields{jj}).';
    end
end
out1 = outib{1};
outfield1 = {'f','z','S','Sii','Se','phi','rms','sn'}.';
outfield2 = setdiff(outfields,[outfield1;'coefv'],'stable');
covfield = fieldnames(outib{1}.coefv{1});
heading = cell(1,ns);
for r = 1:ns
    heading{r} = ['Setup',num2str(r,'%02d')];
end
for jj = 1:length(outfield1)
    for ib = 2:nb
        cf = outib{ib}.(outfield1{jj});
        out1.(outfield1{jj}) = [out1.(outfield1{jj}),cf];
    end
    for r = 1:ns
        if ~iscell(out1.(outfield1{jj}){1})
            out.(heading{r}).(outfield1{jj}) = cell2mat(out1.(outfield1{jj})(r,:));
        else
            for ib = 1:nb
                out.(heading{r}).(outfield1{jj}){ib} = out1.(outfield1{jj}){r,ib}{1};
            end
        end
    end
end
for ib = 2:nb
    out1.coefv = [out1.coefv,outib{ib}.coefv];
end
for r = 1: ns
    for jj = 1:length(covfield)
        out.(heading{r}).coefv.(covfield{jj}) = out1.coefv{r,1}.(covfield{jj});
        for ib = 2:nb
            out.(heading{r}).coefv.(covfield{jj}) = [out.(heading{r}).coefv.(covfield{jj})...
                ,out1.coefv{r,ib}.(covfield{jj})];
        end
    end
end
for jj = 1:length(outfield2)
    for ib = 2:nb
        cf = outib{ib}.(outfield2{jj});
        out1.(outfield2{jj}) = [out1.(outfield2{jj}),cf];
    end
    out.(outfield2{jj}) = out1.(outfield2{jj});
end
end
