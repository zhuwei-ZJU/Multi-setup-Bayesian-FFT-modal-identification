function PHI = initialModeShapeFromSelmat(C, F, localPhi)
% Assemble local spectral mode-shape estimates using selection matrices.
% C{r} has one unit entry per row; its row order matches F{r}'s channels.
% For one mode, use the minimum-eigenvalue construction of the P-EM code.
% For several modes, align local columns on shared DOFs and average them.

ns = numel(C);
n = size(C{1}, 2);
m = size(localPhi{1}, 2);
idx = cell(1, ns);
for r = 1:ns
    [idx{r}, ~] = find(C{r}.');
end

% Relative signs and amplitudes cannot be identified across disconnected
% groups of setups.
connected = false(1, ns);
connected(1) = true;
while true
    previous = connected;
    for r = find(connected)
        for s = find(~connected)
            connected(s) = connected(s) || ...
                ~isempty(intersect(idx{r}, idx{s}));
        end
    end
    if isequal(previous, connected), break; end
end
if ~all(connected)
    error('selmat must connect all setups through shared DOFs.');
end

if m == 1
    B = zeros(n, n);
    for r = 1:ns
        A = real(F{r}.' * conj(F{r}));
        A = (A + A.') / 2;
        peak = max(eig(A));
        B = B + C{r}.' * (peak * eye(size(A, 1)) - A) * C{r};
    end
    B = (B + B.') / 2;
    [V, D] = eig(B);
    [~, order] = sort(diag(D), 'ascend');
    PHI = V(:, order(1));
    PHI = PHI / norm(PHI);
    return;
end

PHI = zeros(n, m);
for j = 1:m
    scale = nan(1, ns);
    scale(1) = 1;
    visited = false(1, ns);
    visited(1) = true;

    % Propagate local signs and scales through the shared-DOF graph.
    while ~all(visited)
        bestScore = -inf;
        bestChild = 0;
        bestScale = 0;
        for parent = find(visited)
            for child = find(~visited)
                [~, ip, ic] = intersect(idx{parent}, idx{child}, 'stable');
                if isempty(ip)
                    continue;
                end
                a = localPhi{parent}(ip, j);
                b = localPhi{child}(ic, j);
                denom = b.' * b;
                score = abs(a.' * b);
                if denom > eps && score > bestScore
                    bestScore = score;
                    bestChild = child;
                    bestScale = scale(parent) * (a.' * b) / denom;
                end
            end
        end
        if bestChild == 0 || bestScore <= eps || abs(bestScale) <= eps
            error(['Cannot align initial mode shapes from selmat: ', ...
                'the setups must share informative DOFs for every mode.']);
        end
        scale(bestChild) = bestScale;
        visited(bestChild) = true;
    end

    total = zeros(n, 1);
    count = zeros(n, 1);
    for r = 1:ns
        total(idx{r}) = total(idx{r}) + scale(r) * localPhi{r}(:, j);
        count(idx{r}) = count(idx{r}) + 1;
    end
    PHI(:, j) = total ./ count;
    lengthPhi = norm(PHI(:, j));
    if lengthPhi <= eps
        error('The assembled initial mode shape has zero norm.');
    end
    PHI(:, j) = PHI(:, j) / lengthPhi;
end
end
