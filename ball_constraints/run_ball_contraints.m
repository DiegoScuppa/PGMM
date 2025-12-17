%% Run all L1-ball-constraints problems

addpath(fileparts(pwd));
savepath;

import pgmm.*;


filenames = {'mushrooms';'a1a';'a9a';'w1a';'w8a';'phishing';'gisette';'splice';'sonar';'madelon'}; 
versions = {"SPG", "PGMM"};

maxit = 50000;	      % Maximum number of iterations
maxfc = 10 * maxit;         % Maximum number of function evaluations
iprint = 0;	              % Print Parameter
epsopt = 1.0E-5; % Tolerance for the convergence criterion

runs = 10;

rng(1) % fix a seed

for f = 1:numel(filenames)

    filename = filenames{f};
    fprintf('\n=== FILE: %s ===\n', filename);
    [X, y, n, d] = readDataAuto(filename); % read data
    x0 = randn(n,1);
    F = @(n,w) fLogReg (n, w, X, y);
    G = @(n,w) gLogReg (n, w, X, y, d);
    options = optimoptions('fminunc', ...
    'Algorithm','quasi-newton', ...        
    'SpecifyObjectiveGradient',true, ...
    'MaxIterations', 1000, ...
    'OptimalityTolerance', 1e-4, ...
    'HessianApproximation','lbfgs');
    % find an unconstrained stationary point
    fun = @(w) fungrad(w, X, y);
    [x,fval,exitflag,output] = fminunc(fun,x0,options); 
   
    results = table();
    k = 1/2;
    R = norm(x,1);
    P = @(n,x) projL1(n, x, k*R);

    for v = 1:numel(versions)
        version = versions{v};

        if v==1
            momentum = false; %SPG
            % run few iterations (the first run is always more time consuming)
            [x, fopt, gpsupn, iter, fcnt, gcnt, spginfo, inform] = ...
                pgmm(n, x0, epsopt, 10, maxfc, iprint, momentum, F, G, P);
        else
            momentum = true; %PGMM
        end

        for run = 1:runs
            x0 = randn(n,1);
            tic;
            [x, fopt, gpsupn, iter, fcnt, gcnt, spginfo, inform] = ...
                pgmm(n, x0, epsopt, maxit, maxfc, iprint, momentum, F, G, P);
            t = toc;
            norm_x = norm(x, ball);
            new_row = {version, run, iter, fcnt, gcnt, t, fopt, norm_x, gpsupn};
            results = [results; new_row];
        end

    end

    results.Properties.VariableNames =  {'Version','Run','Iterations','CostEvals','GradEvals','Time','FinalCost', 'SolutionNorm', 'FinalGradNorm'};
    save(['results/results_',char(filename),'_L1.mat'], 'results');

end


% routines where we define cost/grad/proj

function [f, flag] = fLogReg(n, w, X, y)
% fLogReg  evaluates the objective function of the logistic regression.
% Input:
%   n  - vector w dimension
%   w  - parameters vector (n x 1)
%   X  - (d x n) data matrix
%   y  - (d x 1) labels vector
%
% Output:
%   f     - objective function value
%   flag  - state code (0 = ok)

    % Exponenzial computation
    expon = exp(-y .* (X * w));   % (d x 1) vector

    % Mean of the logistic function
    f = mean(log(1 + expon));

    % flag = 0 means no error
    flag = 0;
end



function [g, flag] = gLogReg(n, w, X, y, d)
% EVALG  evaluates the gradient of the logistic regression.
% Input:
%   n  - vector w dimension
%   w  - parameters vector (n x 1)
%   X  - (d x n) data matrix
%   y  - (d x 1) labels vector
%   d  - vector y dimension
%
% Output:
%   g     - gradient (n x 1)
%   flag  - state code (0 = ok)

    % Exponenzial computation
    expon = exp(-y .* (X * w));     % (d x 1)

    %  -y * exp(-y Xx) / (1 + exp(-y Xx))
    frac = (-y .* expon) ./ (1 + expon);  % (d x 1)

    g = (X' * frac) / d;   % (n x 1)

    flag = 0;
end



function [x, flag] = projL1(n, x, R)
% PROJ_L1  Projection of x onto the L1-norm ball of radius R.
% Input:
%   n  - dimension of vector x
%   x  - parameter vector
%   R  - radius of the L1-norm ball
%
% Output:
%   x     - projected vector
%   flag  - status code (0 = ok)

    % If the L1 norm is already <= R, no projection is needed
    if norm(x, 1) <= R
        flag = 0;
        return;
    end
    
    % Projection algorithm (Duchi et al., 2008)
    u = sort(abs(x), 'descend');
    sv = cumsum(u);
    rho = find(u > (sv - R) ./ (1:length(u))', 1, 'last');
    if isempty(rho)
        rho = length(u);
    end
    theta = (sv(rho) - R) / rho;
    x = sign(x) .* max(abs(x) - theta, 0);

    flag = 0;
end


% routine for cost and grad for fminunc

function [f,g] = fungrad(w, X, y)
    % Exponenzial computation
    expon = exp(-y .* (X * w));   % (d x 1) vector

    % Mean of the logistic function
    f = mean(log(1 + expon));

    %  -y * exp(-y Xx) / (1 + exp(-y Xx))
    frac = (-y .* expon) ./ (1 + expon);  % (d x 1)

    g = mean(X .* frac, 1);          % (1 x n)
    d = size(X, 1);
    g = (X' * frac) / d;   % (n x 1)

end


% routine to read data (saved in the subfolder /problems) or save them

function [X, y, n, d] = readDataAuto(filename)
% Reads a data file where we have previously saved X and y
%
% Output:
%   X - data matrix
%   y - label vector (-1/+1)
%   n - number of features
%   d - number of rows (samples)

    currentFolder = pwd;
    dataFolder = fullfile(currentFolder, "problems", filename);
    if ~isfolder(dataFolder)
        error('The folder containing the data does not exist in the current directory.');
    end

    matFile = fullfile(dataFolder, [filename, '.mat']);

    if isfile(matFile)
        S = load(matFile);
        if isfield(S, 'X') && isfield(S, 'y')
            X = S.X;
            y = S.y;
            [d, n] = size(X);
        else
            error('The file "%s" does not contain the variables X and y.', matFile);
        end

    else
        error('There is no data available for the specified problem.');
    end
end