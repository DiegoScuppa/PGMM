%% Run all box-constraints problems

addpath(fileparts(pwd));
savepath;

% be sure cutest is installed!

import pgmm.*;

problems = ["BDEXP","EXPLIN","EXPLIN2","EXPQUAD","MCCORMCK","PROBPENL",...
    "QRTQUAD","S368","HADAMALS","CHEBYQAD","HS110","LINVERSE","NONSCOMP",...
    "QR3DLS","SCON1LS","DECONVB","BIGGSB1","BQPGABIM","BQPGASIM",...
    "BQPGAUSS","CHENHARK","CVXBQP1","HARKERP2","JNLBRNG1","JNLBRNG2",...
    "JNLBRNGA","JNLBRNGB","NCVXBQP1","NCVXBQP2","NCVXBQP3","NOBNDTOR",...
    "OBSTCLAE","OBSTCLAL","OBSTCLBL","OBSTCLBM","OBSTCLBU","PENTDI",...
    "TORSION1","TORSION2","TORSION3","TORSION4","TORSION5","TORSION6",...
    "TORSIONA","TORSIONB","TORSIONC","TORSIOND","TORSIONE","TORSIONF","ODNAMUR"];

versions = {"SPG", "PGMM"};

maxit = 100000;	      % Maximum number of iterations
maxfc = 10 * maxit;         % Maximum number of function evaluations
iprint = 0;	              % Print Parameter
epsopt = 1.0E-5; % Tolerance for the convergence criterion

runs = 10;

for i = 1:size(problems,2)
    problem_name = problems(i);
    unix(['runcutest -p matlab -D ' char(problem_name)]);
    pb = cutest_setup();
    n = pb.n;
    xlower = pb.bl;
    xupper = pb.bu;
    x0 = pb.x;
    F = @(n,x) fCUTE (n, x);
    G = @(n,x) gCUTE (n, x);
    P = @(n,x) projBox (n, x, xlower, xupper);
   
    results = table();

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
            tic;
            [x, fopt, gpsupn, iter, fcnt, gcnt, spginfo, inform] = ...
                pgmm(n, x0, epsopt, maxit, maxfc, iprint, momentum, F, G, P);
            t = toc;
            new_row = {version, run, iter, fcnt, gcnt, t, fopt,gpsupn};
            results = [results; new_row];
        end

    end

    results.Properties.VariableNames =  {'Version','Run','Iterations','CostEvals','GradEvals','Time','FinalCost','FinalGradNorm'};
    save(['results/results_',char(problem_name),'.mat'], 'results');
    
    cutest_terminate;

end


% routines where we define cost/grad/proj

function [f, flag] = fCUTE(n, x)

    f = cutest_obj(x);
    % flag = 0 means no error
    flag = 0;
end


function [g, flag] = gCUTE(n, x)

    g = cutest_grad(x);

    % Check that the gradient is a column vector (n x 1)
    c = size(g, 2);
    if c == n
        g = g';
    end

    flag = 0;
end


function [x, flag] = projBox(n, x, xlower, xupper)

    x = max(xlower,min(xupper,x));

    % Check that it is a column vector (n x 1)
    c = size(x, 2);
    if c == n
        x = x';
    end

    flag = 0;
end