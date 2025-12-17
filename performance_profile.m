%% ============================================================
%  COMPARISON OF ALGORITHMS
%  Tables: mean ± std over ALL runs
%  Performance profiles: ONLY successful runs
% ============================================================

clear; clc;

algos = {'SPG', 'PGMM'};
tol   = 1e-5;

for icase = 1:2

    %% ================= FOLDERS ===============================
    if icase == 1
        folder = fullfile(fileparts(mfilename('fullpath')), ...
                          'ball_constraints/results');
    else
        folder = fullfile(fileparts(mfilename('fullpath')), ...
                          'box_constraints/results');
    end

    files = dir(fullfile(folder,'results_*.mat'));
    numP  = numel(files);
    numA  = numel(algos);

    latex_file = fullfile(folder,'latex_tables2.tex');
    fid = fopen(latex_file,'w');   % overwrite
    fclose(fid);

    %% ================= MATRICES FOR PERFORMANCE PROFILES =====
    TimePP = NaN(numP,numA);
    IterPP = NaN(numP,numA);
    CostPP = NaN(numP,numA);

    %% ================= LOOP OVER PROBLEMS ====================
    for p = 1:numP

        data = load(fullfile(folder,files(p).name));
        T = data.results;

        % problem name
        [~,fname] = fileparts(files(p).name);
        parts = split(fname,'_');
        problem_name = strjoin(parts(2:end),'_');

        %% ===== STATISTICS PER ALGORITHM =======================
        MeanIter = NaN(numA,1);  StdIter = NaN(numA,1);
        MeanTime = NaN(numA,1);  StdTime = NaN(numA,1);
        MeanCost = NaN(numA,1);  StdCost = NaN(numA,1);
        NumFail  = zeros(numA,1);

        for a = 1:numA

            Ta = T(strcmp(T.Version,algos{a}),:);
            if isempty(Ta), continue; end

            % failures
            fail = Ta.FinalGradNorm > tol;
            NumFail(a) = sum(fail);

            % ===== TABLE STATISTICS (ALL RUNS) ==================
            MeanIter(a) = mean(Ta.Iterations);
            StdIter(a)  = std(Ta.Iterations);

            MeanTime(a) = mean(Ta.Time);
            StdTime(a)  = std(Ta.Time);

            MeanCost(a) = mean(Ta.CostEvals);
            StdCost(a)  = std(Ta.CostEvals);

            % ===== PERFORMANCE PROFILE (ONLY VALID RUNS) ========
            Ta_ok = Ta(~fail,:);
            if ~isempty(Ta_ok)
                TimePP(p,a) = mean(Ta_ok.Time);
                IterPP(p,a) = mean(Ta_ok.Iterations);
                CostPP(p,a) = mean(Ta_ok.CostEvals);
            end
        end

        %% ================== LATEX TABLE =======================
        fid = fopen(latex_file,'a');

        fprintf(fid,'%% =========================================\n');
        fprintf(fid,'%% Problem: %s\n',problem_name);
        fprintf(fid,'\\begin{table}[h]\n');
        fprintf(fid,'\\centering\n');
        fprintf(fid,'\\begin{tabular}{lcccc}\n');
        fprintf(fid,'\\toprule\n');
        fprintf(fid,'Algorithm & Iterations & Time & CostEvals & Failures\\\\\n');
        fprintf(fid,'\\midrule\n');

        for a = 1:numA
            fprintf(fid,...
                '%s & $%.1f \\pm %.1f$ & $%.4f \\pm %.4f$ & $%.1f \\pm %.1f$ & %d \\\\\n',...
                algos{a},...
                MeanIter(a),StdIter(a),...
                MeanTime(a),StdTime(a),...
                MeanCost(a),StdCost(a),...
                NumFail(a));
        end

        fprintf(fid,'\\bottomrule\n');
        fprintf(fid,'\\end{tabular}\n');
        fprintf(fid,'\\caption{Results for problem %s}\n',problem_name);
        fprintf(fid,'\\end{table}\n\n');

        fclose(fid);
    end

    %% ================= PERFORMANCE PROFILES ==================
    plot_performance_profile(TimePP,algos,'Performance Profile (CPU Time)');
    plot_performance_profile(IterPP,algos,'Performance Profile (Iterations)');
    plot_performance_profile(CostPP,algos,'Performance Profile (Cost Evaluations)');

end

%% ============================================================
%  PERFORMANCE PROFILE FUNCTION
% ============================================================
function plot_performance_profile(M,algos,title_str)

    numP = size(M,1);

    best = min(M,[],2,'omitnan');
    rho  = M ./ best;

    tau = linspace(1,10,2000);
    linestyles = {'-','--'};

    figure; hold on;
    for a = 1:size(M,2)
        y = arrayfun(@(t) sum(rho(:,a) <= t,'omitnan') / numP, tau);
        plot(tau,y,'LineWidth',2,'LineStyle', linestyles{a});
    end

    set(gca,'FontSize',16,'TickLabelInterpreter','latex');
    xlabel('$\tau$','Interpreter','latex');
    ylabel('$\rho_a(\tau)$','Interpreter','latex');
    title(title_str,'Interpreter','latex');
    legend(algos,'Location','SouthEast','Interpreter','latex');
    ylim([0 1.01])
    grid on;
end
