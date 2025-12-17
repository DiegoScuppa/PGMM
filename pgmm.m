%     =================================================================
%     File: pgmm.m
%     =================================================================
%
%     =================================================================
%     Module: Projected Gradient Method with Momentum


function [x, f, gpsupn, iter, fcnt, gcnt, pgmminfo, inform] = pgmm (n, x, epsopt, maxit, maxfc, iprint, momentum, evalf, evalg, proj)

%     Subroutine PGMM implements the Projected Gradient Method with
%     Momentum to find a local minimizers of a given function with convex 
%     constraints, described in
%
%     M. Lapucci, G. Liuzzi, S. Lucidi, M. Sciandrone and D. Scuppa, "On
%     the convergence of the projected gradient method with momentum", 2026
%
%     It was written by modyfing the spgm subroutine (Spectral Projected 
%     Gradient Method), described in
%
%     E.G. Birgin, J.M. Martinez and M. Raydan, "Nonmonotone spectral
%     projected gradient methods for convex sets", SIAM Journal on
%     Optimization 10, pp. 1196-1211, 2000.
%
%     The user must supply the external subroutines evalf, evalg and 
%     proj to evaluate the objective function and its gradient and to 
%     project an arbitrary point onto the feasible region.
%
%     Other parameters (i means input, o means output):
%
%     n       (i)   number of variables
%     x       (i/o) initial guess on input, solution on output
%     epsopt  (i)   tolerance for the convergence criterion
%     maxit   (i)   maximum number of iterations
%     maxfc   (i)   maximum number of functional evaluations
%     iprint  (i)   controls output level (0 = no print)
%     momentum(i)   consider the momentum (true/false) (*)
%     evalf   (i)   handle function to evaluate the cost function
%     evalg   (i)   handle function to evaluate the gradient
%     proj    (i)   handle function to compute the projection
%     f       (o)   functional value at the solution
%     gpsupn  (o)   sup-norm of the projected gradient at the solution
%     iter    (o)   number of iterations
%     fcnt    (o)   number of functional evaluations
%     gcnt    (o)   number of gradient evaluations
%     pgmminfo(o)   indicates the reason for stopping
%     inform  (o)   indicates an error in an user supplied subroutine
%
%     (*) momentum = false reduces PGMM to SPG by Birgin-Martinez-Raydan
%
%     pgmminfo:
%
%     0: Small continuous-projected-gradient norm
%     1: Maximum number of iterations reached
%     2: Maximum number of functional evaluations reached
%     3: Stepsize too small
%
%     pgmminfo remains unset if inform is not equal to zero on output
%
%     inform:
%
%       0: ok
%     -90: error in the user supplied evalf subroutine
%     -91: error in the user supplied evalg subroutine
%     -92: error in the user supplied proj  subroutine

   if (iprint ~= 0) 
    fprintf("==============================================================================\n");
    fprintf(" This is the PROJECTED GRADIENT METHOD with MOMENTUM (PGGM)\n");
    fprintf(" for convex-constrained optimization, as described in:\n\n");
    fprintf(" M. Lapucci, G. Liuzzi, S. Lucidi, M. Sciandrone and D. Scuppa,\n");
    fprintf(" On the convergence of the projected gradient method with momentum', 2026\n\n");
    fprintf(" It is based on:\n\n");
    fprintf(" E. G. Birgin, J. M. Martinez and M. Raydan, Algorithm 813: SPG - software\n");	
    fprintf(" for convex-constrained optimization, ACM Transactions on Mathematical\n");
    fprintf(" Software 27, pp. 340-349, 2001.\n");
    fprintf(" ==============================================================================\n");
    fprintf("\n");
    fprintf(" Entry to PGMM.\n");
    fprintf(" Number of Variables:\t%d\n", n);  
   end

   inform = 0;
   iter = 0;
   fcnt = 0;
   gcnt = 0;

   lambda_min = 10^-30;
   lambda_max = 10^30;
   tolStep = 10^-15; % stop if ||x-x_new||^2 < tolStep 
   sts = Inf;
   if momentum % monotone linesearch for PGMM
       M = 1;
   else
       M = 10; % non-monotone line-search for SPG
   end

   lastfv(1:M) = -Inf;
   
   [x, inform] = sproj (proj, n, x, inform);
   if (inform ~= 0) return; end

   [f, inform] = sevalf (evalf, n,x, inform);
   if (inform ~= 0) return; end
   fcnt = fcnt + 1;

   [g, inform] = sevalg (evalg, n, x, inform);
   if (inform ~= 0) return; end
   gcnt = gcnt + 1;
 
   lastfv(M) = f;   

   [gp, inform] = sproj (proj, n, x - g, inform);
   if (inform ~= 0) return; end

   gp = gp - x;

   gpsupn = norm(gp, Inf);

   if (gpsupn ~= 0) lambda = min (lambda_max, max (lambda_min, 1.0 / gpsupn));
   else lambda = 0.0; end

   while (gpsupn > epsopt & iter < maxit & fcnt < maxfc & sts >= tolStep)  

     if(iprint ~= 0) 
      if (mod(iter, 10) == 0) 
       fprintf("\n ITER\t F\t GPSUPNORM\n");
      end
      fprintf(" %d\t %e\t %e\n", iter, f, gpsupn); 
     end

     tabline = fopen ("pgmm-tabline.out", "w");
     fprintf(tabline, " %d  %d  %d %d  %e  %e Abnormal termination. Probably killed by CPU time  limit.\n", n, iter, fcnt, gcnt, f, gpsupn );
     fclose(tabline);

     iter = iter + 1;
     if momentum == false || iter == 1 % SPG by Birgin-Martinez-Raydan
         [d, inform] = sproj (proj, n, x - lambda * g, inform);
         if (inform ~= 0) return; end
         alpha = 1; beta = 0;
         d = d - x;
     else % PGMM
         [gk, inform] = sproj (proj, n, x - lambda * g, inform);
         if (inform ~= 0) return; end
         gk = gk - x;
         [sk, inform] = sproj (proj, n, x +  s, inform);
         if (inform ~= 0) return; end
         sk = sk - x;
         [alpha, beta, inform, fcnt] = computeAlphaBeta(x, f, g, gk, sk, n, evalf, inform, fcnt);
         d = alpha * gk + beta * sk;
         if d'*g > -10^-15 % check if d is a descent direction
             d = gk;
             alpha = 1; beta = 0;
             if (iprint ~= 0) 
             fprintf("\n Not a descent direction: take the projected gradient\n");
             end
         end
     end
     
     

     [fcnt, fnew, xnew, lsinfo, inform] = linesearch (n, x, f, g, d, lastfv, maxfc, fcnt, evalf, inform);

     if (inform ~= 0) return; end 

     if (lsinfo == 2) 
      pgmminfo = 2; 
      if (iprint ~= 0) 
       fprintf("Flag of RGMM: Maximum of functional evaluations reached.\n");
      end
      return;
     end

     f = fnew;

     if(mod(iter, M) == 0) lastfv(M) = f;
     else lastfv(mod(iter, M)) = f; end

     [gnew, inform] = sevalg (evalg, n, xnew, inform);
     gcnt = gcnt + 1;
     if (inform ~= 0) return; end

     s = xnew - x;
     y = gnew - g;
     sts = s' * s;
     sty = s' * y;

     x = xnew;
     g = gnew;    

     [gp, inform] = sproj (proj, n, x - g, inform);
     if (inform ~= 0) return; end

     gp = gp - x; 
     gpsupn = norm(gp, Inf);

     if (sty <= 0) lambda = lambda_max;
     else lambda = min (lambda_max, max(lambda_min, sts / sty)); end
   end

   if(iprint ~= 0) 
    if (mod(iter, 10) == 0) 
     fprintf("\n ITER\t F\t GPSUPNORM\n");
    end
    fprintf(" %d\t %e\t %e\n", iter, f, gpsupn); 
   end

   if (iprint ~= 0) 
    fprintf("\n");
    fprintf ("Number of iterations               : %d\n", iter);
    fprintf ("Number of functional evaluations   : %d\n", fcnt);
    fprintf ("Number of gradient evaluations     : %d\n", gcnt);
    fprintf ("Objective function value           : %e\n", f);
    fprintf ("Sup-norm of the projected gradient : %e\n", gpsupn);
   end

   if (gpsupn <= epsopt) 
    pgmminfo = 0; 
    if (iprint ~= 0)
     fprintf("Flag of PGMM: Solution was found.\n"); 
    end
    return;
   end

   if (iter >= maxit) 
    pgmminfo = 1; 
    if (iprint ~= 0) 
     fprintf("Flag of PGMM: Maximum of iterations reached.\n"); 
    end
    return;
   end

   if (fcnt >= maxfc)
    pgmminfo = 2; 
    if (iprint ~= 0)
     fprintf("Flag of PGMM: Maximum of functional evaluations reached.\n"); 
    end
    return;
   end   

   if (sts < tolStep)
    pgmminfo = 3; 
    if (iprint ~= 0)
     fprintf("Flag of PGMM: Stepsize too small.\n"); 
    end
    return;
   end  
end




function reperr (inform)

%    Reports errors that occours during sevalf, sevalg and sevalg executions.
% 
%    Name   (i/o)  Description:
%
%    inform (i)    Information Parameter

  if (inform == -90)
   fprintf("\n\n*** There was an error in the user supplied EVALF function\n");
  elseif (inform == -91)
   fprintf("\n\n*** There was an error in the user supplied EVALG function\n");
  elseif (inform == -92)
   fprintf("\n\n*** There was an error in the user supplied PROJ function\n");
  end 
end

function [f, inform] = sevalf (evalf, n, x, inform)

%   Function sevalf evaluates the function f at point x.
%
%    Name  (i/o)  Description:
%
%    n     (i)    size of problem.
%
%    x     (i)    point to be evaluate.
%
%    evalf  (i)    handle function to evaluate the cost function.
%
%    f     (o)    Value of f(x).
%
%  inform  (i/o)   Information Parameter.
%                  0 = No problem occours.
%                  -90 = Some problem occours on function evaluation. 

   [f, flag] = evalf(n,x);

   if (flag ~= 0) 
    inform = -90;
    reperr (inform);
   end
end

function [g, inform] = sevalg (evalg, n, x, inform)

%   Function sevalg evaluates the gradient vetor of f at point x.
%
%    Name  (i/o)  Description:
%
%    n     (i)    size of problem.
%
%    x     (i)    point to be evaluate.
%
%    evalg  (i)    handle function to evaluate the gradient.
%
%    g     (o)    Value of f(x).
%
%  inform  (i/o)   Information Parameter.
%                  0 = No problem occours.
%                  -91 = Some problem occours on function evaluation. 

   [g, flag] = evalg(n,x);

   if (flag ~= 0)
    inform = -91;
    reperr (inform);
   end
end

function [x, inform] = sproj (proj, n, x, inform)

%    Function sproj evaluates the projection of point x onto feasible set.
%
%    Name   (i/o)  Description:
%
%    n      (i)    size of problem.
%
%    proj  (i)     handle function to compute the projection.
%
%    x      (i/o)    point to be evaluate.
%
%    inform (i/o)  Information Parameter.
%                  0 = No problem occours.
%                 -92 = Some problem occours on function evaluation. 

   [x, flag] = proj(n,x);

   if (flag ~= 0)
    inform = -92;
    reperr (infom);
   end
end

function[fcnt, fnew, xnew, lsinfo, inform] = linesearch (n, x, f, g, d, lastfv, maxfc, fcnt, evalf, inform)

%     Function linesearch implements a nonmonotone line search with
%     safeguarded quadratic interpolation.
%
%    This version 17 JAN 2000 by E.G.Birgin, J.M.Martinez and M.Raydan.
%    Reformatted 25 FEB 2008 by Tiago Montanher.
%    Final revision 30 APR 2001 by E.G.Birgin, J.M.Martinez and M.Raydan.
%
%    Name  (i/o)  Description:
%
%    n      (i)    size of the problem.
%
%    x      (i)    initial guess.
%
%    f      (i)    function value at the actual point.
%
%    d      (i)    search direction.
%
%    g      (i)    gradient function evaluated at initial guess.
%
%    lastfv (i)    last m function values.
%
%    maxfc  (i)    maximum number of function evaluations.
%
%    evalf  (i)    handle function to evaluate the cost function.
%
%    fcnt   (i/o)  actual number of fucntion evaluations.
%
%    fnew   (o)    function value at the new point.
%
%    xnew   (o)    new point.
%     
%    lsinfo:
%
%     0: Armijo-like criterion satisfied
%     2: Maximum number of functional evaluations reached
%
%    inform (i/o)    Information parameter:
%	            0= no problem occours during function evaluation,
%	           -90=  some problem occours during function evaluation,  

   sigma_min = 0.1;
   sigma_max = 0.9;
   gamma = 10^-4;

   fmax = max(lastfv);
   gtd = g' * d;

   alpha = 1.0;
   xnew = x + alpha * d;

   [fnew, inform] = sevalf(evalf, n, xnew, inform);
   fcnt = fcnt + 1;
   if (inform ~= 0) return; end
  
   while (fnew > fmax + gamma * alpha * gtd & fcnt < maxfc)
     if (alpha <= sigma_min) alpha = 0.5 * alpha;

     else
       a_temp = -0.5 * (alpha^2) * gtd / (fnew - f - alpha * gtd);

       if (a_temp < sigma_min | a_temp > sigma_max * alpha) 
        a_temp = 0.5 * alpha; end

       alpha = a_temp;
     end

     xnew = x + alpha * d;
     [fnew, inform] = sevalf(evalf, n, xnew, inform);
     fcnt = fcnt + 1;
     if (inform ~= 0) return; end
   end 
   
   if (fnew <= fmax + gamma * alpha * gtd) lsinfo = 0;
   else lsinfo = 2; end
end



function [alpha, beta, inform, fcnt] = computeAlphaBeta(x, f, g, gk, sk, n, evalf, inform, fcnt)
%    Function computing the coefficients alpha and beta via interpolation
%
%    Name  (i/o)  Description:
%
%    x      (i)    actual point.
%
%    f      (i)    function value at the actual point.
%
%    g      (i)    gradient at the actual point.
%
%    gk     (i)    projected gradient at the actual point.
%
%    sk     (i)    momentum term at the actual point.
%
%    n      (i)    size of the problem.
%
%    evalf  (i)    handle function to evaluate the cost function.
%
%    alpha  (o)    computed value of alpha.
%
%    beta   (o)    computed value of beta.
%
%    fcnt   (i/o)  actual number of function evaluations.
%
%    inform (i/o)    Information parameter:
%	            0= no problem occours during function/gradient evaluation,
%	           -90=  some problem occours during function/gradient evaluation, 

    g_gk = g'*gk; 
    g_sk = g'*sk; 
    alpha = 0.5; beta = 0.5; % values for interpolation

    [f1, inform] = sevalf(evalf, n, x + beta*sk, inform);
    fcnt = fcnt + 1;
    if (inform ~= 0) return; end

    [f2, inform] = sevalf(evalf, n, x + alpha*gk, inform);
    fcnt = fcnt + 1;
    if (inform ~= 0) return; end

    [f3, inform] = sevalf(evalf, n, x + alpha*gk + beta*sk, inform);
    fcnt = fcnt + 1;
    if (inform ~= 0) return; end

    H = [2*(f2 - f - alpha*g_gk) / alpha^2;
         (f3 + f - f1 - f2) / alpha / beta;
         2*(f1 - f - beta*g_sk) / beta^2];

    [alpha, beta] = quadratic_simplex(H(1), H(2), H(3), g_gk, g_sk);
end
        


function [a,b] = quadratic_simplex(t,u,w,y,h)
    %function to solve:  min 0.5*(a;b)^T(t,u;u,w)(a;b)+(y;h)^t(a;b)  with 0<=a, 0<=b, a+b<=1

    %verify if matrix is positive definite, if so, compute the minimum and verify if it's suitable
    det = t*w-u^2; %determinant of the 2x2 matrix
    if t>0 && det>0 % case matrix is positive definite
        a = (-w*y+u*h) / det;
        b = (u*y-t*h) / det;
        if a>=0 && b>=0 && a+b<=1 % verify (a,b) satisfies the constraints
            return
        end
    end

    %else we investigate the boundary minimum on the vertices
    f_min = 0; a = 0; b = 0;

    f_trial = t/2 + y;
    if f_trial < f_min
        f_min = f_trial;
        a = 1; b = 0;
    end

    f_trial = w/2 + h;
    if f_trial < f_min
        f_min = f_trial;
        a= 0; b = 1;
    end

    %minimum on the edge {(alpha,0)}
    if t>0
        alpha_trial = -y / t;
        if alpha_trial > 0 && alpha_trial < 1
            f_trial = -y^2 / (2*t);
            if f_trial < f_min
                f_min = f_trial;
                a = alpha_trial; b = 0;
            end
        end
    end

    if w>0
    %minimum on the edge {(0,alpha)}
        alpha_trial = -h/w;
        if alpha_trial>0 && alpha_trial<1
            f_trial = -h^2 / (2*w);
            if f_trial < f_min
                f_min = f_trial;
                a = 0; b = alpha_trial;
            end
        end
    end

    if t-2*u+w>0
    %minimum on the edge {(alpha,1-alpha)}
        alpha_trial = -(u-w+y-h) / (t-2*u+w);
        if alpha_trial>0 && alpha_trial<1
            f_trial = 0.5*(t-2*u+w)*alpha_trial^2 + (u-w+y-h)*alpha_trial+(w/2+h);
            if f_trial < f_min
                f_min = f_trial;
                a = alpha_trial;
                b = 1 - alpha_trial;
            end
        end
    end
end




