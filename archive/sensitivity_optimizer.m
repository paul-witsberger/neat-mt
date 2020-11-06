function [x, F, xmul, Fmul, INFO] = sensitivity_optimizer()
    % Clear everything to avoid SNOPT issues
    clear all
    close all
    
    % Set SNOPT settings
    snscreen on
    snprint('paul_test.out');  % By default, screen output is off;
    snseti('Major Iteration limit', 3000);
    
    k = 0; % Not all derivatives are provided
    snseti('Derivative option', k);
    
    % Load trajectory
    [t, states, controls] = read_traj();
    
    % Set optimization variables
    xl = -5; xu = 5;
    xlow   = [xl, xl, xl, xl,  1000]';
    xupp   = [xu, xu, xu, xu, 10000]';
    x      = rand() .* (xupp - xlow) + xlow;
    x(2) = 3;
    xmul   = [];
    xstate = [];
    Flow   = [ -Inf, 0, 0, 0, 0, 0]';
    Fupp   = [  Inf, 5, 0, 0, 0, 0]';
    Fmul   = [];
    Fstate = [];
    
    % Call optimizer
    [x, F, INFO, xmul, Fmul] = snopt( x, xlow, xupp, xmul, xstate, Flow, ...
                                      Fupp, Fmul, Fstate, @get_sensitivity );
    snend;
    % Print results to screen
    fprintf('\n')
    disp('Final delta V''s')
    disp(x(1:4))
    disp('Final TOF')
    disp(F(1))
    disp('Final delta V magnitude')
    disp(F(2))
    disp('Final constraint violations')
    disp(F(3:6))
end

function s = get_sensitivity(x)
    states = x(:, 1:6);
    controls = x(1:end-1, 7:9);
    for i = 1:4*n
        for j = 1:2*(n-1)
            jac(i, j) = ur(j) / xr(i);
        end
    end
    s = sqrt(sum(jac .^ 2));
end

function [t, states, controls] = read_traj()
    % Read data from hdf5 file
    t = h5read('C:\Users\pawit\OneDrive\Documents\Classes\Research\neat-mt\traj_data.hdf5', '/t');
    states = h5read('C:\Users\pawit\OneDrive\Documents\Classes\Research\neat-mt\traj_data.hdf5', '/x')';
    controls = h5read('C:\Users\pawit\OneDrive\Documents\Classes\Research\neat-mt\traj_data.hdf5', '/u')';
end