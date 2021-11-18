function dx = cloudODE(t,x,Xi,nVars,polyorder,strWhat)
%CLOUDODE Calculates the next step in the cloud ODE, vector with rank r
%entries
%   t: time
%   x: position, data vector of a state
%   Xi: sparse model of the states
%   dx: next state of the ODE

if ~exist('strWhat','var')
    strWhat = 'polynomial';
end

dx = buildTheta(x',nVars,polyorder,strWhat)*Xi;
dx = dx';

end

