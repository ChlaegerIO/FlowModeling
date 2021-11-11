function dx = cloudODE(t,x,Xi,nVars,polyorder)
%CLOUDODE Calculates the next step in the cloud ODE, vector with rank r
%entries
%   t: time
%   x: position, data vector of a state
%   Xi: sparse model of the states
%   dx: next state of the ODE

dx = buildTheta(x',nVars,polyorder)*Xi;
dx = dx';

end

