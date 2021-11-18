function yout = buildTheta(yin,nVars,polyorder,strWhat)
%buildTheta makes the underlaying functions or polynomials for Theta
%   yin: data of the system for theta(x)
%   nVars: number of different variables e.g. x,y,...
%   polyorder: polynomial order, maximal 5 is implemented

% TODO: add sin(x), radial function, ...
 if ~exist('strWhat','var')
      strWhat = 'polynomial';
 end

% trigonometric functions
if strcmp(strWhat,'trigonometric')
    ind = 1;
    % sin
    for i=1:nVars
        yout(:,ind) = sin(yin(:,i));
        ind = ind+1;
    end
    
    % cos
    for i=1:nVars
        yout(:,ind) = cos(yin(:,i));
        ind = ind+1;
    end
end


% polynomial
if strcmp(strWhat,'polynomial')
    n = size(yin,1);
    ind = 1;
    % poly order 0
    yout(:,ind) = ones(n,1);
    ind = ind+1;
    
    % poly order 1
    for i=1:nVars
        yout(:,ind) = yin(:,i);
        ind = ind+1;
    end
    
    if(polyorder>=2)    % poly order 2
        for i=1:nVars
            for j=i:nVars
                yout(:,ind) = yin(:,i).*yin(:,j);
                ind = ind+1;
            end
        end
    end
    
    if(polyorder>=3)    % poly order 3
        for i=1:nVars
            for j=i:nVars
                for k=j:nVars
                    yout(:,ind) = yin(:,i).*yin(:,j).*yin(:,k);
                    ind = ind+1;
                end
            end
        end
    end
    
    if(polyorder>=4)    % poly order 4
        for i=1:nVars
            for j=i:nVars
                for k=j:nVars
                    for l=k:nVars
                        yout(:,ind) = yin(:,i).*yin(:,j).*yin(:,k).*yin(:,l);
                        ind = ind+1;
                    end
                end
            end
        end
    end
    
    if(polyorder>=5)    % poly order 5
        for i=1:nVars
            for j=i:nVars
                for k=j:nVars
                    for l=k:nVars
                        for m=l:nVars
                            yout(:,ind) = yin(:,i).*yin(:,j).*yin(:,k).*yin(:,l).*yin(:,m);
                            ind = ind+1;
                        end
                    end
                end
            end
        end
    end
end