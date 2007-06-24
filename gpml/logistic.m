function [out1, out2, out3, out4] = logistic(y, f, var)

% logistic - logistic likelihood function. The expression for the likelihood is
% logistic(t) = 1./(1+exp(-t)).
%
% Three modes are provided, for computing likelihoods, derivatives and moments
% respectively, see likelihoods.m for the details. In general, care is taken
% to avoid numerical issues when the arguments are extreme. The moments
% \int f^k cumGauss(y,f) N(f|mu,var) df are calculated via Gauss-Hermite
% quadrature.
%
% Copyright (c) 2007 Carl Edward Rasmussen and Hannes Nickisch, 2007-06-25.

if nargin>1, y=sign(y); end                         % allow only +/- 1 as values

if nargin == 2                                     % (log) likelihood evaluation
    
  if numel(y)>0, yf = y.*f; else yf = f; end     % product of latents and labels

  out1 = 1./(1+exp(-yf));                                           % likelihood
  if nargout>1
    out2 = yf;
    ok = -35<yf;
    out2(ok) = -log(1+exp(-yf(ok)));                         % log of likelihood
  end

elseif nargin == 3 

  if strcmp(var,'deriv')                         % derivatives of log likelihood

    if numel(y)==0, y=1; end
    yf = y.*f;                                   % product of latents and labels
     
    s    = -yf; 
    ps   = max(0,s); 
    out1 = -sum(ps+log(exp(-ps)+exp(s-ps)));          % lp = -sum(log(1+exp(s)))
    if nargout>1 % dlp - first derivatives
      s    = min(0,f); 
      p    = exp(s)./(exp(s)+exp(s-f));                     % p = 1./(1+exp(-f))
      out2 = (y+1)/2-p;                      % dlp, derivative of log likelihood
      if nargout>2                      % d2lp, 2nd derivative of log likelihood
        out3 = -exp(2*s-f)./(exp(s)+exp(s-f)).^2;
        if nargout>3                    % d3lp, 3rd derivative of log likelihood
          out4 = 2*out3.*(0.5-p);
        end
      end
    end

  else                                                         % compute moments
        
    mu = f;                             % 2nd argument is the mean of a Gaussian

    N = 20;                                    % 20 yields precalculated weights
    [f,w] = gauher(N); 
    sz = size(mu);

    f0 = sqrt(var(:))*f'+repmat(mu(:),[1,N]);               % center values of f

    if numel(y)==0       % calculate the likelihood values. If empty, assume y=1
      sig = logistic([],f0);
    else                                                      % include y values
      sig = logistic( repmat(y(:),[1,N]), f0 );
    end

%   out1 = reshape(sig*w, sz);        % much less accurate than the method below
    out1 = erfint(mu,var); out1(y==-1) = 1-out1(y==-1);      % zeroth raw moment
    
    if nargout>1
      out2 = reshape(f0.*sig*w, sz);                          % first raw moment
      if nargout>2
        out3 = reshape(f0.*f0.*sig*w, sz);                   % second raw moment
      end
    end
    
  end

else
  error('No valid input provided.')    
end



% The erfint function approximates "\int logistic(t) N(t|mu, var) dt" by 
%   1/2 + \sum_{i=1}^5 (c_i/2) \int erf(lambda_i t) N(t| mu, var) dt
% = 1/2 + \sum_{i=1}^5 (c_i/2) erf(mu/v_i), where  v_i^2 = 2*var+lambda_i^(-2)
% The inputs mu and var are column vectors.

function pout = erfint(mu, var)

    ilam2 = [0.44 0.41 0.40 0.39 0.36].^(-2);

    c = [1.146480988574439e+02; -1.508871030070582e+03; 2.676085036831241e+03;  
        -1.356294962039222e+03;  7.543285642111850e+01                        ];

    V = sqrt(2*var*ones(1,5) + ones(length(var),1)*ilam2);
    pout = (1+erf(mu*ones(1,5)./V)*c)/2;
