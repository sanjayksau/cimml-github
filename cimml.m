 function model = cimml( X, Y, param)
%%Code structure derived from LSML, multi label learning method
    lambdaC          = param.lambdaC; % missing labels |YR-Y|
    lambdaR          = param.lambdaR; % regularization of R |R|_1
    lambdaW          = param.lambdaW; % regularization of W |W|_1
    lambdaL          = param.lambdaL; % regularization of graph laplacian WLW'
    lambdaI          = param.lambdaI; % regularization of instance similarity (XW)'LinstXW
    beta             = param.beta;    % adaptive coeff for cost sensitive matrix
    rho              = param.rho;
    maxIter          = param.maxIter;
    miniLossMargin   = param.minimumLossMargin;
    J =  (Y~=0); %observed indices
    num_dim   = size(X,2);
    num_class = size(Y,2);
    XTX = X'*X;
    XTY = X'*Y;
    YTY = Y'*Y;
    [~, Linst] = computeLaplacianForInstanceSimilarity(X);
    XTLinstX = X' * Linst * X;
    W   = (XTX + rho*eye(num_dim)) \ (XTY); %zeros(num_dim,num_class); % 
    W_1 = W; W_k = W;
    C = zeros(num_class,num_class); %eye(num_class,num_class);
    C_1 = C;
    B = computeCostSensitiveMatrix(Y, beta);
    maxB2 = max(B.^2, [], 'all');
    iter = 1; oldloss = 10^9;
    bk = 1; bk_1 = 1; 
    %https://math.stackexchange.com/questions/351544/bounded-partial-derivatives-imply-continuity/412704#412704
    LipW2 = 3*maxB2 * norm(XTX)^2 +3* norm(lambdaI * XTLinstX)^2;
    LipR2  = 2*maxB2 * norm(YTY)^2 + 2* norm(lambdaC*YTY)^2;
    while iter <= maxIter
       L = diag(sum(C,2)) - C;
       lapterm = 3 * norm(lambdaL*(L+L'))^2; %laplacian term of LipW2 
       Lip = sqrt( LipR2 + LipW2 + lapterm);
       

      %% update C
       C_k  = C + (bk_1 - 1)/bk * (C - C_1);
       Gc_k = C_k - 1/Lip * gradientOfC(YTY, W, C_k, lambdaC, B, X, Y, J);
       C_1  = C;
       C    = softthres(Gc_k,lambdaR/Lip); 
       C    = max(C,0);
       
       W_k  = W + (bk_1 - 1)/bk * (W - W_1);
       Gw_x_k = W_k - 1/Lip * gradientOfW(W_k,C,lambdaL, XTLinstX, lambdaI, B, X, Y, J);
       W_1  = W;
       W    = softthres(Gw_x_k,lambdaW/Lip);       
       bk_1   = bk;
       bk     = (1 + sqrt(4*bk^2 + 1))/2;

       LS = (X*W - Y*C);
       DiscriminantLoss = 0.5 * norm((LS .* LS).*B, 1);
       LS = (Y*C - Y) .* J;
       CorrelationLoss = trace(LS' * LS);
       CorrelationLoss2 = trace(W*L*W');
       InstanceSimilarityLoss = trace((X*W)' * Linst * (X*W));
       sparesW    = sum(sum(W~=0));
       sparesC    = sum(sum(C~=0));
       totalloss = DiscriminantLoss + lambdaC*CorrelationLoss + ...
            lambdaW*sparesW + lambdaR*sparesC + lambdaL*CorrelationLoss2 +...
            lambdaI*InstanceSimilarityLoss;
      

       if abs((oldloss - totalloss)/oldloss) <= miniLossMargin
           break;
       elseif totalloss - oldloss >=0
           break;
       elseif totalloss <=0
           break;
       else
           oldloss = totalloss;
       end
       iter=iter+1;
    end
    model.W = W;
    model.C = C;
end

function W = softthres(W_t,lambda)
    W = max(W_t-lambda,0) - max(-W_t-lambda,0);  
end

function gradient = gradientOfW(W, C,lambdaL, XTLinstX, lambdaI, B, X, Y, J)
    L = diag(sum(C,2)) - C;
    gradient = X' * ((X*W).*B) - X' * ((Y*C).*B) + lambdaL*W*(L + L') + + lambdaI * XTLinstX * W; 
end

function gradient = gradientOfC(YTY,W, C, lambdaC, B, X, Y, J)
    gradient = Y' * ((Y * C) .* B)+ lambdaC * YTY*C - Y' * ((X*W) .* B) - lambdaC*YTY;
end

function [S, Linst] = computeLaplacianForInstanceSimilarity(X)
    S = exp(-squareform(pdist(X))); 
    Linst = diag(sum(S, 2)) - S; %keep everything, no k-nn
end

function [B] = computeCostSensitiveMatrix(Y, beta)
    ones_sum = sum(Y == 1);
    neg_ones_sum = sum(Y == -1);
    zeros_sum = sum(Y == 0);
    weighted_val = (neg_ones_sum + beta * zeros_sum)./(ones_sum + beta * zeros_sum);
    B = ones(size(Y));
    cols=size(Y, 2);
    for col=1:cols
        B(Y(:, col) == 1, col) = weighted_val(col);
    end  
end 