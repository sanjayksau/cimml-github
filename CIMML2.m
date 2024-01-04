function model = CIMML2( X, Y, optmParameter)
%    Syntax
%
%      [model] = CIMML2( X, Y, optmParameter)
%       model.W = ;
%       model.C =  ; %paper will have R instead of C
%
%    Input
%       X               - a n by d data matrix, n is the number of instances and d is the number of features 
%       Y               - a n by l label matrix, n is the number of instances and l is the number of labels
%       optmParameter   - the optimization parameters for CIMML, a struct variable with several fields, 
%
%    Output
%
%       model    -  the model coefficients model.W and model.R
   %% optimization parameters
    lambdaC          = optmParameter.lambdaC; % missing labels |YR-Y|
    lambdaR          = optmParameter.lambdaR; % regularization of R |R|_1
    lambdaW          = optmParameter.lambdaW; % regularization of W |W|_1
    lambdaL          = optmParameter.lambdaL; % regularization of graph laplacian WLW'
    lambdaI          = optmParameter.lambdaI; % regularization of instance similarity (XW)'LinstXW
    
    beta             = optmParameter.beta;    % adaptive coeff for cost sensitive matrix
    
    rho              = optmParameter.rho;
    eta              = optmParameter.eta;
    isBacktracking   = optmParameter.isBacktracking;
    maxIter          = optmParameter.maxIter;
    miniLossMargin   = optmParameter.minimumLossMargin;

    J =  (Y~=0); %observed indices
    %J = ones(size(Y));
    
    num_dim   = size(X,2);
    num_class = size(Y,2);
    XTX = X'*X;
    XTY = X'*Y;
    YTY = Y'*Y;
    [~, Linst] = computeLaplacianForInstanceSimilarity(X);
    XTLinstX = X' * Linst * X;
    
   %% initialization
    %W   = (XTX + rho*eye(num_dim)) \ (XTY); %zeros(num_dim,num_class); % 
    W = zeros(num_dim,num_class);
    W_1 = W; W_k = W;
    C = zeros(num_class,num_class); %eye(num_class,num_class);
    C_1 = C;
    
    B = computeCostSensitiveMatrix(Y, beta);
    %B = B .^ 0.5; 
    maxB2 = max(B.^2, [], 'all');
        
    iter = 1; oldloss = 10^9;
    bk = 1; bk_1 = 1; 
    
    %https://math.stackexchange.com/questions/351544/bounded-partial-derivatives-imply-continuity/412704#412704

    
    LipW2 = 3*maxB2 * norm(XTX)^2 +3* norm(lambdaI * XTLinstX)^2;
    LipR2  = 2*maxB2 * norm(YTY)^2 + 2* norm(lambdaC*YTY)^2;

    
    while iter <= maxIter
       L = diag(sum(C,2)) - C;
       lapterm = 3 * norm(lambdaL*(L+L'))^2; %laplacian term of LipW2 
       %Lip^2 = sqrt(LipW^2 + LipR^2);
       %or Lip^2 = sqrt(2) * max(LipR, LipW)
       Lip = sqrt( LipR2 + LipW2 + lapterm);
       %Lip = sqrt(Lip);
          
       %LipW_2 = 3 * norm(lambdaL*(L+L')); 
       %LipW = sqrt(LipW_1 + LipW_2);
       %Lip = sqrt(2) * max(LipR, LipW);
       

      %% update C
       C_k  = C + (bk_1 - 1)/bk * (C - C_1);
       Gc_k = C_k - 1/Lip * gradientOfC(YTY, W, C_k, lambdaC, B, X, Y, J);
       C_1  = C;
       C    = softthres(Gc_k,lambdaR/Lip); 
       C    = max(C,0);
       
      %% update W
       W_k  = W + (bk_1 - 1)/bk * (W - W_1);
       %Gw_x_k = W_k - 1/Lip * gradientOfW(XTX,XTY,W_k,C,lambda4);
       Gw_x_k = W_k - 1/Lip * gradientOfW(W_k,C,lambdaL, XTLinstX, lambdaI, B, X, Y, J);
       W_1  = W;
       W    = softthres(Gw_x_k,lambdaW/Lip);
       
       bk_1   = bk;
       bk     = (1 + sqrt(4*bk^2 + 1))/2;
      
      %% Loss
       
       %HingeL = max(0, (E - (Y*C).*(X*W))) .* J ;
       LS = (X*W - Y*C);
       DiscriminantLoss = 0.5 * norm((LS .* LS).*B, 1);
       %DiscriminantLoss = trace(LS'* LS);
       %DiscriminantLoss = norm((X*W - Y*C) .* LS, 1);
 
       LS = (Y*C - Y) .* J;
       CorrelationLoss = trace(LS' * LS);
       CorrelationLoss2 = trace(W*L*W');
       InstanceSimilarityLoss = trace((X*W)' * Linst * (X*W));
       sparesW    = sum(sum(W~=0));
       sparesC    = sum(sum(C~=0));
       totalloss = DiscriminantLoss + lambdaC*CorrelationLoss + ...
            lambdaW*sparesW + lambdaR*sparesC + lambdaL*CorrelationLoss2 +...
            lambdaI*InstanceSimilarityLoss;
      
       loss(iter,1) = totalloss;
       %sparesCLoss(iter, 1) = sparesC;
       %sparesWLoss(iter, 1) = sparesW;
       %dLoss(iter, 1) = DiscriminantLoss;
       %corrLoss(iter, 1) = CorrelationLoss;
       %corr2Loss(iter, 1) = CorrelationLoss2;
       instsimLoss(iter, 1) = InstanceSimilarityLoss;
       
       %fprintf('%0.4f  %0.4f  %0.4f  %0.4f  %0.4f %0.4f\n', DiscriminantLoss, ...
       %    CorrelationLoss, sparesC, sparesW, CorrelationLoss2, InstanceSimilarityLoss );
       fprintf('%0.4f  %0.4f  %0.4f  %0.4f  %0.4f  %0.4f\n', DiscriminantLoss, ...
           lambdaC*CorrelationLoss, lambdaW*sparesW, lambdaR*sparesC,    ...
           lambdaL*CorrelationLoss2, lambdaI* InstanceSimilarityLoss );
       fprintf('oldloss %0.4f newloss %0.4f\n', oldloss, totalloss);

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
    %plot(instsimLoss);
    %hold
    plot(loss);
    model.W = W;
    model.C = C;
    model.loss = loss;
    model.optmParameter = optmParameter;
end

%% soft thresholding operator
function W = softthres(W_t,lambda)
    W = max(W_t-lambda,0) - max(-W_t-lambda,0);  
end

function gradient = gradientOfW(W, C,lambdaL, XTLinstX, lambdaI, B, X, Y, J)
    L = diag(sum(C,2)) - C;
    %gradient = XTX*W - XTY*C + lambda4*W*(L + L') + lambda5 * XTLinstX * W;
    gradient = X' * ((X*W).*B) - X' * ((Y*C).*B) + lambdaL*W*(L + L') + + lambdaI * XTLinstX * W; 
end

function gradient = gradientOfC(YTY,W, C, lambdaC, B, X, Y, J)
    %gradient = (lambda1+1)*YTY*C - XTY'*W - lambda1*YTY;
    gradient = Y' * ((Y * C) .* B)+ lambdaC * YTY*C - Y' * ((X*W) .* B) - lambdaC*YTY;
end


function [S, Linst] = computeLaplacianForInstanceSimilarity(X)
%input X: n x d
    S = exp(-squareform(pdist(X))); 
    Linst = diag(sum(S, 2)) - S; %keep everything, no k-nn
%     %similarity x_i, x_j in [0, 1]
%     S = exp(-squareform(pdist(X))); 
%     knn = 5;
%     %keep only the top k-nn
%     n = size(X, 1);
%     for col1=1:n %probably can be done for all of matrix at once.
%         [~, idx] = maxk(S(:, col1), knn);
%         S(setdiff(1:end, idx), col1) = 0;
%     end
%     Linst = diag(sum(S, 2)) - S;
end


function [B] = computeCostSensitiveMatrix(Y, beta)
    ones_sum = sum(Y == 1);
    neg_ones_sum = sum(Y == -1);
    zeros_sum = sum(Y == 0);
    weighted_val = (neg_ones_sum + beta * zeros_sum)./(ones_sum + beta * zeros_sum);
    
    B = ones(size(Y));
    cols=size(Y, 2);
    for col=1:cols
        %Update weight of positive labels
        B(Y(:, col) == 1, col) = weighted_val(col);
    end  
end 

% OLD
% function [B]= computeCostSensitiveMatrix(Y, beta)
%     ones_sum = sum(Y == 1);
%     neg_ones_sum = sum(Y == 0); %-1 in CPNL
% 
%     weighted_val = (neg_ones_sum ./ ones_sum) .^ beta;
%     cols = size(Y, 2);
%     
%     B = ones(size(Y));
%     for col=1:cols
%         %Update weight of positive labels
%         B(Y(:,col) == 1, col) = weighted_val(col);
%     end
% end


