function [target, totalNum, totaldeleteNum, realpercent,obsTarget_index]= getIncompleteTarget(target, percent, bQuiet)
% delete the elements in the target matrix 'oldtarget' by the given percent
% oldtarget : N by L data matrix
% percent   : 10%, 20%, 30%, 40%, 50%
%Comment: Modifies labels with 1(presetn) to 0 , not labels which are absent,
%Comment: Sanjay, added commented code to not only remove 1 but -1 as well.
obsTarget_index = ones(size(target));

%totalNum = numel(target); %better and shorter if both pos and neg are considered for incompletion: sanjay
totalNum = sum(sum(target ==1));
totalNum = totalNum + sum(sum(target == -1));%sanjay,for 1 and -1 as wel
totaldeleteNum = 0;
[N,~] = size(target);
realpercent = 0;
maxIteration = 50;
factor = 2;
count=0;
while realpercent < percent
    if maxIteration == 0
        factor = 1;
        maxIteration = 10;
        if count==1
            break;
        end
        count = count+1;
    else
        maxIteration = maxIteration - 1;
    end
    for i=1:N
        %index = find(target(i,:)==1);
        index = find(target(i,:)==1 | target(i,:)==-1); %sanjay
        if length(index) >= factor % can be set to be 1 if the real missing rate can not reach the pre-set value
            deleteNum = round(1 + rand*(length(index)-1));%至少保证该样本有个类别标签
            totaldeleteNum = totaldeleteNum + deleteNum;
            realpercent = totaldeleteNum/totalNum;
            if realpercent >= percent
                break;
            end
            if deleteNum > 0
                index = index(randperm(length(index)));
                target(i,index(1:deleteNum)) = 0; %key step, convert known(-1,1) to unknown (0)
                obsTarget_index(i,index(1:deleteNum))=0;
            end
        end
    end
end

if bQuiet == 0
    fprintf('\n  Totoal Number of Labeled Entities : %d\n ',totalNum);  
    fprintf('Number of Deleted Labeled Entities : %d\n ',totaldeleteNum);  
    fprintf('        Given percent/Real percent : %.2f / %.4f\n', percent,totaldeleteNum/totalNum);  
end
end