function c = getMealCoeffs(t,carbs,cutPoints,C)
    c = zeros(3,1);
    m = size(cutPoints,1);
    
    if (t < cutPoints(1,1))
        fprintf(1,'%f , %f',t,cutPoints(1,1));
       assert(false,'Cutpoints for meal model linearization improperly chosen'); 
    end
    for i = 1:(m-1)
        if (t >= cutPoints(i,1) && t < cutPoints(i+1,1))
            c = carbs*C(:,i);
            return 
        end
    end
    c = carbs*C(:,m);
end

