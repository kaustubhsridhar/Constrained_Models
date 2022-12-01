function mode = initializeInsulinControl(G)
    if (G <= 80)
        mode  = 0;
    elseif (G <= 120)
        mode =1;
    elseif (G <= 180)
        mode = 2;
    elseif (G <= 300)
        mode = 3;
    else 
        mode = 4;
    end
end