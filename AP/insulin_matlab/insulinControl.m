function [newMode,I] = insulinControl(iMode, G)
    newMode = iMode;
    switch iMode
        case  0,
            if (G >= 80)
                newMode = 1;
            end
        case 1,
            if ( G <= 75)
                newMode = 0;
            elseif (G >= 120)
                newMode = 2;
            end
        case 2,
            if (G <= 115)
                newMode = 1;
            elseif (G>= 180)
                newMode = 3;
            end
        case 3,
            if (G <= 175)
                newMode = 2;
            elseif (G >= 300)
                newMode = 4;
            end
        case 4,
            if (G <= 295)
                newMode = 3;
            end
    end
    
    switch newMode
        case 0,
            I = 0.05;
        case 1,
            I = 0.1;
        case 2,
            I = 0.2;
        case 3,
            I  = 0.5;
        case 4,
            I = 1.4;
    end
    
end