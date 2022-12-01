function [times,gValues,gsValues,iValues, mealValues] = simulateClosedLoop(G0,carbs,cutPoints,C)
%% G0 = values of carbohydrates
    simTime = 720;
    startTime = 0;
    step = 5;
    x=[ 0;72.43; 141.15; 162.45; 1.9152*G0; 3.20; 5.50; 100.25; 100.25;G0]; %% Initial state vector
    times=zeros(simTime,1);
    gValues=zeros(simTime,1);
    gsValues=zeros(simTime,1);
    iValues = zeros(simTime,1);
    mealValues = zeros(simTime, 1);
    iMode = initializeInsulinControl(G0);
    options=odeset('MaxStep',0.5);
    for t = startTime:step:simTime
        %% compute coefficients for meal input
        c = getMealCoeffs(t,carbs,cutPoints,C); %% Piece wise polynomial function on time.
        %% Compute sensor value = x10 + [-10,10] %% time varying uncertainty
        Gs = x(10,1) - 10 + 20 * rand; %% Simulate Random Noise [-10,10] on glucose value
        gsValues(t+1,1) = Gs; 
        %% Compute the control value for next 5 minutes
        I = insulinControl(iMode,Gs); %% Calculate insulin
        iValues(t+1:t+step,:)= I * ones(step,1);
        %% Do a simulation with that value of the control Input I and meal input c1*t^2 + c2 * t + c3
        mealValues(t+1:t+step,:) = c(1)*(t)^2 + c(2)*(t) + c(3);
        dallaManModel = @(s,x) dallaManModelODE(I,c,t,s,x); %%@(s,x) dallaManODEOrig(params,c,I,t,s,x);   
        [~,S] = ode45(dallaManModel, 1:1:step, x, options);
        times(t+1:t+step,:) = t+1:t+step;
        gValues(t+1:t+step,:) = 0.522*S(:,5);
        x = S(end,:)';
    end
    
end