function [y] = dallaManODEOrig(params,c,IIR,t0,t,x)
        
        %% Sources
        %% DM2007: Dalla Man et al. Meal Simulation Model of the Glucose-Insulin System IEEE Trans. on BME, 54(10), 2007
        %% DM2006: Dalla Man et al. A system model for oral glucose absorption: validation on gold standard data. IEEE Trans. on BME. 53(12), 2006.
        %% Inputs
        
        %% tMeal: Meal time.
        %% D: meal carbs amount.
        %% IIR: subcutaneousinsulin infusion signal.
        y=zeros(10,1);
        
        %%D = mCarbs;
        IIR = IIR*100/params.weight;
        Ra = c(1,1) * (t0+t)^2 + c(2,1)*(t0+t) + c(3,1);
      %%  fprintf(1,'%f %f\n',t,Ra);
        %% State Variables
        
        %% X: remote chamber insulin conc. X(0) = 0
        %% Isc1: Inslin in Subcutaneous chamber. Isc1(0) = params.Isc1ss
        %% Isc2: Insulin in Subcutaneous chamber2. Isc2(0) = params.Isc2ss
        %% Gt: glucose levels in rapidly equilibriating tissues, Gt(0) = params.Gtb
        %% Gp: glucose level in the plasma. Gp(0) = params.Gpb
        %% Il: Insulin in Liver. Il(0) = params.Ilb
        %% Ip: Insulin in Plasma. Ip(0) = params.Ipb
        %% I1: Insulin in a remote chamber. I1(0) = params.Ib
        %% Id: "Delayed" insulin signal. Id(0) = params.Ib
        %% Gs: sensor glucose reading. Gs(0) = Gb
        
        X = x(1,1);
        Isc1 = x(2,1);
        Isc2 = x(3,1);
        Gt = x(4,1);
        Gp = x(5,1);
        Il = x(6,1);
        Ip = x(7,1);
        I1 = x(8,1);
        Id = x(9,1);
        %%GUT Qsto1 = x(10,1);
        %%GUT Qsto2 = x(11,1);
        %% GUT Qgut = x(12,1);
        Gs = x(10,1);
        
        %% Calculated Variables
        %% EGP  : Endogenous Glucose Production
        %% HE: Hepatic extraction of insulin
        %% m3: Liver insulin extraction rate
        %% Ra: Rate of appearance of glucose
        %% Uii: Insulin indepenendent utilization of glucose
        %% Uid: Insulin dependent utliziation of glucose.
        %% Vm, Km: Linear functions of remote insulin concentrations.
        %% R: rate of appearance of insulin in the plasma.
        %% E: renal glucose clearnance.
        %% G: plasma glucose concentration (mg/dl)
        %% I: plasma insulin concentration (pmol/L)
        %% Qsto: glucose in stomuch
        %% Ra: Rate of appearance of glucose in blood.
        
        %% parameters
        %%
        
        %% Equation (1) in DM2007
        G = Gp/params.Vg;
        %% Equation (3) in DM2007fileName
        I = Ip/params.Vi;
        
        %% Equation (10) in DM2007
        EGP = params.kp1 - params.kp2 * Gp - params.kp3 * Id ;
        
        %% Equation (8) in DM2007
        
        %% Equation (8) in DM 2007
        %%Db = Sb;
        %% Equation (6) in DM 2007
        
        %% Equation (2) of DM-GIM-07
        %%S = params.ka1 * Isc1 + params.ka2 * Isc2;
        S = 0; %% type 1 diabetes no secretion.
        insulinRateOfAppearance = params.ka1 * Isc1 + params.ka2 * Isc2;
        
        %% Equation (4) in DM 2007
        %%HE = -params.m5 * S + params.m6;
        
        HE = params.m6;
        
        %% Equation(5) in DM 2007
        m3 = HE * params.m1 / ( 1- HE);
        
        %% Equation (13) in DM 2007
        %%GUT Ra = params.f * params.kabs * Qgut / params.weight;
        
        %% Utilization model
        %% Eq. (14) in DM 2007
        Uii = params.Fcns;
        %% Equation (16) in DM 2007
        Vm = params.Vm0 + params.Vmx * X;
        %% Equation (17) in DM 2007. Note: According to note after (17), Kmx = 0
        Km = params.Km0 ;
        %% Equation (15) of DM2007
        Uid = Vm * Gt / (Km + Gt);
        %% Equation (18) of DM2007
        d_X = - params.p2u * X + params.p2u * ( I - params.Ib);
        
        %% Insulin Input Model
        %% Equation (1) of DM-GIM-07
        d_Isc1 = - (params.kd + params.ka1) * Isc1 + IIR;
        %% Equation(1) of DM-GIM-07
        d_Isc2 = params.kd * Isc1 - params.ka2 * Isc2;
        
        %% Glucose Renal Model
        %% Equation (27) of DM 2007
        if ( Gp > params.ke2)
            E = params.ke1 * (Gp - params.ke2);
        else
            E = 0;
        end
        %% Glucose Model
        %% Equation (1) of DM 2007
        d_Gp = EGP + Ra - Uii - E - params.k1 * Gp + params.k2 * Gt;
        d_Gt = - Uid + params.k1 * Gp - params.k2 * Gt;
        d_Gs = 1/params.Ts*( G - Gs);
        %% Insulin action model
        %% Equation (3) of DM 2007
        d_Il = - (params.m1 + m3) * Il + params.m2 * Ip + S;
        d_Ip = - (params.m2 + params.m4) * Ip + params.m1 * Il + insulinRateOfAppearance ;
        
        %% insulin transport model
        %% Equation (11) of DM 2007
        d_I1 = -params.ki * ( I1 - I);
        d_Id = -params.ki * (Id - I1 );
        
        % %% Equation to model an impuse response function for meal absorption.
        % %% Gut absorption submodel
        % D = mCarbs;
        % if (t > tMeal && t < tMeal + mDuration)
        %     dImp = mCarbs/mDuration;
        %     D = mCarbs * ( t- tMeal)/mDuration;
        % else
        %     dImp = 0;
        %
        % end
        % %% Meal Gut absorption submodel taken from  DM2006
        %
        % Qsto = Qsto1 + Qsto2;
        % alpha = 5 / ( 2 * mCarbs * (1-params.b));
        % beta = 5/ (2 * mCarbs * params.c);
        %
        % kempt =  params.kmin + (params.kmax - params.kmin)/2 * ( 2+  tanh( alpha * (Qsto - params.b * D) ) - tanh( beta * (Qsto - params.c * D)));
        % d_Qsto1 = - params.kmax * Qsto1 + dImp;
        % d_Qsto2 = - kempt * Qsto2 + params.kmax * Qsto1;
        % d_Qgut = - params.kabs * Qgut + kempt * Qsto2;
        
        y(1,1)=d_X;
        y(2,1)=d_Isc1;
        y(3,1) = d_Isc2 ;
        y(4,1) = d_Gt ;
        y(5,1) = d_Gp;
        y(6,1) = d_Il ;
        y(7,1) = d_Ip ;
        y(8,1) = d_I1;
        y(9,1) = d_Id ;
        % y(10,1) = d_Qsto1 ;
        % y(11,1) = d_Qsto2;
        % y(12,1) = d_Qgut;
        y(10,1) = d_Gs;
        
        
        
        
    end

function [ y ] = simplifiedODE(t,x)

X = x(1,1);
Isc1 = x(2,1);
Isc2 = x(3,1);
Gt = x(4,1);
Gp = x(5,1);
Il = x(6,1);
Ip = x(7,1);
I1 = x(8,1);
Id = x(9,1);
Gs = x(10,1);


%% Sriram: Renal clearance dropped.
d_X = - 0.0278 * X +0.0278 * ( 18.2129 * Ip - 100.25);
d_Isc1 = - 0.0171 * Isc1 + IIR;
d_Isc2 = 0.0152 * Isc1 - 0.0078 * Isc2;
d_Gp =  4.7314 - 0.0047 * Gp - 0.0121 * Id + Ra - 1 - 0.0581*Gp +  0.0871*Gt ;%%EGP + Ra - Uii - E - params.k1 * Gp + params.k2 * Gt;
d_Gt = -(3.2267+0.0313*X) * Gt / 253.52 + 0.0581 *Gp - 0.0871* Gt;  %% - Uid + params.k1 * Gp - params.k2 * Gt;
d_Gs = 0.1*( 0.5221 * Gp - Gs);
d_Il = - 0.4219 * Il + 0.2250* Ip;
d_Ip = - 0.3150 * Ip + 0.1545 * Il + 0.0019 * Isc1 + 0.0078 * Isc2 ;
d_I1 = -0.0046 * ( I1 - 18.2129 * Ip);
d_Id = -0.0046 * (Id - I1 );


y(1,1)=d_X;
y(2,1)=d_Isc1;
y(3,1) = d_Isc2 ;
y(4,1) = d_Gt ;
y(5,1) = d_Gp;
y(6,1) = d_Il ;
y(7,1) = d_Ip ;
y(8,1) = d_I1;
y(9,1) = d_Id ;
% y(10,1) = d_Qsto1 ;
% y(11,1) = d_Qsto2;
% y(12,1) = d_Qgut;
y(10,1) = d_Gs;
end
