function [ y ] = dallaManModelODE(IIR,c,t0,t,x)
m = size(c,1);
assert(m == 3);
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
d_Isc1 = - 0.0171 * Isc1 + 100*IIR/102.3;
d_Isc2 = 0.0152 * Isc1 - 0.0078 * Isc2;
d_Gp =  4.7314 - 0.0047 * Gp - 0.0121 * Id + c(1)*(t0+t)^2 + c(2)*(t0+t) + c(3) - 1 - 0.0581*Gp +  0.0871*Gt ;%%EGP + Ra - Uii - E - params.k1 * Gp + params.k2 * Gt;
d_Gt = -0.0039* (3.2267+0.0313*X) * Gt * ( 1 - 0.0026 * Gt + 2.5097e-06 *Gt^2) + 0.0581 *Gp - 0.0871* Gt;  %% - Uid + params.k1 * Gp - params.k2 * Gt;
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
y(10,1) = d_Gs;
end

