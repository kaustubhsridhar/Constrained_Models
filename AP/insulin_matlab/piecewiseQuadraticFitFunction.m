function  coeffs=piecewiseQuadraticFitFunction(x,y,cutPoints )

m = size(cutPoints,1);
plot(x,y);
hold on;
coeffs=zeros(3,m);
for i=1:(m-1)
   t0 = cutPoints(i,1);
   t1 = cutPoints(i+1,1);
   mid = floor((t0+t1)/2);
   A = [ t0^2 t0 1;
         t1^2 t1 1;
         mid^2 mid 1];
   b = [y(t0+1,1); y(t1+1,1); y(mid+1,1)];
   coeffs(:,i)=inv(A)*b;
   slice=t0:t1;
   plot(slice,coeffs(1,i)*slice.^2 + coeffs(2,i)*slice + coeffs(3,i));
   hold on;
end

end

