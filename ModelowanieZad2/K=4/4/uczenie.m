%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.948354e+002; foe(n+1)=2.960846e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.768609e+002; foe(n+1)=1.780326e+002; krok(n+1)=4.966995e-004; ng(n+1)=9.060727e+002;
n=2; farx(n+1)=6.480080e+001; foe(n+1)=6.548173e+001; krok(n+1)=2.452306e-002; ng(n+1)=2.228960e+002;
n=3; farx(n+1)=5.106664e+001; foe(n+1)=5.440512e+001; krok(n+1)=2.114007e-004; ng(n+1)=9.978825e+002;
n=4; farx(n+1)=4.836028e+001; foe(n+1)=5.336069e+001; krok(n+1)=6.694872e-004; ng(n+1)=1.711122e+002;
n=5; farx(n+1)=1.753584e+001; foe(n+1)=3.887450e+001; krok(n+1)=4.980847e-003; ng(n+1)=1.800045e+002;
n=6; farx(n+1)=1.630838e+001; foe(n+1)=3.137570e+001; krok(n+1)=4.375565e-003; ng(n+1)=1.495703e+003;
n=7; farx(n+1)=1.759019e+001; foe(n+1)=2.945079e+001; krok(n+1)=7.088910e-003; ng(n+1)=1.115551e+003;
n=8; farx(n+1)=1.660975e+001; foe(n+1)=2.803035e+001; krok(n+1)=3.581322e-003; ng(n+1)=3.788557e+002;
n=9; farx(n+1)=1.809160e+001; foe(n+1)=2.568087e+001; krok(n+1)=1.257761e-002; ng(n+1)=2.489187e+002;
n=10; farx(n+1)=1.760490e+001; foe(n+1)=2.538797e+001; krok(n+1)=2.173143e-003; ng(n+1)=3.003331e+002;
n=11; farx(n+1)=1.627447e+001; foe(n+1)=2.290812e+001; krok(n+1)=3.323730e-002; ng(n+1)=2.605542e+002;
n=12; farx(n+1)=1.557537e+001; foe(n+1)=2.178776e+001; krok(n+1)=3.065383e-003; ng(n+1)=3.284166e+002;
n=13; farx(n+1)=1.386656e+001; foe(n+1)=2.097453e+001; krok(n+1)=7.444102e-003; ng(n+1)=2.547920e+002;
n=14; farx(n+1)=9.802298e+000; foe(n+1)=1.848347e+001; krok(n+1)=8.346822e-003; ng(n+1)=3.635148e+002;
n=15; farx(n+1)=5.773331e+000; foe(n+1)=1.473967e+001; krok(n+1)=2.521242e-002; ng(n+1)=3.193594e+002;
n=16; farx(n+1)=4.069101e+000; foe(n+1)=1.248223e+001; krok(n+1)=4.464645e-003; ng(n+1)=7.556349e+002;
n=17; farx(n+1)=3.583182e+000; foe(n+1)=1.206157e+001; krok(n+1)=1.238677e-002; ng(n+1)=3.786576e+002;
n=18; farx(n+1)=2.845767e+000; foe(n+1)=1.132677e+001; krok(n+1)=1.932256e-002; ng(n+1)=6.268418e+002;
n=19; farx(n+1)=2.167483e+000; foe(n+1)=9.028855e+000; krok(n+1)=8.973318e-002; ng(n+1)=5.547065e+002;
n=20; farx(n+1)=1.983983e+000; foe(n+1)=8.562859e+000; krok(n+1)=4.000365e-002; ng(n+1)=1.079958e+002;
n=21; farx(n+1)=1.844595e+000; foe(n+1)=7.935528e+000; krok(n+1)=1.091056e-001; ng(n+1)=2.631920e+002;
n=22; farx(n+1)=1.591786e+000; foe(n+1)=6.737609e+000; krok(n+1)=1.238533e-001; ng(n+1)=8.919740e+002;
n=23; farx(n+1)=1.565373e+000; foe(n+1)=5.345371e+000; krok(n+1)=1.482616e-001; ng(n+1)=9.448780e+002;
n=24; farx(n+1)=8.809214e-001; foe(n+1)=3.725656e+000; krok(n+1)=4.536903e-001; ng(n+1)=7.222565e+002;
n=25; farx(n+1)=6.830331e-001; foe(n+1)=2.869153e+000; krok(n+1)=1.666377e-001; ng(n+1)=9.025895e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=6.694934e-001; foe(n+1)=2.665469e+000; krok(n+1)=2.717293e-005; ng(n+1)=3.260995e+002;
n=27; farx(n+1)=6.673599e-001; foe(n+1)=2.612963e+000; krok(n+1)=2.647223e-005; ng(n+1)=1.671986e+002;
n=28; farx(n+1)=6.361570e-001; foe(n+1)=2.351395e+000; krok(n+1)=1.107642e-004; ng(n+1)=1.757477e+002;
n=29; farx(n+1)=6.060643e-001; foe(n+1)=2.162941e+000; krok(n+1)=2.583925e-003; ng(n+1)=4.808455e+001;
n=30; farx(n+1)=6.004864e-001; foe(n+1)=1.855899e+000; krok(n+1)=4.708341e-003; ng(n+1)=4.804158e+001;
n=31; farx(n+1)=5.820193e-001; foe(n+1)=1.702712e+000; krok(n+1)=1.779489e-003; ng(n+1)=1.491268e+002;
n=32; farx(n+1)=5.877760e-001; foe(n+1)=1.524186e+000; krok(n+1)=1.042882e-002; ng(n+1)=2.940864e+002;
n=33; farx(n+1)=5.561507e-001; foe(n+1)=1.346548e+000; krok(n+1)=1.883428e-002; ng(n+1)=9.537553e+001;
n=34; farx(n+1)=5.487323e-001; foe(n+1)=1.308777e+000; krok(n+1)=2.308292e-002; ng(n+1)=8.172816e+001;
n=35; farx(n+1)=4.806831e-001; foe(n+1)=1.188780e+000; krok(n+1)=3.712452e-002; ng(n+1)=1.266030e+002;
n=36; farx(n+1)=3.969041e-001; foe(n+1)=1.073724e+000; krok(n+1)=1.387866e-001; ng(n+1)=1.384194e+002;
n=37; farx(n+1)=4.069747e-001; foe(n+1)=1.011686e+000; krok(n+1)=5.082412e-002; ng(n+1)=6.979732e+001;
n=38; farx(n+1)=4.128523e-001; foe(n+1)=9.776808e-001; krok(n+1)=1.223774e-002; ng(n+1)=1.374632e+002;
n=39; farx(n+1)=3.903622e-001; foe(n+1)=9.438022e-001; krok(n+1)=4.953895e-002; ng(n+1)=1.390529e+002;
n=40; farx(n+1)=3.978226e-001; foe(n+1)=9.022547e-001; krok(n+1)=1.545805e-001; ng(n+1)=6.304545e+001;
n=41; farx(n+1)=3.883492e-001; foe(n+1)=8.807063e-001; krok(n+1)=6.073322e-002; ng(n+1)=1.007325e+002;
n=42; farx(n+1)=3.630578e-001; foe(n+1)=8.349805e-001; krok(n+1)=4.992693e-001; ng(n+1)=9.497500e+001;
n=43; farx(n+1)=3.642514e-001; foe(n+1)=8.143973e-001; krok(n+1)=1.634459e-001; ng(n+1)=1.109417e+002;
n=44; farx(n+1)=3.650790e-001; foe(n+1)=7.955757e-001; krok(n+1)=2.591726e-001; ng(n+1)=3.667776e+001;
n=45; farx(n+1)=3.664181e-001; foe(n+1)=7.825999e-001; krok(n+1)=2.431780e-001; ng(n+1)=6.985663e+001;
n=46; farx(n+1)=3.656663e-001; foe(n+1)=7.521974e-001; krok(n+1)=4.543926e-001; ng(n+1)=5.020571e+001;
n=47; farx(n+1)=3.565963e-001; foe(n+1)=7.267953e-001; krok(n+1)=4.268310e-001; ng(n+1)=4.596612e+001;
n=48; farx(n+1)=3.537799e-001; foe(n+1)=6.934289e-001; krok(n+1)=4.635137e-001; ng(n+1)=7.971836e+001;
n=49; farx(n+1)=3.314059e-001; foe(n+1)=6.765915e-001; krok(n+1)=3.556863e-001; ng(n+1)=5.916868e+001;
n=50; farx(n+1)=3.188223e-001; foe(n+1)=6.599977e-001; krok(n+1)=3.745525e-001; ng(n+1)=2.238733e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)