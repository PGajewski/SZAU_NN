%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.668866e+002; foe(n+1)=2.653538e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.725706e+002; foe(n+1)=1.712214e+002; krok(n+1)=4.927090e-004; ng(n+1)=1.202560e+003;
n=2; farx(n+1)=5.860557e+001; foe(n+1)=6.035308e+001; krok(n+1)=9.041785e-003; ng(n+1)=4.250120e+002;
n=3; farx(n+1)=5.533053e+001; foe(n+1)=5.610218e+001; krok(n+1)=1.892781e-003; ng(n+1)=3.099878e+002;
n=4; farx(n+1)=3.364313e+001; foe(n+1)=5.087027e+001; krok(n+1)=3.814009e-002; ng(n+1)=7.083027e+001;
n=5; farx(n+1)=8.152867e+000; foe(n+1)=2.361130e+001; krok(n+1)=2.001208e-003; ng(n+1)=4.777019e+002;
n=6; farx(n+1)=7.475571e+000; foe(n+1)=2.091020e+001; krok(n+1)=2.064885e-004; ng(n+1)=9.395232e+002;
n=7; farx(n+1)=7.055547e+000; foe(n+1)=1.955495e+001; krok(n+1)=8.121118e-004; ng(n+1)=1.155192e+003;
n=8; farx(n+1)=7.417835e+000; foe(n+1)=1.827693e+001; krok(n+1)=3.660956e-003; ng(n+1)=7.326139e+002;
n=9; farx(n+1)=5.895981e+000; foe(n+1)=1.582519e+001; krok(n+1)=1.012183e-002; ng(n+1)=1.153057e+002;
n=10; farx(n+1)=4.424304e+000; foe(n+1)=1.259333e+001; krok(n+1)=1.495979e-002; ng(n+1)=4.292690e+002;
n=11; farx(n+1)=4.090246e+000; foe(n+1)=1.213531e+001; krok(n+1)=7.348901e-004; ng(n+1)=6.555765e+002;
n=12; farx(n+1)=3.407684e+000; foe(n+1)=1.134823e+001; krok(n+1)=2.391214e-003; ng(n+1)=6.634302e+002;
n=13; farx(n+1)=2.894604e+000; foe(n+1)=1.028863e+001; krok(n+1)=5.294040e-003; ng(n+1)=1.124800e+003;
n=14; farx(n+1)=2.558914e+000; foe(n+1)=9.765521e+000; krok(n+1)=3.128460e-003; ng(n+1)=8.907508e+002;
n=15; farx(n+1)=2.160378e+000; foe(n+1)=9.110263e+000; krok(n+1)=1.193441e-002; ng(n+1)=5.657454e+002;
n=16; farx(n+1)=1.748239e+000; foe(n+1)=8.643619e+000; krok(n+1)=5.062842e-003; ng(n+1)=1.346842e+002;
n=17; farx(n+1)=1.425871e+000; foe(n+1)=7.888580e+000; krok(n+1)=1.086709e-002; ng(n+1)=6.509614e+002;
n=18; farx(n+1)=1.321450e+000; foe(n+1)=7.435850e+000; krok(n+1)=1.046070e-003; ng(n+1)=5.605551e+002;
n=19; farx(n+1)=1.265220e+000; foe(n+1)=6.952659e+000; krok(n+1)=8.029022e-003; ng(n+1)=7.460067e+002;
n=20; farx(n+1)=1.121197e+000; foe(n+1)=6.506882e+000; krok(n+1)=7.366454e-003; ng(n+1)=4.687578e+002;
n=21; farx(n+1)=1.015861e+000; foe(n+1)=5.601957e+000; krok(n+1)=5.915528e-003; ng(n+1)=6.942842e+002;
n=22; farx(n+1)=1.005260e+000; foe(n+1)=5.295010e+000; krok(n+1)=3.625910e-003; ng(n+1)=3.786752e+002;
n=23; farx(n+1)=1.050446e+000; foe(n+1)=4.572152e+000; krok(n+1)=6.970444e-003; ng(n+1)=4.286367e+002;
n=24; farx(n+1)=1.159997e+000; foe(n+1)=3.891085e+000; krok(n+1)=4.766707e-003; ng(n+1)=4.531684e+002;
n=25; farx(n+1)=1.194456e+000; foe(n+1)=3.489909e+000; krok(n+1)=5.915258e-003; ng(n+1)=5.139822e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=1.170184e+000; foe(n+1)=3.240353e+000; krok(n+1)=1.787154e-005; ng(n+1)=6.046127e+002;
n=27; farx(n+1)=1.164684e+000; foe(n+1)=3.203369e+000; krok(n+1)=1.302585e-005; ng(n+1)=2.134702e+002;
n=28; farx(n+1)=1.130699e+000; foe(n+1)=3.143671e+000; krok(n+1)=7.394073e-004; ng(n+1)=4.775426e+001;
n=29; farx(n+1)=7.825219e-001; foe(n+1)=2.274057e+000; krok(n+1)=4.266621e-003; ng(n+1)=6.988945e+001;
n=30; farx(n+1)=6.331234e-001; foe(n+1)=1.637881e+000; krok(n+1)=8.155298e-003; ng(n+1)=2.498787e+002;
n=31; farx(n+1)=6.095341e-001; foe(n+1)=1.427447e+000; krok(n+1)=2.191235e-002; ng(n+1)=5.260987e+001;
n=32; farx(n+1)=6.083780e-001; foe(n+1)=1.365643e+000; krok(n+1)=1.469780e-003; ng(n+1)=2.769325e+002;
n=33; farx(n+1)=5.946685e-001; foe(n+1)=1.292070e+000; krok(n+1)=1.940049e-003; ng(n+1)=2.873570e+002;
n=34; farx(n+1)=5.462187e-001; foe(n+1)=1.174266e+000; krok(n+1)=1.669364e-002; ng(n+1)=7.179771e+001;
n=35; farx(n+1)=5.165083e-001; foe(n+1)=1.088251e+000; krok(n+1)=2.331822e-002; ng(n+1)=3.022546e+002;
n=36; farx(n+1)=4.838024e-001; foe(n+1)=1.029548e+000; krok(n+1)=1.129495e-002; ng(n+1)=1.311016e+002;
n=37; farx(n+1)=4.696191e-001; foe(n+1)=9.784103e-001; krok(n+1)=1.571994e-002; ng(n+1)=2.463382e+002;
n=38; farx(n+1)=4.540687e-001; foe(n+1)=9.358048e-001; krok(n+1)=1.289397e-002; ng(n+1)=1.065040e+002;
n=39; farx(n+1)=4.582483e-001; foe(n+1)=9.066032e-001; krok(n+1)=2.362755e-002; ng(n+1)=1.553017e+002;
n=40; farx(n+1)=4.426355e-001; foe(n+1)=8.806181e-001; krok(n+1)=2.421456e-002; ng(n+1)=1.275342e+002;
n=41; farx(n+1)=4.423288e-001; foe(n+1)=8.522651e-001; krok(n+1)=3.786424e-002; ng(n+1)=1.298910e+002;
n=42; farx(n+1)=4.462950e-001; foe(n+1)=7.947873e-001; krok(n+1)=1.283701e-001; ng(n+1)=7.008146e+001;
n=43; farx(n+1)=4.435768e-001; foe(n+1)=7.554943e-001; krok(n+1)=5.857529e-002; ng(n+1)=1.023828e+002;
n=44; farx(n+1)=4.434862e-001; foe(n+1)=7.444029e-001; krok(n+1)=9.876893e-003; ng(n+1)=1.132486e+002;
n=45; farx(n+1)=4.230127e-001; foe(n+1)=6.909504e-001; krok(n+1)=9.963479e-002; ng(n+1)=6.694980e+001;
n=46; farx(n+1)=4.018766e-001; foe(n+1)=6.636729e-001; krok(n+1)=3.252259e-002; ng(n+1)=7.721635e+001;
n=47; farx(n+1)=3.718448e-001; foe(n+1)=6.334526e-001; krok(n+1)=8.776600e-002; ng(n+1)=5.579420e+001;
n=48; farx(n+1)=3.609665e-001; foe(n+1)=6.221748e-001; krok(n+1)=1.818608e-002; ng(n+1)=8.045011e+001;
n=49; farx(n+1)=3.530706e-001; foe(n+1)=6.034135e-001; krok(n+1)=3.542231e-002; ng(n+1)=5.077508e+001;
n=50; farx(n+1)=3.528487e-001; foe(n+1)=5.741664e-001; krok(n+1)=1.115271e-001; ng(n+1)=3.625465e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
