%uczenie predyktora oe
clear all;
n=0; farx(n+1)=1.841373e+002; foe(n+1)=1.869252e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.558492e+002; foe(n+1)=1.591992e+002; krok(n+1)=5.885714e-004; ng(n+1)=8.563355e+002;
n=2; farx(n+1)=6.030226e+001; foe(n+1)=6.288790e+001; krok(n+1)=4.260904e-003; ng(n+1)=5.656006e+002;
n=3; farx(n+1)=5.612002e+001; foe(n+1)=5.716761e+001; krok(n+1)=2.254077e-003; ng(n+1)=3.268057e+002;
n=4; farx(n+1)=4.818675e+001; foe(n+1)=5.580847e+001; krok(n+1)=1.303720e-002; ng(n+1)=5.222051e+001;
n=5; farx(n+1)=2.362022e+001; foe(n+1)=4.516904e+001; krok(n+1)=1.305012e-002; ng(n+1)=1.683124e+002;
n=6; farx(n+1)=1.544041e+001; foe(n+1)=4.051973e+001; krok(n+1)=3.888965e-003; ng(n+1)=4.420935e+002;
n=7; farx(n+1)=1.122689e+001; foe(n+1)=3.815068e+001; krok(n+1)=8.160960e-004; ng(n+1)=7.395458e+002;
n=8; farx(n+1)=8.447015e+000; foe(n+1)=3.347489e+001; krok(n+1)=2.590317e-002; ng(n+1)=1.217567e+003;
n=9; farx(n+1)=8.203912e+000; foe(n+1)=3.217395e+001; krok(n+1)=1.935115e-004; ng(n+1)=1.841495e+003;
n=10; farx(n+1)=7.384213e+000; foe(n+1)=2.859222e+001; krok(n+1)=3.382411e-003; ng(n+1)=1.944441e+003;
n=11; farx(n+1)=5.318262e+000; foe(n+1)=2.416287e+001; krok(n+1)=8.017859e-003; ng(n+1)=2.518717e+003;
n=12; farx(n+1)=4.175429e+000; foe(n+1)=1.849010e+001; krok(n+1)=1.037622e-003; ng(n+1)=2.155734e+003;
n=13; farx(n+1)=3.821402e+000; foe(n+1)=1.610395e+001; krok(n+1)=1.499941e-003; ng(n+1)=1.324510e+003;
n=14; farx(n+1)=3.215240e+000; foe(n+1)=1.416472e+001; krok(n+1)=1.725161e-002; ng(n+1)=8.058415e+002;
n=15; farx(n+1)=2.899124e+000; foe(n+1)=1.275623e+001; krok(n+1)=2.530456e-003; ng(n+1)=7.113929e+002;
n=16; farx(n+1)=2.364484e+000; foe(n+1)=1.153168e+001; krok(n+1)=2.883948e-003; ng(n+1)=5.101843e+002;
n=17; farx(n+1)=2.292447e+000; foe(n+1)=1.062897e+001; krok(n+1)=9.899545e-003; ng(n+1)=4.662875e+002;
n=18; farx(n+1)=2.307673e+000; foe(n+1)=1.003396e+001; krok(n+1)=8.899418e-003; ng(n+1)=5.855351e+002;
n=19; farx(n+1)=2.271817e+000; foe(n+1)=8.809595e+000; krok(n+1)=2.835564e-002; ng(n+1)=7.977197e+002;
n=20; farx(n+1)=2.200983e+000; foe(n+1)=7.530302e+000; krok(n+1)=6.425309e-003; ng(n+1)=8.642724e+002;
n=21; farx(n+1)=2.170367e+000; foe(n+1)=6.390801e+000; krok(n+1)=1.569591e-002; ng(n+1)=6.753442e+002;
n=22; farx(n+1)=2.219452e+000; foe(n+1)=6.031224e+000; krok(n+1)=3.157424e-003; ng(n+1)=4.458300e+002;
n=23; farx(n+1)=2.060217e+000; foe(n+1)=5.493927e+000; krok(n+1)=2.720095e-002; ng(n+1)=3.931662e+002;
n=24; farx(n+1)=1.355996e+000; foe(n+1)=4.171218e+000; krok(n+1)=3.743558e-002; ng(n+1)=1.973796e+002;
n=25; farx(n+1)=1.007276e+000; foe(n+1)=3.352091e+000; krok(n+1)=6.345427e-002; ng(n+1)=3.919234e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=9.650369e-001; foe(n+1)=3.067675e+000; krok(n+1)=1.961230e-005; ng(n+1)=4.145170e+002;
n=27; farx(n+1)=9.531484e-001; foe(n+1)=3.036675e+000; krok(n+1)=6.666596e-005; ng(n+1)=8.412283e+001;
n=28; farx(n+1)=9.471744e-001; foe(n+1)=2.984426e+000; krok(n+1)=2.154098e-004; ng(n+1)=6.827017e+001;
n=29; farx(n+1)=8.768309e-001; foe(n+1)=2.825111e+000; krok(n+1)=1.620122e-003; ng(n+1)=4.574665e+001;
n=30; farx(n+1)=7.800358e-001; foe(n+1)=2.456734e+000; krok(n+1)=1.056509e-002; ng(n+1)=2.867766e+001;
n=31; farx(n+1)=7.188844e-001; foe(n+1)=2.288914e+000; krok(n+1)=2.147336e-002; ng(n+1)=1.249025e+002;
n=32; farx(n+1)=6.855522e-001; foe(n+1)=2.131802e+000; krok(n+1)=2.254943e-002; ng(n+1)=2.308057e+002;
n=33; farx(n+1)=7.032226e-001; foe(n+1)=2.043503e+000; krok(n+1)=8.202462e-003; ng(n+1)=2.315839e+002;
n=34; farx(n+1)=6.341394e-001; foe(n+1)=1.740679e+000; krok(n+1)=2.254943e-002; ng(n+1)=1.261380e+002;
n=35; farx(n+1)=5.632181e-001; foe(n+1)=1.396603e+000; krok(n+1)=2.965791e-002; ng(n+1)=1.341750e+002;
n=36; farx(n+1)=5.466686e-001; foe(n+1)=1.252633e+000; krok(n+1)=1.349675e-002; ng(n+1)=2.031763e+002;
n=37; farx(n+1)=5.332427e-001; foe(n+1)=1.181432e+000; krok(n+1)=9.955600e-003; ng(n+1)=1.280351e+002;
n=38; farx(n+1)=4.815683e-001; foe(n+1)=1.084003e+000; krok(n+1)=3.039725e-002; ng(n+1)=5.221713e+001;
n=39; farx(n+1)=4.642927e-001; foe(n+1)=1.036731e+000; krok(n+1)=4.130221e-002; ng(n+1)=6.790748e+001;
n=40; farx(n+1)=4.318329e-001; foe(n+1)=9.478446e-001; krok(n+1)=4.285764e-002; ng(n+1)=1.000399e+002;
n=41; farx(n+1)=3.876786e-001; foe(n+1)=8.028291e-001; krok(n+1)=7.539806e-002; ng(n+1)=1.662049e+002;
n=42; farx(n+1)=3.817767e-001; foe(n+1)=7.652498e-001; krok(n+1)=2.847603e-002; ng(n+1)=1.151031e+002;
n=43; farx(n+1)=3.774723e-001; foe(n+1)=7.246552e-001; krok(n+1)=5.286105e-002; ng(n+1)=3.481193e+001;
n=44; farx(n+1)=3.728194e-001; foe(n+1)=6.978704e-001; krok(n+1)=8.509079e-002; ng(n+1)=6.829440e+001;
n=45; farx(n+1)=3.704847e-001; foe(n+1)=6.213100e-001; krok(n+1)=3.136673e-001; ng(n+1)=7.087250e+001;
n=46; farx(n+1)=3.710917e-001; foe(n+1)=6.114257e-001; krok(n+1)=2.705929e-002; ng(n+1)=8.994858e+001;
n=47; farx(n+1)=3.602068e-001; foe(n+1)=5.909136e-001; krok(n+1)=2.203031e-001; ng(n+1)=4.135767e+001;
n=48; farx(n+1)=3.515462e-001; foe(n+1)=5.568008e-001; krok(n+1)=2.373843e-001; ng(n+1)=2.519815e+001;
n=49; farx(n+1)=3.343374e-001; foe(n+1)=5.358340e-001; krok(n+1)=3.057702e-001; ng(n+1)=4.575842e+001;
n=50; farx(n+1)=3.315814e-001; foe(n+1)=5.214926e-001; krok(n+1)=9.137743e-002; ng(n+1)=8.520722e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
