%uczenie predyktora oe
clear all;
n=0; farx(n+1)=3.089557e+002; foe(n+1)=3.033119e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.676015e+002; foe(n+1)=1.647661e+002; krok(n+1)=5.127695e-004; ng(n+1)=1.106610e+003;
n=2; farx(n+1)=5.826123e+001; foe(n+1)=6.059256e+001; krok(n+1)=6.081190e-003; ng(n+1)=3.985316e+002;
n=3; farx(n+1)=5.499525e+001; foe(n+1)=5.655835e+001; krok(n+1)=1.457199e-003; ng(n+1)=2.616090e+002;
n=4; farx(n+1)=4.069450e+001; foe(n+1)=5.315537e+001; krok(n+1)=2.219272e-002; ng(n+1)=5.844404e+001;
n=5; farx(n+1)=1.592309e+001; foe(n+1)=4.348555e+001; krok(n+1)=3.516052e-003; ng(n+1)=3.297250e+002;
n=6; farx(n+1)=8.278353e+000; foe(n+1)=3.059092e+001; krok(n+1)=2.716773e-003; ng(n+1)=1.146651e+003;
n=7; farx(n+1)=8.308329e+000; foe(n+1)=3.010365e+001; krok(n+1)=8.083878e-005; ng(n+1)=2.414571e+003;
n=8; farx(n+1)=8.308076e+000; foe(n+1)=2.998811e+001; krok(n+1)=4.636412e-004; ng(n+1)=2.245224e+003;
n=9; farx(n+1)=9.636731e+000; foe(n+1)=2.595661e+001; krok(n+1)=4.756638e-003; ng(n+1)=2.364211e+003;
n=10; farx(n+1)=9.957180e+000; foe(n+1)=2.536575e+001; krok(n+1)=3.127663e-003; ng(n+1)=1.666386e+003;
n=11; farx(n+1)=1.101974e+001; foe(n+1)=2.239430e+001; krok(n+1)=8.751130e-003; ng(n+1)=1.396552e+003;
n=12; farx(n+1)=1.127255e+001; foe(n+1)=2.193091e+001; krok(n+1)=2.013418e-003; ng(n+1)=5.236851e+002;
n=13; farx(n+1)=1.084727e+001; foe(n+1)=1.813817e+001; krok(n+1)=3.178877e-002; ng(n+1)=5.209273e+002;
n=14; farx(n+1)=1.053684e+001; foe(n+1)=1.760442e+001; krok(n+1)=2.739044e-003; ng(n+1)=2.018649e+002;
n=15; farx(n+1)=9.703335e+000; foe(n+1)=1.670796e+001; krok(n+1)=1.131000e-002; ng(n+1)=2.016821e+002;
n=16; farx(n+1)=5.345526e+000; foe(n+1)=1.255878e+001; krok(n+1)=1.660195e-002; ng(n+1)=2.058970e+002;
n=17; farx(n+1)=4.767667e+000; foe(n+1)=1.213662e+001; krok(n+1)=6.026360e-003; ng(n+1)=2.431473e+002;
n=18; farx(n+1)=3.814510e+000; foe(n+1)=1.156659e+001; krok(n+1)=6.293742e-003; ng(n+1)=1.184118e+002;
n=19; farx(n+1)=1.335262e+000; foe(n+1)=8.956881e+000; krok(n+1)=1.111520e-002; ng(n+1)=3.884084e+002;
n=20; farx(n+1)=1.233398e+000; foe(n+1)=8.771606e+000; krok(n+1)=1.207660e-003; ng(n+1)=3.303585e+002;
n=21; farx(n+1)=1.079302e+000; foe(n+1)=8.547843e+000; krok(n+1)=1.089156e-002; ng(n+1)=2.295716e+002;
n=22; farx(n+1)=1.054721e+000; foe(n+1)=8.255631e+000; krok(n+1)=2.592196e-002; ng(n+1)=1.095700e+002;
n=23; farx(n+1)=9.661157e-001; foe(n+1)=8.026217e+000; krok(n+1)=3.320390e-002; ng(n+1)=1.066734e+002;
n=24; farx(n+1)=9.810079e-001; foe(n+1)=7.832138e+000; krok(n+1)=8.625198e-002; ng(n+1)=8.704803e+001;
n=25; farx(n+1)=9.841678e-001; foe(n+1)=7.678152e+000; krok(n+1)=5.160128e-002; ng(n+1)=2.384190e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=9.859775e-001; foe(n+1)=7.632806e+000; krok(n+1)=1.428638e-005; ng(n+1)=1.638310e+002;
n=27; farx(n+1)=1.001667e+000; foe(n+1)=7.614404e+000; krok(n+1)=3.505202e-004; ng(n+1)=3.097102e+001;
n=28; farx(n+1)=1.008974e+000; foe(n+1)=7.603675e+000; krok(n+1)=2.135248e-004; ng(n+1)=2.807048e+001;
n=29; farx(n+1)=1.075543e+000; foe(n+1)=6.987517e+000; krok(n+1)=3.376031e-003; ng(n+1)=4.407065e+001;
n=30; farx(n+1)=1.295951e+000; foe(n+1)=6.413271e+000; krok(n+1)=2.223039e-002; ng(n+1)=3.519712e+002;
n=31; farx(n+1)=1.415784e+000; foe(n+1)=5.883139e+000; krok(n+1)=7.138862e-004; ng(n+1)=7.300525e+002;
n=32; farx(n+1)=1.429876e+000; foe(n+1)=5.647711e+000; krok(n+1)=1.003628e-003; ng(n+1)=4.124305e+002;
n=33; farx(n+1)=1.334154e+000; foe(n+1)=5.407237e+000; krok(n+1)=1.076348e-002; ng(n+1)=3.297357e+002;
n=34; farx(n+1)=1.330886e+000; foe(n+1)=4.513233e+000; krok(n+1)=1.384176e-002; ng(n+1)=6.629511e+002;
n=35; farx(n+1)=1.391322e+000; foe(n+1)=4.222078e+000; krok(n+1)=1.195779e-003; ng(n+1)=6.379532e+002;
n=36; farx(n+1)=1.575565e+000; foe(n+1)=3.847653e+000; krok(n+1)=5.485375e-003; ng(n+1)=6.514182e+002;
n=37; farx(n+1)=1.583332e+000; foe(n+1)=3.677830e+000; krok(n+1)=6.019169e-003; ng(n+1)=2.411324e+002;
n=38; farx(n+1)=1.531821e+000; foe(n+1)=3.523626e+000; krok(n+1)=9.829303e-003; ng(n+1)=1.881148e+002;
n=39; farx(n+1)=1.456481e+000; foe(n+1)=3.145540e+000; krok(n+1)=3.586741e-002; ng(n+1)=2.847413e+002;
n=40; farx(n+1)=1.255279e+000; foe(n+1)=2.574030e+000; krok(n+1)=1.450364e-002; ng(n+1)=3.561285e+002;
n=41; farx(n+1)=8.777384e-001; foe(n+1)=2.140310e+000; krok(n+1)=2.835564e-002; ng(n+1)=1.510363e+002;
n=42; farx(n+1)=6.893828e-001; foe(n+1)=1.780462e+000; krok(n+1)=3.898185e-003; ng(n+1)=3.379940e+002;
n=43; farx(n+1)=6.581607e-001; foe(n+1)=1.667196e+000; krok(n+1)=3.280985e-002; ng(n+1)=1.482393e+002;
n=44; farx(n+1)=5.491612e-001; foe(n+1)=1.247520e+000; krok(n+1)=7.936896e-002; ng(n+1)=3.084568e+002;
n=45; farx(n+1)=5.125211e-001; foe(n+1)=1.197818e+000; krok(n+1)=7.782134e-003; ng(n+1)=8.505802e+001;
n=46; farx(n+1)=4.735151e-001; foe(n+1)=1.145002e+000; krok(n+1)=1.638666e-002; ng(n+1)=1.048270e+002;
n=47; farx(n+1)=4.025926e-001; foe(n+1)=1.030896e+000; krok(n+1)=3.712452e-002; ng(n+1)=1.543783e+002;
n=48; farx(n+1)=4.313061e-001; foe(n+1)=9.308867e-001; krok(n+1)=6.460817e-002; ng(n+1)=1.754283e+002;
n=49; farx(n+1)=4.178390e-001; foe(n+1)=8.573415e-001; krok(n+1)=5.914179e-001; ng(n+1)=6.173305e+001;
n=50; farx(n+1)=4.139730e-001; foe(n+1)=8.300931e-001; krok(n+1)=1.392836e-001; ng(n+1)=1.289550e+002;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)