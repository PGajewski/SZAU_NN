%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.123824e+002; foe(n+1)=2.143970e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.610877e+002; foe(n+1)=1.644705e+002; krok(n+1)=4.988661e-004; ng(n+1)=1.034609e+003;
n=2; farx(n+1)=5.853437e+001; foe(n+1)=6.674945e+001; krok(n+1)=6.607631e-003; ng(n+1)=4.762143e+002;
n=3; farx(n+1)=5.476207e+001; foe(n+1)=6.176472e+001; krok(n+1)=1.704775e-003; ng(n+1)=3.739809e+002;
n=4; farx(n+1)=4.808246e+001; foe(n+1)=6.050893e+001; krok(n+1)=1.153579e-002; ng(n+1)=6.575793e+001;
n=5; farx(n+1)=1.458040e+001; foe(n+1)=5.051433e+001; krok(n+1)=1.086709e-002; ng(n+1)=2.300977e+002;
n=6; farx(n+1)=7.386234e+000; foe(n+1)=4.700176e+001; krok(n+1)=6.200700e-004; ng(n+1)=1.540204e+003;
n=7; farx(n+1)=5.148107e+000; foe(n+1)=4.551932e+001; krok(n+1)=7.138862e-004; ng(n+1)=3.196108e+003;
n=8; farx(n+1)=3.779077e+000; foe(n+1)=4.304970e+001; krok(n+1)=5.202718e-004; ng(n+1)=4.986575e+003;
n=9; farx(n+1)=3.607194e+000; foe(n+1)=4.185953e+001; krok(n+1)=1.675019e-003; ng(n+1)=7.755860e+003;
n=10; farx(n+1)=3.594414e+000; foe(n+1)=4.144603e+001; krok(n+1)=7.740461e-004; ng(n+1)=8.262528e+003;
n=11; farx(n+1)=3.504423e+000; foe(n+1)=4.044711e+001; krok(n+1)=2.063028e-003; ng(n+1)=7.798045e+003;
n=12; farx(n+1)=3.340018e+000; foe(n+1)=3.673061e+001; krok(n+1)=3.550402e-003; ng(n+1)=6.130535e+003;
n=13; farx(n+1)=3.579536e+000; foe(n+1)=3.488956e+001; krok(n+1)=2.676681e-004; ng(n+1)=3.906581e+003;
n=14; farx(n+1)=4.064967e+000; foe(n+1)=3.366027e+001; krok(n+1)=2.120860e-003; ng(n+1)=4.369319e+003;
n=15; farx(n+1)=5.868694e+000; foe(n+1)=2.981257e+001; krok(n+1)=1.602144e-003; ng(n+1)=5.634122e+003;
n=16; farx(n+1)=7.669704e+000; foe(n+1)=2.663241e+001; krok(n+1)=9.497806e-004; ng(n+1)=4.518771e+003;
n=17; farx(n+1)=8.739513e+000; foe(n+1)=2.327623e+001; krok(n+1)=3.303815e-003; ng(n+1)=2.290806e+003;
n=18; farx(n+1)=8.427214e+000; foe(n+1)=2.257419e+001; krok(n+1)=4.977800e-003; ng(n+1)=5.999847e+002;
n=19; farx(n+1)=7.883746e+000; foe(n+1)=2.064148e+001; krok(n+1)=1.839173e-002; ng(n+1)=6.374182e+002;
n=20; farx(n+1)=6.502270e+000; foe(n+1)=1.797669e+001; krok(n+1)=1.859267e-002; ng(n+1)=6.366751e+002;
n=21; farx(n+1)=4.232802e+000; foe(n+1)=1.296844e+001; krok(n+1)=5.107684e-003; ng(n+1)=5.858216e+002;
n=22; farx(n+1)=3.985764e+000; foe(n+1)=1.253598e+001; krok(n+1)=1.116161e-003; ng(n+1)=6.071153e+002;
n=23; farx(n+1)=2.970832e+000; foe(n+1)=9.131930e+000; krok(n+1)=2.525939e-002; ng(n+1)=7.748707e+002;
n=24; farx(n+1)=2.703233e+000; foe(n+1)=8.246073e+000; krok(n+1)=1.529718e-003; ng(n+1)=7.600165e+002;
n=25; farx(n+1)=2.332677e+000; foe(n+1)=6.853273e+000; krok(n+1)=8.346822e-003; ng(n+1)=2.756391e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=2.381501e+000; foe(n+1)=6.709785e+000; krok(n+1)=5.162212e-005; ng(n+1)=2.858158e+002;
n=27; farx(n+1)=2.380339e+000; foe(n+1)=6.624479e+000; krok(n+1)=2.609844e-005; ng(n+1)=2.541179e+002;
n=28; farx(n+1)=2.242274e+000; foe(n+1)=6.033767e+000; krok(n+1)=1.848518e-004; ng(n+1)=2.725386e+002;
n=29; farx(n+1)=1.912587e+000; foe(n+1)=4.785467e+000; krok(n+1)=6.106204e-004; ng(n+1)=2.495416e+002;
n=30; farx(n+1)=1.135249e+000; foe(n+1)=2.736700e+000; krok(n+1)=1.175883e-002; ng(n+1)=1.221164e+002;
n=31; farx(n+1)=1.078595e+000; foe(n+1)=2.609915e+000; krok(n+1)=1.460933e-003; ng(n+1)=4.529434e+002;
n=32; farx(n+1)=8.759275e-001; foe(n+1)=2.097363e+000; krok(n+1)=7.191676e-003; ng(n+1)=3.715253e+002;
n=33; farx(n+1)=8.221898e-001; foe(n+1)=1.893505e+000; krok(n+1)=1.477175e-002; ng(n+1)=2.313391e+002;
n=34; farx(n+1)=8.426541e-001; foe(n+1)=1.608125e+000; krok(n+1)=2.357847e-002; ng(n+1)=1.158991e+002;
n=35; farx(n+1)=8.304690e-001; foe(n+1)=1.552474e+000; krok(n+1)=2.804162e-003; ng(n+1)=2.528726e+002;
n=36; farx(n+1)=7.624381e-001; foe(n+1)=1.459003e+000; krok(n+1)=1.032555e-002; ng(n+1)=2.283817e+002;
n=37; farx(n+1)=6.515535e-001; foe(n+1)=1.317500e+000; krok(n+1)=4.256083e-002; ng(n+1)=1.492993e+002;
n=38; farx(n+1)=6.169148e-001; foe(n+1)=1.286286e+000; krok(n+1)=1.481850e-002; ng(n+1)=1.028999e+002;
n=39; farx(n+1)=5.782419e-001; foe(n+1)=1.254709e+000; krok(n+1)=1.053841e-002; ng(n+1)=1.304742e+002;
n=40; farx(n+1)=5.467784e-001; foe(n+1)=1.216131e+000; krok(n+1)=3.426151e-002; ng(n+1)=5.814570e+001;
n=41; farx(n+1)=5.130616e-001; foe(n+1)=1.172078e+000; krok(n+1)=1.640492e-002; ng(n+1)=1.569279e+002;
n=42; farx(n+1)=4.798196e-001; foe(n+1)=1.123326e+000; krok(n+1)=4.000365e-002; ng(n+1)=1.055055e+002;
n=43; farx(n+1)=4.639211e-001; foe(n+1)=1.089639e+000; krok(n+1)=2.835564e-002; ng(n+1)=1.426917e+002;
n=44; farx(n+1)=4.563021e-001; foe(n+1)=1.043451e+000; krok(n+1)=3.192003e-002; ng(n+1)=9.596848e+001;
n=45; farx(n+1)=4.559873e-001; foe(n+1)=1.026874e+000; krok(n+1)=1.019133e-002; ng(n+1)=1.298280e+002;
n=46; farx(n+1)=4.599114e-001; foe(n+1)=9.686355e-001; krok(n+1)=6.299996e-002; ng(n+1)=7.151292e+001;
n=47; farx(n+1)=4.669035e-001; foe(n+1)=9.119778e-001; krok(n+1)=5.335388e-002; ng(n+1)=1.007685e+002;
n=48; farx(n+1)=4.727699e-001; foe(n+1)=8.658941e-001; krok(n+1)=7.032104e-003; ng(n+1)=2.735159e+002;
n=49; farx(n+1)=4.786363e-001; foe(n+1)=8.288742e-001; krok(n+1)=9.939393e-003; ng(n+1)=1.773860e+002;
n=50; farx(n+1)=4.811269e-001; foe(n+1)=8.060776e-001; krok(n+1)=8.693674e-002; ng(n+1)=1.148893e+002;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
