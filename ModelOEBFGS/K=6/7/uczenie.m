%uczenie predyktora oe
clear all;
n=0; farx(n+1)=1.816311e+002; foe(n+1)=1.814509e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.654802e+002; foe(n+1)=1.649038e+002; krok(n+1)=7.010405e-004; ng(n+1)=5.056044e+002;
n=2; farx(n+1)=7.164963e+001; foe(n+1)=6.817531e+001; krok(n+1)=5.767896e-003; ng(n+1)=4.137138e+002;
n=3; farx(n+1)=6.762291e+001; foe(n+1)=6.252786e+001; krok(n+1)=1.413789e-003; ng(n+1)=3.466644e+002;
n=4; farx(n+1)=5.740137e+001; foe(n+1)=6.099536e+001; krok(n+1)=1.401574e-002; ng(n+1)=5.440890e+001;
n=5; farx(n+1)=1.917966e+001; foe(n+1)=5.265375e+001; krok(n+1)=7.772060e-003; ng(n+1)=1.941098e+002;
n=6; farx(n+1)=6.850864e+000; foe(n+1)=4.706430e+001; krok(n+1)=1.100811e-003; ng(n+1)=1.187765e+003;
n=7; farx(n+1)=4.943116e+000; foe(n+1)=4.567609e+001; krok(n+1)=4.532388e-004; ng(n+1)=3.436021e+003;
n=8; farx(n+1)=4.732073e+000; foe(n+1)=4.533395e+001; krok(n+1)=2.752027e-004; ng(n+1)=5.158719e+003;
n=9; farx(n+1)=4.204216e+000; foe(n+1)=3.141916e+001; krok(n+1)=1.450364e-002; ng(n+1)=5.407739e+003;
n=10; farx(n+1)=4.190709e+000; foe(n+1)=3.125434e+001; krok(n+1)=3.632499e-005; ng(n+1)=2.398656e+003;
n=11; farx(n+1)=4.361357e+000; foe(n+1)=3.049520e+001; krok(n+1)=2.778799e-003; ng(n+1)=2.203668e+003;
n=12; farx(n+1)=5.495569e+000; foe(n+1)=2.543797e+001; krok(n+1)=5.095665e-003; ng(n+1)=2.588758e+003;
n=13; farx(n+1)=5.965370e+000; foe(n+1)=2.347710e+001; krok(n+1)=1.997747e-003; ng(n+1)=1.369311e+003;
n=14; farx(n+1)=6.415359e+000; foe(n+1)=2.156546e+001; krok(n+1)=3.625910e-003; ng(n+1)=1.027532e+003;
n=15; farx(n+1)=6.366269e+000; foe(n+1)=1.796014e+001; krok(n+1)=4.587536e-003; ng(n+1)=1.097676e+003;
n=16; farx(n+1)=6.093203e+000; foe(n+1)=1.648690e+001; krok(n+1)=6.935224e-004; ng(n+1)=9.732493e+002;
n=17; farx(n+1)=4.347014e+000; foe(n+1)=1.361053e+001; krok(n+1)=2.243330e-002; ng(n+1)=5.414583e+002;
n=18; farx(n+1)=3.344968e+000; foe(n+1)=1.185039e+001; krok(n+1)=1.477175e-002; ng(n+1)=4.959519e+002;
n=19; farx(n+1)=2.637140e+000; foe(n+1)=1.029273e+001; krok(n+1)=3.888965e-003; ng(n+1)=9.301194e+002;
n=20; farx(n+1)=2.086921e+000; foe(n+1)=9.113291e+000; krok(n+1)=4.174641e-003; ng(n+1)=1.081685e+003;
n=21; farx(n+1)=1.614719e+000; foe(n+1)=8.126249e+000; krok(n+1)=5.858547e-003; ng(n+1)=1.292471e+003;
n=22; farx(n+1)=1.474235e+000; foe(n+1)=7.029158e+000; krok(n+1)=2.447519e-002; ng(n+1)=1.019479e+003;
n=23; farx(n+1)=1.411369e+000; foe(n+1)=6.441324e+000; krok(n+1)=1.142380e-002; ng(n+1)=7.628820e+002;
n=24; farx(n+1)=1.505162e+000; foe(n+1)=5.719736e+000; krok(n+1)=1.931884e-002; ng(n+1)=3.583334e+002;
n=25; farx(n+1)=1.411811e+000; foe(n+1)=4.958872e+000; krok(n+1)=2.842041e-002; ng(n+1)=2.602848e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=1.442149e+000; foe(n+1)=4.783648e+000; krok(n+1)=2.114892e-005; ng(n+1)=2.877548e+002;
n=27; farx(n+1)=1.356092e+000; foe(n+1)=4.599468e+000; krok(n+1)=2.996980e-004; ng(n+1)=1.061500e+002;
n=28; farx(n+1)=1.319988e+000; foe(n+1)=4.388643e+000; krok(n+1)=1.874208e-004; ng(n+1)=1.282244e+002;
n=29; farx(n+1)=1.292277e+000; foe(n+1)=3.436144e+000; krok(n+1)=3.093608e-004; ng(n+1)=2.645629e+002;
n=30; farx(n+1)=1.040052e+000; foe(n+1)=3.008442e+000; krok(n+1)=1.336226e-002; ng(n+1)=2.515384e+002;
n=31; farx(n+1)=8.800772e-001; foe(n+1)=2.539469e+000; krok(n+1)=1.199953e-002; ng(n+1)=5.484011e+002;
n=32; farx(n+1)=8.371239e-001; foe(n+1)=2.167979e+000; krok(n+1)=1.785641e-002; ng(n+1)=1.449335e+002;
n=33; farx(n+1)=8.584992e-001; foe(n+1)=1.984476e+000; krok(n+1)=9.201482e-003; ng(n+1)=2.059759e+002;
n=34; farx(n+1)=8.510515e-001; foe(n+1)=1.869812e+000; krok(n+1)=2.580064e-002; ng(n+1)=8.293850e+001;
n=35; farx(n+1)=8.231448e-001; foe(n+1)=1.776498e+000; krok(n+1)=7.366454e-003; ng(n+1)=1.140040e+002;
n=36; farx(n+1)=7.457761e-001; foe(n+1)=1.614593e+000; krok(n+1)=2.527804e-002; ng(n+1)=1.432873e+002;
n=37; farx(n+1)=7.292900e-001; foe(n+1)=1.520759e+000; krok(n+1)=1.488820e-002; ng(n+1)=1.420804e+002;
n=38; farx(n+1)=7.040758e-001; foe(n+1)=1.433907e+000; krok(n+1)=3.252259e-002; ng(n+1)=9.529859e+001;
n=39; farx(n+1)=7.005673e-001; foe(n+1)=1.388190e+000; krok(n+1)=1.625772e-002; ng(n+1)=1.791840e+002;
n=40; farx(n+1)=6.575184e-001; foe(n+1)=1.184133e+000; krok(n+1)=2.788178e-002; ng(n+1)=1.569940e+002;
n=41; farx(n+1)=6.518004e-001; foe(n+1)=1.167239e+000; krok(n+1)=1.131031e-002; ng(n+1)=1.576068e+002;
n=42; farx(n+1)=6.108469e-001; foe(n+1)=1.089326e+000; krok(n+1)=3.847317e-002; ng(n+1)=7.759638e+001;
n=43; farx(n+1)=5.592474e-001; foe(n+1)=1.016477e+000; krok(n+1)=7.596282e-002; ng(n+1)=1.272804e+002;
n=44; farx(n+1)=4.942096e-001; foe(n+1)=8.942031e-001; krok(n+1)=4.703530e-002; ng(n+1)=2.245794e+002;
n=45; farx(n+1)=4.796807e-001; foe(n+1)=8.620490e-001; krok(n+1)=6.442938e-002; ng(n+1)=1.733170e+001;
n=46; farx(n+1)=4.639313e-001; foe(n+1)=8.427607e-001; krok(n+1)=5.737813e-002; ng(n+1)=7.802329e+001;
n=47; farx(n+1)=4.532707e-001; foe(n+1)=8.265020e-001; krok(n+1)=9.326940e-002; ng(n+1)=5.413573e+001;
n=48; farx(n+1)=4.560986e-001; foe(n+1)=7.889698e-001; krok(n+1)=3.477470e-001; ng(n+1)=5.904425e+001;
n=49; farx(n+1)=4.719030e-001; foe(n+1)=7.403315e-001; krok(n+1)=2.894475e-001; ng(n+1)=7.444701e+001;
n=50; farx(n+1)=4.825822e-001; foe(n+1)=6.989970e-001; krok(n+1)=1.818254e-001; ng(n+1)=9.032327e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
