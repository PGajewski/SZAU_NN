%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.473664e+002; foe(n+1)=2.446824e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
n=1; farx(n+1)=1.714183e+002; foe(n+1)=1.698631e+002; krok(n+1)=5.045766e-004; ng(n+1)=6.391357e+002;
n=2; farx(n+1)=7.385687e+001; foe(n+1)=7.569349e+001; krok(n+1)=1.109636e-002; ng(n+1)=2.008870e+002;
n=3; farx(n+1)=5.880722e+001; foe(n+1)=6.100153e+001; krok(n+1)=3.473499e-004; ng(n+1)=5.049125e+002;
n=4; farx(n+1)=5.596980e+001; foe(n+1)=6.046012e+001; krok(n+1)=2.391558e-003; ng(n+1)=5.275071e+001;
n=5; farx(n+1)=5.537693e+001; foe(n+1)=6.023577e+001; krok(n+1)=8.786586e-004; ng(n+1)=6.108474e+001;
n=6; farx(n+1)=5.481700e+001; foe(n+1)=6.007010e+001; krok(n+1)=4.912482e-004; ng(n+1)=6.182877e+001;
n=7; farx(n+1)=5.410947e+001; foe(n+1)=5.991021e+001; krok(n+1)=9.349870e-004; ng(n+1)=4.933958e+001;
n=8; farx(n+1)=5.354455e+001; foe(n+1)=5.975180e+001; krok(n+1)=4.863834e-004; ng(n+1)=5.878003e+001;
n=9; farx(n+1)=5.281555e+001; foe(n+1)=5.959216e+001; krok(n+1)=9.349870e-004; ng(n+1)=4.803154e+001;
n=10; farx(n+1)=5.225885e+001; foe(n+1)=5.943150e+001; krok(n+1)=4.748903e-004; ng(n+1)=5.922054e+001;
n=11; farx(n+1)=5.153594e+001; foe(n+1)=5.926991e+001; krok(n+1)=9.247269e-004; ng(n+1)=4.823493e+001;
n=12; farx(n+1)=5.098956e+001; foe(n+1)=5.910743e+001; krok(n+1)=4.651747e-004; ng(n+1)=5.986086e+001;
n=13; farx(n+1)=5.027612e+001; foe(n+1)=5.894414e+001; krok(n+1)=9.152390e-004; ng(n+1)=4.873087e+001;
n=14; farx(n+1)=4.973713e+001; foe(n+1)=5.877905e+001; krok(n+1)=4.576195e-004; ng(n+1)=6.071365e+001;
n=15; farx(n+1)=4.905775e+001; foe(n+1)=5.861601e+001; krok(n+1)=8.713328e-004; ng(n+1)=4.954756e+001;
n=16; farx(n+1)=4.852511e+001; foe(n+1)=5.845226e+001; krok(n+1)=4.576195e-004; ng(n+1)=6.039751e+001;
n=17; farx(n+1)=4.787639e+001; foe(n+1)=5.829015e+001; krok(n+1)=8.336543e-004; ng(n+1)=5.055410e+001;
n=18; farx(n+1)=4.735503e+001; foe(n+1)=5.812588e+001; krok(n+1)=4.482666e-004; ng(n+1)=6.098110e+001;
n=19; farx(n+1)=4.670617e+001; foe(n+1)=5.796029e+001; krok(n+1)=8.336543e-004; ng(n+1)=5.122311e+001;
n=20; farx(n+1)=4.619324e+001; foe(n+1)=5.779295e+001; krok(n+1)=4.375567e-004; ng(n+1)=6.240910e+001;
n=21; farx(n+1)=4.555783e+001; foe(n+1)=5.762520e+001; krok(n+1)=8.145474e-004; ng(n+1)=5.214708e+001;
n=22; farx(n+1)=4.505328e+001; foe(n+1)=5.745667e+001; krok(n+1)=4.297012e-004; ng(n+1)=6.327966e+001;
n=23; farx(n+1)=4.442698e+001; foe(n+1)=5.728645e+001; krok(n+1)=8.011837e-004; ng(n+1)=5.308575e+001;
n=24; farx(n+1)=4.392987e+001; foe(n+1)=5.711566e+001; krok(n+1)=4.209206e-004; ng(n+1)=6.447837e+001;
n=25; farx(n+1)=4.331363e+001; foe(n+1)=5.694311e+001; krok(n+1)=7.861003e-004; ng(n+1)=5.410482e+001;
n=26; farx(n+1)=4.281952e+001; foe(n+1)=5.676999e+001; krok(n+1)=4.168272e-004; ng(n+1)=6.571403e+001;
n=27; farx(n+1)=4.223461e+001; foe(n+1)=5.659823e+001; krok(n+1)=7.432430e-004; ng(n+1)=5.564012e+001;
n=28; farx(n+1)=4.174767e+001; foe(n+1)=5.642519e+001; krok(n+1)=4.114508e-004; ng(n+1)=6.617143e+001;
n=29; farx(n+1)=4.117114e+001; foe(n+1)=5.625060e+001; krok(n+1)=7.286672e-004; ng(n+1)=5.676627e+001;
n=30; farx(n+1)=4.069062e+001; foe(n+1)=5.607535e+001; krok(n+1)=4.031350e-004; ng(n+1)=6.736082e+001;
n=31; farx(n+1)=4.012240e+001; foe(n+1)=5.589794e+001; krok(n+1)=7.138862e-004; ng(n+1)=5.809472e+001;
n=32; farx(n+1)=3.964599e+001; foe(n+1)=5.572025e+001; krok(n+1)=3.968650e-004; ng(n+1)=6.863909e+001;
n=33; farx(n+1)=3.909677e+001; foe(n+1)=5.554142e+001; krok(n+1)=6.847610e-004; ng(n+1)=5.971222e+001;
n=34; farx(n+1)=3.862336e+001; foe(n+1)=5.536280e+001; krok(n+1)=3.939441e-004; ng(n+1)=6.939276e+001;
n=35; farx(n+1)=3.809124e+001; foe(n+1)=5.518293e+001; krok(n+1)=6.592938e-004; ng(n+1)=6.147528e+001;
n=36; farx(n+1)=3.762605e+001; foe(n+1)=5.500214e+001; krok(n+1)=3.831728e-004; ng(n+1)=7.058599e+001;
n=37; farx(n+1)=3.708270e+001; foe(n+1)=5.481690e+001; krok(n+1)=6.694872e-004; ng(n+1)=6.258974e+001;
n=38; farx(n+1)=3.662384e+001; foe(n+1)=5.462967e+001; krok(n+1)=3.684136e-004; ng(n+1)=7.336726e+001;
n=39; farx(n+1)=3.608079e+001; foe(n+1)=5.443983e+001; krok(n+1)=6.629006e-004; ng(n+1)=6.396592e+001;
n=40; farx(n+1)=3.562861e+001; foe(n+1)=5.424857e+001; krok(n+1)=3.569431e-004; ng(n+1)=7.544018e+001;
n=41; farx(n+1)=3.507931e+001; foe(n+1)=5.405320e+001; krok(n+1)=6.647718e-004; ng(n+1)=6.532151e+001;
n=42; farx(n+1)=3.463448e+001; foe(n+1)=5.385588e+001; krok(n+1)=3.423805e-004; ng(n+1)=7.823849e+001;
n=43; farx(n+1)=3.407182e+001; foe(n+1)=5.365285e+001; krok(n+1)=6.738436e-004; ng(n+1)=6.655270e+001;
n=44; farx(n+1)=3.363040e+001; foe(n+1)=5.344829e+001; krok(n+1)=3.310624e-004; ng(n+1)=8.138284e+001;
n=45; farx(n+1)=3.307409e+001; foe(n+1)=5.324118e+001; krok(n+1)=6.586506e-004; ng(n+1)=6.837460e+001;
n=46; farx(n+1)=3.263618e+001; foe(n+1)=5.303212e+001; krok(n+1)=3.229906e-004; ng(n+1)=8.381367e+001;
n=47; farx(n+1)=3.209174e+001; foe(n+1)=5.282139e+001; krok(n+1)=6.359070e-004; ng(n+1)=7.036039e+001;
n=48; farx(n+1)=3.165902e+001; foe(n+1)=5.260941e+001; krok(n+1)=3.149159e-004; ng(n+1)=8.586252e+001;
n=49; farx(n+1)=3.111617e+001; foe(n+1)=5.239297e+001; krok(n+1)=6.254538e-004; ng(n+1)=7.218998e+001;
n=50; farx(n+1)=3.068637e+001; foe(n+1)=5.217567e+001; krok(n+1)=3.067806e-004; ng(n+1)=8.862299e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)