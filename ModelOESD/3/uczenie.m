%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.479035e+002; foe(n+1)=2.465921e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
n=1; farx(n+1)=1.817856e+002; foe(n+1)=1.808439e+002; krok(n+1)=5.033545e-004; ng(n+1)=5.408478e+002;
n=2; farx(n+1)=1.760024e+002; foe(n+1)=1.759040e+002; krok(n+1)=3.891067e-003; ng(n+1)=9.291446e+001;
n=3; farx(n+1)=1.730576e+002; foe(n+1)=1.727638e+002; krok(n+1)=1.187780e-003; ng(n+1)=1.388922e+002;
n=4; farx(n+1)=1.677975e+002; foe(n+1)=1.672423e+002; krok(n+1)=1.131319e-003; ng(n+1)=1.954659e+002;
n=5; farx(n+1)=1.583795e+002; foe(n+1)=1.579023e+002; krok(n+1)=1.093065e-003; ng(n+1)=2.695245e+002;
n=6; farx(n+1)=1.440231e+002; foe(n+1)=1.433344e+002; krok(n+1)=1.072587e-003; ng(n+1)=3.345351e+002;
n=7; farx(n+1)=1.242027e+002; foe(n+1)=1.244832e+002; krok(n+1)=9.152390e-004; ng(n+1)=4.449140e+002;
n=8; farx(n+1)=1.036129e+002; foe(n+1)=1.041460e+002; krok(n+1)=9.317073e-004; ng(n+1)=4.379877e+002;
n=9; farx(n+1)=8.466108e+001; foe(n+1)=8.677261e+001; krok(n+1)=7.286945e-004; ng(n+1)=5.120384e+002;
n=10; farx(n+1)=7.186125e+001; foe(n+1)=7.454559e+001; krok(n+1)=7.941269e-004; ng(n+1)=3.906126e+002;
n=11; farx(n+1)=6.348724e+001; foe(n+1)=6.735913e+001; krok(n+1)=6.078605e-004; ng(n+1)=3.819625e+002;
n=12; farx(n+1)=5.919478e+001; foe(n+1)=6.363871e+001; krok(n+1)=6.946998e-004; ng(n+1)=2.422569e+002;
n=13; farx(n+1)=5.663900e+001; foe(n+1)=6.187837e+001; krok(n+1)=5.510555e-004; ng(n+1)=2.057369e+002;
n=14; farx(n+1)=5.532000e+001; foe(n+1)=6.106305e+001; krok(n+1)=6.586506e-004; ng(n+1)=1.211815e+002;
n=15; farx(n+1)=5.438472e+001; foe(n+1)=6.067145e+001; krok(n+1)=5.353361e-004; ng(n+1)=9.753677e+001;
n=16; farx(n+1)=5.371843e+001; foe(n+1)=6.045108e+001; krok(n+1)=7.010405e-004; ng(n+1)=6.087690e+001;
n=17; farx(n+1)=5.313336e+001; foe(n+1)=6.030002e+001; krok(n+1)=5.656595e-004; ng(n+1)=5.335157e+001;
n=18; farx(n+1)=5.260458e+001; foe(n+1)=6.017035e+001; krok(n+1)=7.581588e-004; ng(n+1)=4.522898e+001;
n=19; farx(n+1)=5.209835e+001; foe(n+1)=6.004863e+001; krok(n+1)=5.750926e-004; ng(n+1)=4.583669e+001;
n=20; farx(n+1)=5.162102e+001; foe(n+1)=5.992897e+001; krok(n+1)=7.590926e-004; ng(n+1)=4.424925e+001;
n=21; farx(n+1)=5.114166e+001; foe(n+1)=5.981015e+001; krok(n+1)=5.685319e-004; ng(n+1)=4.584683e+001;
n=22; farx(n+1)=5.068668e+001; foe(n+1)=5.969170e+001; krok(n+1)=7.499704e-004; ng(n+1)=4.469057e+001;
n=23; farx(n+1)=5.022097e+001; foe(n+1)=5.957360e+001; krok(n+1)=5.623096e-004; ng(n+1)=4.609995e+001;
n=24; farx(n+1)=4.978038e+001; foe(n+1)=5.945575e+001; krok(n+1)=7.394410e-004; ng(n+1)=4.493683e+001;
n=25; farx(n+1)=4.932505e+001; foe(n+1)=5.933809e+001; krok(n+1)=5.557084e-004; ng(n+1)=4.627855e+001;
n=26; farx(n+1)=4.889637e+001; foe(n+1)=5.922074e+001; krok(n+1)=7.285994e-004; ng(n+1)=4.510569e+001;
n=27; farx(n+1)=4.845173e+001; foe(n+1)=5.910349e+001; krok(n+1)=5.469456e-004; ng(n+1)=4.638325e+001;
n=28; farx(n+1)=4.802810e+001; foe(n+1)=5.898623e+001; krok(n+1)=7.286945e-004; ng(n+1)=4.512642e+001;
n=29; farx(n+1)=4.759541e+001; foe(n+1)=5.886882e+001; krok(n+1)=5.333277e-004; ng(n+1)=4.678056e+001;
n=30; farx(n+1)=4.717644e+001; foe(n+1)=5.875134e+001; krok(n+1)=7.273797e-004; ng(n+1)=4.506793e+001;
n=31; farx(n+1)=4.675257e+001; foe(n+1)=5.863407e+001; krok(n+1)=5.252174e-004; ng(n+1)=4.698978e+001;
n=32; farx(n+1)=4.634449e+001; foe(n+1)=5.851680e+001; krok(n+1)=7.138862e-004; ng(n+1)=4.517301e+001;
n=33; farx(n+1)=4.592533e+001; foe(n+1)=5.840018e+001; krok(n+1)=5.251366e-004; ng(n+1)=4.689365e+001;
n=34; farx(n+1)=4.552916e+001; foe(n+1)=5.828381e+001; krok(n+1)=7.010405e-004; ng(n+1)=4.552517e+001;
n=35; farx(n+1)=4.512040e+001; foe(n+1)=5.816706e+001; krok(n+1)=5.126539e-004; ng(n+1)=4.722633e+001;
n=36; farx(n+1)=4.472991e+001; foe(n+1)=5.805027e+001; krok(n+1)=6.946998e-004; ng(n+1)=4.545572e+001;
n=37; farx(n+1)=4.432643e+001; foe(n+1)=5.793403e+001; krok(n+1)=5.104038e-004; ng(n+1)=4.718053e+001;
n=38; farx(n+1)=4.394623e+001; foe(n+1)=5.781774e+001; krok(n+1)=6.822427e-004; ng(n+1)=4.572003e+001;
n=39; farx(n+1)=4.354903e+001; foe(n+1)=5.770156e+001; krok(n+1)=5.045766e-004; ng(n+1)=4.737582e+001;
n=40; farx(n+1)=4.317836e+001; foe(n+1)=5.758555e+001; krok(n+1)=6.696507e-004; ng(n+1)=4.592905e+001;
n=41; farx(n+1)=4.278761e+001; foe(n+1)=5.746959e+001; krok(n+1)=4.987505e-004; ng(n+1)=4.754714e+001;
n=42; farx(n+1)=4.242351e+001; foe(n+1)=5.735359e+001; krok(n+1)=6.621248e-004; ng(n+1)=4.606718e+001;
n=43; farx(n+1)=4.203944e+001; foe(n+1)=5.723759e+001; krok(n+1)=4.912716e-004; ng(n+1)=4.786928e+001;
n=44; farx(n+1)=4.167983e+001; foe(n+1)=5.712153e+001; krok(n+1)=6.586506e-004; ng(n+1)=4.624701e+001;
n=45; farx(n+1)=4.130425e+001; foe(n+1)=5.700510e+001; krok(n+1)=4.792372e-004; ng(n+1)=4.839594e+001;
n=46; farx(n+1)=4.094772e+001; foe(n+1)=5.688843e+001; krok(n+1)=6.553529e-004; ng(n+1)=4.634530e+001;
n=47; farx(n+1)=4.057693e+001; foe(n+1)=5.677200e+001; krok(n+1)=4.744783e-004; ng(n+1)=4.867049e+001;
n=48; farx(n+1)=4.022757e+001; foe(n+1)=5.665550e+001; krok(n+1)=6.459813e-004; ng(n+1)=4.672590e+001;
n=49; farx(n+1)=3.986220e+001; foe(n+1)=5.653878e+001; krok(n+1)=4.674935e-004; ng(n+1)=4.909188e+001;
n=50; farx(n+1)=3.952091e+001; foe(n+1)=5.642221e+001; krok(n+1)=6.328553e-004; ng(n+1)=4.705480e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
