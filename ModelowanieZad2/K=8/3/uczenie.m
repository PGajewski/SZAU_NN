%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.038366e+002; foe(n+1)=1.996039e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.750805e+002; foe(n+1)=1.715240e+002; krok(n+1)=5.723719e-004; ng(n+1)=7.290915e+002;
n=2; farx(n+1)=5.683957e+001; foe(n+1)=5.703357e+001; krok(n+1)=8.510569e-003; ng(n+1)=3.944304e+002;
n=3; farx(n+1)=5.495378e+001; foe(n+1)=5.513239e+001; krok(n+1)=1.120856e-003; ng(n+1)=1.975245e+002;
n=4; farx(n+1)=4.654094e+001; foe(n+1)=5.336076e+001; krok(n+1)=6.920879e-003; ng(n+1)=8.460745e+001;
n=5; farx(n+1)=1.098401e+001; foe(n+1)=2.892569e+001; krok(n+1)=9.152723e-003; ng(n+1)=2.225274e+002;
n=6; farx(n+1)=8.448263e+000; foe(n+1)=2.495353e+001; krok(n+1)=1.060828e-003; ng(n+1)=9.405434e+002;
n=7; farx(n+1)=8.158188e+000; foe(n+1)=2.318361e+001; krok(n+1)=1.146661e-003; ng(n+1)=1.290831e+003;
n=8; farx(n+1)=8.309636e+000; foe(n+1)=2.145062e+001; krok(n+1)=1.224233e-003; ng(n+1)=1.440392e+003;
n=9; farx(n+1)=7.541729e+000; foe(n+1)=1.889612e+001; krok(n+1)=2.729179e-002; ng(n+1)=7.418927e+002;
n=10; farx(n+1)=7.367802e+000; foe(n+1)=1.713902e+001; krok(n+1)=1.533559e-002; ng(n+1)=3.837182e+002;
n=11; farx(n+1)=6.759878e+000; foe(n+1)=1.364618e+001; krok(n+1)=1.512907e-003; ng(n+1)=1.121070e+003;
n=12; farx(n+1)=6.484962e+000; foe(n+1)=1.308284e+001; krok(n+1)=2.378319e-003; ng(n+1)=3.460189e+002;
n=13; farx(n+1)=4.834510e+000; foe(n+1)=1.014257e+001; krok(n+1)=2.900728e-002; ng(n+1)=4.825189e+002;
n=14; farx(n+1)=4.561863e+000; foe(n+1)=9.917156e+000; krok(n+1)=1.651908e-003; ng(n+1)=3.095595e+002;
n=15; farx(n+1)=3.191997e+000; foe(n+1)=8.683112e+000; krok(n+1)=1.302239e-002; ng(n+1)=2.315955e+002;
n=16; farx(n+1)=2.451133e+000; foe(n+1)=7.732890e+000; krok(n+1)=1.920058e-003; ng(n+1)=6.529602e+002;
n=17; farx(n+1)=2.197882e+000; foe(n+1)=7.487335e+000; krok(n+1)=5.038654e-003; ng(n+1)=1.134268e+002;
n=18; farx(n+1)=1.811808e+000; foe(n+1)=6.955891e+000; krok(n+1)=7.525650e-003; ng(n+1)=2.643876e+002;
n=19; farx(n+1)=1.519300e+000; foe(n+1)=6.119140e+000; krok(n+1)=5.172508e-003; ng(n+1)=4.260822e+002;
n=20; farx(n+1)=1.393008e+000; foe(n+1)=5.422611e+000; krok(n+1)=7.444102e-003; ng(n+1)=4.617590e+002;
n=21; farx(n+1)=1.390774e+000; foe(n+1)=5.067700e+000; krok(n+1)=1.587460e-003; ng(n+1)=4.682971e+002;
n=22; farx(n+1)=1.267592e+000; foe(n+1)=4.633253e+000; krok(n+1)=1.082825e-002; ng(n+1)=2.805541e+002;
n=23; farx(n+1)=1.153040e+000; foe(n+1)=4.095298e+000; krok(n+1)=1.477175e-002; ng(n+1)=3.561291e+002;
n=24; farx(n+1)=1.053909e+000; foe(n+1)=3.617129e+000; krok(n+1)=4.038010e-003; ng(n+1)=4.420900e+002;
n=25; farx(n+1)=9.815941e-001; foe(n+1)=2.878850e+000; krok(n+1)=2.015462e-002; ng(n+1)=4.582395e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=9.752373e-001; foe(n+1)=2.622826e+000; krok(n+1)=1.044207e-005; ng(n+1)=7.160270e+002;
n=27; farx(n+1)=9.791219e-001; foe(n+1)=2.584403e+000; krok(n+1)=1.104522e-005; ng(n+1)=2.481742e+002;
n=28; farx(n+1)=1.011688e+000; foe(n+1)=2.478154e+000; krok(n+1)=1.736749e-004; ng(n+1)=1.072092e+002;
n=29; farx(n+1)=9.931052e-001; foe(n+1)=2.393134e+000; krok(n+1)=1.916949e-003; ng(n+1)=4.607150e+001;
n=30; farx(n+1)=9.113076e-001; foe(n+1)=2.093243e+000; krok(n+1)=1.574999e-002; ng(n+1)=5.388442e+001;
n=31; farx(n+1)=9.259226e-001; foe(n+1)=1.826314e+000; krok(n+1)=1.012568e-002; ng(n+1)=3.851409e+002;
n=32; farx(n+1)=8.496762e-001; foe(n+1)=1.581615e+000; krok(n+1)=1.894771e-002; ng(n+1)=2.279293e+002;
n=33; farx(n+1)=7.247420e-001; foe(n+1)=1.473981e+000; krok(n+1)=1.974372e-002; ng(n+1)=3.832142e+001;
n=34; farx(n+1)=5.865171e-001; foe(n+1)=1.283405e+000; krok(n+1)=6.555717e-003; ng(n+1)=2.702039e+002;
n=35; farx(n+1)=5.693218e-001; foe(n+1)=1.252006e+000; krok(n+1)=6.970444e-003; ng(n+1)=1.302814e+002;
n=36; farx(n+1)=5.209115e-001; foe(n+1)=1.145292e+000; krok(n+1)=1.600986e-002; ng(n+1)=2.012398e+002;
n=37; farx(n+1)=4.848249e-001; foe(n+1)=1.061482e+000; krok(n+1)=2.351648e-002; ng(n+1)=1.165675e+002;
n=38; farx(n+1)=4.750899e-001; foe(n+1)=1.031805e+000; krok(n+1)=9.328627e-003; ng(n+1)=9.791327e+001;
n=39; farx(n+1)=4.421328e-001; foe(n+1)=9.301782e-001; krok(n+1)=2.857938e-002; ng(n+1)=9.785375e+001;
n=40; farx(n+1)=4.318937e-001; foe(n+1)=8.708235e-001; krok(n+1)=1.785858e-002; ng(n+1)=1.682951e+002;
n=41; farx(n+1)=4.175312e-001; foe(n+1)=8.378358e-001; krok(n+1)=9.289554e-003; ng(n+1)=1.014605e+002;
n=42; farx(n+1)=3.934771e-001; foe(n+1)=7.701001e-001; krok(n+1)=3.712452e-002; ng(n+1)=2.069331e+002;
n=43; farx(n+1)=3.937439e-001; foe(n+1)=6.945483e-001; krok(n+1)=5.286105e-002; ng(n+1)=1.396740e+002;
n=44; farx(n+1)=3.780149e-001; foe(n+1)=6.323801e-001; krok(n+1)=3.006802e-002; ng(n+1)=1.692878e+002;
n=45; farx(n+1)=3.752401e-001; foe(n+1)=6.167047e-001; krok(n+1)=2.727640e-002; ng(n+1)=1.062696e+002;
n=46; farx(n+1)=3.649628e-001; foe(n+1)=5.985246e-001; krok(n+1)=3.735290e-002; ng(n+1)=8.006331e+001;
n=47; farx(n+1)=3.279472e-001; foe(n+1)=5.304182e-001; krok(n+1)=1.072012e-001; ng(n+1)=3.379334e+001;
n=48; farx(n+1)=3.210979e-001; foe(n+1)=5.215961e-001; krok(n+1)=2.142359e-002; ng(n+1)=1.106095e+002;
n=49; farx(n+1)=3.042048e-001; foe(n+1)=4.767569e-001; krok(n+1)=1.328156e-001; ng(n+1)=4.689481e+001;
n=50; farx(n+1)=2.992235e-001; foe(n+1)=4.553267e-001; krok(n+1)=8.061847e-002; ng(n+1)=9.875189e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
