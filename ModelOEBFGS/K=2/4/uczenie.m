%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.894911e+002; foe(n+1)=2.862797e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.819488e+002; foe(n+1)=1.774322e+002; krok(n+1)=5.053235e-004; ng(n+1)=6.313455e+002;
n=2; farx(n+1)=7.970239e+001; foe(n+1)=6.186919e+001; krok(n+1)=4.171530e-002; ng(n+1)=6.797406e+001;
n=3; farx(n+1)=7.828560e+001; foe(n+1)=6.124503e+001; krok(n+1)=1.279942e-003; ng(n+1)=7.239031e+001;
n=4; farx(n+1)=8.892183e+000; foe(n+1)=3.224308e+001; krok(n+1)=4.568872e-002; ng(n+1)=5.292669e+001;
n=5; farx(n+1)=7.441688e+000; foe(n+1)=3.140559e+001; krok(n+1)=1.395478e-005; ng(n+1)=7.240657e+002;
n=6; farx(n+1)=7.961172e+000; foe(n+1)=3.023407e+001; krok(n+1)=9.338225e-003; ng(n+1)=9.538490e+002;
n=7; farx(n+1)=8.320708e+000; foe(n+1)=2.856708e+001; krok(n+1)=3.818612e-003; ng(n+1)=6.824361e+002;
n=8; farx(n+1)=7.695609e+000; foe(n+1)=2.587201e+001; krok(n+1)=4.236653e-003; ng(n+1)=5.706855e+002;
n=9; farx(n+1)=7.389789e+000; foe(n+1)=2.339731e+001; krok(n+1)=4.285764e-002; ng(n+1)=3.811764e+002;
n=10; farx(n+1)=5.588631e+000; foe(n+1)=1.570619e+001; krok(n+1)=2.833785e-001; ng(n+1)=3.467228e+002;
n=11; farx(n+1)=4.677247e+000; foe(n+1)=1.470616e+001; krok(n+1)=1.196783e-001; ng(n+1)=1.211325e+002;
n=12; farx(n+1)=2.916138e+000; foe(n+1)=1.134547e+001; krok(n+1)=4.150498e-001; ng(n+1)=1.279679e+002;
n=13; farx(n+1)=2.192434e+000; foe(n+1)=9.484303e+000; krok(n+1)=1.571889e-001; ng(n+1)=4.473221e+002;
n=14; farx(n+1)=1.201351e+000; foe(n+1)=7.248177e+000; krok(n+1)=4.013589e-001; ng(n+1)=4.944181e+002;
n=15; farx(n+1)=1.128954e+000; foe(n+1)=6.726968e+000; krok(n+1)=3.743213e-001; ng(n+1)=1.901808e+002;
n=16; farx(n+1)=1.248302e+000; foe(n+1)=6.611536e+000; krok(n+1)=1.992226e+000; ng(n+1)=3.300731e+001;
n=17; farx(n+1)=1.176267e+000; foe(n+1)=6.154047e+000; krok(n+1)=4.123480e+000; ng(n+1)=4.976388e+001;
n=18; farx(n+1)=7.751315e-001; foe(n+1)=5.244579e+000; krok(n+1)=8.892951e-001; ng(n+1)=2.198436e+002;
n=19; farx(n+1)=5.916377e-001; foe(n+1)=4.903507e+000; krok(n+1)=2.538306e-001; ng(n+1)=2.837734e+002;
n=20; farx(n+1)=5.261519e-001; foe(n+1)=4.475070e+000; krok(n+1)=2.991387e-001; ng(n+1)=6.381873e+002;
n=21; farx(n+1)=5.010224e-001; foe(n+1)=4.294192e+000; krok(n+1)=5.045340e-001; ng(n+1)=3.454269e+002;
n=22; farx(n+1)=5.108571e-001; foe(n+1)=3.982497e+000; krok(n+1)=6.446279e-001; ng(n+1)=2.551183e+002;
n=23; farx(n+1)=5.101661e-001; foe(n+1)=3.920781e+000; krok(n+1)=2.277746e-001; ng(n+1)=1.082812e+002;
n=24; farx(n+1)=4.883193e-001; foe(n+1)=3.826433e+000; krok(n+1)=4.887823e-001; ng(n+1)=4.992149e+001;
n=25; farx(n+1)=4.787364e-001; foe(n+1)=3.681577e+000; krok(n+1)=5.695627e-001; ng(n+1)=1.294182e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=4.788841e-001; foe(n+1)=3.614922e+000; krok(n+1)=3.896489e-006; ng(n+1)=3.198883e+002;
n=27; farx(n+1)=4.776620e-001; foe(n+1)=3.613471e+000; krok(n+1)=7.022416e-005; ng(n+1)=1.166223e+001;
n=28; farx(n+1)=4.786602e-001; foe(n+1)=3.606931e+000; krok(n+1)=3.070097e-004; ng(n+1)=1.109655e+001;
n=29; farx(n+1)=4.490088e-001; foe(n+1)=3.549794e+000; krok(n+1)=1.779489e-003; ng(n+1)=1.583308e+001;
n=30; farx(n+1)=4.461654e-001; foe(n+1)=3.504695e+000; krok(n+1)=5.123414e-003; ng(n+1)=4.702003e+001;
n=31; farx(n+1)=4.251640e-001; foe(n+1)=3.434870e+000; krok(n+1)=5.344961e-002; ng(n+1)=1.060717e+002;
n=32; farx(n+1)=4.089628e-001; foe(n+1)=3.307507e+000; krok(n+1)=2.476948e-002; ng(n+1)=1.217598e+002;
n=33; farx(n+1)=3.898270e-001; foe(n+1)=3.239441e+000; krok(n+1)=5.005537e-002; ng(n+1)=1.900162e+002;
n=34; farx(n+1)=3.804125e-001; foe(n+1)=3.186882e+000; krok(n+1)=5.349375e-002; ng(n+1)=1.865381e+002;
n=35; farx(n+1)=3.469477e-001; foe(n+1)=2.999431e+000; krok(n+1)=4.108715e-001; ng(n+1)=2.768168e+002;
n=36; farx(n+1)=3.217751e-001; foe(n+1)=2.809970e+000; krok(n+1)=1.032026e-001; ng(n+1)=1.385836e+002;
n=37; farx(n+1)=3.070206e-001; foe(n+1)=2.686418e+000; krok(n+1)=4.352152e-001; ng(n+1)=2.077977e+002;
n=38; farx(n+1)=2.947870e-001; foe(n+1)=2.618674e+000; krok(n+1)=4.641165e-001; ng(n+1)=4.024542e+001;
n=39; farx(n+1)=2.952441e-001; foe(n+1)=2.552229e+000; krok(n+1)=3.380830e-001; ng(n+1)=1.357712e+002;
n=40; farx(n+1)=2.973111e-001; foe(n+1)=2.476187e+000; krok(n+1)=8.159604e-001; ng(n+1)=1.656716e+002;
n=41; farx(n+1)=2.961320e-001; foe(n+1)=2.455326e+000; krok(n+1)=2.594656e-001; ng(n+1)=9.225472e+001;
n=42; farx(n+1)=2.940732e-001; foe(n+1)=2.440349e+000; krok(n+1)=1.634459e-001; ng(n+1)=5.827922e+001;
n=43; farx(n+1)=2.942744e-001; foe(n+1)=2.433659e+000; krok(n+1)=3.435737e-001; ng(n+1)=3.410370e+001;
n=44; farx(n+1)=2.980008e-001; foe(n+1)=2.427023e+000; krok(n+1)=3.268918e-001; ng(n+1)=5.873013e+001;
n=45; farx(n+1)=3.000074e-001; foe(n+1)=2.414229e+000; krok(n+1)=8.094987e-001; ng(n+1)=8.566867e+001;
n=46; farx(n+1)=2.995128e-001; foe(n+1)=2.401134e+000; krok(n+1)=1.155850e+000; ng(n+1)=8.931642e+000;
n=47; farx(n+1)=2.987500e-001; foe(n+1)=2.382258e+000; krok(n+1)=8.728449e-001; ng(n+1)=7.369131e+001;
n=48; farx(n+1)=2.977142e-001; foe(n+1)=2.362647e+000; krok(n+1)=1.485218e+000; ng(n+1)=7.132194e+001;
n=49; farx(n+1)=2.983900e-001; foe(n+1)=2.357201e+000; krok(n+1)=1.009068e+000; ng(n+1)=5.285905e+001;
n=50; farx(n+1)=2.981960e-001; foe(n+1)=2.353928e+000; krok(n+1)=2.127187e+000; ng(n+1)=2.838944e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)