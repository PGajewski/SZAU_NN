%uczenie predyktora oe
clear all;
n=0; farx(n+1)=1.648811e+002; foe(n+1)=1.708138e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.521446e+002; foe(n+1)=1.573599e+002; krok(n+1)=6.856719e-004; ng(n+1)=4.777278e+002;
n=2; farx(n+1)=7.243959e+001; foe(n+1)=7.470282e+001; krok(n+1)=4.312903e-003; ng(n+1)=3.883691e+002;
n=3; farx(n+1)=6.583161e+001; foe(n+1)=6.435550e+001; krok(n+1)=1.810600e-003; ng(n+1)=4.881121e+002;
n=4; farx(n+1)=6.774682e+001; foe(n+1)=6.224685e+001; krok(n+1)=4.644777e-003; ng(n+1)=2.697019e+002;
n=5; farx(n+1)=3.486734e+001; foe(n+1)=5.573209e+001; krok(n+1)=5.286105e-002; ng(n+1)=4.831143e+001;
n=6; farx(n+1)=1.766905e+001; foe(n+1)=5.156664e+001; krok(n+1)=5.548179e-003; ng(n+1)=5.578170e+002;
n=7; farx(n+1)=9.405221e+000; foe(n+1)=4.905785e+001; krok(n+1)=1.581052e-003; ng(n+1)=1.184206e+003;
n=8; farx(n+1)=7.024675e+000; foe(n+1)=4.764766e+001; krok(n+1)=7.878882e-004; ng(n+1)=2.415929e+003;
n=9; farx(n+1)=5.756255e+000; foe(n+1)=4.595954e+001; krok(n+1)=8.018005e-004; ng(n+1)=2.914753e+003;
n=10; farx(n+1)=5.646877e+000; foe(n+1)=4.524551e+001; krok(n+1)=3.878507e-003; ng(n+1)=4.344152e+003;
n=11; farx(n+1)=5.756267e+000; foe(n+1)=4.432580e+001; krok(n+1)=2.883948e-003; ng(n+1)=4.914317e+003;
n=12; farx(n+1)=6.039532e+000; foe(n+1)=4.273525e+001; krok(n+1)=2.469223e-003; ng(n+1)=5.125371e+003;
n=13; farx(n+1)=6.167052e+000; foe(n+1)=4.221753e+001; krok(n+1)=3.205178e-004; ng(n+1)=4.199406e+003;
n=14; farx(n+1)=6.774531e+000; foe(n+1)=4.100547e+001; krok(n+1)=4.587536e-003; ng(n+1)=3.649649e+003;
n=15; farx(n+1)=7.528026e+000; foe(n+1)=3.925932e+001; krok(n+1)=1.944482e-003; ng(n+1)=2.693066e+003;
n=16; farx(n+1)=8.062549e+000; foe(n+1)=3.835391e+001; krok(n+1)=1.273916e-003; ng(n+1)=2.134509e+003;
n=17; farx(n+1)=9.677631e+000; foe(n+1)=3.640443e+001; krok(n+1)=7.860345e-003; ng(n+1)=2.617322e+003;
n=18; farx(n+1)=1.166961e+001; foe(n+1)=3.429103e+001; krok(n+1)=5.355898e-003; ng(n+1)=2.189648e+003;
n=19; farx(n+1)=1.421592e+001; foe(n+1)=3.029865e+001; krok(n+1)=1.111520e-002; ng(n+1)=1.259329e+003;
n=20; farx(n+1)=1.387638e+001; foe(n+1)=2.727644e+001; krok(n+1)=4.388300e-002; ng(n+1)=3.850012e+002;
n=21; farx(n+1)=1.344206e+001; foe(n+1)=2.607539e+001; krok(n+1)=4.548255e-003; ng(n+1)=6.764122e+002;
n=22; farx(n+1)=1.227744e+001; foe(n+1)=2.492372e+001; krok(n+1)=7.251821e-003; ng(n+1)=1.609935e+002;
n=23; farx(n+1)=9.381133e+000; foe(n+1)=2.206482e+001; krok(n+1)=1.238677e-002; ng(n+1)=4.395349e+002;
n=24; farx(n+1)=4.926769e+000; foe(n+1)=1.349303e+001; krok(n+1)=3.130011e-002; ng(n+1)=9.399607e+002;
n=25; farx(n+1)=4.544460e+000; foe(n+1)=1.285947e+001; krok(n+1)=2.716773e-003; ng(n+1)=7.365571e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=4.559458e+000; foe(n+1)=1.146564e+001; krok(n+1)=1.684728e-005; ng(n+1)=1.409607e+003;
n=27; farx(n+1)=4.465289e+000; foe(n+1)=9.625431e+000; krok(n+1)=1.984325e-004; ng(n+1)=3.651924e+002;
n=28; farx(n+1)=4.210716e+000; foe(n+1)=8.979332e+000; krok(n+1)=8.904993e-005; ng(n+1)=3.354245e+002;
n=29; farx(n+1)=3.131577e+000; foe(n+1)=6.490206e+000; krok(n+1)=2.442482e-003; ng(n+1)=3.146652e+002;
n=30; farx(n+1)=1.892725e+000; foe(n+1)=4.381475e+000; krok(n+1)=4.483426e-003; ng(n+1)=1.438709e+002;
n=31; farx(n+1)=1.356141e+000; foe(n+1)=3.125849e+000; krok(n+1)=2.431442e-003; ng(n+1)=1.029922e+003;
n=32; farx(n+1)=1.142973e+000; foe(n+1)=2.631194e+000; krok(n+1)=9.899545e-003; ng(n+1)=2.354034e+002;
n=33; farx(n+1)=1.037175e+000; foe(n+1)=2.337193e+000; krok(n+1)=5.655154e-003; ng(n+1)=2.676544e+002;
n=34; farx(n+1)=9.144194e-001; foe(n+1)=2.111766e+000; krok(n+1)=1.797739e-002; ng(n+1)=1.275364e+002;
n=35; farx(n+1)=7.757283e-001; foe(n+1)=1.910791e+000; krok(n+1)=5.699196e-003; ng(n+1)=2.128009e+002;
n=36; farx(n+1)=6.702715e-001; foe(n+1)=1.735943e+000; krok(n+1)=9.630946e-003; ng(n+1)=1.867240e+002;
n=37; farx(n+1)=6.021566e-001; foe(n+1)=1.614970e+000; krok(n+1)=8.402186e-003; ng(n+1)=1.408241e+002;
n=38; farx(n+1)=5.597840e-001; foe(n+1)=1.462329e+000; krok(n+1)=1.077104e-002; ng(n+1)=9.623502e+001;
n=39; farx(n+1)=5.272204e-001; foe(n+1)=1.264683e+000; krok(n+1)=2.047906e-002; ng(n+1)=3.169585e+002;
n=40; farx(n+1)=5.275283e-001; foe(n+1)=1.202166e+000; krok(n+1)=4.663178e-003; ng(n+1)=1.779552e+002;
n=41; farx(n+1)=5.457524e-001; foe(n+1)=1.120579e+000; krok(n+1)=1.846372e-002; ng(n+1)=1.284723e+002;
n=42; farx(n+1)=5.265248e-001; foe(n+1)=1.059385e+000; krok(n+1)=7.000904e-002; ng(n+1)=1.128885e+002;
n=43; farx(n+1)=5.097231e-001; foe(n+1)=1.037738e+000; krok(n+1)=1.851676e-002; ng(n+1)=1.592167e+002;
n=44; farx(n+1)=4.511070e-001; foe(n+1)=1.005938e+000; krok(n+1)=7.919636e-002; ng(n+1)=2.654625e+001;
n=45; farx(n+1)=4.400120e-001; foe(n+1)=9.866279e-001; krok(n+1)=3.072094e-002; ng(n+1)=8.124127e+001;
n=46; farx(n+1)=4.337867e-001; foe(n+1)=9.676878e-001; krok(n+1)=5.142836e-002; ng(n+1)=3.823176e+001;
n=47; farx(n+1)=4.444495e-001; foe(n+1)=9.393709e-001; krok(n+1)=6.306675e-002; ng(n+1)=1.235182e+002;
n=48; farx(n+1)=4.347803e-001; foe(n+1)=9.147558e-001; krok(n+1)=2.978305e-001; ng(n+1)=1.871345e+001;
n=49; farx(n+1)=4.363646e-001; foe(n+1)=8.625511e-001; krok(n+1)=1.538927e-001; ng(n+1)=1.058768e+002;
n=50; farx(n+1)=4.255009e-001; foe(n+1)=8.473372e-001; krok(n+1)=1.428686e-001; ng(n+1)=6.110384e+001;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)