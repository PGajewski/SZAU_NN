%uczenie predyktora oe
clear all;
n=0; farx(n+1)=1.816818e+002; foe(n+1)=1.830941e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.727026e+002; foe(n+1)=1.743083e+002; krok(n+1)=6.603743e-004; ng(n+1)=3.389080e+002;
n=2; farx(n+1)=6.544828e+001; foe(n+1)=6.723673e+001; krok(n+1)=1.196180e-002; ng(n+1)=2.670456e+002;
n=3; farx(n+1)=6.075796e+001; foe(n+1)=6.123061e+001; krok(n+1)=1.072587e-003; ng(n+1)=3.473834e+002;
n=4; farx(n+1)=5.764535e+001; foe(n+1)=6.052307e+001; krok(n+1)=3.460293e-003; ng(n+1)=6.579282e+001;
n=5; farx(n+1)=1.676176e+001; foe(n+1)=4.650878e+001; krok(n+1)=1.423591e-002; ng(n+1)=1.342249e+002;
n=6; farx(n+1)=9.437569e+000; foe(n+1)=4.331538e+001; krok(n+1)=1.255562e-003; ng(n+1)=8.827407e+002;
n=7; farx(n+1)=8.282362e+000; foe(n+1)=4.253513e+001; krok(n+1)=1.624224e-003; ng(n+1)=1.522162e+003;
n=8; farx(n+1)=5.298865e+000; foe(n+1)=3.245399e+001; krok(n+1)=3.881922e-003; ng(n+1)=2.057488e+003;
n=9; farx(n+1)=5.224916e+000; foe(n+1)=3.223052e+001; krok(n+1)=1.333447e-004; ng(n+1)=2.933488e+003;
n=10; farx(n+1)=6.629919e+000; foe(n+1)=2.961601e+001; krok(n+1)=2.262062e-002; ng(n+1)=2.739555e+003;
n=11; farx(n+1)=8.035815e+000; foe(n+1)=2.688630e+001; krok(n+1)=2.059282e-003; ng(n+1)=1.819416e+003;
n=12; farx(n+1)=8.277615e+000; foe(n+1)=2.586555e+001; krok(n+1)=3.334617e-003; ng(n+1)=6.153859e+002;
n=13; farx(n+1)=8.218995e+000; foe(n+1)=2.482859e+001; krok(n+1)=6.543079e-003; ng(n+1)=3.364580e+002;
n=14; farx(n+1)=7.665317e+000; foe(n+1)=2.149896e+001; krok(n+1)=7.251821e-003; ng(n+1)=5.959907e+002;
n=15; farx(n+1)=7.393250e+000; foe(n+1)=1.977180e+001; krok(n+1)=8.948804e-003; ng(n+1)=2.983050e+002;
n=16; farx(n+1)=6.985547e+000; foe(n+1)=1.880287e+001; krok(n+1)=1.533559e-002; ng(n+1)=1.472935e+002;
n=17; farx(n+1)=5.364422e+000; foe(n+1)=1.615955e+001; krok(n+1)=8.097460e-002; ng(n+1)=2.236670e+002;
n=18; farx(n+1)=3.865345e+000; foe(n+1)=1.322423e+001; krok(n+1)=4.388300e-002; ng(n+1)=3.269085e+002;
n=19; farx(n+1)=3.406142e+000; foe(n+1)=1.220387e+001; krok(n+1)=1.965861e-002; ng(n+1)=2.602337e+002;
n=20; farx(n+1)=2.458176e+000; foe(n+1)=9.690246e+000; krok(n+1)=4.138006e-002; ng(n+1)=3.965618e+002;
n=21; farx(n+1)=2.286042e+000; foe(n+1)=9.105338e+000; krok(n+1)=5.927401e-002; ng(n+1)=3.203519e+002;
n=22; farx(n+1)=2.173153e+000; foe(n+1)=8.326419e+000; krok(n+1)=5.983917e-002; ng(n+1)=2.645485e+002;
n=23; farx(n+1)=2.161946e+000; foe(n+1)=7.598008e+000; krok(n+1)=6.528949e-002; ng(n+1)=2.157341e+002;
n=24; farx(n+1)=1.908268e+000; foe(n+1)=7.208206e+000; krok(n+1)=2.475890e-001; ng(n+1)=1.106999e+002;
n=25; farx(n+1)=1.325525e+000; foe(n+1)=6.132505e+000; krok(n+1)=3.340461e-001; ng(n+1)=3.149706e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=1.320418e+000; foe(n+1)=5.966755e+000; krok(n+1)=3.452947e-005; ng(n+1)=1.911919e+002;
n=27; farx(n+1)=1.311569e+000; foe(n+1)=5.904717e+000; krok(n+1)=4.063871e-005; ng(n+1)=1.158013e+002;
n=28; farx(n+1)=1.366903e+000; foe(n+1)=5.756761e+000; krok(n+1)=1.500113e-004; ng(n+1)=1.092441e+002;
n=29; farx(n+1)=1.329325e+000; foe(n+1)=5.134697e+000; krok(n+1)=6.402952e-003; ng(n+1)=3.587460e+001;
n=30; farx(n+1)=1.185415e+000; foe(n+1)=4.510938e+000; krok(n+1)=1.462925e-002; ng(n+1)=1.111488e+002;
n=31; farx(n+1)=1.086703e+000; foe(n+1)=3.923501e+000; krok(n+1)=5.637357e-003; ng(n+1)=4.278019e+002;
n=32; farx(n+1)=1.019725e+000; foe(n+1)=3.650803e+000; krok(n+1)=6.319511e-003; ng(n+1)=1.548272e+002;
n=33; farx(n+1)=1.019423e+000; foe(n+1)=3.154359e+000; krok(n+1)=1.142218e-002; ng(n+1)=3.490814e+002;
n=34; farx(n+1)=9.845711e-001; foe(n+1)=3.037462e+000; krok(n+1)=1.184956e-002; ng(n+1)=1.065530e+002;
n=35; farx(n+1)=9.254384e-001; foe(n+1)=2.804149e+000; krok(n+1)=9.375382e-003; ng(n+1)=1.695970e+002;
n=36; farx(n+1)=8.657007e-001; foe(n+1)=2.332409e+000; krok(n+1)=1.389524e-002; ng(n+1)=2.024106e+002;
n=37; farx(n+1)=8.427544e-001; foe(n+1)=2.267476e+000; krok(n+1)=2.128042e-002; ng(n+1)=9.209753e+001;
n=38; farx(n+1)=8.220033e-001; foe(n+1)=2.138580e+000; krok(n+1)=2.083339e-002; ng(n+1)=1.347007e+002;
n=39; farx(n+1)=8.413078e-001; foe(n+1)=2.026940e+000; krok(n+1)=1.143200e-001; ng(n+1)=8.652007e+001;
n=40; farx(n+1)=8.301403e-001; foe(n+1)=1.875048e+000; krok(n+1)=7.689657e-002; ng(n+1)=6.294126e+001;
n=41; farx(n+1)=7.997347e-001; foe(n+1)=1.773628e+000; krok(n+1)=5.918252e-002; ng(n+1)=7.804916e+001;
n=42; farx(n+1)=7.264998e-001; foe(n+1)=1.642995e+000; krok(n+1)=2.587256e-001; ng(n+1)=8.847882e+001;
n=43; farx(n+1)=6.824672e-001; foe(n+1)=1.588577e+000; krok(n+1)=1.630613e-001; ng(n+1)=1.136904e+002;
n=44; farx(n+1)=6.314451e-001; foe(n+1)=1.549253e+000; krok(n+1)=1.053314e-001; ng(n+1)=6.730383e+001;
n=45; farx(n+1)=5.903987e-001; foe(n+1)=1.515641e+000; krok(n+1)=2.114442e-001; ng(n+1)=7.038567e+001;
n=46; farx(n+1)=5.367220e-001; foe(n+1)=1.409004e+000; krok(n+1)=2.367770e-001; ng(n+1)=5.005803e+001;
n=47; farx(n+1)=5.369708e-001; foe(n+1)=1.331127e+000; krok(n+1)=2.268451e-001; ng(n+1)=3.701129e+001;
n=48; farx(n+1)=5.373822e-001; foe(n+1)=1.284640e+000; krok(n+1)=1.003397e-001; ng(n+1)=1.450117e+002;
n=49; farx(n+1)=4.949808e-001; foe(n+1)=1.227470e+000; krok(n+1)=6.666685e-001; ng(n+1)=6.572643e+001;
n=50; farx(n+1)=4.806636e-001; foe(n+1)=1.183411e+000; krok(n+1)=4.317671e-001; ng(n+1)=1.082512e+002;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
