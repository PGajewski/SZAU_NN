%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.055556e+002; foe(n+1)=2.003608e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.819725e+002; foe(n+1)=1.778426e+002; krok(n+1)=5.504055e-004; ng(n+1)=5.486018e+002;
n=2; farx(n+1)=9.354810e+001; foe(n+1)=7.063516e+001; krok(n+1)=2.219272e-002; ng(n+1)=2.867207e+002;
n=3; farx(n+1)=8.231636e+001; foe(n+1)=6.379606e+001; krok(n+1)=3.072763e-004; ng(n+1)=7.894612e+002;
n=4; farx(n+1)=7.990845e+001; foe(n+1)=6.303549e+001; krok(n+1)=7.740461e-004; ng(n+1)=1.898907e+002;
n=5; farx(n+1)=2.326897e+001; foe(n+1)=5.223680e+001; krok(n+1)=1.488820e-002; ng(n+1)=1.070436e+002;
n=6; farx(n+1)=9.747261e+000; foe(n+1)=4.510990e+001; krok(n+1)=4.505183e-003; ng(n+1)=1.528123e+003;
n=7; farx(n+1)=7.857724e+000; foe(n+1)=4.306859e+001; krok(n+1)=2.752027e-004; ng(n+1)=5.033973e+003;
n=8; farx(n+1)=7.007737e+000; foe(n+1)=4.132062e+001; krok(n+1)=2.827499e-003; ng(n+1)=7.290388e+003;
n=9; farx(n+1)=6.979915e+000; foe(n+1)=3.840211e+001; krok(n+1)=1.757317e-003; ng(n+1)=8.554699e+003;
n=10; farx(n+1)=7.355236e+000; foe(n+1)=3.768614e+001; krok(n+1)=4.031350e-004; ng(n+1)=8.161550e+003;
n=11; farx(n+1)=8.183076e+000; foe(n+1)=3.701431e+001; krok(n+1)=1.532691e-003; ng(n+1)=6.774448e+003;
n=12; farx(n+1)=9.366819e+000; foe(n+1)=3.529441e+001; krok(n+1)=3.516052e-003; ng(n+1)=5.669995e+003;
n=13; farx(n+1)=1.000513e+001; foe(n+1)=3.460767e+001; krok(n+1)=1.102111e-003; ng(n+1)=3.242207e+003;
n=14; farx(n+1)=1.170072e+001; foe(n+1)=3.273865e+001; krok(n+1)=4.889122e-003; ng(n+1)=2.423776e+003;
n=15; farx(n+1)=1.357730e+001; foe(n+1)=3.015172e+001; krok(n+1)=2.302721e-003; ng(n+1)=2.430036e+003;
n=16; farx(n+1)=1.453198e+001; foe(n+1)=2.863250e+001; krok(n+1)=3.096185e-003; ng(n+1)=1.479726e+003;
n=17; farx(n+1)=1.482763e+001; foe(n+1)=2.707607e+001; krok(n+1)=8.166461e-003; ng(n+1)=4.431109e+002;
n=18; farx(n+1)=1.375714e+001; foe(n+1)=2.594209e+001; krok(n+1)=9.879138e-003; ng(n+1)=3.420929e+002;
n=19; farx(n+1)=1.228536e+001; foe(n+1)=2.432948e+001; krok(n+1)=1.109636e-002; ng(n+1)=4.902754e+002;
n=20; farx(n+1)=1.042831e+001; foe(n+1)=2.208367e+001; krok(n+1)=1.226153e-002; ng(n+1)=3.844131e+002;
n=21; farx(n+1)=7.480772e+000; foe(n+1)=1.687779e+001; krok(n+1)=6.099924e-003; ng(n+1)=8.945058e+002;
n=22; farx(n+1)=6.928199e+000; foe(n+1)=1.583237e+001; krok(n+1)=6.497382e-004; ng(n+1)=5.769332e+002;
n=23; farx(n+1)=5.495191e+000; foe(n+1)=1.437016e+001; krok(n+1)=3.222183e-002; ng(n+1)=5.412166e+002;
n=24; farx(n+1)=4.437945e+000; foe(n+1)=1.315761e+001; krok(n+1)=1.086709e-002; ng(n+1)=4.843042e+002;
n=25; farx(n+1)=3.827181e+000; foe(n+1)=1.248366e+001; krok(n+1)=2.983603e-003; ng(n+1)=8.020892e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=3.771992e+000; foe(n+1)=1.224905e+001; krok(n+1)=8.001301e-006; ng(n+1)=6.555646e+002;
n=27; farx(n+1)=3.639085e+000; foe(n+1)=1.210883e+001; krok(n+1)=6.738910e-005; ng(n+1)=1.815915e+002;
n=28; farx(n+1)=3.581586e+000; foe(n+1)=1.092603e+001; krok(n+1)=6.326141e-004; ng(n+1)=1.703167e+002;
n=29; farx(n+1)=3.140212e+000; foe(n+1)=9.998621e+000; krok(n+1)=3.729911e-004; ng(n+1)=2.559890e+002;
n=30; farx(n+1)=2.716708e+000; foe(n+1)=8.863669e+000; krok(n+1)=1.255562e-003; ng(n+1)=2.256656e+002;
n=31; farx(n+1)=2.358027e+000; foe(n+1)=7.468119e+000; krok(n+1)=6.023109e-003; ng(n+1)=7.342916e+002;
n=32; farx(n+1)=2.224233e+000; foe(n+1)=5.745103e+000; krok(n+1)=9.019129e-003; ng(n+1)=8.140868e+002;
n=33; farx(n+1)=2.091621e+000; foe(n+1)=5.099089e+000; krok(n+1)=4.696827e-003; ng(n+1)=8.580829e+002;
n=34; farx(n+1)=1.807321e+000; foe(n+1)=4.627325e+000; krok(n+1)=1.542549e-002; ng(n+1)=5.524219e+002;
n=35; farx(n+1)=1.469128e+000; foe(n+1)=3.787657e+000; krok(n+1)=4.352613e-003; ng(n+1)=1.238755e+003;
n=36; farx(n+1)=1.401173e+000; foe(n+1)=3.601160e+000; krok(n+1)=5.034661e-004; ng(n+1)=8.102534e+002;
n=37; farx(n+1)=1.254546e+000; foe(n+1)=3.069835e+000; krok(n+1)=1.311143e-002; ng(n+1)=8.354699e+002;
n=38; farx(n+1)=1.125324e+000; foe(n+1)=2.739972e+000; krok(n+1)=7.100804e-003; ng(n+1)=4.338495e+002;
n=39; farx(n+1)=9.358851e-001; foe(n+1)=2.478214e+000; krok(n+1)=1.835014e-002; ng(n+1)=1.150505e+002;
n=40; farx(n+1)=7.798331e-001; foe(n+1)=2.187091e+000; krok(n+1)=1.481850e-002; ng(n+1)=5.852091e+002;
n=41; farx(n+1)=7.736938e-001; foe(n+1)=2.149125e+000; krok(n+1)=3.813746e-003; ng(n+1)=2.247113e+002;
n=42; farx(n+1)=7.568515e-001; foe(n+1)=2.058825e+000; krok(n+1)=1.448480e-002; ng(n+1)=3.913416e+001;
n=43; farx(n+1)=7.469614e-001; foe(n+1)=1.939894e+000; krok(n+1)=3.542231e-002; ng(n+1)=2.930053e+002;
n=44; farx(n+1)=7.736895e-001; foe(n+1)=1.852141e+000; krok(n+1)=2.900728e-002; ng(n+1)=5.056481e+002;
n=45; farx(n+1)=7.898105e-001; foe(n+1)=1.757552e+000; krok(n+1)=5.584770e-002; ng(n+1)=1.589868e+002;
n=46; farx(n+1)=7.406767e-001; foe(n+1)=1.627825e+000; krok(n+1)=1.280543e-001; ng(n+1)=1.372419e+002;
n=47; farx(n+1)=6.537310e-001; foe(n+1)=1.514865e+000; krok(n+1)=7.317146e-002; ng(n+1)=1.517249e+002;
n=48; farx(n+1)=6.509535e-001; foe(n+1)=1.485892e+000; krok(n+1)=2.452306e-002; ng(n+1)=1.257692e+002;
n=49; farx(n+1)=7.748840e-001; foe(n+1)=1.355960e+000; krok(n+1)=2.393567e-001; ng(n+1)=1.531787e+002;
n=50; farx(n+1)=7.622248e-001; foe(n+1)=1.309978e+000; krok(n+1)=2.097129e-002; ng(n+1)=1.154878e+002;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
