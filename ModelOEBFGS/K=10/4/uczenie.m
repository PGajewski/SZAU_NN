%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.372392e+002; foe(n+1)=2.263830e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.797643e+002; foe(n+1)=1.713677e+002; krok(n+1)=5.322828e-004; ng(n+1)=9.362076e+002;
n=2; farx(n+1)=7.171547e+001; foe(n+1)=6.380522e+001; krok(n+1)=9.636858e-003; ng(n+1)=4.104819e+002;
n=3; farx(n+1)=7.033536e+001; foe(n+1)=6.321342e+001; krok(n+1)=1.198026e-003; ng(n+1)=1.287209e+002;
n=4; farx(n+1)=5.829302e+001; foe(n+1)=6.200347e+001; krok(n+1)=7.314625e-003; ng(n+1)=7.980636e+001;
n=5; farx(n+1)=3.439602e+000; foe(n+1)=2.488359e+001; krok(n+1)=1.127471e-002; ng(n+1)=1.955509e+002;
n=6; farx(n+1)=3.178496e+000; foe(n+1)=2.461035e+001; krok(n+1)=9.318742e-006; ng(n+1)=2.975007e+003;
n=7; farx(n+1)=4.087456e+000; foe(n+1)=1.369361e+001; krok(n+1)=4.235232e-002; ng(n+1)=3.485723e+003;
n=8; farx(n+1)=4.152022e+000; foe(n+1)=1.351062e+001; krok(n+1)=4.220039e-004; ng(n+1)=6.544478e+002;
n=9; farx(n+1)=4.208228e+000; foe(n+1)=1.328056e+001; krok(n+1)=2.469785e-003; ng(n+1)=4.087393e+002;
n=10; farx(n+1)=4.054621e+000; foe(n+1)=1.104685e+001; krok(n+1)=6.019169e-003; ng(n+1)=5.207414e+002;
n=11; farx(n+1)=3.814718e+000; foe(n+1)=1.058774e+001; krok(n+1)=6.130766e-003; ng(n+1)=2.670732e+002;
n=12; farx(n+1)=3.614522e+000; foe(n+1)=1.025236e+001; krok(n+1)=1.529718e-003; ng(n+1)=3.061735e+002;
n=13; farx(n+1)=2.462305e+000; foe(n+1)=9.162743e+000; krok(n+1)=4.852216e-002; ng(n+1)=1.238607e+002;
n=14; farx(n+1)=2.051871e+000; foe(n+1)=8.645522e+000; krok(n+1)=7.663457e-004; ng(n+1)=8.138273e+002;
n=15; farx(n+1)=1.671674e+000; foe(n+1)=8.144060e+000; krok(n+1)=6.767654e-004; ng(n+1)=9.428980e+002;
n=16; farx(n+1)=1.283904e+000; foe(n+1)=7.285255e+000; krok(n+1)=1.672561e-002; ng(n+1)=1.088723e+003;
n=17; farx(n+1)=1.148599e+000; foe(n+1)=7.003684e+000; krok(n+1)=4.046350e-003; ng(n+1)=4.569590e+002;
n=18; farx(n+1)=9.065522e-001; foe(n+1)=6.116531e+000; krok(n+1)=5.711090e-003; ng(n+1)=5.264177e+002;
n=19; farx(n+1)=8.448198e-001; foe(n+1)=5.933795e+000; krok(n+1)=4.267031e-003; ng(n+1)=2.826597e+002;
n=20; farx(n+1)=7.335125e-001; foe(n+1)=5.233003e+000; krok(n+1)=1.054870e-002; ng(n+1)=5.328249e+002;
n=21; farx(n+1)=7.031040e-001; foe(n+1)=5.024627e+000; krok(n+1)=1.772228e-003; ng(n+1)=4.517135e+002;
n=22; farx(n+1)=7.009730e-001; foe(n+1)=4.966328e+000; krok(n+1)=8.099144e-003; ng(n+1)=3.228752e+002;
n=23; farx(n+1)=7.005623e-001; foe(n+1)=4.447553e+000; krok(n+1)=2.803148e-002; ng(n+1)=2.867417e+002;
n=24; farx(n+1)=6.840077e-001; foe(n+1)=4.271904e+000; krok(n+1)=3.660956e-003; ng(n+1)=2.575146e+002;
n=25; farx(n+1)=6.432150e-001; foe(n+1)=3.800263e+000; krok(n+1)=2.883948e-003; ng(n+1)=5.675858e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=6.494998e-001; foe(n+1)=3.645580e+000; krok(n+1)=6.202616e-006; ng(n+1)=5.408923e+002;
n=27; farx(n+1)=6.532444e-001; foe(n+1)=3.610123e+000; krok(n+1)=1.761549e-005; ng(n+1)=2.336410e+002;
n=28; farx(n+1)=6.581388e-001; foe(n+1)=3.432854e+000; krok(n+1)=1.528320e-003; ng(n+1)=5.765286e+001;
n=29; farx(n+1)=5.937008e-001; foe(n+1)=3.076052e+000; krok(n+1)=8.567570e-004; ng(n+1)=1.176214e+002;
n=30; farx(n+1)=7.053965e-001; foe(n+1)=2.194141e+000; krok(n+1)=2.954351e-002; ng(n+1)=3.230453e+001;
n=31; farx(n+1)=7.724725e-001; foe(n+1)=1.977318e+000; krok(n+1)=1.130223e-003; ng(n+1)=6.487108e+002;
n=32; farx(n+1)=7.587385e-001; foe(n+1)=1.887372e+000; krok(n+1)=1.729712e-002; ng(n+1)=1.148916e+002;
n=33; farx(n+1)=7.102732e-001; foe(n+1)=1.670673e+000; krok(n+1)=4.799811e-002; ng(n+1)=3.265529e+002;
n=34; farx(n+1)=6.978138e-001; foe(n+1)=1.614277e+000; krok(n+1)=2.774090e-003; ng(n+1)=3.157345e+002;
n=35; farx(n+1)=6.875236e-001; foe(n+1)=1.523447e+000; krok(n+1)=1.142380e-002; ng(n+1)=2.265774e+002;
n=36; farx(n+1)=6.867349e-001; foe(n+1)=1.488426e+000; krok(n+1)=1.388291e-002; ng(n+1)=1.809605e+002;
n=37; farx(n+1)=6.741860e-001; foe(n+1)=1.436006e+000; krok(n+1)=4.134280e-002; ng(n+1)=3.428015e+001;
n=38; farx(n+1)=6.734723e-001; foe(n+1)=1.376956e+000; krok(n+1)=1.284535e-002; ng(n+1)=2.267389e+002;
n=39; farx(n+1)=6.588269e-001; foe(n+1)=1.291855e+000; krok(n+1)=2.107682e-002; ng(n+1)=2.342081e+002;
n=40; farx(n+1)=6.705316e-001; foe(n+1)=1.221731e+000; krok(n+1)=2.038894e-002; ng(n+1)=1.704750e+002;
n=41; farx(n+1)=6.196972e-001; foe(n+1)=1.139064e+000; krok(n+1)=2.643052e-002; ng(n+1)=1.512278e+002;
n=42; farx(n+1)=5.931163e-001; foe(n+1)=1.108367e+000; krok(n+1)=7.609550e-003; ng(n+1)=2.235388e+002;
n=43; farx(n+1)=5.576526e-001; foe(n+1)=1.085081e+000; krok(n+1)=2.165649e-002; ng(n+1)=6.045268e+001;
n=44; farx(n+1)=5.034311e-001; foe(n+1)=1.047586e+000; krok(n+1)=2.340953e-002; ng(n+1)=1.742231e+002;
n=45; farx(n+1)=4.894396e-001; foe(n+1)=1.034149e+000; krok(n+1)=1.254988e-002; ng(n+1)=1.362890e+002;
n=46; farx(n+1)=4.824345e-001; foe(n+1)=9.940409e-001; krok(n+1)=8.005809e-002; ng(n+1)=8.054159e+001;
n=47; farx(n+1)=4.686442e-001; foe(n+1)=9.677162e-001; krok(n+1)=3.413297e-002; ng(n+1)=1.475556e+002;
n=48; farx(n+1)=4.513637e-001; foe(n+1)=9.454528e-001; krok(n+1)=4.076532e-002; ng(n+1)=1.064255e+002;
n=49; farx(n+1)=4.388344e-001; foe(n+1)=9.140001e-001; krok(n+1)=9.464413e-002; ng(n+1)=8.898382e+001;
n=50; farx(n+1)=4.252127e-001; foe(n+1)=8.896928e-001; krok(n+1)=5.213103e-002; ng(n+1)=1.359611e+002;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
