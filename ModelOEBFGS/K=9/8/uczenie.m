%uczenie predyktora oe
clear all;
n=0; farx(n+1)=2.210500e+002; foe(n+1)=2.255502e+002; wspucz(n+1)=0.000000e+000; ng(n+1)=0.000000e+000;
%odnowa zmiennej metryki
n=1; farx(n+1)=1.669366e+002; foe(n+1)=1.728853e+002; krok(n+1)=4.912716e-004; ng(n+1)=1.042088e+003;
n=2; farx(n+1)=4.866413e+001; foe(n+1)=6.002040e+001; krok(n+1)=9.659422e-003; ng(n+1)=4.431886e+002;
n=3; farx(n+1)=4.758982e+001; foe(n+1)=5.891323e+001; krok(n+1)=1.078226e-003; ng(n+1)=1.809874e+002;
n=4; farx(n+1)=4.140278e+001; foe(n+1)=5.782489e+001; krok(n+1)=5.915258e-003; ng(n+1)=7.837349e+001;
n=5; farx(n+1)=2.852112e+000; foe(n+1)=2.120384e+001; krok(n+1)=1.153579e-002; ng(n+1)=2.057595e+002;
n=6; farx(n+1)=2.535620e+000; foe(n+1)=2.087995e+001; krok(n+1)=4.412088e-005; ng(n+1)=1.726612e+003;
n=7; farx(n+1)=2.535090e+000; foe(n+1)=2.061429e+001; krok(n+1)=1.350546e-003; ng(n+1)=2.159816e+003;
n=8; farx(n+1)=2.695576e+000; foe(n+1)=1.397824e+001; krok(n+1)=1.163808e-002; ng(n+1)=1.945778e+003;
n=9; farx(n+1)=2.781005e+000; foe(n+1)=1.342796e+001; krok(n+1)=1.093065e-003; ng(n+1)=9.583864e+002;
n=10; farx(n+1)=2.718069e+000; foe(n+1)=1.131759e+001; krok(n+1)=4.213325e-003; ng(n+1)=1.024932e+003;
n=11; farx(n+1)=2.415693e+000; foe(n+1)=1.033978e+001; krok(n+1)=7.599313e-003; ng(n+1)=2.709966e+002;
n=12; farx(n+1)=2.284412e+000; foe(n+1)=9.306114e+000; krok(n+1)=1.041486e-002; ng(n+1)=4.968518e+002;
n=13; farx(n+1)=2.291788e+000; foe(n+1)=8.406378e+000; krok(n+1)=6.669234e-003; ng(n+1)=4.191069e+002;
n=14; farx(n+1)=2.264415e+000; foe(n+1)=7.784707e+000; krok(n+1)=8.174750e-003; ng(n+1)=1.985149e+002;
n=15; farx(n+1)=2.235910e+000; foe(n+1)=7.168735e+000; krok(n+1)=4.478448e-003; ng(n+1)=5.219690e+002;
n=16; farx(n+1)=2.123546e+000; foe(n+1)=6.719612e+000; krok(n+1)=1.063635e-002; ng(n+1)=1.737974e+002;
n=17; farx(n+1)=1.990178e+000; foe(n+1)=6.435326e+000; krok(n+1)=5.269205e-003; ng(n+1)=6.461095e+002;
n=18; farx(n+1)=1.596477e+000; foe(n+1)=5.301643e+000; krok(n+1)=2.015462e-002; ng(n+1)=8.490389e+002;
n=19; farx(n+1)=1.418539e+000; foe(n+1)=3.901948e+000; krok(n+1)=7.887845e-003; ng(n+1)=9.384449e+002;
n=20; farx(n+1)=1.403977e+000; foe(n+1)=3.672056e+000; krok(n+1)=4.815209e-004; ng(n+1)=4.335307e+002;
n=21; farx(n+1)=1.360967e+000; foe(n+1)=3.481940e+000; krok(n+1)=1.071441e-002; ng(n+1)=4.356323e+002;
n=22; farx(n+1)=1.348712e+000; foe(n+1)=3.359171e+000; krok(n+1)=7.242402e-003; ng(n+1)=2.630568e+002;
n=23; farx(n+1)=1.209376e+000; foe(n+1)=3.084084e+000; krok(n+1)=4.486659e-002; ng(n+1)=9.631466e+001;
n=24; farx(n+1)=1.173568e+000; foe(n+1)=3.004770e+000; krok(n+1)=3.416397e-003; ng(n+1)=2.441660e+002;
n=25; farx(n+1)=1.129401e+000; foe(n+1)=2.886468e+000; krok(n+1)=7.739087e-003; ng(n+1)=2.325243e+002;
%odnowa zmiennej metryki
n=26; farx(n+1)=1.122323e+000; foe(n+1)=2.822174e+000; krok(n+1)=1.806577e-005; ng(n+1)=2.208921e+002;
n=27; farx(n+1)=1.066936e+000; foe(n+1)=2.542994e+000; krok(n+1)=1.550175e-004; ng(n+1)=1.756006e+002;
n=28; farx(n+1)=1.070607e+000; foe(n+1)=2.538037e+000; krok(n+1)=4.204485e-005; ng(n+1)=4.995091e+001;
n=29; farx(n+1)=8.987126e-001; foe(n+1)=2.032051e+000; krok(n+1)=1.734425e-002; ng(n+1)=2.413537e+001;
n=30; farx(n+1)=8.814222e-001; foe(n+1)=1.801669e+000; krok(n+1)=1.290032e-002; ng(n+1)=1.692812e+002;
n=31; farx(n+1)=8.760234e-001; foe(n+1)=1.653448e+000; krok(n+1)=4.907442e-003; ng(n+1)=2.883047e+002;
n=32; farx(n+1)=8.715163e-001; foe(n+1)=1.607370e+000; krok(n+1)=6.130766e-003; ng(n+1)=9.165903e+001;
n=33; farx(n+1)=8.110033e-001; foe(n+1)=1.495716e+000; krok(n+1)=4.446078e-002; ng(n+1)=8.246407e+001;
n=34; farx(n+1)=7.902437e-001; foe(n+1)=1.469662e+000; krok(n+1)=1.213054e-002; ng(n+1)=9.800124e+001;
n=35; farx(n+1)=7.648699e-001; foe(n+1)=1.439102e+000; krok(n+1)=1.547817e-002; ng(n+1)=1.202915e+002;
n=36; farx(n+1)=7.167036e-001; foe(n+1)=1.396849e+000; krok(n+1)=3.638604e-002; ng(n+1)=5.603155e+001;
n=37; farx(n+1)=6.454267e-001; foe(n+1)=1.313158e+000; krok(n+1)=5.016986e-002; ng(n+1)=1.633048e+002;
n=38; farx(n+1)=6.150497e-001; foe(n+1)=1.290356e+000; krok(n+1)=2.426108e-002; ng(n+1)=1.074529e+002;
n=39; farx(n+1)=5.651027e-001; foe(n+1)=1.239144e+000; krok(n+1)=1.345452e-002; ng(n+1)=1.144812e+002;
n=40; farx(n+1)=5.529683e-001; foe(n+1)=1.213083e+000; krok(n+1)=1.111520e-002; ng(n+1)=1.020167e+002;
n=41; farx(n+1)=5.317797e-001; foe(n+1)=1.173509e+000; krok(n+1)=4.756755e-002; ng(n+1)=4.529210e+001;
n=42; farx(n+1)=5.191172e-001; foe(n+1)=1.129383e+000; krok(n+1)=2.219272e-002; ng(n+1)=1.277637e+002;
n=43; farx(n+1)=5.022762e-001; foe(n+1)=1.059657e+000; krok(n+1)=3.022871e-002; ng(n+1)=1.432956e+002;
n=44; farx(n+1)=4.838099e-001; foe(n+1)=1.003459e+000; krok(n+1)=2.366103e-002; ng(n+1)=7.632101e+001;
n=45; farx(n+1)=4.903633e-001; foe(n+1)=9.743268e-001; krok(n+1)=2.015462e-002; ng(n+1)=5.903251e+001;
n=46; farx(n+1)=4.988969e-001; foe(n+1)=9.519070e-001; krok(n+1)=2.429096e-002; ng(n+1)=1.559142e+002;
n=47; farx(n+1)=4.952987e-001; foe(n+1)=9.354383e-001; krok(n+1)=4.864952e-002; ng(n+1)=4.983308e+001;
n=48; farx(n+1)=4.935838e-001; foe(n+1)=9.200594e-001; krok(n+1)=2.741202e-002; ng(n+1)=8.068550e+001;
n=49; farx(n+1)=4.945831e-001; foe(n+1)=8.909754e-001; krok(n+1)=5.507794e-002; ng(n+1)=7.570324e+001;
n=50; farx(n+1)=4.975113e-001; foe(n+1)=8.676943e-001; krok(n+1)=4.852216e-002; ng(n+1)=1.320899e+002;

figure; semilogy(farx,'b'); hold on; semilogy(foe,'r'); xlabel('Iteracje'); ylabel('Earx, Eoe'); legend('Earx','Eoe'); title('Uczenie predyktora OE');
figure; subplot(2,1,1); semilogy(krok); xlabel('Iteracje'); ylabel('Krok');
subplot(2,1,2); semilogy(ng); xlabel('Iteracje'); ylabel('Norma gradientu');
Earx=farx(n+1)
Eoe=foe(n+1)
