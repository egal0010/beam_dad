L'idée ici est de voir si E[g_L]-->I_T (I_T=espérance eig totale) quand L --> \inf. Le calcul de I se fait par double nested monte-carlo, avec n_inner et n_outer très grand. Ce test est fait afin de voir si l'implémentation de g_L est bonne ou non. 

On a les paramètres suivants : T = 1, \rho et \sigma connus, et le beam fixé.

(beam_dad) egalarraga@beamsoperaserver1:~/beam_dad$ /home/egalarraga/.conda/envs/beam_dad/bin/python /home/egalarraga/beam_dad/I_nm_vs_g_L.py
Device: cuda
rho known : 0.5
sigma     : 0.5

===========================================================================
T=1 | KNOWN rho | FIXED sigma | FIXED eta
===========================================================================
batch   1/20
batch   2/20
batch   3/20
batch   4/20
batch   5/20
batch   6/20
batch   7/20
batch   8/20
batch   9/20
batch  10/20
batch  11/20
batch  12/20
batch  13/20
batch  14/20
batch  15/20
batch  16/20
batch  17/20
batch  18/20
batch  19/20
batch  20/20

===========================================================================
RESULTS
===========================================================================
Reference EIG : 0.777506 +/- 0.031856 nats (approx. 95% MC interval)

       L |       E[g_L] |       2 SE |   gap to EIG |   log(L+1)
-----------------------------------------------------------------
       1 |     0.318447 |   0.016201 |     0.459059 |   0.693147
       5 |     0.595870 |   0.025249 |     0.181636 |   1.791759
      10 |     0.677749 |   0.028499 |     0.099757 |   2.397895
      30 |     0.748173 |   0.031466 |     0.029333 |   3.433987
      70 |     0.766167 |   0.031709 |     0.011339 |   4.262680
     128 |     0.771079 |   0.031764 |     0.006427 |   4.859812
     300 |     0.775566 |   0.031760 |     0.001940 |   5.707110
    1000 |     0.777244 |   0.031783 |     0.000262 |   6.908755
    3000 |     0.777861 |   0.031866 |    -0.000355 |   8.006701
    5000 |     0.778067 |   0.031884 |    -0.000561 |   8.517393
   10000 |     0.777858 |   0.031867 |    -0.000352 |   9.210440

===========================================================================
CHECK EXISTING estimate_eig_for_eta()
===========================================================================
Existing estimator mean : 0.786983 nats
Reference EIG           : 0.777506 nats
Difference              : 0.009477 nats
(beam_dad) egalarraga@beamsoperaserver1:~/beam_dad$ 

on voit que c'est bon....