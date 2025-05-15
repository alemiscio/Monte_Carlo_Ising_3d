import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

plt.ion()

from TD_Ising_Monte_Carlo import *


field1 = load_field_npy("Results/3d_40x500x500_field_configurations/field_config_0_1000.npy")
#display_spin_field(field1,2).show()
#plot_two_point_epsilon(field1, 39, 100000, 0)
#plot_two_point_sigma(field1, 39, 10000, 1)
#plot_two_point_sigma(field1, 39, 100, 2)
#export_two_point_data_txt(compute_two_point_r_profile_3d(field1,39,0),"two_point_profile.txt")




j= 0

for i in range(2000,4000,20):
    field1 =  load_field_npy("Results/3d_40x500x500_field_configurations/field_config_{jj}_{ii}.npy".format(ii = i,jj = j))
    print(i,sep=" ", end=" ", flush=True)
    #display_spin_field(field1).show()
    #plot_Thermal_two_point_r_sigma(field1, 60, 40, f"test.png")
    #plot_Thermal_two_point_t_sigma(field1, 39, 40, f"testt.png")
    export_two_point_data_txt( compute_two_point_epsilon_profile(field1, 60,1),"Results/3d_40x500x500_field_configurations/tp/tp_e_x_{jj}_{ii}.txt".format(jj = j,ii = i))
