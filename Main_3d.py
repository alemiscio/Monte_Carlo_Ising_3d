import numpy as np

import time

from TD_Ising_Monte_Carlo import  *

# --- Main simulation ---
start = time.time()

beta_c = 0.221654  # 1004.4486

beta = beta_c

step_max = 4000+1;

cutoff = 1000-1;

#field = random_spin_field(30, 500, 500)
field = random_spin_field(40, 500, 500)
#display_spin_field(field,1).show()

end = time.time()
elapsed = end - start


print(f"Random field in  {elapsed:.3f} seconds.")
start = time.time()

field = ising_step_3d(field, beta)
end = time.time()
elapsed = end - start
print(f"One step takes  {elapsed:.3f} seconds.")

for j in range(100):
    print("Random with j = {jj} starting...".format(jj=j))
    for i in range(step_max):
        field = ising_step_3d(field, beta)
        if i % 5 == 0:
            print(i, " magnetization : ", magnetization(field))

        if i % 20 == 0 and i>cutoff :
            #print(i, " magnetization : ", magnetization(field))
            plot_two_point_sigma(field, 20, 1000000, 0)
            plot_two_point_epsilon(field, 20, 100000, 0)
            save_field_npy(field, filename="Results/3d_40x500x500_field_configurations/field_config_{jj}_{ii}.npy".format(ii = i,jj = j))

    end = time.time()
    elapsed = end - start
    print(f"Full process until now in  {elapsed:.3f} seconds.")

