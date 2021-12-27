# lammps_param_sweep
 Run parameter sweeps for varying geometries in Lammps


EXAMPLE

def get_aa_geom(param): \n
    a=2.529 \n
    sep=float(param["layer_sep"]) \n
    atoms_obj=fg.shift.make_graphene(stacking=['A','A'],cell_type='rect', \n
                        n_layer=2,n_1=5,n_2=5,lat_con=0.0,a_nn=a/np.sqrt(3), \n
                        sep=sep,sym=["B",'Ti'],mass=[12.01,12.02],h_vac=5) \n
    return atoms_obj \n

template="static_calc.in" \n
#AA \n
geom_gen=get_aa_geom \n
proj_name="AA_calcs" \n
\n
l_=np.array([3.0,3.2,3.35,3.5,3.65,3.8,4.0,4.5,5.0,10]) \n
params={"layer_sep":l_} \n
\n
proj=lammps_project(template,geom_gen,params,\ \n
               project_name=proj_name) \n
            
proj.run_lammps_all() \n