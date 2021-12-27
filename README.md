# lammps_param_sweep
 Run parameter sweeps for varying geometries in Lammps


EXAMPLE

def get_aa_geom(param):
    a=2.529
    sep=float(param["layer_sep"])
    atoms_obj=fg.shift.make_graphene(stacking=['A','A'],cell_type='rect',
                        n_layer=2,n_1=5,n_2=5,lat_con=0.0,a_nn=a/np.sqrt(3),
                        sep=sep,sym=["B",'Ti'],mass=[12.01,12.02],h_vac=5)
    return atoms_obj

template="static_calc.in" 
#AA
geom_gen=get_aa_geom
proj_name="AA_calcs"

l_=np.array([3.0,3.2,3.35,3.5,3.65,3.8,4.0,4.5,5.0,10])
params={"layer_sep":l_}

proj=lammps_project(template,geom_gen,params,\
               project_name=proj_name)
            
proj.run_lammps_all()