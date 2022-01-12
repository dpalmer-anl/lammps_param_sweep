import ase

import scipy.optimize
import numpy as np
import flatgraphene as fg
import subprocess 
import lammps_logfile
import os

def eval_energy(df, delta,C,C0,C2,C4,z0,A6,A8,A10):
    """ 
    """
    energy = []
    #for it, row in df.iterrows():
    for it in range(np.shape(df)[0]):
        #atoms = generate_geometry.create_graphene_geom(row['d'], row['disregistry'])
        #calc = ase.calculators.lj.LennardJones(sigma=sigma, epsilon=epsilon)
        #atoms.calc=calc
        #x=row['disregistry']
        #sep=row['d']
        x=df[it,0]
        sep=df[it,1]
        a=2.529
        stacking=np.array([[0,0],[x,0]])
        atoms=fg.shift.make_graphene(stacking=stacking,cell_type='rect',
                        n_layer=2,n_1=5,n_2=5,lat_con=0.0,a_nn=a/np.sqrt(3),
                        sep=sep,sym=["B",'Ti'],mass=[12.01,12.02],h_vac=5)
        # num_atoms=atoms.get_global_number_of_atoms()
        temp_energy_atom=get_energy_interlayer(delta,C,C0,C2,C4,z0,A6,A8,A10,atoms)
        energy.append(temp_energy_atom)

    #lj_en =  np.asarray(energy)- np.min(energy) + np.min(df['energy'])
    lj_en= np.asarray(energy,dtype=np.float64)
    e_inf=np.min(lj_en)
    lj_en-=e_inf
    
    with open("constant_trace.txt","a+") as f:
        const=[str(delta),str(C),str(C0),str(C2),str(C4),str(z0),str(A6),str(A8),str(A10)]
        new_line=" ".join(const)+" \n"
        f.write(new_line)
    return lj_en
        


def fit(df, delta=0.578,C=73.28797,C0=15.71,C2=12.29,C4=4.933,z0=3.34,A6=-0.257119,A8=0.397049,A10=0.6390464):
    e_inf=np.min(df['energy'].to_numpy())
    ydata = df['energy'].to_numpy()-e_inf
    
    xdata=np.stack((df["disregistry"].to_numpy(),df["d"].to_numpy()),axis=1)
    popt, pcov = scipy.optimize.curve_fit(eval_energy, xdata , ydata, p0=(delta,C,C0,C2,C4,z0,A6,A8,A10))
    return popt
    
    
def write_lammps(fname,ase_obj):
    cell=np.array(ase_obj.get_cell())
    rx_=" ".join(map(str,cell[0,:]))
    ry_=" ".join(map(str,cell[1,:]))
    rz_=" ".join(map(str,cell[2,:]))
    
    xyz=ase_obj.get_positions()
    natom=np.shape(xyz)[0]
    
    with open(fname,'w+') as f:
        skew=cell[1,0]
        if np.isclose(cell[0,0]/2,skew,atol=1e-5):
            skew-=1e-5
        f.write(fname+ " (written by ASE)      \n\n")
        f.write(str(natom)+" 	 atoms \n")
        f.write("2 atom types \n")
        f.write("0.0      "+str(cell[0,0])+"  xlo xhi \n")
        f.write("0.0      "+str(cell[1,1])+ " ylo yhi \n")
        f.write("0.0      "+str(cell[2,2])+" zlo zhi \n")
        f.write("    "+str(skew)+"                       0                       0  xy xz yz \n\n\n")
        f.write("Atoms \n\n")
        
        m1=ase_obj.get_masses()[0]
        for i,a in enumerate(ase_obj):
            if a.mass==m1:
                atom_type="1"
            else:
                atom_type="2"
                
            f.write(str(i+1)+" "+atom_type+" "+atom_type+" 0 ")
            pos=np.array(a.position)
            str_pos=" ".join(map(str,pos))
            f.write(str_pos+" \n")
            
def get_energy_interlayer(delta,C,C0,C2,C4,z0,A6,A8,A10,atoms_obj):
    const=[str(delta),str(C),str(C0),str(C2),str(C4),str(z0),str(A6),str(A8),str(A10)]
    with open("KC_insp.txt","r+") as f:
        with open("temp.txt","w+") as g:
            
            lines=f.readlines()
            new_line="C C "+" ".join(const)+" 1.0 2.0 \n"
            for l in lines:
                if "C C" in l:
                    g.write(new_line)
                else:
                    g.write(l)

    subprocess.call("rm -f KC_insp.txt",shell=True)
    subprocess.call("mv temp.txt KC_insp.txt",shell=True)        
    write_lammps("ab.data", atoms_obj)
    subprocess.call("~/lammps/src/lmp_serial<test_input.md",shell=True)

    log_file="log.test"
    log = lammps_logfile.File(log_file)
    Evdw=float(log.get("v_Evdw"))
    e_tb=float(log.get("v_latteE"))
    energy=e_tb+Evdw
    num_atoms=atoms_obj.get_global_number_of_atoms()
    energy*=1/num_atoms
    # e_inf=-1838.865/200
    # energy-=e_inf
        
        
    return energy


if __name__=="__main__":
    import load
    df = load.load_data()
    import matplotlib.pyplot as plt
    import seaborn as sns
    if os.path.exists("constant_trace.txt"):
        subprocess.call("rm -f constant_trace.txt",shell=True)
    g = sns.FacetGrid(hue='disregistry', data =df, height=3)
    g.map(plt.errorbar,'d', 'energy', 'energy_err', marker='o', mew=1, mec='k')
    g.add_legend()
    plt.xlabel("Interlayer distance (Angstroms)")
    plt.ylabel("Energy (eV/atom)")
    plt.savefig("qmc_data.pdf", bbox_inches='tight')

    const = fit(df)
    print(const)
    df['lj_en'] = eval_energy(df,const)
    g = sns.FacetGrid(hue='disregistry', col='disregistry',data =df)
    g.map( plt.plot,'d','lj_en')
    g.map( plt.errorbar,'d','energy', 'energy_err', marker='o', mew=1, mec='k', linestyle="")
    print(df)
    g.add_legend()
    plt.show()