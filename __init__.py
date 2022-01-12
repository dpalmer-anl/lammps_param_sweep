# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 11:33:18 2021

@author: danpa
"""

import numpy as np
import shutil
import os
import subprocess
import ase.io
import lammps_logfile
from glob import glob
import pandas as pd

import lammps_param_sweep

class lammps_project:
    
    def __init__(self,template,geom_gen,params,project_name="lammps_calcs",metadata_file="metadata.txt",global_md_file="global_metadata.txt"):
        
        #T,A,Sep=np.meshgrid(theta_,a_,sep_)
        self.params=params
        self.template=template
        self.project_loc="/".join(self.template.split("/")[:-1])
        self.ori_cwd=os.getcwd()
        os.chdir(self.project_loc)
        self.template=self.template.split("/")[-1]
        self.project_name=project_name
        self.global_md_file=global_md_file
        self.param_keys=list(params.keys())
        self.num_params=len(self.param_keys)
        self.param_vals=np.stack(tuple(params.values()),axis=1)
        #self.param_mesh=np.meshgrid(*self.param_vals)
        #self.flat_param_mesh=self.flatten_mesh()
        self.md_keys=self.get_md_keys()
        self.geom_gen=geom_gen
        self.metadata_file=os.path.join(self.project_name,metadata_file)
        
        if not os.path.exists(self.project_name):
            os.mkdir(self.project_name)
        
        #write metadata keys
        
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file,"w+") as f:
                keys=["directory","uuid","num_atoms"]+self.param_keys+self.md_keys+["\n"]
                f.write(" ".join(keys))
    
    def set_global_metadata(self,params):
        keys=params.keys()
        values=params.values()
        
        with open(self.global_md_file,"w+") as f:
            k_str=" ".join(keys)
            f.write(k_str)
            md=[str(d) for d in values]
            f.write(" ".join(md))
            
    def flatten_mesh(self):
        flat_param=[]
        for i in range(self.num_params):
            flat_arr=self.param_mesh[i].reshape(np.prod(np.shape(self.param_mesh[i])))
            flat_param.append(flat_arr)
        return flat_param
        
    def remove_all_instances(self,l,val):
        try:
            while True:
                l.remove(val)
        except ValueError:
            pass  
        return l

    def get_md_keys(self):
    
         with open(self.template,"r") as f:
            lines=f.readlines()
            #add translation to map key words in input file to key words in log file
            translation={"step":"Step","pe":"PotEng","ke":"KinEng","etotal":"TotEng",\
                         "temp":"Temp","epair":"E_pair"}
            for l in lines:
                if "thermo_style" in l:
                    l=l.replace("\n","")
                    keys=l.split(" ")
                    keys=self.remove_all_instances(keys,"")
                    keys.remove("thermo_style")
                    keys.remove("custom")
            for i,k in enumerate(keys):
                if k in translation.keys():
                    new_key=translation[k]
                    keys[i]=new_key
            return keys
        
    def run_lammps_all(self,**kwargs):
        """run lammps simulations for given parameters with given template file """
        
        for i in range(np.shape(self.param_vals)[0]):
            param=dict(zip(self.param_keys,self.param_vals[i,:]))
            uuid,folder=self.generate_data_file(param)
            self.run_lammps(uuid,folder)
            self.generate_metadata(uuid,folder,**kwargs)
        os.chdir(self.ori_cwd)
        
    def run_lammps_cluster(self,batch_template):
        
        for i in range(np.shape(self.param_vals)[0]):
            param=dict(zip(self.param_keys,self.param_vals[i,:]))
            uuid,folder=self.generate_data_file(param)
            des_dir=self.generate_input_file(uuid,folder)
            self.submit_batch_file(batch_template,des_dir)
            
            
    def submit_batch_file(self,batch_template,des_dir):
        path=des_dir.split("/")
        folder=os.getcwd()
        input_file=des_dir
        subprocess.call("sed -i s+DIR+"+folder+"+g "+batch_template,shell=True)
        subprocess.call("sed -i s+INPUTFILE+"+input_file+"+g "+batch_template,shell=True)
        subprocess.call("sbatch "+batch_template,shell=True)
        subprocess.call("sed -i s+"+folder+"+DIR+g "+batch_template,shell=True)
        subprocess.call("sed -i s+"+input_file+"+INPUTFILE+g "+batch_template,shell=True)
        
    def generate_input_file(self,uuid,folder):
        in_file="input_"+str(uuid)+".in"
        des_dir=os.path.join(folder,in_file)
        shutil.copy(self.template,des_dir)
        
        log_file= os.path.join(folder,"log."+str(uuid))
        dump_file= os.path.join(folder,"dump."+str(uuid))
        data_file= os.path.join(folder,"coords"+str(uuid)+".data")
        
        subprocess.call("sed -i s+DATAFILE+"+data_file+"+g "+des_dir,shell=True)
        subprocess.call("sed -i s+LOGFILE+"+log_file+"+g "+des_dir,shell=True)
        subprocess.call("sed -i s+DUMPFILE+"+dump_file+"+g "+des_dir,shell=True)
        return des_dir
    
    def generate_data_file(self,param):
    
        #put files into common dir
        atoms_obj=self.geom_gen(param)
        
        
        num_atoms=atoms_obj.get_global_number_of_atoms()
        
        uuid=hash(tuple(param.values()))
        folder=os.path.join(self.project_name,"calc_"+str(uuid))
        
        if not os.path.exists(folder):
            os.mkdir(folder)
            
        fname=os.path.join(folder,"coords"+str(uuid)+".data")
        #NOTE temporary fix for triclinic cells, won't need this in future
        #ase.io.write("bad_cell.data",atoms_obj,format="lammps-data",atom_style="full")
        write_lammps(fname, atoms_obj)
        
        with open(self.metadata_file,"a") as f:
            param_str=[] #str(element) for element in param.values()
            for k in self.param_keys:
                param_str.append(str(param[k]))
            entries=[folder,str(uuid),str(num_atoms)]+param_str+[""]
            f.write(" ".join(entries))
        
        return uuid,folder
    
    def run_lammps(self,uuid,folder):
        """Generates lammps file based on template, places in same file as associated
        coordinate file, and runs lammps simulation. Dump and log file will also be
        in this directory. Edits log, and dump file locations"""
        
        des_dir=self.generate_input_file(uuid,folder)
        input_file=des_dir.split("/")[-1]
        subprocess.call("~/lammps/src/lmp_serial<"+des_dir,shell=True)
        
    def get_thermo_md(self,folder,uuid,md_type="final"):
    
        log_file=folder+"/log."+str(uuid)
        log = lammps_logfile.File(log_file)
        md=[]
    
        for i,k in enumerate(self.md_keys):
            if md_type=="final":
                md.append( log.get(k)[-1])
            elif md_type=="mean":
                md.append( np.mean(log.get(k)))
            else:
                print("md_type must be final or mean")
        
        return md
    
    def generate_metadata(self,uuid,folder):
        
        thermo_data=self.get_thermo_md(folder,uuid)
        
        with open(self.metadata_file,"a") as f:
            md=thermo_data+["\n"]
            md=[str(k) for k in md]
            f.write(" ".join(md))

    def scrape_metadata(self,**kwargs):
        md_files=glob.glob("*/metadata*",recursive=True)
        for f in md_files:
            loc=f.split("/")[:-1]
            loc="/".join(loc)
            db=analyze_db(loc)
            log_files=glob.glob("*/log*",recursive=True)
            
            for log in log_files:
                folder=log.split("/")
                uuid=folder[-1].split("_")[-1]
                folder="/".join(folder[:-1])
                md=self.get_thermo_md(folder,uuid)
                
                row=np.array(db["uuid"]==int(uuid))
                row=np.concatenate((row,md))
                db.loc[db["uuid"]==int(uuid)] = row
                
            db.to_csv(f)
                
                
                
########################## Analysis Class ###################################

class analyze_db:
    
    def __init__(self,project_loc,metadata_file="metadata.txt",global_md_file="global_metadata.txt"):
        self.project_loc=project_loc
        self.metadata_file=metadata_file
        self.global_md_file=global_md_file
        os.chdir(self.project_loc)
        
        self.read_metadata()
        #self.read_global_metadata()
        
        
    def read_metadata(self):
        data = pd.read_csv(self.metadata_file, sep=" ",header=0)
        self.md_df=data
        return data
    
    def read_global_metadata(self):
        data = pd.read_csv(self.global_md_file, sep=" ",header=0)
        self.global_md_df=data
        return data
    
    def get_file(self,param,filetype,float_tol=1e-1):
        """filetype= log,dump,coords,input if filetype is folder, just return folder location """
        try:
            df=self.md_df
        except:
            self.read_metadata()
            df=self.md_df

        for key in param:
            if isinstance(param[key],float):
                indices=df.index[(df[key] > param[key]-float_tol) & \
                                 (df[key] < param[key]+float_tol)]
            else:
                indices=df.index[df[key] == param[key]]
                
            df=df.loc[indices]
        
        dirs=df["directory"].tolist()
        uuids=df["uuid"].tolist()
        
        if filetype=="folder":
            return dirs
        
        files=[]
        for i,d in enumerate(dirs):
            path=os.path.join(dirs,"*"+filetype+"*")
            f=glob.glob(path)
            files.append(f)
        
        return files
    
    def load_logFile(self,filename):
        """load lammps logfile into pandas df """
        return lammps_logfile.File(filename)
    
    def load_coordFile(self,filename):
        atom_obj=ase.io.read(filename,format="lammps-data")
        return atom_obj
    
    def load_dumpFile(self,filename):
        atom_obj=ase.io.read(filename,format="lammps-dump-text")
        return atom_obj

###################### Convert to ASE DB #####################################
def to_AseDB(project):
    return None

def round_float_down(num):
    num_str=str(num)
    last_digit=int(num_str[-1])
    if last_digit%2 !=0:
        num_str[-1]=int(last_digit-1)
    return float(num_str)
   
def write_lammps(fname,ase_obj):
    cell=np.array(ase_obj.get_cell())
    rx_=" ".join(map(str,cell[0,:]))
    ry_=" ".join(map(str,cell[1,:]))
    rz_=" ".join(map(str,cell[2,:]))
    
    xyz=ase_obj.get_positions()
    natom=np.shape(xyz)[0]
    
    with open(fname,'w+') as f:
        skew=cell[1,0]
        if np.abs(cell[0,0]/2)<np.abs(skew) and np.isclose(cell[0,0]/2,skew,atol=1e-5):
            skew=round_float_down(cell[0,0]/2) #1e-5
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
            
            
if __name__=="__main__":
    import flatgraphene as fg
    atoms_obj=a=2.529
    sep=3.35
    t=27.8
    p_found, q_found, theta_comp = fg.twist.find_p_q(t)
    atoms_obj=fg.twist.make_graphene(cell_type="hex",n_layer=2,
                                        p=p_found,q=q_found,lat_con=0.0,a_nn=a/np.sqrt(3),sym=["B","Ti"],
                                        mass=[12.01,12.02],sep=sep,h_vac=5.5)
    
    write_lammps("C:/Users/danpa/Documents/research/latte_tools/Lammps_DB/projects/benchmarks/coords27_8_new.data",atoms_obj)
    