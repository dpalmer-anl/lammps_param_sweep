# To be used with the latte-lib input file.  

units		metal
atom_style	full
atom_modify    sort 0 0.0  # This is to avoid sorting the coordinates

#read_data lammpsdataAB.data
read_data ab.data

group top type 1
group bottom type 2

set group top mol 1
set group bottom mol 2

mass 1 12.0100
mass 2 12.0200

velocity	all create 0.0 87287 loop geom
# Interaction potential for carbon atoms
######################## Potential defition ########################
#pair_style       hybrid/overlay kolmogorov/crespi/full 16.0 0 table linear 500
pair_style       hybrid/overlay reg/dep/poly 16.0 1 table linear 500
#pair_coeff       * *   rebo	CH.rebo        C C # chemical
pair_coeff	1 1 table parameters_potentials/lammps/porezag_correction/porezag_c-c.table POREZAG_C
pair_coeff       * *   reg/dep/poly  KC_insp.txt   C C # long-range #need to add in KC correction here
#pair_coeff       * *   kolmogorov/crespi/full  CH_taper.txt   C C # long-range #need to add in KC correction here
pair_coeff      2 2 table parameters_potentials/lammps/porezag_correction/porezag_c-c.table POREZAG_C
####################################################################

neighbor	2.0 bin
neigh_modify	delay 0 one 10000
#delete_atoms overlap 0.4 all all

compute interlayer top pe/tally bottom
compute 0 all pair reg/dep/poly
#compute 0 all pair kolmogorov/crespi/full
variable Evdw  equal c_0[1]
variable Erep  equal c_0[2]

variable latteE equal "(ke + f_2)"
variable kinE equal "ke"
variable potE equal "f_2"

timestep 0.00025
thermo 1
thermo_style   custom step pe ke etotal temp epair v_Erep v_Evdw c_interlayer v_latteE

fix		1 all nve

fix   2 all latte NULL
log log.test
variable latteE equal "(ke + f_2)"
variable kinE equal "ke"
variable potE equal "f_2"
#variable myT equal "temp"


run 0
#minimize       1e-11 1e-12 30 10000000
