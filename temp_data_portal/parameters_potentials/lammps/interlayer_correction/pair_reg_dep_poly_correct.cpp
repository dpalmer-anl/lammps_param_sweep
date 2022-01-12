// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Wengen Ouyang (Tel Aviv University)
   e-mail: w.g.ouyang at gmail dot com
   based on previous versions by Jaap Kroes

   This is a complete version of the potential described in
   [Kolmogorov & Crespi, Phys. Rev. B 71, 235415 (2005)]
------------------------------------------------------------------------- */

#include "pair_reg_dep_poly.h"

#include "atom.h"
#include "citeme.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "interlayer_taper.h"
#include "memory.h"
#include "my_page.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "potential_file_reader.h"
#include <iostream>
#include <fstream>
#include <string.h>
#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace InterLayer;
#define MAXLINE 1024
#define DELTA 4
#define PGDELTA 1

static const char cite_kc[] =
  "@Article{Ouyang2018\n"
  " author = {W. Ouyang, D. Mandelli, M. Urbakh, and O. Hod},\n"
  " title = {Nanoserpents: Graphene Nanoribbon Motion on Two-Dimensional Hexagonal Materials},\n"
  " journal = {Nano Letters},\n"
  " volume =  18,\n"
  " pages =   {6009}\n"
  " year =    2018,\n"
  "}\n\n";

/* ---------------------------------------------------------------------- */

PairRegDepPoly::PairRegDepPoly(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  centroidstressflag = CENTROID_NOTAVAIL;
  unit_convert_flag = utils::get_supported_conversions(utils::ENERGY);
  if (lmp->citeme) lmp->citeme->add(cite_kc);

  nextra = 2;
  pvector = new double[nextra];

  // initialize element to parameter maps
  params = nullptr;
  cutKCsq = nullptr;

  nmax = 0;
  maxlocal = 0;
  KC_numneigh = nullptr;
  KC_firstneigh = nullptr;
  ipage = nullptr;
  pgsize = oneatom = 0;

  normal = nullptr;
  dnormal = nullptr;
  dnormdri = nullptr;

  // always compute energy offset
  offset_flag = 1;

  // turn off the taper function by default
  tap_flag = 0;
}

/* ---------------------------------------------------------------------- */

PairRegDepPoly::~PairRegDepPoly()
{
  memory->destroy(KC_numneigh);
  memory->sfree(KC_firstneigh);
  delete [] ipage;
  delete [] pvector;
  memory->destroy(normal);
  memory->destroy(dnormal);
  memory->destroy(dnormdri);

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    //memory->destroy(cut);
    memory->destroy(offset);
  }

  memory->destroy(params);
  memory->destroy(elem2param);
  memory->destroy(cutKCsq);
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairRegDepPoly::allocate()
{
  allocated = 1;
  int n = atom->ntypes+1;

  memory->create(setflag, n, n,"pair:setflag");
  for (int i = 1; i < n; i++)
    for (int j = i; j < n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq, n, n,"pair:cutsq");
  //memory->create(cut, n, n,"pair:cut");
  memory->create(offset, n, n,"pair:offset");
  map = new int[n];
  //map = new int[atom->ntypes+1];
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairRegDepPoly::settings(int narg, char **arg)
{
  if (narg < 1 || narg > 2) error->all(FLERR, "Illegal pair_style command");
  if (!utils::strmatch(force->pair_style, "^hybrid/overlay"))
    error->all(FLERR, "Pair style kolmogorov/crespi/full must be used as sub-style with hybrid/overlay");

  cut_global = utils::numeric(FLERR, arg[0], false, lmp);
  if (narg == 2) tap_flag = utils::numeric(FLERR, arg[1], false, lmp);
  
  // reset cutoffs that have been explicitly set

  //if (allocated) {
  //  int i,j;
  //  for (i = 1; i <= atom->ntypes; i++)
  //    for (j = i; j <= atom->ntypes; j++)
  //      if (setflag[i][j]) cut[i][j] = cut_global;
  //      cutsq[i][j]=cut_global; 
	// This line was added because cutsq[i][j] gets set to zero somewhere
	// not needed in pair_kolmogorov_crespi_full.cpp
  //}
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairRegDepPoly::coeff(int narg, char **arg)
{
  if (!allocated) allocate();
  map_element2type(narg - 3, arg + 3);
  read_file(arg[2]);
}


/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairRegDepPoly::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");
  if (!offset_flag) error->all(FLERR, "Must use 'pair_modify shift yes' with this pair style");
  if (offset_flag && (cut_global > 0.0)) {
  //if (offset_flag && (cut[i][j] > 0.0)) {
    int iparam_ij = elem2param[map[i]][map[j]];
    Param &p = params[iparam_ij];
    //offset[i][j] = -(p.A6*pow(p.z0/cut[i][j],6) + p.A8*pow(p.z0/cut[i][j],8)
    //		     + p.A10*pow(p.z0/cut[i][j],10));
    offset[i][j] = -(p.A6*pow(p.z0/cut_global,6) + p.A8*pow(p.z0/cut_global,8)
	     + p.A10*pow(p.z0/cut_global,10));
  } else offset[i][j] = 0.0;
  offset[j][i] = offset[i][j];

  return cut_global;
}
/* ----------------------------------------------------------------------
   read Kolmogorov-Crespi potential file new version
------------------------------------------------------------------------- */
void PairRegDepPoly::read_file(char *filename)
{
  memory->sfree(params);
  params = nullptr;
  nparams = maxparam = 0;
  int NPARAMS_PER_LINE = 13; // open file on proc 0

  if (comm->me == 0) {
    PotentialFileReader reader(lmp, filename, "pair/reg/dep/poly", unit_convert_flag);
    char *line;

    // transparently convert units for supported conversions

    int unit_convert = reader.get_unit_convert();
    double conversion_factor = utils::get_conversion_factor(utils::ENERGY, unit_convert);

    while ((line = reader.next_line(NPARAMS_PER_LINE))) {

      try {
        ValueTokenizer values(line);

        std::string iname = values.next_string();
        std::string jname = values.next_string();

        // ielement,jelement = 1st args
        // if both args are in element list, then parse this line
        // else skip to next entry in file
        int ielement, jelement;

        for (ielement = 0; ielement < nelements; ielement++)
          if (iname == elements[ielement]) break;
        if (ielement == nelements) continue;
        for (jelement = 0; jelement < nelements; jelement++)
          if (jname == elements[jelement]) break;
        if (jelement == nelements) continue;

        // expand storage, if needed

        if (nparams == maxparam) {
          maxparam += DELTA;
          params = (Param *) memory->srealloc(params, maxparam * sizeof(Param), "pair:params");

          // make certain all addional allocated storage is initialized
          // to avoid false positives when checking with valgrind

          memset(params + nparams, 0, DELTA * sizeof(Param));
        }

        // load up parameter settings and error check their values

	params[nparams].ielement = ielement;
        params[nparams].jelement = jelement;
        params[nparams].delta    = values.next_double();
        params[nparams].C        = values.next_double();
        params[nparams].C0       = values.next_double();
        params[nparams].C2       = values.next_double();
        params[nparams].C4       = values.next_double();
        params[nparams].z0       = values.next_double();
        params[nparams].A6       = values.next_double();
        params[nparams].A8       = values.next_double();
        params[nparams].A10      = values.next_double();
        // S provides a convenient scaling of all energies
        params[nparams].S = values.next_double();
        params[nparams].rcut = values.next_double();

      } catch (TokenizerException &e) {
        error->one(FLERR, e.what());
      }
      
      // energies in meV further scaled by S
      
      double meV = 1.0e-3*params[nparams].S;
      if (unit_convert) meV *= conversion_factor;

      params[nparams].C   *= meV;
      params[nparams].C0  *= meV;
      params[nparams].C2  *= meV;
      params[nparams].C4  *= meV;
      params[nparams].A6  *= meV;
      params[nparams].A8  *= meV;
      params[nparams].A10 *= meV;
      
      // precompute some quantities
      params[nparams].delta2inv = pow(params[nparams].delta,-2);
      params[nparams].z06 = pow(params[nparams].z0, 6);
      params[nparams].z08 = pow(params[nparams].z0,8);
      params[nparams].z010 = pow(params[nparams].z0,10);

      nparams++;
    }
  }

  MPI_Bcast(&nparams, 1, MPI_INT, 0, world);
  MPI_Bcast(&maxparam, 1, MPI_INT, 0, world);

  if (comm->me != 0) {
    params = (Param *) memory->srealloc(params, maxparam * sizeof(Param), "pair:params");
  }

  MPI_Bcast(params, maxparam * sizeof(Param), MPI_BYTE, 0, world);

  memory->destroy(elem2param);
  memory->destroy(cutKCsq);
  memory->create(elem2param, nelements, nelements, "pair:elem2param");
  memory->create(cutKCsq, nelements, nelements, "pair:cutKCsq");
  for (int i = 0; i < nelements; i++) {
    for (int j = 0; j < nelements; j++) {
      int n = -1;
      for (int m = 0; m < nparams; m++) {
        if (i == params[m].ielement && j == params[m].jelement) {
          if (n >= 0) error->all(FLERR, "KC potential file has duplicate entry");
          n = m;
        }
      }
      if (n < 0) error->all(FLERR, "Potential file is missing an entry");
      elem2param[i][j] = n;
     
      cutKCsq[i][j] = params[n].rcut * params[n].rcut;
      //std::cout << cutKCsq[i][j] << std::endl;
    }
  }
}

/* ----------------------------------------------------------------------
   read Kolmogorov-Crespi potential file old version
------------------------------------------------------------------------- */

//void PairRegDepPoly::read_file_old_version(char *filename)
//{
//  int params_per_line = 13;
//  char **words = new char*[params_per_line+1];
//  memory->sfree(params);
//  params = nullptr;
//  nparams = maxparam = 0;
//
//  // open file on proc 0
//
//if (comm->me == 0) {
//    PotentialFileReader reader(lmp, filename, "kolmogorov/crespi/full", unit_convert_flag);
//    char *line;
//
//    // transparently convert units for supported conversions
//
//    int unit_convert = reader.get_unit_convert();
//    double conversion_factor = utils::get_conversion_factor(utils::ENERGY, unit_convert);
//
//  // Old function	
//  FILE *fp;
//  if (comm->me == 0) {
//    fp = utils::open_potential(filename,lmp,nullptr);
//    if (fp == nullptr) {
//      char str[128];
//      snprintf(str,128,"Cannot open RDP potential file %s",filename);
//      error->one(FLERR,str);
//    }
//  }
//
//  // read each line out of file, skipping blank lines or leading '#'
//  // store line of params if all 3 element tags are in element list
//
//  int i,j,n,m,nwords,ielement,jelement;
//  char line[MAXLINE],*ptr;
//  int eof = 0;
//
//  while (1) {
//    if (comm->me == 0) {
//      ptr = fgets(line,MAXLINE,fp);
//      if (ptr == nullptr) {
//        eof = 1;
//        fclose(fp);
//      } else n = strlen(line) + 1;
//    }
//    MPI_Bcast(&eof,1,MPI_INT,0,world);
//    if (eof) break;
//    MPI_Bcast(&n,1,MPI_INT,0,world);
//    MPI_Bcast(line,n,MPI_CHAR,0,world);
//
//    // strip comment, skip line if blank
//
//    if ((ptr = strchr(line,'#'))) *ptr = '\0';
//    nwords = utils::count_words(line);
//    if (nwords == 0) continue;
//
//    // concatenate additional lines until have params_per_line words
//
//    while (nwords < params_per_line) {
//      n = strlen(line);
//      if (comm->me == 0) {
//        ptr = fgets(&line[n],MAXLINE-n,fp);
//        if (ptr == nullptr) {
//          eof = 1;
//          fclose(fp);
//        } else n = strlen(line) + 1;
//      }
//      MPI_Bcast(&eof,1,MPI_INT,0,world);
//      if (eof) break;
//      MPI_Bcast(&n,1,MPI_INT,0,world);
//      MPI_Bcast(line,n,MPI_CHAR,0,world);
//      if ((ptr = strchr(line,'#'))) *ptr = '\0';
//      nwords = utils::count_words(line);
//    }
//
//    if (nwords != params_per_line)
//      error->all(FLERR,"Insufficient format in KC potential file");
//
//    // words = ptrs to all words in line
//
//    nwords = 0;
//    words[nwords++] = strtok(line," \t\n\r\f");
//    while ((words[nwords++] = strtok(nullptr," \t\n\r\f"))) continue;
//
//    // ielement,jelement = 1st args
//    // if these 2 args are in element list, then parse this line
//    // else skip to next line (continue)
//
//    for (ielement = 0; ielement < nelements; ielement++)
//      if (strcmp(words[0],elements[ielement]) == 0) break;
//    if (ielement == nelements) continue;
//    for (jelement = 0; jelement < nelements; jelement++)
//      if (strcmp(words[1],elements[jelement]) == 0) break;
//    if (jelement == nelements) continue;
//
//    // load up parameter settings and error check their values
//
//    if (nparams == maxparam) {
//      maxparam += DELTA;
//      params = (Param *) memory->srealloc(params,maxparam*sizeof(Param),
//                                          "pair:params");
//
//      // make certain all addional allocated storage is initialized
//      // to avoid false positives when checking with valgrind
//
//      memset(params + nparams, 0, DELTA*sizeof(Param));
//    }
//
//    params[nparams].ielement = ielement;
//    params[nparams].jelement = jelement;
//    params[nparams].delta    = atof(words[2]);
//    params[nparams].C        = atof(words[3]);
//    params[nparams].C0       = atof(words[4]);
//    params[nparams].C2       = atof(words[5]);
//    params[nparams].C4       = atof(words[6]);
//    params[nparams].z0       = atof(words[7]);
//    params[nparams].A6       = atof(words[8]);
//    params[nparams].A8       = atof(words[9]);
//    params[nparams].A10      = atof(words[10]);
//    // S provides a convenient scaling of all energies
//    params[nparams].S        = atof(words[11]);
//    params[nparams].rcut     = atof(words[12]);
//
//    // energies in meV further scaled by S
//    double meV = 1.0e-3*params[nparams].S;
//    params[nparams].C   *= meV;
//    params[nparams].C0  *= meV;
//    params[nparams].C2  *= meV;
//    params[nparams].C4  *= meV;
//    params[nparams].A6  *= meV;
//    params[nparams].A8  *= meV;
//    params[nparams].A10 *= meV;
//
//    // precompute some quantities
//    params[nparams].delta2inv = pow(params[nparams].delta,-2);
//    params[nparams].z06 = pow(params[nparams].z0,6);
//    params[nparams].z08 = pow(params[nparams].z0,8);
//    params[nparams].z010 = pow(params[nparams].z0,10);
//
//    nparams++;
//    //if(nparams >= pow(atom->ntypes,3)) break;
//  }
//  memory->destroy(elem2param);
//  memory->destroy(cutKCsq);
//  memory->create(elem2param,nelements,nelements,"pair:elem2param");
//  memory->create(cutKCsq,nelements,nelements,"pair:cutKCsq");
//  for (int i = 0; i < nelements; i++) {
//    for (int j = 0; j < nelements; j++) {
//      int n = -1;
//      for (int m = 0; m < nparams; m++) {
//        if (i == params[m].ielement && j == params[m].jelement) {
//          if (n >= 0) error->all(FLERR,"Potential file has duplicate entry");
//          n = m;
//        }
//      }
//      if (n < 0) error->all(FLERR,"Potential file is missing an entry");
//      elem2param[i][j] = n; 
//      
//      cutKCsq[i][j] = params[n].rcut*params[n].rcut;
//      
//    }
//  }
//  delete [] words;
//}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairRegDepPoly::init_style()
{
  if (force->newton_pair == 0)
    error->all(FLERR,"Pair style reg/dep/pair requires newton pair on");
  if (!atom->molecule_flag)
    error->all(FLERR,"Pair style reg/dep/pair requires atom attribute molecule");

  // need a full neighbor list, including neighbors of ghosts

  int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->ghost = 1;

  // local KC neighbor list
  // create pages if first time or if neighbor pgsize/oneatom has changed

  int create = 0;
  if (ipage == nullptr) create = 1;
  if (pgsize != neighbor->pgsize) create = 1;
  if (oneatom != neighbor->oneatom) create = 1;

  if (create) {
    delete [] ipage;
    pgsize = neighbor->pgsize;
    oneatom = neighbor->oneatom;

    int nmypage= comm->nthreads;
    ipage = new MyPage<int>[nmypage];
    for (int i = 0; i < nmypage; i++)
      ipage[i].init(oneatom,pgsize,PGDELTA);
  }
}

/* ---------------------------------------------------------------------- */

void PairRegDepPoly::compute(int eflag, int vflag)
{
  ev_init(eflag,vflag);
  pvector[0] = pvector[1] = 0.0;

  // Build full neighbor list
  KC_neigh();
  // Calculate the normals and its derivatives
  calc_normal();
  // Calculate the van der Waals force and energy
  calc_FvdW(eflag,vflag);

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   van der Waals forces and energy
------------------------------------------------------------------------- */

void PairRegDepPoly::calc_FvdW(int eflag, int /* vflag */)
{
  int i,j,ii,jj,inum,jnum,itype,jtype,k,kk;
  tagint itag, jtag;
  double prodnorm1,fkcx,fkcy,fkcz;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair,fpair1;
  double rsq,r,rho_ijsq,rho_ij_by_delta_sq,expf,sumP,Tap,dTap,Vkc;
  double r2inv,r6inv,r8inv,r10inv,r12inv;
  double frho_ij,sumC1,sumC11,sumCff,fsum,rho_ij;
  int *ilist,*jlist,*numneigh,**firstneigh;
  int *KC_neighs_i;

  evdwl = 0.0;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;
  double dprodnorm1[3] = {0.0, 0.0, 0.0};
  double fp1[3] = {0.0, 0.0, 0.0};
  double fprod1[3] = {0.0, 0.0, 0.0};
  double delkj[3] = {0.0, 0.0, 0.0};
  double fk[3] = {0.0, 0.0, 0.0};

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  //calculate exp(-lambda*(r-z0))*[epsilon/2 + f(rho_ij)]
  // loop over neighbors of owned atoms
  // std::cout << atom->type << std::endl;
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    itag = tag[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = type[j];
      jtag = tag[j];

      // two-body interactions from full neighbor list, skip half of them
      if (itag > jtag) {
        if ((itag + jtag) % 2 == 0) continue;
      } else if (itag < jtag) {
        if ((itag + jtag) % 2 == 1) continue;
      } else {
        if (x[j][2] < ztmp) continue;
        if (x[j][2] == ztmp && x[j][1] < ytmp) continue;
        if (x[j][2] == ztmp && x[j][1] == ytmp && x[j][0] < xtmp) continue;
      }

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      
      //
      // only include the interaction between different layers
      //std::cout << cutsq[itype][jtype] << std::endl;
      //cutsq[itype][jtype]=10;
       
      if (rsq < cutsq[itype][jtype] && atom->molecule[i] != atom->molecule[j]) {
        
        int iparam_ij = elem2param[map[itype]][map[jtype]];
        Param& p = params[iparam_ij];

        r = sqrt(rsq);
	r2inv = 1/rsq; // r^(-2)
	r6inv = r2inv*r2inv*r2inv; // r^(-6)
	r8inv = r6inv*r2inv; // r^(-8)
	r10inv = r8inv*r2inv; // r^(-10)
	r12inv = r6inv*r6inv; // r^(-12) (only used for derivatives)

        // turn on/off taper function
        if (tap_flag) {
          Tap = calc_Tap(r,sqrt(cutsq[itype][jtype]));
          dTap = calc_dTap(r,sqrt(cutsq[itype][jtype]));
        } else {Tap = 1.0; dTap = 0.0;}

        // calculate the transverse distance
        prodnorm1 = normal[i][0]*delx + normal[i][1]*dely + normal[i][2]*delz; // r_ij \cdot n_i
        rho_ijsq = rsq - prodnorm1*prodnorm1;  // rho_ij^2
        rho_ij_by_delta_sq = rho_ijsq*p.delta2inv; // (rho_ij/delta)^2

        // store exponential part of f(rho)
        expf = exp(-rho_ij_by_delta_sq);

	//compute polynomial pairwise term, where z0n = z0^(n)
	sumP = p.A6*p.z06*r6inv + p.A8*p.z08*r8inv + p.A10*p.z010*r10inv; // C6*(r/z0)^-6 + C8*(r/z0)^-8 + C10*(r/z0)^-10

        sumC1 = p.C0 + p.C2*rho_ij_by_delta_sq + p.C4*rho_ij_by_delta_sq*rho_ij_by_delta_sq; //C0 + C2*(rho_ij/delta)^2 + C4*(rho_ij/delta)^4, polynomial part of registry dependence
        sumC11 = (p.C2 + 2.0*p.C4*rho_ij_by_delta_sq); //derivative of the term above with respect to (rho_ij/delta)^2 
        frho_ij = expf*sumC1;
        sumCff = 0.5*p.C + frho_ij; //"half" of the registry dependent term [C+f(rho_ij)+f(rho_ji)]
	Vkc = -sumCff*sumP; //potential energy
        

	/*---
	compute force via F=-d(Vkc)/dr
	-d(Vkc)/dr = sumCff*d(sumP)/dr + d(sumCff)/dr*sumP
 	           = sumCff*d(sumP)/dr + sumP * d((rho_ij/delta)^2)/dr * d(sumCff)/d((rho_ij/delta)^2)
 	           = sumCff*d(sumP)/dr + sumP * (1/delta)^2 * d(rho_ij^2)/dr * d(sumCff)/d((rho_ij/delta)^2)
 	           = sumCff*d(sumP)/dr + d(rho_ij^2)/dr *(sumP * (1/delta)^2 *  d(sumCff)/d((rho_ij/delta)^2)
		  *all of the above is before multiplication by 1/r for projections via delx, etc.*

	  fpair  = sumCff*d(sumP)/dr * (1/r)
	  fpair1 = sumP * (1/delta)^2 *  d(sumCff)/d((rho_ij/delta)^2 * [2.0]
	  *the [2.0] and lack of (1/r) term in fpair1 is anomalous (I don't know
	  precisely why they are (not) there, but they are necessary based on
	  understanding of equivalent terms in pair_kolmogorov_crespi_full.cpp*
	---*/

	// product rule derivative terms of d(Vkc)/dr (divided by r)
	fpair = -(6.0*p.A6*p.z06*r8inv + 8.0*p.A8*p.z08*r10inv + 10.0*p.A10*p.z010*r12inv)*sumCff;
        fpair1 = -2.0*sumP*p.delta2inv*expf*(sumC1 - sumC11);
        fsum = fpair + fpair1;

	// I WANT TO HAVE TO CHANGE NOTHING BEYOND THIS COMMENT
	//After everything appears in order, change some names to make more sense (Vkc->VvdW)
        // derivatives of the product of rij and ni, the result is a vector
        dprodnorm1[0] = dnormdri[0][0][i]*delx + dnormdri[1][0][i]*dely + dnormdri[2][0][i]*delz;
        dprodnorm1[1] = dnormdri[0][1][i]*delx + dnormdri[1][1][i]*dely + dnormdri[2][1][i]*delz;
        dprodnorm1[2] = dnormdri[0][2][i]*delx + dnormdri[1][2][i]*dely + dnormdri[2][2][i]*delz;
        fp1[0] = prodnorm1*normal[i][0]*fpair1;
        fp1[1] = prodnorm1*normal[i][1]*fpair1;
        fp1[2] = prodnorm1*normal[i][2]*fpair1;
        fprod1[0] = prodnorm1*dprodnorm1[0]*fpair1;
        fprod1[1] = prodnorm1*dprodnorm1[1]*fpair1;
        fprod1[2] = prodnorm1*dprodnorm1[2]*fpair1;
        fkcx = (delx*fsum - fp1[0])*Tap - Vkc*dTap*delx/r;
        fkcy = (dely*fsum - fp1[1])*Tap - Vkc*dTap*dely/r;
        fkcz = (delz*fsum - fp1[2])*Tap - Vkc*dTap*delz/r;

        f[i][0] += fkcx - fprod1[0]*Tap;
        f[i][1] += fkcy - fprod1[1]*Tap;
        f[i][2] += fkcz - fprod1[2]*Tap;
        f[j][0] -= fkcx;
        f[j][1] -= fkcy;
        f[j][2] -= fkcz;

        // calculate the forces acted on the neighbors of atom i from atom j
        KC_neighs_i = KC_firstneigh[i];
        for (kk = 0; kk < KC_numneigh[i]; kk++) {
          k = KC_neighs_i[kk];
          if (k == i) continue;
          // derivatives of the product of rij and ni respect to rk, k=0,1,2, where atom k is the neighbors of atom i
          dprodnorm1[0] = dnormal[0][0][kk][i]*delx + dnormal[1][0][kk][i]*dely + dnormal[2][0][kk][i]*delz;
          dprodnorm1[1] = dnormal[0][1][kk][i]*delx + dnormal[1][1][kk][i]*dely + dnormal[2][1][kk][i]*delz;
          dprodnorm1[2] = dnormal[0][2][kk][i]*delx + dnormal[1][2][kk][i]*dely + dnormal[2][2][kk][i]*delz;
          fk[0] = (-prodnorm1*dprodnorm1[0]*fpair1)*Tap;
          fk[1] = (-prodnorm1*dprodnorm1[1]*fpair1)*Tap;
          fk[2] = (-prodnorm1*dprodnorm1[2]*fpair1)*Tap;
          f[k][0] += fk[0];
          f[k][1] += fk[1];
          f[k][2] += fk[2];
          delkj[0] = x[k][0] - x[j][0];
          delkj[1] = x[k][1] - x[j][1];
          delkj[2] = x[k][2] - x[j][2];
          if (evflag) ev_tally_xyz(k,j,nlocal,newton_pair,0.0,0.0,fk[0],fk[1],fk[2],delkj[0],delkj[1],delkj[2]);
        }
      if (eflag) {
	  //std::cout << "Vkc"  << std::endl;    
          //std::cout << Vkc  << std::endl;
          if (tap_flag){
		  pvector[0] += evdwl = Tap*Vkc;
	  }
          else  pvector[0] += evdwl = Vkc - offset[itype][jtype];
          //std::cout << "evdwl"  << std::endl;
	  //std::cout << evdwl << std::endl;
      }//end if (eflag)
      
      if (evflag) ev_tally_xyz(i,j,nlocal,newton_pair,evdwl,0.0,fkcx,fkcy,fkcz,delx,dely,delz);
      } //end if (rsq < cutsq[itype][jtype] && atom->molecule[i] != atom->molecule[j])


    } // loop over jj
  } // loop over ii
}

/* ----------------------------------------------------------------------
 create neighbor list from main neighbor list for calculating the normals
------------------------------------------------------------------------- */

void PairRegDepPoly::KC_neigh()
{
  int i,j,ii,jj,n,allnum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int *ilist,*jlist,*numneigh,**firstneigh;
  int *neighptr;

  double **x = atom->x;
  int *type = atom->type;

  if (atom->nmax > maxlocal) {
    maxlocal = atom->nmax;
    memory->destroy(KC_numneigh);
    memory->sfree(KC_firstneigh);
    //memory->create(KC_numneigh,maxlocal,"KolmogorovCrespiFull:numneigh");
    //KC_firstneigh = (int **) memory->smalloc(maxlocal*sizeof(int *),
    //                                         "KolmogorovCrespiFull:firstneigh");
    memory->create(KC_numneigh,maxlocal,"PairRegDepPoly:numneigh");
    KC_firstneigh = (int **) memory->smalloc(maxlocal*sizeof(int *),
                                             "PairRegDepPoly:firstneigh");
  }

  allnum = list->inum + list->gnum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // store all KC neighs of owned and ghost atoms
  // scan full neighbor list of I

  ipage->reset();
  
  for (ii = 0; ii < allnum; ii++) {
    i = ilist[ii];

    n = 0;
    neighptr = ipage->vget();

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = map[type[i]];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;
      jtype = map[type[j]];
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz; 
      
      if (rsq != 0 && rsq < cutKCsq[itype][jtype] && atom->molecule[i] == atom->molecule[j]) {
        neighptr[n++] = j;
      }
    }

    KC_firstneigh[i] = neighptr;
    KC_numneigh[i] = n;
    if (n > 3) error->one(FLERR,"There are too many neighbors for some atoms, please check your configuration");

    ipage->vgot(n);
    if (ipage->status())
      error->one(FLERR,"Neighbor list overflow, boost neigh_modify one");
  }
}

/* ----------------------------------------------------------------------
   Calculate the normals for each atom
------------------------------------------------------------------------- */
void PairRegDepPoly::calc_normal()
{
  int i,j,ii,jj,inum,jnum;
  int cont,id,ip,m;
  double nn,xtp,ytp,ztp,delx,dely,delz,nn2;
  int *ilist,*jlist;
  double pv12[3],pv31[3],pv23[3],n1[3],dni[3],dnn[3][3],vet[3][3],dpvdri[3][3];
  double dn1[3][3][3],dpv12[3][3][3],dpv23[3][3][3],dpv31[3][3][3];

  double **x = atom->x;

  // grow normal array if necessary

  if (atom->nmax > nmax) {
    memory->destroy(normal);
    memory->destroy(dnormal);
    memory->destroy(dnormdri);
    nmax = atom->nmax;
    //memory->create(normal,nmax,3,"KolmogorovCrespiFull:normal");
    //memory->create(dnormdri,3,3,nmax,"KolmogorovCrespiFull:dnormdri");
    //memory->create(dnormal,3,3,3,nmax,"KolmogorovCrespiFull:dnormal");
    memory->create(normal,nmax,3,"PairRegDepPoly:normal");
    memory->create(dnormdri,3,3,nmax,"PairRegDepPoly:dnormdri");
    memory->create(dnormal,3,3,3,nmax,"PairRegDepPoly:dnormal");
  }

  inum = list->inum;
  ilist = list->ilist;
  //Calculate normals
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];

    //   Initialize the arrays
    for (id = 0; id < 3; id++) {
      pv12[id] = 0.0;
      pv31[id] = 0.0;
      pv23[id] = 0.0;
      n1[id] = 0.0;
      dni[id] = 0.0;
      normal[i][id] = 0.0;
      for (ip = 0; ip < 3; ip++) {
        vet[ip][id] = 0.0;
        dnn[ip][id] = 0.0;
        dpvdri[ip][id] = 0.0;
        dnormdri[ip][id][i] = 0.0;
        for (m = 0; m < 3; m++) {
          dpv12[ip][id][m] = 0.0;
          dpv31[ip][id][m] = 0.0;
          dpv23[ip][id][m] = 0.0;
          dn1[ip][id][m] = 0.0;
          dnormal[ip][id][m][i] = 0.0;
        }
      }
    }

    xtp = x[i][0];
    ytp = x[i][1];
    ztp = x[i][2];

    cont = 0;
    jlist = KC_firstneigh[i];
    jnum = KC_numneigh[i];
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = x[j][0] - xtp;
      dely = x[j][1] - ytp;
      delz = x[j][2] - ztp;
      vet[cont][0] = delx;
      vet[cont][1] = dely;
      vet[cont][2] = delz;
      cont++;
    }

    if (cont <= 1) {
      normal[i][0] = 0.0;
      normal[i][1] = 0.0;
      normal[i][2] = 1.0;
      // derivatives of normal vector is zero
      for (id = 0; id < 3; id++) {
        for (ip = 0; ip < 3; ip++) {
          dnormdri[id][ip][i] = 0.0;
          for (m = 0; m < 3; m++) {
            dnormal[id][ip][m][i] = 0.0;
          }
        }
      }
    }
    else if (cont == 2) {
      // for the atoms at the edge who has only two neighbor atoms
      pv12[0] = vet[0][1]*vet[1][2] - vet[1][1]*vet[0][2];
      pv12[1] = vet[0][2]*vet[1][0] - vet[1][2]*vet[0][0];
      pv12[2] = vet[0][0]*vet[1][1] - vet[1][0]*vet[0][1];
      dpvdri[0][0] = 0.0;
      dpvdri[0][1] = vet[0][2]-vet[1][2];
      dpvdri[0][2] = vet[1][1]-vet[0][1];
      dpvdri[1][0] = vet[1][2]-vet[0][2];
      dpvdri[1][1] = 0.0;
      dpvdri[1][2] = vet[0][0]-vet[1][0];
      dpvdri[2][0] = vet[0][1]-vet[1][1];
      dpvdri[2][1] = vet[1][0]-vet[0][0];
      dpvdri[2][2] = 0.0;

      // derivatives respect to the first neighbor, atom k
      dpv12[0][0][0] =  0.0;
      dpv12[0][1][0] =  vet[1][2];
      dpv12[0][2][0] = -vet[1][1];
      dpv12[1][0][0] = -vet[1][2];
      dpv12[1][1][0] =  0.0;
      dpv12[1][2][0] =  vet[1][0];
      dpv12[2][0][0] =  vet[1][1];
      dpv12[2][1][0] = -vet[1][0];
      dpv12[2][2][0] =  0.0;

      // derivatives respect to the second neighbor, atom l
      dpv12[0][0][1] =  0.0;
      dpv12[0][1][1] = -vet[0][2];
      dpv12[0][2][1] =  vet[0][1];
      dpv12[1][0][1] =  vet[0][2];
      dpv12[1][1][1] =  0.0;
      dpv12[1][2][1] = -vet[0][0];
      dpv12[2][0][1] = -vet[0][1];
      dpv12[2][1][1] =  vet[0][0];
      dpv12[2][2][1] =  0.0;

      // derivatives respect to the third neighbor, atom n
      for (id = 0; id < 3; id++) {
        for (ip = 0; ip < 3; ip++) {
          dpv12[id][ip][2] = 0.0;
        }
      }

      n1[0] = pv12[0];
      n1[1] = pv12[1];
      n1[2] = pv12[2];
      // the magnitude of the normal vector
      nn2 = n1[0]*n1[0] + n1[1]*n1[1] + n1[2]*n1[2];
      nn = sqrt(nn2);
      if (nn == 0) error->one(FLERR,"The magnitude of the normal vector is zero");
      // the unit normal vector
      normal[i][0] = n1[0]/nn;
      normal[i][1] = n1[1]/nn;
      normal[i][2] = n1[2]/nn;
      // derivatives of nn, dnn:3x1 vector
      dni[0] = (n1[0]*dpvdri[0][0] + n1[1]*dpvdri[1][0] + n1[2]*dpvdri[2][0])/nn;
      dni[1] = (n1[0]*dpvdri[0][1] + n1[1]*dpvdri[1][1] + n1[2]*dpvdri[2][1])/nn;
      dni[2] = (n1[0]*dpvdri[0][2] + n1[1]*dpvdri[1][2] + n1[2]*dpvdri[2][2])/nn;
      // derivatives of unit vector ni respect to ri, the result is 3x3 matrix
      for (id = 0; id < 3; id++) {
        for (ip = 0; ip < 3; ip++) {
          dnormdri[id][ip][i] = dpvdri[id][ip]/nn - n1[id]*dni[ip]/nn2;
        }
      }

      // derivatives of non-normalized normal vector, dn1:3x3x3 array
      for (id = 0; id < 3; id++) {
        for (ip = 0; ip < 3; ip++) {
          for (m = 0; m < 3; m++) {
            dn1[id][ip][m] = dpv12[id][ip][m];
          }
        }
      }
      // derivatives of nn, dnn:3x3 vector
      // dnn[id][m]: the derivative of nn respect to r[id][m], id,m=0,1,2
      // r[id][m]: the id's component of atom m
      for (m = 0; m < 3; m++) {
        for (id = 0; id < 3; id++) {
          dnn[id][m] = (n1[0]*dn1[0][id][m] + n1[1]*dn1[1][id][m] + n1[2]*dn1[2][id][m])/nn;
        }
      }
      // dnormal[id][ip][m][i]: the derivative of normal[id] respect to r[ip][m], id,ip=0,1,2
      // for atom m, which is a neighbor atom of atom i, m=0,jnum-1
      for (m = 0; m < 3; m++) {
        for (id = 0; id < 3; id++) {
          for (ip = 0; ip < 3; ip++) {
            dnormal[id][ip][m][i] = dn1[id][ip][m]/nn - n1[id]*dnn[ip][m]/nn2;
          }
        }
      }
    }
//##############################################################################################

    else if (cont == 3) {
      // for the atoms at the edge who has only two neighbor atoms
      pv12[0] = vet[0][1]*vet[1][2] - vet[1][1]*vet[0][2];
      pv12[1] = vet[0][2]*vet[1][0] - vet[1][2]*vet[0][0];
      pv12[2] = vet[0][0]*vet[1][1] - vet[1][0]*vet[0][1];
      // derivatives respect to the first neighbor, atom k
      dpv12[0][0][0] =  0.0;
      dpv12[0][1][0] =  vet[1][2];
      dpv12[0][2][0] = -vet[1][1];
      dpv12[1][0][0] = -vet[1][2];
      dpv12[1][1][0] =  0.0;
      dpv12[1][2][0] =  vet[1][0];
      dpv12[2][0][0] =  vet[1][1];
      dpv12[2][1][0] = -vet[1][0];
      dpv12[2][2][0] =  0.0;
      // derivatives respect to the second neighbor, atom l
      dpv12[0][0][1] =  0.0;
      dpv12[0][1][1] = -vet[0][2];
      dpv12[0][2][1] =  vet[0][1];
      dpv12[1][0][1] =  vet[0][2];
      dpv12[1][1][1] =  0.0;
      dpv12[1][2][1] = -vet[0][0];
      dpv12[2][0][1] = -vet[0][1];
      dpv12[2][1][1] =  vet[0][0];
      dpv12[2][2][1] =  0.0;

      // derivatives respect to the third neighbor, atom n
      for (id = 0; id < 3; id++) {
        for (ip = 0; ip < 3; ip++) {
          dpv12[id][ip][2] = 0.0;
        }
      }

      pv31[0] = vet[2][1]*vet[0][2] - vet[0][1]*vet[2][2];
      pv31[1] = vet[2][2]*vet[0][0] - vet[0][2]*vet[2][0];
      pv31[2] = vet[2][0]*vet[0][1] - vet[0][0]*vet[2][1];
      // derivatives respect to the first neighbor, atom k
      dpv31[0][0][0] =  0.0;
      dpv31[0][1][0] = -vet[2][2];
      dpv31[0][2][0] =  vet[2][1];
      dpv31[1][0][0] =  vet[2][2];
      dpv31[1][1][0] =  0.0;
      dpv31[1][2][0] = -vet[2][0];
      dpv31[2][0][0] = -vet[2][1];
      dpv31[2][1][0] =  vet[2][0];
      dpv31[2][2][0] =  0.0;
      // derivatives respect to the third neighbor, atom n
      dpv31[0][0][2] =  0.0;
      dpv31[0][1][2] =  vet[0][2];
      dpv31[0][2][2] = -vet[0][1];
      // derivatives of pv13[1] to rn
      dpv31[1][0][2] = -vet[0][2];
      dpv31[1][1][2] =  0.0;
      dpv31[1][2][2] =  vet[0][0];
      // derivatives of pv13[2] to rn
      dpv31[2][0][2] =  vet[0][1];
      dpv31[2][1][2] = -vet[0][0];
      dpv31[2][2][2] =  0.0;

      // derivatives respect to the second neighbor, atom l
      for (id = 0; id < 3; id++) {
        for (ip = 0; ip < 3; ip++) {
          dpv31[id][ip][1] = 0.0;
        }
      }

      pv23[0] = vet[1][1]*vet[2][2] - vet[2][1]*vet[1][2];
      pv23[1] = vet[1][2]*vet[2][0] - vet[2][2]*vet[1][0];
      pv23[2] = vet[1][0]*vet[2][1] - vet[2][0]*vet[1][1];
      // derivatives respect to the second neighbor, atom k
      for (id = 0; id < 3; id++) {
        for (ip = 0; ip < 3; ip++) {
          dpv23[id][ip][0] = 0.0;
        }
      }
      // derivatives respect to the second neighbor, atom l
      dpv23[0][0][1] =  0.0;
      dpv23[0][1][1] =  vet[2][2];
      dpv23[0][2][1] = -vet[2][1];
      dpv23[1][0][1] = -vet[2][2];
      dpv23[1][1][1] =  0.0;
      dpv23[1][2][1] =  vet[2][0];
      dpv23[2][0][1] =  vet[2][1];
      dpv23[2][1][1] = -vet[2][0];
      dpv23[2][2][1] =  0.0;
      // derivatives respect to the third neighbor, atom n
      dpv23[0][0][2] =  0.0;
      dpv23[0][1][2] = -vet[1][2];
      dpv23[0][2][2] =  vet[1][1];
      dpv23[1][0][2] =  vet[1][2];
      dpv23[1][1][2] =  0.0;
      dpv23[1][2][2] = -vet[1][0];
      dpv23[2][0][2] = -vet[1][1];
      dpv23[2][1][2] =  vet[1][0];
      dpv23[2][2][2] =  0.0;

//############################################################################################
      // average the normal vectors by using the 3 neighboring planes
      n1[0] = (pv12[0] + pv31[0] + pv23[0])/cont;
      n1[1] = (pv12[1] + pv31[1] + pv23[1])/cont;
      n1[2] = (pv12[2] + pv31[2] + pv23[2])/cont;
      // the magnitude of the normal vector
      nn2 = n1[0]*n1[0] + n1[1]*n1[1] + n1[2]*n1[2];
      nn = sqrt(nn2);
      if (nn == 0) error->one(FLERR,"The magnitude of the normal vector is zero");
      // the unit normal vector
      normal[i][0] = n1[0]/nn;
      normal[i][1] = n1[1]/nn;
      normal[i][2] = n1[2]/nn;

      // for the central atoms, dnormdri is always zero
      for (id = 0; id < 3; id++) {
        for (ip = 0; ip < 3; ip++) {
          dnormdri[id][ip][i] = 0.0;
        }
      } // end of derivatives of normals respect to atom i

      // derivatives of non-normalized normal vector, dn1:3x3x3 array
      for (id = 0; id < 3; id++) {
        for (ip = 0; ip < 3; ip++) {
          for (m = 0; m < 3; m++) {
            dn1[id][ip][m] = (dpv12[id][ip][m] + dpv23[id][ip][m] + dpv31[id][ip][m])/cont;
          }
        }
      }
      // derivatives of nn, dnn:3x3 vector
      // dnn[id][m]: the derivative of nn respect to r[id][m], id,m=0,1,2
      // r[id][m]: the id's component of atom m
      for (m = 0; m < 3; m++) {
        for (id = 0; id < 3; id++) {
          dnn[id][m] = (n1[0]*dn1[0][id][m] + n1[1]*dn1[1][id][m] + n1[2]*dn1[2][id][m])/nn;
        }
      }
      // dnormal[id][ip][m][i]: the derivative of normal[id] respect to r[ip][m], id,ip=0,1,2
      // for atom m, which is a neighbor atom of atom i, m=0,jnum-1
      for (m = 0; m < 3; m++) {
        for (id = 0; id < 3; id++) {
          for (ip = 0; ip < 3; ip++) {
            dnormal[id][ip][m][i] = dn1[id][ip][m]/nn - n1[id]*dnn[ip][m]/nn2;
          }
        }
      }
    }
    else {
      error->one(FLERR,"There are too many neighbors for calculating normals");
    }

//##############################################################################################
  }
}

/* ---------------------------------------------------------------------- */

double PairRegDepPoly::single(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
                         double /*factor_coul*/, double factor_lj,
                         double &fforce)
{
  double r,r2inv,r6inv,r8inv,r10inv,r12inv,forcelj,philj;
  double Tap,dTap,Vkc,fpair;

  int iparam_ij = elem2param[map[itype]][map[jtype]];
  Param& p = params[iparam_ij];

  // pair_kolmogorov_crespi_full.cpp's version of this
  // function only computes the attractive polynomial
  // energy and force in single(), with no regard for
  // registry dependence
  // this implementation does the same, even though
  // the registry dependence is on the atttractive
  // term now
  r = sqrt(rsq);
  // turn on/off taper function
  if (tap_flag) {
    Tap = calc_Tap(r,sqrt(cutsq[itype][jtype]));
    dTap = calc_dTap(r,sqrt(cutsq[itype][jtype]));
  } else {Tap = 1.0; dTap = 0.0;}


  r2inv = 1.0/rsq;
  r6inv = r2inv*r2inv*r2inv;
  r8inv = r2inv*r6inv;
  r10inv = r8inv*r2inv;
  r12inv = r6inv*r6inv;

  Vkc = -(p.z06*r6inv + p.z08*r8inv + p.z010*r10inv);
  // derivatives
  // DON'T FORGET THE CONSTANTS IF THEY ARE NOT ALREADY PART OF THE p.z0* terms!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  fpair = -(6.0*p.z06*r8inv);
  forcelj = fpair;
  fforce = factor_lj*(forcelj*Tap - Vkc*dTap/r);

  if (tap_flag) philj = Vkc*Tap;
  else philj = Vkc - offset[itype][jtype];
  return factor_lj*philj;
}
