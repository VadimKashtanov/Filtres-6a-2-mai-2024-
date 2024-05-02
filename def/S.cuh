#pragma once

#include "marchee.cuh"

#include "insts.cuh"

#define L2_regularisation 0.000001 //0.00001

#define SCORE_Y_COEF_BRUIT 0.00
#define ENTREE_COEF_BRUIT  0.10

#define P_S      2.00
#define P_somme  1.00
#define K_fonc(x) (powf(x,0.25))

#define  S(y,w) (powf(y-w,P_S  )/P_S)
#define dS(y,w) (powf(y-w,P_S-1)    )
#define K(p1,p0,alea) K_fonc(fabs( 				\
	100*(1-alea*SCORE_Y_COEF_BRUIT)*(p1/p0 - 1)	\
))

#define ___SCORE(y,p1,p0,alea,u) (K(p1,p0,alea) *  S(y, sng(p1/p0 - 1)) * u)
#define _dySCORE(y,p1,p0,alea,u) (K(p1,p0,alea) * dS(y, sng(p1/p0 - 1)) * u)

//	----

static float APRES_SCORE(float somme) {
	return powf(somme, P_somme) / P_somme;
};

static float dAPRES_SCORE(float somme) {
	return powf(somme, P_somme - 1);
};

//	----

static __device__ float cuda_SCORE(float y, float p1, float p0, float alea, float u) {
	return ___SCORE(y,p1,p0,alea,u);
};

static __device__ float cuda_dSCORE(float y, float p1, float p0, float alea, float u) {
	return  _dySCORE(y,p1,p0,alea,u);
};

//	=================================

//	S(x)
float   nvidia_somme_score (uint EXACTE, float * y, float * u_max, float * u_min, uint depart, uint T, uint _t_MODE, uint GRAINE);
float   nvidia_score_finale(float somme, uint T, uint _t_MODE, uint GRAINE);
//	dS(x)/dx
float d_nvidia_score_finale(float somme, uint T, uint _t_MODE, uint GRAINE);
void  d_nvidia_somme_score (uint EXACTE, float d_somme, float * y, float * dy, float * u_max, float * u_min, uint depart, uint T, uint _t_MODE, uint GRAINE);

//	=================================

//	Prediction : pourcent%

float nvidia_prediction(float * y, uint depart, uint T, uint _t_MODE, uint GRAINE);