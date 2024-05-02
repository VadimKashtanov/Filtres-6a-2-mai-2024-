#include "S.cuh"

#include "../../impl_tmpl/tmpl_etc.cu"

//	   (a-min)/(max-min) + (b-min)/(max-min) + (c-min)/(max-min)
//	= ( (a-min)+(b-min)+(c-min)) / (max - min)
//	= ( a+b+c - 3*min) / (max-min)

#define L 3

static __global__ void kerd_nvidia_score_somme(
	uint EXACTE,
	uint _t_MODE, uint GRAINE,
	float * y,
	float * _s_max, float * _s_min,
	uint t0, uint T,
	float * score, float * _PRIXS)
{
	uint t = threadIdx.x + blockIdx.x * blockDim.x;
	//
	if (t < T) {
		float s = 0;	//******:!!!!!! normer u_max, u_min. mettre =1 quand y=0
		//				//  puis resoudre les nan
		float u = 1.0;
		float s_max;
		float s_min;
		//
		FOR(0, mega_t, MEGA_T) {
			uint depart_plus_t = t_MODE(
				_t_MODE, GRAINE,
				t0, t0+T*MEGA_T,
				t, mega_t,
				T, MEGA_T
			);

			float p1 = _PRIXS[depart_plus_t+1];
			float p0 = _PRIXS[depart_plus_t  ];

			float _y = y[(mega_t*T + t)*1 + 0];
			//
			uint _alea = PSEUDO_ALEA_cuda((1+GRAINE+mega_t*T+t));
			float alea = 2*((float)(_alea%1000))/1000.0 - 1.0;
			if (EXACTE) alea = 0;
			//
			u += u * _y * (p1/p0-1) * L;
			if (u < 0) u = 0.0;
			//
			//
			float _s = cuda_SCORE(_y, p1, p0, alea, u);
			s += _s;
			//
			if (mega_t == 0) {
				s_max = _s;
				s_min = _s;
			}
			//
			if (s_max < _s) s_max = _s;
			if (s_min > _s) s_min = _s;
		}
		//
		//if (s_max == s_min) s_min = 0;
		//
		_s_max[t] = s_max;
		_s_min[t] = s_min;
		//
		score[t] = s;///s_max;///s_max;//(s - MEGA_T*s_min) / (s_max - s_min);
		//printf("[%i,%f],\n", t, 1/s_max);
	}
};

#define HORIZON 32

static __global__ void kerd_addition_horizontale(
	float * vecteur, uint T, float * somme_finale)
{
	uint thx = threadIdx.x;
	uint t = threadIdx.x + blockIdx.x * blockDim.x;
	//
	uint __BLOQUE = blockDim.x;
	//
	if (t < T) {
		uint depart_bloque = 2*(t - (t% __BLOQUE));
		//
		for (uint mul=1; mul <= HORIZON;) {
			if (thx % mul == 0) {
				// a = b + c
				uint a = depart_bloque + 2*thx;
				uint b = depart_bloque + 2*thx;
				uint c = depart_bloque + 2*thx + 2*(mul)/2;
				//
				if (!(a < T)) assert(0);
				if (!(b < T)) assert(0);
				if (!(c < T)) assert(0);
				//
				vecteur[a] = vecteur[b] + vecteur[c];
			}
			__syncthreads();
			mul *= 2;
		}
		//
		if (thx == 0) atomicAdd(&somme_finale[0], vecteur[depart_bloque+0]);
	};
};

float nvidia_somme_score(
	uint EXACTE,
	float * y,
	float * s_max, float * s_min,
	uint depart, uint T,
	uint _t_MODE, uint GRAINE)
{
	ASSERT(T % (HORIZON*2) == 0);
	//
	float * somme_score__d = cudalloc<float>(T);
	float * somme_score_finale__d = cudalloc<float>(1);
	CONTROLE_CUDA(cudaMemset(somme_score_finale__d, 0, sizeof(float)*1));
	CONTROLE_CUDA(cudaMemset(somme_score__d, 0, sizeof(float)*T));
	
	//	--- Calcule du Score ---
	kerd_nvidia_score_somme<<<dim3(KERD(T,1)),dim3(1)>>>(
		EXACTE,
		_t_MODE, GRAINE,
		y,
		s_max, s_min,
		depart, T,
		somme_score__d, cuda_MARCHEE_DE_TRADE
	);
	ATTENDRE_CUDA();

	//	--- Somme Horizontale ---
	kerd_addition_horizontale<<<dim3(KERD(T/2,HORIZON)),dim3(HORIZON)>>>(
		somme_score__d,
		T, somme_score_finale__d
	);
	ATTENDRE_CUDA();

	//	Gpu vers Cpu
	float * somme_score = gpu_vers_cpu<float>(somme_score_finale__d, 1);
	float somme = somme_score[0];
	//
	CONTROLE_CUDA(cudaFree(somme_score__d));
	CONTROLE_CUDA(cudaFree(somme_score_finale__d));
	free(somme_score);
	//
	return somme;
};

float  nvidia_score_finale(float somme, uint T, uint _t_MODE, uint GRAINE) {
	float moyenne = somme / (float)(1 * T * MEGA_T);
	return APRES_SCORE(moyenne);
};

//	===============================================================

float d_nvidia_score_finale(float somme, uint T, uint _t_MODE, uint GRAINE) {
	float moyenne = somme / (float)(1 * T * MEGA_T);
	return dAPRES_SCORE(moyenne) / (float)(1 * T * MEGA_T);
};

//	===============================================================

static __global__ void kerd_nvidia_score_dpowf(
	uint EXACTE,
	uint _t_MODE, uint GRAINE,
	float dS, float * y, float * dy,
	float * _s_max, float * _s_min,
	uint t0, uint T,
	float * _PRIXS)
{
	uint _t = threadIdx.x + blockIdx.x * blockDim.x;

	if (_t < T) {
		float u = 1.0;
		//
		float s_max = _s_max[_t];
		float s_min = _s_min[_t];
		//
		//printf("(%i,%f),\n", _t, 1/s_max);
		//
		FOR(0, mega_t, MEGA_T) {
			uint depart_plus_t = t_MODE(
				_t_MODE, GRAINE,
				t0, t0+T*MEGA_T,
				_t, mega_t,
				T, MEGA_T
			);
			//
			float p1 = _PRIXS[depart_plus_t+1];
			float p0 = _PRIXS[depart_plus_t  ];
			//
			float _y = y[(mega_t*T + _t)*1 + 0];
			//
			uint _alea = PSEUDO_ALEA_cuda((1+GRAINE+mega_t*T+_t));
			float alea = 2*((float)(_alea%1000))/1000.0 - 1.0;
			if (EXACTE) alea = 0;
			//
			u += u * _y * (p1/p0 - 1) * L;
			if (u < 0) u = 0.0;
			//
			//printf("t=%i %f\n", _t, 1/s_max);
			float _dy = cuda_dSCORE(_y, p1, p0, alea, u);///s_max;// / (s_max - s_min);
			//
			atomicAdd(&dy[(mega_t*T + _t)*1+0], dS * _dy);
			//atomicAdd car certaines fonction prennent y[-1] comme entree
		}
	}
};

void d_nvidia_somme_score(
	uint EXACTE,
	float d_score,
	float * y, float * dy,
	float * u_max, float * u_min,
	uint depart, uint T,
	uint _t_MODE, uint GRAINE)
{
	kerd_nvidia_score_dpowf<<<dim3(KERD(T,1)), dim3(1)>>>(
		EXACTE,
		_t_MODE, GRAINE,
		d_score,
		y, dy,
		u_max, u_min,
		depart, T,
		cuda_MARCHEE_DE_TRADE
	);
	ATTENDRE_CUDA();
};