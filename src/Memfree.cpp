#include "Head.h"
#include "Data.h"
#include "Memfree.h"

void Memfree_global_kspace()
{
	free(K);
	for(int i=0;i<CplxDofs;i++)
	{
		free(kindex[i]);
	}
	free(kindex);
}

void Memfree_global_data()
{
	free(u_R);
	free(u_F);
	free(NCpt);
	for(int i=0;i<dimPhy;i++)
	{
		free(ProjMatrix[i]);
	}
	free(ProjMatrix);
}

void Memfree_local_kspace()
{
	for(int i=0;i<CplxDofs_local;i++)
	{
		free(PK_local[i]);
		free(kindex_local[i]);
	}
	free(PK_local);
	free(kindex_local);
	free(k_square_local);
	free(epmckt2_local);
	free(ikt2_local);
}

void Memfree_local_data()
{
	free(u_R_local);
	free(u_F_local);
	free(u2_R_local);
	free(u3_R_local);
	free(u4_R_local);
	free(nln_R_local);
	free(nln_F_local);
	free(Laplas_R_local);
	free(Laplas_F_local);
	fftw_free(fftw_Ctmp_local);
	fftw_free(fftw_Rtmp_local);

	for(int i=0;i<2;i++)
	{
		fftw_free(Phi0[i]);
	}

	fftw_free(Phi0);
	fftw_free(tv);

}

void Memfree_Option(Option *option)
{
	for(int i=0;i<option->k;i++)
	{
		fftw_free(option->V[i]);
	}
	fftw_free(option->V);
	free(option);
}






