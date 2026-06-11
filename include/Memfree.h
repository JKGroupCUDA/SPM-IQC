#ifndef __Memfree_h
#define __Memfree_h

#include "Head.h"

extern void Memfree_global_kspace();

extern void Memfree_global_data();

extern void Memfree_local_kspace();

extern void Memfree_local_data();

extern void Memfree_Option(Option *option);

#endif
