#ifndef FIRESTORM_H
#define FIRESTORM_H

#if defined(__clang__)
#define UNROLL _Pragma("clang loop unroll(full)")
#elif defined(__GNUC__)
#define UNROLL _Pragma("GCC unroll 16")
#else
#define UNROLL
#endif

#define S_MR 12
#define S_NR 8
#define D_MR 6
#define D_NR 8

#define S_MV (S_MR / 4)
#define S_NV (S_NR / 4)
#define D_MV (D_MR / 2)
#define D_NV (D_NR / 2)

#endif
