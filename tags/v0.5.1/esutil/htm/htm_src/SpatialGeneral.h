#if !defined (_SpatialGeneral_h)
#define _SpatialGeneral_h
//# SpatialGeneral.h
//#
//# This file contains all SDSS Science Archive global information,
//# including Global Type Definitions, Global Macros, and Global Inlines
//#
//# Author:		Peter Z. Kunszt
//#	
//# Creation:		October 19, 1999
//#
//# (c) Copyright The Johns Hopkins University 1999
//# All Rights Reserved
//#
//# The software and information contained herein are proprietary to The
//# Johns Hopkins University, Copyright 1995, 1996. This software is furnished
//# pursuant to a written license agreement and may be used, copied,
//# transmitted, and stored only in accordance with the terms of such
//# license and with the inclusion of the above copyright notice.  This
//# software and information or any other copies thereof may not be
//# provided or otherwise made available to any other person.
//#
//# Modification History:


// using stdint is safer, especially for OSX.
// ESS
#include <stdint.h>

typedef uint8_t uint8;
typedef int8_t int8;

typedef uint16_t uint16;
typedef int16_t int16;

typedef uint32_t uint32;
typedef int32_t int32;

typedef uint64_t uint64;
typedef int64_t int64;

typedef float			float32;
typedef double			float64;

#define IDSIZE                     64
#define HTMNAMEMAX                 32
#define HTMMAXDEPTH                25
#define HTMMAXKEEP		   12
#define HTMMAXBIT		   14

// emulate the standard bool type where not supported by compiler

#  if !defined(__sgi) && !defined(__linux) && !defined(_WIN32)
#    ifdef __unix
/*
 * The following ifndef must ALWAYS be present since C++ may use
 * _BOOL_EXISTS to prevent the header from trying to redefine a
 * C++ reserved word.
 */
#      ifndef _BOOL_EXISTS
#         define _BOOL_EXISTS
#         ifndef bool

typedef unsigned char           bool;
const bool                      false = 0;
const bool                      true = 1;
#define bool(x) ((x) ? true : false)

#         endif
#      endif  /* _BOOL_EXISTS */
#    endif  /* __unix */
#  endif  /* __sgi && __linux */

// Global Math Constants

const float64 gPi = 3.1415926535897932385E0 ;
const float64 gPr = gPi/180.0; 
const float64 gEpsilon = 1.0E-15;
// CVSversion = "$Name:  $";

//
// To Simplify Porting, define our platforms:
// Digital UNIX -> SpatialDigitalUnix
// SGI          -> SpatialSGI
// WINNT        -> SpatialWinNT
//
// These are set here, and then included by everything else.

// Flag SpatialStandardTemplate: Defined to indicating proper method
// for explicit template instantiation.  If a compiler supports the
// standard explicit template instantiation, define.
//

#if defined(__unix)

#   if defined(__osf__)
#      define SpatialDigitalUnix 1
#      define SpatialStandardTemplate 1
#   elif defined(__sgi)
#      define SpatialSGI 1
#      define SpatialPragmaTemplateSGI 1
#   elif defined(__sun)
#      define SpatialSUN 1
#      define SpatialStandardTemplate 1
#   elif defined(__linux)
#      define SpatialLinux 1
#      define SpatialStandardTemplate 1
#   endif
#   define LINKAGE 

#elif defined(_WIN32)
#   define SpatialWinNT 1
#   define SpatialWINTemplate 1
// This warning is about template instances being exported in the dll...
#   pragma warning(disable: 4251)

// Define LINKAGE for NT VC++6 complier

#   ifdef _EXPORTING
#	define LINKAGE	__declspec(dllexport)
#   endif
#   ifdef _IMPORTING
#	define	LINKAGE	__declspec(dllimport)
#   endif

#endif  /* _WIN32 */

#ifndef SpatialWinNT
#   define IDHIGHBIT  0x8000000000000000LL
#   define IDHIGHBIT2 0x4000000000000000LL
#   ifdef SpatialDigitalUnix
#      define PRINTID(x) printf("%lu",(x))
#      define PRINTID_HEX(x) printf("%lX",(x))
#   else
#      define PRINTID(x) printf("%llu",(x))
#      define PRINTID_HEX(x) printf("%llX",(x))
#   endif
#else
#   define PRINTID(x) printf("%I64u",(x))
#   define PRINTID_HEX(x) printf("%I64X",(x))
#endif

#endif /* SPATIALGENERAL_H */
