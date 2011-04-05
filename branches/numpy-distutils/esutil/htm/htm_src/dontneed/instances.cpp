//#     Filename:       instances.cpp
//#
//#     The instances needed for the SpatialMap package.
//#
//#     Author:         Peter Z. Kunszt
//#     
//#     Date:           October 23, 1998
//#
//#
//#
//# (c) Copyright The Johns Hopkins University 1998
//# All Rights Reserved
//#
//# The software and information contained herein are proprietary to The
//# Johns Hopkins University, Copyright 1998.  This software is furnished
//# pursuant to a written license agreement and may be used, copied,
//# transmitted, and stored only in accordance with the terms of such
//# license and with the inclusion of the above copyright notice.  This
//# software and information or any other copies thereof may not be
//# provided or otherwise made available to any other person.
//#
//#
//#     Modification History:
//#
#include <SpatialGeneral.h>
#include <SpatialInterface.h>
#include <SpatialConvex.h>

// In the SX environment this file is part of the sxGeneral library.
#ifndef SXDB
#include <VarStr.hpp>
#endif

#if defined(SpatialStandardTemplate)

// The sparc has a strange way of not explicitly defining the subclasses...
#if defined(SpatialSUN) && !defined(SpatialLinux)
template class ValVec<QuadNode>;
template class ValVec<Layer>;
#else
template class ValVec<SpatialIndex::QuadNode>;
template class ValVec<SpatialIndex::Layer>;
#endif

template class ValVec<BitList>;
template class ValVec<SpatialVector>;
template class ValVec<SpatialConstraint>;
template class ValVec<SpatialConvex>;
template class ValVec<int16>;
template class ValVec<int32>;
template class ValVec<uint8>;
#ifndef SXDB
template class ValVec<uint16>;
#endif
template class ValVec<uint32>;
template class ValVec<uint64>;
#ifdef SpatialDigitalUnix
template class ValVec<size_t>;
#endif
template class ValVec<htmRange>;
template class ValVec<htmPolyCorner>;

#elif defined(SpatialPragmaTemplateSGI)

#pragma instantiate ValVec<SpatialIndex::QuadNode>
#pragma instantiate ValVec<SpatialIndex::Layer>
#pragma instantiate ValVec<BitList>
#pragma instantiate ValVec<SpatialVector>
#pragma instantiate ValVec<SpatialConstraint>
#pragma instantiate ValVec<SpatialConvex>
#pragma instantiate ValVec<uint8>
#ifndef SXDB
#pragma instantiate ValVec<uint16>
#endif
#pragma instantiate ValVec<uint32>
#pragma instantiate ValVec<uint64>
#pragma instantiate ValVec<htmRange>
#pragma instantiate ValVec<htmPolyCorner>

#elif defined(SpatialWINTemplate)

#include <VarVecDef.h>

//template class LINKAGE ValVec<uint64>;
//template class LINKAGE ValVec<uint32>;
template class ValVec<uint64>;
template class ValVec<uint32>;

#endif
