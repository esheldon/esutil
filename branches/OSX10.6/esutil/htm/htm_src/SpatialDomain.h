#ifndef _SpatialDomain_h
#define _SpatialDomain_h
//#     Filename:       SpatialDomain.h
//#
//#     Classes defined here: SpatialDomain
//#
//#
//#     Author:         Peter Z. Kunszt
//#     
//#     Date:           October 16, 1998
//#
//#
//#
//# (c) Copyright The Johns Hopkins University 1998-1999
//# All Rights Reserved
//#
//# The software and information contained herein are proprietary to The
//# Johns Hopkins University, Copyright 1999.  This software is furnished
//# pursuant to a written license agreement and may be used, copied,
//# transmitted, and stored only in accordance with the terms of such
//# license and with the inclusion of the above copyright notice.  This
//# software and information or any other copies thereof may not be
//# provided or otherwise made available to any other person.
//#
//#


#include "SpatialConvex.h"
#include "BitList.h"

//########################################################################
//
// Spatial Domain class
//
// 

/** A spatial domain is a list of spatial convexes. So we can have
 really disjoint pieces of the sky defined by a domain.  */

//class LINKAGE SpatialDomain {
class SpatialDomain {
public:
  /// Constructor
  SpatialDomain(const SpatialIndex * idx = 0);

  /// Destructor
  ~SpatialDomain();

  /// Set index pointer
  void setIndex(const SpatialIndex *);

  /// Add a convex
  void add(SpatialConvex &);

  /// Simplify the Domain, remove redundancies
  void simplify();

  /** Intersect with index. 
      Return the bitlist of the leafnodes that are
      partially and fully intersected by this domain. */
  bool intersect(const SpatialIndex * idx, 
		 BitList & partial, BitList & full);

  /// Same intersection, but return vectors of ids instead of bitlists.
  bool intersect(const SpatialIndex * idx, 
		 ValVec<uint64> & partial, ValVec<uint64> & full);

  /// Same intersection, but return just a list of IDs not level depth
  bool intersect(const SpatialIndex * idx, ValVec<uint64> & idlist);

  /// numConvexes: give back the number of convexes
  size_t numConvexes();

  /// [] operator: give back convex
  SpatialConvex & operator [](size_t i);

  /// read from stream
  void read(std::istream&);

  /// set ra,dec,d from user
  void setRaDecD(float64 ra, float64 dec, float64 d);

  /// write to stream
  void write(std::ostream&) const;

  const SpatialIndex * index; 		/// A pointer to the index

  static void ignoreCrLf(std::istream &);
protected:
  ValVec<SpatialConvex> convexes_;      /// The vector of convexes

public:
  static uint64 topBit_;
};

#include "SpatialDomain.hxx"
#endif
