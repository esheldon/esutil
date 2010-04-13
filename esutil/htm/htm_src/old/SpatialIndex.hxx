//#     Filename:       SpatialIndex.hxx
//#
//#     H Implementations for spatialindex
//#
//#
//#     Author:         Peter Z. Kunszt, based on A. Szalay s code
//#
//#     Date:           October 15, 1998
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

/////////////leafCount//////////////////////////////////////
// leafCount: return number of leaf nodes
inline uint64
SpatialIndex::leafCount() const
{
  return leaves_;
}

/////////////NVERTICES////////////////////////////////////
// nVertices: return number of vertices
inline size_t
SpatialIndex::nVertices() const
{
  return vertices_.length();
}

//////////////////LEAFNUMBERBYID///////////////////////////////////////////
//
inline uint32
SpatialIndex::leafNumberById(uint64 id) const{
  if(maxlevel_ > HTMMAXBIT)
    throw SpatialInterfaceError("SpatialIndex:leafNumberById","BitList may only be used up to level HTMMAXBIT deep");

  return (uint32)(id - leafCount());
}

//////////////////IDBYLEAFNUMBER///////////////////////////////////////////
//
inline uint64
SpatialIndex::idByLeafNumber(uint32 n) const{
  uint64 l = leafCount();
  l += n;
  return l;
}

//////////////////NAMEBYLEAFNUMBER////////////////////////////////////////
//
inline char *
SpatialIndex::nameByLeafNumber(uint32 n, char * name) const{
  return nameById(idByLeafNumber(n), name);
}

//////////////////IDBYPOINT////////////////////////////////////////////////
// Find a leaf node where a ra/dec points to
//

inline uint64
SpatialIndex::idByPoint(const float64 & ra, const float64 & dec) const {
  SpatialVector v(ra,dec);
  return idByPoint(v);
}

//////////////////NAMEBYPOINT//////////////////////////////////////////////
// Find a leaf node where a ra/dec points to, return its name
//

inline char*
SpatialIndex::nameByPoint(const float64 & ra, const float64 & dec, 
			  char* name) const {
  return nameById(idByPoint(ra,dec), name);
}

//////////////////NAMEBYPOINT//////////////////////////////////////////////
// Find a leaf node where v points to, return its name
//

inline char*
SpatialIndex::nameByPoint(SpatialVector & vector, char* name) const {
  return nameById(idByPoint(vector),name);
}
