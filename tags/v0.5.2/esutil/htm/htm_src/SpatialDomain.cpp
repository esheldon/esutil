//#     Filename:       SpatialDomain.cpp
//#
//#     The SpatialDomain
//#     classes are defined here.
//#
//#     Author:         Peter Z. Kunszt based on A. Szalay's code
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
#include "SpatialDomain.h"
#include "SpatialException.h"

#define COMMENT '#'
// ===========================================================================
//
// Member functions for class SpatialDomain
//
// ===========================================================================

/////////////CONSTRUCTOR//////////////////////////////////
//
// Initialize
//
SpatialDomain::SpatialDomain(const SpatialIndex * i) :
  index(i) {
}

/////////////DESTRUCTOR///////////////////////////////////
//
SpatialDomain::~SpatialDomain()
{
}

/////////////SETINDEX/////////////////////////////////////
//
void
SpatialDomain::setIndex(const SpatialIndex * idx)
{
  index = idx;
}

/////////////ADD//////////////////////////////////////////
//
void
SpatialDomain::add(SpatialConvex & c)
{
  convexes_.append(c);
}


/////////////INTERSECT////////////////////////////////////
//
bool
SpatialDomain::intersect(const SpatialIndex * idx, 
			 BitList & partial, BitList & full) {
  index = idx;

  if ( idx->maxlevel_ > 10 )
    throw SpatialException("Intersection with Bitlists more than 10 levels deep is impractical.");
  size_t i;
  // initialize empty lists
  full.clear(); partial.clear();
  full.trim();  partial.trim();
  full.set((uint32)index->leafCount()-1, false);
  partial.set((uint32)index->leafCount()-1, false);

  for(i = 0; i < convexes_.length(); i++)  // intersect every convex
    convexes_[i].intersect(index, &partial, &full);

  return true;
}

/////////////INTERSECT////////////////////////////////////
//
bool
SpatialDomain::intersect(const SpatialIndex * idx, 
			 ValVec<uint64> & partial, ValVec<uint64> & full) {
  index = idx;

  size_t i;
  // initialize empty lists
  full.cut(full.length());
  partial.cut(partial.length());

  for(i = 0; i < convexes_.length(); i++)  // intersect every convex
    convexes_[i].intersect(index, &partial, &full);

  partial.sort(compUint64);
  full.sort(compUint64);
  return true;
}

/////////////INTERSECT////////////////////////////////////
//
bool
SpatialDomain::intersect(const SpatialIndex * idx, 
			 ValVec<uint64> & idList) {
  index = idx;

  size_t i;
  // initialize empty lists
  idList.cut(idList.length());

  for(i = 0; i < convexes_.length(); i++)  // intersect every convex
    convexes_[i].intersect(index, &idList);

  topBit_ = 1;
  size_t n = (index->maxlevel_+2) * 2 - 1 ;
  topBit_ = topBit_ << n;
  idList.sort(compRange);
  return true;
}

void
SpatialDomain::ignoreCrLf(std::istream &in) {
  char c = in.peek();
  while (c == 10 || c == 13) {
    in.ignore();
    c = in.peek();
  }
}
/////////////READ/////////////////////////////////////////
//
void
SpatialDomain::read(std::istream &in) {
  size_t nconv;
  char comstr[20];

  while(in.peek() == COMMENT)  // ignore comments
      in.ignore(10000,'\n');
  in >> nconv; 
  ignoreCrLf(in);
  for(size_t i = 0; i < nconv; i++) {

    if(in.peek() == COMMENT) // here comes a command
      in >> comstr;

    if(strcmp(comstr,"#TRIANGLE")==0) {
      SpatialVector v1,v2,v3;
      in >> v1;
      in >> v2;
      in >> v3;
      SpatialConvex cvx(&v1,&v2,&v3);
      add(cvx);
      ignoreCrLf(in);
    } else if(strcmp(comstr,"#RECTANGLE")==0) {
      SpatialVector v1,v2,v3,v4;
      in >> v1;
      in >> v2;
      in >> v3;
      in >> v4;
      SpatialConvex cvx(&v1,&v2,&v3,&v4);
      add(cvx);
      ignoreCrLf(in);
    } else if(strcmp(comstr,"#TRIANGLE_RADEC")==0) {
      float64 ra1,ra2,ra3;
      float64 dec1,dec2,dec3;
      in >> ra1 >> dec1;
      in >> ra2 >> dec2;
      in >> ra3 >> dec3;
      SpatialVector v1(ra1,dec1);
      SpatialVector v2(ra2,dec2);
      SpatialVector v3(ra3,dec3);
      SpatialConvex cvx(&v1,&v2,&v3);
      add(cvx);
      ignoreCrLf(in);
    } else if(strcmp(comstr,"#RECTANGLE_RADEC")==0) {
      float64 ra1,ra2,ra3,ra4;
      float64 dec1,dec2,dec3,dec4;
      in >> ra1 >> dec1;
      in >> ra2 >> dec2;
      in >> ra3 >> dec3;
      in >> ra4 >> dec4;
      SpatialVector v1(ra1,dec1);
      SpatialVector v2(ra2,dec2);
      SpatialVector v3(ra3,dec3);
      SpatialVector v4(ra4,dec4);
      SpatialConvex cvx(&v1,&v2,&v3,&v4);
      add(cvx);
      ignoreCrLf(in);
    } else  if(strcmp(comstr,"#CONVEX_RADEC")==0) {
      SpatialConvex conv;
      conv.readRaDec(in);
      add(conv);
    } else {
      SpatialConvex conv;
      in >> conv;
      add(conv);
    }
    comstr[0] = 0;
  }
}

/////////////set ra,dec E.S.S./////////////////////////////////////////
//

void
SpatialDomain::setRaDecD(float64 ra, float64 dec, float64 d) {

  SpatialConvex conv;
  conv.setRaDecD(ra, dec, d);
  add(conv);
}

/////////////Write////////////////////////////////////////
//
void
SpatialDomain::write(std::ostream &out) const {
  out << "#DOMAIN" << "\n";
  out << convexes_.length() << "\n";
  for (size_t i = 0; i < convexes_.length() ; i++)
    out << convexes_[i];
}

/////////////COMPUINT64///////////////////////////////////
// compare ids
//
int 
compUint64(const void* v1, const void* v2) {
  return (  ( *((uint64 *)v1) < *((uint64 *)v2) ) ? -1 :
	    ( ( *((uint64 *)v1) > *((uint64 *)v2) ) ? 1 : 0 ) );
}

/////////////COMPRANGE///////////////////////////////////
// compare ids
//
int 
compRange(const void* v1, const void* v2) {
  uint64 a = *((uint64 *)v1);
  uint64 b = *((uint64 *)v2);

  while( (a & SpatialDomain::topBit_) == 0 ) a = a << 2 ;
  while( (b & SpatialDomain::topBit_) == 0 ) b = b << 2 ;

  return (  ( a < b ) ? -1 : ( ( a > b ) ? 1 : 0 ) );
}

uint64 SpatialDomain::topBit_ = 0;
