//#     Filename:       SpatialDomain.hxx
//#
//#     H declaratinos for SpatialDomain
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

inline
SpatialConvex &
SpatialDomain::operator [](size_t i) {
  return convexes_[i];
}

inline
size_t
SpatialDomain::numConvexes() {
  return convexes_.length();
}

/////////////>>///////////////////////////////////////////
// read from istream
//
inline
std::istream& operator >>( std::istream& in, SpatialDomain & c) {
  c.read(in);
  return(in);
}

/////////////<<///////////////////////////////////////////
// write to ostream
//
inline
std::ostream& operator <<( std::ostream& out, const SpatialDomain & c) {
  c.write(out);
  return(out);
}

extern  int compUint64(const void*, const void*);
extern  int compRange (const void*, const void*);
