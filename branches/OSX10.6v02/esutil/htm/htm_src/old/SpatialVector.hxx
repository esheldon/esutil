//#     Filename:       SpatialVector.hxx
//#
//#     Standard 3-d vector class: .h implementations
//#
//#
//#     Author:         Peter Z. Kunszt
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
// Friend operators
SpatialVector operator *(float64, const SpatialVector&);
SpatialVector operator *(int, const SpatialVector&);
SpatialVector operator *(const SpatialVector&, float64);
SpatialVector operator *(const SpatialVector&, int);

// inline functions

inline
float64 SpatialVector::x() const {
  return x_;
}

inline
float64 SpatialVector::y() const {
  return y_;
}

inline
float64 SpatialVector::z() const {
  return z_;
}

/////////////>>///////////////////////////////////////////
// read from istream
//
inline
std::istream& operator >>( std::istream& in, SpatialVector & v) {
  v.read(in);
  return(in);
}

/////////////<<///////////////////////////////////////////
// write to ostream
//
inline
std::ostream& operator <<( std::ostream& out, const SpatialVector & v) {
  v.write(out);
  return(out);
}
