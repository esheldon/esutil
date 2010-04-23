//#     Filename:       SpatialConvex.hxx
//#
//#     H definitions for  SpatialConvex
//#
//#
//#     Author:         Peter Z. Kunszt, based on A. Szalay's code
//#     
//#     Date:           October 16, 1998
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
extern std::istream& operator >>( std::istream&, SpatialConvex &);
extern std::ostream& operator <<( std::ostream&, const SpatialConvex &);


inline
SpatialConstraint &
SpatialConvex::operator [](size_t i) {
  return constraints_[i];
}

inline
size_t
SpatialConvex::numConstraints() {
  return constraints_.length();
}
