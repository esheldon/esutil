//#     Filename:       SpatialConstraint.hxx
//#
//#     H implementations for spatialconstraint
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
extern std::istream& operator >>( std::istream&, SpatialConstraint &);
extern std::ostream& operator <<( std::ostream&, const SpatialConstraint &);

// 
inline
SpatialVector &
SpatialConstraint::v() {
  return a_;
}

inline
float64
SpatialConstraint::d() const {
  return d_;
}

inline
void
SpatialConstraint::setVector(SpatialVector &v) {
  a_.set(v.x(),v.y(),v.z());
}

inline
void
SpatialConstraint::setDistance(float64 d) {
  d_ = d;
}

