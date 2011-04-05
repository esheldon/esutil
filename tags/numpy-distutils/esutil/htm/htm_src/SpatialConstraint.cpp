//#     Filename:       SpatialConstraint.cpp
//#
//#     The SpatialConstraint, SpatialSign
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
#include "SpatialConstraint.h"
#include "SpatialException.h"

#define COMMENT '#'

// ===========================================================================
//
// Member functions for class SpatialSign
//
// ===========================================================================

/////////////CONSTRUCTOR//////////////////////////////////
//
SpatialSign::SpatialSign(Sign sign) : sign_(sign) {
}

/////////////COPY CONSTRUCTOR/////////////////////////////
//
SpatialSign::SpatialSign(const SpatialSign & oldSign) : sign_(oldSign.sign_) {
}

/////////////ASSIGNMENT///////////////////////////////////
//
SpatialSign &
SpatialSign::operator =(const SpatialSign & oldSign) {
  if( & oldSign != this)sign_ = oldSign.sign_;
  return *this;
}

// ===========================================================================
//
// Member functions for class SpatialConstraint
//
// ===========================================================================

/////////////CONSTRUCTOR//////////////////////////////////
//
SpatialConstraint::SpatialConstraint(SpatialVector a, float64 d) :
  a_(a), d_(d)
{
  a_.normalize();
  s_ = acos(d_);
  if(d_ <= -gEpsilon) sign_ = nEG;
  if(d_ >=  gEpsilon) sign_ = pOS;
}

/////////////COPY CONSTRUCTOR/////////////////////////////
//
SpatialConstraint::SpatialConstraint(const SpatialConstraint & old) :
  a_(old.a_), d_(old.d_), s_(old.s_) {
  sign_ = old.sign_;
}

/////////////ASSIGNMENT///////////////////////////////////
//
SpatialConstraint &
SpatialConstraint::operator =(const SpatialConstraint & old)
{
  if ( &old != this ) { // beware of self-assignment
    a_ = old.a_;
    d_ = old.d_;
    s_ = old.s_;
    sign_ = old.sign_;
  }
  return *this;
}

/////////////CONTAINS/////////////////////////////////////
// check whether a vector is inside this
//
bool 
SpatialConstraint::contains(const SpatialVector v) {
    if ( acos(v * a_) < s_ ) return true;
    return false;
}

/////////////INVERT///////////////////////////////////////
//
void
SpatialConstraint::invert() {
  d_ = -d_;
  s_ = acos(d_);
  if(sign_ == nEG) sign_ = pOS;
  if(sign_ == pOS) sign_ = nEG;
}

/////////////READ/////////////////////////////////////////
//
void
SpatialConstraint::read(std::istream &in) {

  in.setf(std::ios::skipws);
  while(in.peek() == COMMENT)  // ignore comments
      in.ignore(10000,'\n');
  in >> a_ >> d_ ;
  if(!in.good())
    throw SpatialFailure("SpatialConstraint:read: Could not read constraint");
  a_.normalize();
  s_ = acos(d_);
  if     (d_ <= -gEpsilon) sign_ = nEG;
  else if(d_ >=  gEpsilon) sign_ = pOS;
  else                sign_ = zERO;
}


/////////////READ/////////////////////////////////////////
//
void
SpatialConstraint::readRaDec(std::istream &in) {

  while(in.peek() == COMMENT)  // ignore comments
      in.ignore(10000,'\n');
  float64 ra,dec;
  in >> ra >> dec >> d_ ; in.ignore();
  a_.set(ra,dec);
  s_ = acos(d_);
  if     (d_ <= -gEpsilon) sign_ = nEG;
  else if(d_ >=  gEpsilon) sign_ = pOS;
  else                sign_ = zERO;
}

/////////////set ra,dec E.S.S./////////////////////////////////////////
//

void
SpatialConstraint::setRaDecD(float64 ra, float64 dec, float64 d) {
  
  a_.set(ra,dec);
  d_ = d;
  s_ = acos(d_);

  if     (d_ <= -gEpsilon) sign_ = nEG;
  else if(d_ >=  gEpsilon) sign_ = pOS;
  else                sign_ = zERO;
}


/////////////WRITE////////////////////////////////////////
//
void
SpatialConstraint::write(std::ostream &out) const {
  size_t p = out.precision();
  out.precision(16);
  out << a_ << ' ' << d_ << "\n";
  out.precision(p);
}

/////////////>>///////////////////////////////////////////
// read from istream
//
std::istream& operator >>( std::istream& in, SpatialConstraint & c) {
  c.read(in);
  return(in);
}

/////////////<<///////////////////////////////////////////
// write to ostream
//
std::ostream& operator <<( std::ostream& out, const SpatialConstraint & c) {
  c.write(out);
  return(out);
}
