#ifndef _SpatialConstraint_h
#define _SpatialConstraint_h
//#     Filename:       SpatialConstraint.h
//#
//#     Classes defined here: SpatialConstraint SpatialSign
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

#include "SpatialVector.h"

//########################################################################
//#
//# Spatial Sign helper class

/**
   The sign class is inherited by Constraint and Convex. Assignment and
   copy operators are used in both scopes.
*/

//class LINKAGE SpatialSign {
class SpatialSign {
public:
  enum Sign {
    nEG,			// All constraints negative or zero
    zERO,			// All constraints zero
    pOS,			// All constraints positive or zero
    mIXED			// At least one pos and one neg
  };

  /// Constructor
  SpatialSign(Sign sign = zERO);

  /// Copy constructor
  SpatialSign(const SpatialSign &);

  /// Assignment
  SpatialSign & operator =(const SpatialSign &);

protected:
  /// Sign value
  Sign sign_;
};

//########################################################################
//#
//# Spatial Constraint class
//#
/**
   The Constraint is really a cone on the sky-sphere. It is characterized
   by its direction a_, the opening angle s_ and its cosine -- the distance
   of the plane intersecting the sphere and the sphere center.
   If d_ = 0, we have a half-sphere. If it is negative, we have a 'hole'
   i.e. the room angle is larger than 90degrees.

   Example: positive distance
<pre>
.                   ____
.                ---    ---
.               /        /|\
.              /        / |=\
.             |        /  |==|     this side is in the convex.
.            |        /\s |===|
.            |------------|---| -> direction a
.            |        \   |===|
.             |        \  |==|
.              \        \ |=/
.               \        \|/
.                ---____---
.
.
.                     <-d-> is positive (s < 90)

</pre>
 Example: negative distance
<pre>
.                   ____
.                ---====---
.  this side is /========/|\
.  in the      /========/=| \
.  convex     |==== s__/==|  |
.            |===== / /===|   |
.  dir. a <- |------------|---|  'hole' in the sphere
.            |========\===|   |
.             |========\==|  |
.              \========\=| /
.               \========\|/
.                ---____---
.
.
.                     <-d-> is negative (s > 90)
</pre>
 for d=0 we have a half-sphere. Combining such, we get triangles, rectangles
 etc on the sphere surface (pure ZERO convexes)

*/

//class LINKAGE SpatialConstraint : public SpatialSign {
class SpatialConstraint : public SpatialSign {
public:
  /// Constructor
  SpatialConstraint() {};

  /// Initialization constructor
  SpatialConstraint(SpatialVector, float64);

  /// Copy constructor
  SpatialConstraint(const SpatialConstraint &);

  /// Assignment
  SpatialConstraint & operator =(const SpatialConstraint &);

  /// set vector
  void setVector(SpatialVector &);

  /// set distance
  void setDistance(float64);

  /// Invert
  void invert();

  /// check whether a vector is inside this
  bool contains(const SpatialVector v);

  /// give back vector
  SpatialVector & v() ;

  /// give back distance
  float64 d() const ;

  /// read
  void read(std::istream &in);

  /// read
  void readRaDec(std::istream &in);

  /// set ra,dec,d  E.S.S.
  void setRaDecD(float64 ra, float64 dec, float64 d);

  /// write
  void write(std::ostream &out) const;

private:
  SpatialVector a_;			// normal vector
  float64       d_;			// distance from origin
  float64       s_;			// cone angle in radians

  friend class SpatialIndex;
  friend class SpatialConvex;
  friend class SpatialDomain;
  friend class sxSpatialDomain;
};

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

#endif
