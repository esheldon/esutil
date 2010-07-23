//#     Filename:       SpatialInterface.cpp
//#
//#     The htmInterface class is defined here.
//#
//#     Author:         Peter Z. Kunszt 
//#
//#     Date:           August 30 , 2000
//#
//#
//#
//# (c) Copyright The Johns Hopkins University 2000
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
#include "SpatialInterface.h"
#include "string.h"
#include "stdlib.h"
#include "math.h"

#ifdef SpatialSGI
extern long long atoll (const char *str);
#endif

//==============================================================
//
// These are the implementations of the htm interface.
//
//==============================================================

///////////CONSTRUCTOR///////////////////////


htmInterface::htmInterface(
        size_t depth, 
        size_t savedepth) {
    index_ = NULL;
    t_ = NULL;
    init(depth, savedepth);
}
void htmInterface::init(size_t depth, size_t savedepth) {
    if (index_) delete index_;
    if (t_) delete t_;
    index_ = new SpatialIndex(depth, savedepth);
}

///////////DESTRUCTOR////////////////////////
htmInterface::~htmInterface() {
  delete index_;
  if(t_) delete t_;
}

///////////LOOKUP METHODS////////////////////

uint64 htmInterface::lookupIDCmd(char *str) {

  cmd_ = str;
  if(t_)delete t_;
  t_ = new VarStrToken(cmd_);

  float64 v[3];
  cmdCode code = getCode();

  if(code == NAME) {
    VarStr token = t_->next();
    if(token.empty())
      throw SpatialInterfaceError("htmInterface:lookupIDCmd: expected Name");

    return index_->idByName(token.data());
  }

  getDepth();
  if(! parseVec(code, v) )
    throw SpatialInterfaceError("htmInterface:lookupIDCmd: Expect vector in Command. ", cmd_.data());

  if( code == J2000 )
    return lookupID(v[0], v[1]);
  return lookupID(v[0], v[1], v[2]);

}

const char * htmInterface::lookupNameCmd(char *str) {

  cmd_ = str;
  if(t_)delete t_;
  t_ = new VarStrToken(cmd_);

  float64 v[3];
  cmdCode code = getCode();

  if(code == ID) {
    uint64 id = getInt64();
    index_->nameById(id, name_);
  } else {
    getDepth();

  if(! parseVec(code, v) )
    throw SpatialInterfaceError("htmInterface:lookupNameCmd: Expect vector in Command. ", cmd_.data());

    if( code == J2000 )
      index_->nameByPoint(v[0], v[1], name_);
    else {
      SpatialVector tv(v[0], v[1], v[2]);
      index_->nameByPoint(tv, name_);
    }
  }

  return name_;
}

// get the depth, which is the first item in the first character argument.
#if defined(__sun)
cmdCode
#else
htmInterface::cmdCode
#endif
htmInterface::getCode() {

  cmdCode code;

  // parse incoming string. expect to have an integer indicating the
  // depth at first position.
  VarStr token = t_->next();

  if     ( token == "J2000" )
    code = J2000;
  else if( token == "CARTESIAN" )
    code = CARTESIAN;
  else if( token == "NAME" )
    code = NAME;
  else if( token == "ID" )
    code = ID;
  else if( token == "DOMAIN" )
    code = HTMDOMAIN;
  else
    throw SpatialInterfaceError("htmInterface:getCode: Unexpected command",token);

  return code;
}

void htmInterface::getDepth() {

  size_t depth = getInteger();
  if(depth > HTMMAXDEPTH)
    throw SpatialInterfaceError("htmInterface:getDepth: Depth too large: Max is HTMMAXDEPTH");

  changeDepth(depth);
}

// get an integer out of the command string
int32 htmInterface::getInteger() {

  if(!t_)
    throw SpatialFailure("htmInterface:getInteger: No command to parse");

  // parse incoming string. expect to have an integer.
  const VarStr &token = t_->next();
  if(!isInteger(token))
    throw SpatialInterfaceError("htmInterface:getInteger: Expected integer at first position of Command. ",cmd_.data());

  return atoi(token.data());
}

// get an integer out of the command string
uint64 htmInterface::getInt64() {

  if(!t_)
    throw SpatialFailure("htmInterface:getInt64: No command to parse");

  // parse incoming string. expect to have an integer.
  const VarStr &token = t_->next();
  if(!isInteger(token))
    throw SpatialInterfaceError("htmInterface:getInt64: Expected integer at first position of Command. ",cmd_.data());
#ifdef SpatialWinNT
  return _atoi64(token.data());
#elif defined(SpatialDigitalUnix)
  return atol(token.data());
#else
  return atoll(token.data());
#endif
}

// get an integer out of the command string
float64 htmInterface::getFloat() {

  if(!t_)
    throw SpatialFailure("htmInterface:getFloat: No command to parse");

  // parse incoming string. expect to have an integer.
  const VarStr &token = t_->next();
  if(!isFloat(token))
    throw SpatialInterfaceError("htmInterface:getFloat: Expected float at first position of Command. ",cmd_.data());

  return atof(token.data());
}


// parse the string, returning the number of floats
// that have been in the string.
bool
htmInterface::parseVec( cmdCode code, float64 *v) {

  VarStr  token;
  size_t  i = 0, len;

  if(code == J2000)
    len = 2;
  else if(code == CARTESIAN)
    len = 3;
  else
    throw SpatialInterfaceError("htmInterface:parseVec: Expected code J2000 or CARTESIAN.");

  // parse the next len positions
  while( i < len  ) {
    token = t_->next();
    if(token.empty())break;

    if(!isFloat(token))
      throw SpatialInterfaceError("htmInterface:parse: Expected float at this position of Command. ",cmd_.data());
    if(i == len)
      throw SpatialInterfaceError("htmInterface:parse: Expect less floats in Command. ", cmd_.data());
    v[i++] = atof(token.data());
  }

  if(i < len)
    return false;
  return true;

}

// check whether string is an integer
bool htmInterface::isInteger(const VarStr &str) {
  if(str.empty()) return false;
  uint32 len = str.length();
  return (strspn(str.data(),"+0123456789") == len) ? true : false ;
}

// check whether string is a float
bool htmInterface::isFloat(const VarStr &str) {
  if(str.empty()) return false;
  uint32 len = str.length();
  return (strspn(str.data(),"+-.e0123456789") == len) ? true : false ;
}

// check whether an id is in a range
bool htmInterface::inRange( const ValVec<htmRange> &range, int64 id) {
  size_t len = range.length() - 1;

  // completely outside range?
  if(size_t(id) < range(0).lo || size_t(id) > range(len).hi)return false;

  // check each range
  for(size_t i = 0; i <= len; i++)
    if(size_t(id) <= range(i).hi && size_t(id) >= range(i).lo) return true;
  return false;
}

// print the range
void 
htmInterface::printRange( const ValVec<htmRange> &range) {
 
  //for(size_t i = 0; i < range.length(); i++)
    //cout << SpatialIndex::nameById(range(i).lo) << ":" 
    //	 << SpatialIndex::nameById(range(i).hi) 
    //	 << "   " << range(i).lo << " - " << range(i).hi << "\n";
}  

//////////////////////CIRCLEREGION METHODS//////////////////////

const ValVec<htmRange> & 
htmInterface::circleRegion( float64 ra,
			    float64 dec,
			    float64 rad ) {

  SpatialDomain domain;
  SpatialConvex convex;
  float64 d = cos(gPi * rad/10800.0);
  SpatialConstraint c(SpatialVector(ra,dec),d);

  convex.add(c);
  domain.add(convex);
  domain.intersect(index_, idList_);

  range_.cut(range_.length());
  makeRange();

  return range_;
}

const ValVec<htmRange> & 
htmInterface::circleRegion( float64 x,
			    float64 y,
			    float64 z,
			    float64 rad ) {

  SpatialDomain domain;
  SpatialConvex convex;
  float64 d = cos(gPi * rad/10800.0);
  SpatialConstraint c(SpatialVector(x,y,z),d);

  convex.add(c);
  domain.add(convex);
  domain.intersect(index_, idList_);

  range_.cut(range_.length());
  makeRange();

  return range_;
}

const ValVec<htmRange> & 
htmInterface::circleRegionCmd( char *str ) {

  cmd_ = str;
  if(t_)delete t_;
  t_ = new VarStrToken(cmd_);

  float64 v[3];
  float64 d;

  cmdCode code = getCode();
  getDepth();
  if(! parseVec(code, v) )
    throw SpatialInterfaceError("htmInterface:circleRegionCmd: Expect vector in Command. ", cmd_.data());
  d = getFloat();

  if( code == J2000 )
    return circleRegion(v[0], v[1], d);

  return circleRegion(v[0], v[1], v[2], d);
}

//////////////////ConvexHull///////////////////////
const ValVec<htmRange> & 
htmInterface::convexHull( ValVec<float64> ra,
			  ValVec<float64> dec ) {

  if(ra.length() != dec.length())
    throw SpatialBoundsError("htmInterface:convexHull: ra and dec list are not equal size");

  polyCorners_.cut(polyCorners_.length());
  for(size_t i = 0; i < ra.length(); i++) {
    SpatialVector v(ra(i),dec(i));
    setPolyCorner(v);
  }

  return doHull();
}

const ValVec<htmRange> & 
htmInterface::convexHull( ValVec<float64> x,
			  ValVec<float64> y,
			  ValVec<float64> z ) {

  if(x.length() != y.length() || x.length() != z.length())
    throw SpatialBoundsError("htmInterface:convexHull: x,y,z lists are not equal size");

  polyCorners_.cut(polyCorners_.length());
  for(size_t i = 0; i < x.length(); i++) {
    SpatialVector v(x(i),y(i),z(i));
    setPolyCorner(v);
  }

  return doHull();
}

const ValVec<htmRange> & 
htmInterface::convexHullCmd( char *str ) {

  cmd_ = str;
  if(t_)delete t_;
  t_ = new VarStrToken(cmd_);

  float64 v[3];

  cmdCode code = getCode();
  getDepth();

  polyCorners_.cut(polyCorners_.length());

  // the next positions give the coordinate
  while(  parseVec( code, v ) ) {
    if(code == J2000) {
      SpatialVector tv(v[0],v[1]);
      setPolyCorner(tv);
    } else {
      SpatialVector tv(v[0],v[1],v[2]);
      setPolyCorner(tv);
    }
  }

  return doHull();
}


const ValVec<htmRange> &
htmInterface::doHull() {

  if(polyCorners_.length() < 3)
    throw SpatialInterfaceError("htmInterface:convexHull: empty hull: points on one line");

  SpatialVector v;
  SpatialConvex x;
  SpatialDomain d;

  // The constraint we have for each side is a 0-constraint (great circle)
  // passing through the 2 corners. Since we are in counterclockwise order,
  // the vector product of the two successive corners just gives the correct
  // constraint.
  size_t i, len = polyCorners_.length();
  for(i = 0; i < len; i++) {
    v = polyCorners_[i].c_ ^ polyCorners_[ i == len-1 ? 0 : i + 1].c_;
#ifdef DIAG
    cerr << v << " " << i << "," << i+1 << "\n";
#endif
    v.normalize();
    SpatialConstraint c(v,0);
    x.add(c);
  }
  d.add(x);
  d.intersect(index_, idList_);

  range_.cut(range_.length());
  makeRange();

  return range_;
}

//*******************************
//
// polygon processing : generate the convex
// hull of the points given in POLY - and then generate
// the proper x,y,z constraint.
//

// get corner - pop off two last items from stack, which must be
// numbers - update poly list.
void 
htmInterface::setPolyCorner(SpatialVector &v) {

    size_t i,len = polyCorners_.length();
    // test for already existing points
    for(i = 0; i < len; i++)
        if(v == polyCorners_[i].c_)return;

    if(len < 2) {
        // just append first two points.
        len = polyCorners_.insert(1);
        polyCorners_[len-1].c_ = v;
    } else if (len == 2) {
        // first polygon: triangle. set correct orientation.
        if( (polyCorners_[0].c_ ^ polyCorners_[1].c_)*v > 0 ) {
            polyCorners_.insert(1);
            polyCorners_[2].c_ = v;
        } else if( (polyCorners_[0].c_ ^ polyCorners_[1].c_)*v < 0 ) {
            polyCorners_.insert(1,1);
            polyCorners_[1].c_ = v;
        }
    } else {
        //
        // Now we set the flags for the existing polygon.
        // if the new point is inside (i.e. to the left) of 
        // the half-sphere defined by the points polyCorners_[i],[i+1]
        // we set polyCorners_[i].inside_ to true.
        //
        // if it is outside, and the previous side was also outside,
        // set the replace_ flag to true(this corner will be dropped)
        // (be careful on the edges - that's the trackoutside flag)

        bool polyTrackOutside = false;
        for(i = 0 ; i < len; i++) {
            polyCorners_[i].replace_ = false;
            polyCorners_[i].inside_  = false;

            // test if new point is inside the constraint given by a,b
            if( (polyCorners_[i].c_ ^ polyCorners_[i+1==len ? 0 : i+1].c_)*v > 0 ) {
                polyCorners_[i].inside_ = true;
                polyTrackOutside = false;
            } else {
                if(polyTrackOutside)
                    polyCorners_[i].replace_ = true;
                polyTrackOutside = true;
            }
        }
        if(polyTrackOutside && !polyCorners_[0].inside_)
            polyCorners_[0].replace_ = true;

#ifdef DIAG
        for(i = 0; i < len; i++)
            cerr << i << " : " 
                << (polyCorners_[i].replace_ ? "replace" : "keep   ")
                << (polyCorners_[i].inside_ ? "inside  : " : "outside : ")
                << polyCorners_[i].c_ << "\n";
#endif
        // now delete all corners that have the 'replace' flag set
        i = 0;
        while(i < len) {
            if(polyCorners_[i].replace_) {
                polyCorners_.remove(i); // remove returns new length
                len--;
            } else i++;
        }

        // now find first corner that is not inside (there is only one)
        // and insert the point after that.
        // if all points are inside we did nothing...

        for(i = 0; i < len; i++) {
            if(!polyCorners_[i].inside_) {
#ifdef DIAG
                cerr << "QL: Insert after " << i << " length = " << len << "\n";
#endif
                if(i == len-1) // append if last
                    polyCorners_.insert(1);
                else
                    polyCorners_.insert(1,len-i-1);
                polyCorners_[i+1].c_ = v;
                break;
            }
        }
    }
#ifdef DIAG
    cerr << "QL: Polygon: now " << polyCorners_.length() << "\n";
    for(i = 0; i < polyCorners_.length(); i++)
        cerr << polyCorners_[i].c_ << "\n";
#endif
}


void htmInterface::makeRange() {

  if(idList_.length() == 0)return;
  uint64 level = 1;
  size_t depth = index_->maxlevel_;
  size_t n =  (depth+2) * 2 - 1;
  level = level << n;

  htmRange r;
  size_t i,j=0;

  // make the first range
  r.lo = idList_(0);
  r.hi = r.lo;

  while ( (r.lo & level) == 0 ) {
    r.lo = r.lo << 2;
    r.hi = (r.hi << 2) + 3;
  }
  range_.append(r);


  for(i= 1; i < idList_.length(); i++ ) {
    r.lo = idList_(i);
    r.hi = r.lo;

    while ( (r.lo & level) == 0 ) {
      r.lo = r.lo << 2;
      r.hi = (r.hi << 2) + 3;
    }


    // handle overlapping ranges: next lo just after previous hi
    if(r.lo <= range_(j).hi + 1) {
      if(r.hi > range_(j).hi) // replace hi with new one if higher
	range_(j).hi = r.hi;
      continue;
    }

    j++;
    range_.append(r);
  }

}

//////////////////////////domain/////////////////////////
const ValVec<htmRange> & 
htmInterface::domain( SpatialDomain & domain ) {
  domain.intersect(index_, idList_);
  range_.cut(range_.length());
  makeRange();
  return range_;
}

const ValVec<htmRange> & 
htmInterface::domainCmd( char *str ) {

  cmd_ = str;
  if(t_)delete t_;
  t_ = new VarStrToken(cmd_);

  cmdCode code = getCode();
  if(code != HTMDOMAIN)
    throw SpatialInterfaceError("htmInterface:domainCmd: missing keyword HTMDOMAIN");
  getDepth();

  int32 nx,nc;
  nx = getInteger();

  SpatialDomain dom;
  for(int32 i = 0 ; i < nx; i++ ) {
    SpatialConvex convex;
    nc = getInteger();
    for(int32 j = 0; j < nc; j++ ) {
      float64 x = getFloat();
      float64 y = getFloat();
      float64 z = getFloat();
      float64 d = getFloat();
      SpatialConstraint c(SpatialVector(x,y,z),d);
      convex.add(c);
    }
    dom.add(convex);
  }

  return domain(dom);
}

