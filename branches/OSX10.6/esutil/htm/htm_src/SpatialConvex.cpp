//#     Filename:       SpatialConvex.cpp
//#
//#     The SpatialConvex
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
//#define DIAGNOSE
#include "SpatialConvex.h"

#define N(n)	index_->nodes_.vector_[(n)]		 // the node[n]
#define NC(n,m)	index_->nodes_.vector_[(n)].childID_[(m)]// the children n->m
#define NV(m)   index_->nodes_.vector_[id].v_[(m)]       // the vertices of n
#define V(m)    index_->vertices_.vector_[(m)]           // the vertex vector m

#define SGN(x) ( (x)<0? -1: ( (x)>0? 1:0 ) )		 // signum
#define IOFFSET 9
#define COMMENT '#'
// ===========================================================================
//
// Member functions for class SpatialConvex
//
// ===========================================================================


SpatialConvex::SpatialConvex()
{
}

/////////////COPY CONSTRUCTOR/////////////////////////////
//
SpatialConvex::SpatialConvex(const SpatialConvex & c) :
	index_(c.index_), 
	boundingCircle_(c.boundingCircle_), 
	addlevel_(c.addlevel_), 
	full_(c.full_),  
	partial_(c.partial_),  
	flist_(c.flist_), 
	plist_(c.plist_)
{
  constraints_ = c.constraints_;
  corners_ = c.corners_;
  bitresult_ = c.bitresult_;
  sign_ = c.sign_;
}

/////////////ASSIGNMENT///////////////////////////////////
//
SpatialConvex&
SpatialConvex::operator =(const SpatialConvex & c)
{
  if(&c == this)return *this;
  index_ = c.index_;
  addlevel_ = c.addlevel_;
  full_ = c.full_;
  partial_ = c.partial_;
  flist_ = c.flist_;
  plist_ = c.plist_;
  boundingCircle_ = c.boundingCircle_;
  constraints_ = c.constraints_;
  corners_ = c.corners_;
  bitresult_ = c.bitresult_;
  sign_ = c.sign_;
  return *this;
}

/////////////CONSTRUCTOR FROM A TRIANGLE//////////////////
//
// Initialize domain from a triangle. The corners of these vectors
// form a triangle, so we just add three ZERO convexes to the domain
// where the direction is given by the cross product of the corners.
// Of course, the sign has to be determined (on which side of the triangle
// are we?) If the three points lie on one line, no convexes are added.
//
SpatialConvex::SpatialConvex(const SpatialVector * v1,
			     const SpatialVector * v2,
			     const SpatialVector * v3)
{
  SpatialVector a1 = (*v2) ^ (*v3); // set directions of half-spheres
  SpatialVector a2 = (*v3) ^ (*v1);
  SpatialVector a3 = (*v1) ^ (*v2);
  float64 s1 = a1 * (*v1);          // we really need only the signs of these
  float64 s2 = a2 * (*v2);
  float64 s3 = a3 * (*v3);

  if(s1 * s2 * s3) {                // this is nonzero if not on one line
    if(s1 < 0.0L) a1 = (-1) * a1 ;  // change sign if necessary
    if(s2 < 0.0L) a2 = (-1) * a2 ;
    if(s3 < 0.0L) a3 = (-1) * a3 ;
    constraints_.append(SpatialConstraint(a1,0.0)); // we don't care about the
    constraints_.append(SpatialConstraint(a2,0.0)); // order since all angles are
    constraints_.append(SpatialConstraint(a3,0.0)); // 90 degrees.
  }
  sign_ = zERO;
}

/////////////CONSTRUCTOR FROM A RECTANGLE/////////////////
//
// Initialize convex from a rectangle. The vectors that form a rectangle
// may be in any order, the code finds the edges by itself.
// If one of the vectors lies within the triangle formed by the
// other three vectors, the previous constructor is used.
//
SpatialConvex::SpatialConvex(const SpatialVector * v1,
			     const SpatialVector * v2,
			     const SpatialVector * v3,
			     const SpatialVector * v4)
{
  int i,j,k,l,m;  // indices
  // to simplify things, copy input into a 4-array
  const SpatialVector *v[4] = {v1,v2,v3,v4};
  SpatialVector d[6];
  float64 s[6][2];
  for (i = 0, k = 0; i < 4 ; i++)
    for (j = i+1; j < 4; j++, k++) {    // set directions of half-spheres
      d[k] = (*v[i]) ^ (*v[j]);    // two of these are diagonals.
      d[k].normalize();
      for (l = 0, m = 0; l < 4; l++)
	if(l != i && l != j)s[k][m++] = d[k] * (*v[l]); // set the 'sign'
    }

  // the sides are characterized by having both other corners
  // to the same (inner) side. so it is easy to find the edges.
  // again, the sign has to be taken care of -> direction of d
  // the nice thing here is that if one of the corners is inside
  // a triangles formed by the other three, only 3 constraints get
  // added.
  for(i = 0; i < 6; i++)
    if(s[i][0] * s[i][1] > 0.0) // not >= because we don't want aligned corners
      constraints_.append(SpatialConstraint((s[i][0] > 0.0 ? 
					        d[i] : (-1 * d[i])),
					    0.0));

  // Special cases: 1
  // if three of the corners are aligned, we end up with
  // only two constraints. Find the third and append it.
  // Indeed, there are 3 identical constraints among the d[],
  // so the first that qualifies gets appended.
  if(constraints_.length() == 2) {
    for(i = 0; i < 6; i++)
      if(s[i][0] == 0.0 || s[i][1] == 0.0) {
	constraints_.append(SpatialConstraint( ((s[i][0]+s[i][1]) > 0.0 ? 
					          d[i] : (-1 * d[i])), 
					       0.0));
	break;
      }
  }
  // Special cases: 2
  // if all four corners are aligned, no constraints have been appended.
  sign_ = zERO;
}

/////////////ADD//////////////////////////////////////////
//
void
SpatialConvex::add(SpatialConstraint & c)
{
  constraints_.append(c);
  // order constraints: by ascending opening angle. Since we append
  // always at the end, we only need one ordering sweep starting at
  // the end
  for ( size_t i = constraints_.length() - 1; i > 0; i-- ) {
    if ( constraints_.vector_[i].s_ <  constraints_.vector_[i-1].s_ ) {
      SpatialConstraint tmp( constraints_.vector_[i] );
      constraints_.vector_[i] = constraints_.vector_[i-1];
      constraints_.vector_[i-1] = tmp;
    }
  }

  if(constraints_.length() == 1) {  // first constraint
    sign_ = c.sign_;
    return;
  }

  switch (sign_) {
  case nEG:
    if(c.sign_ == pOS) sign_ = mIXED;
    break;
  case pOS:
    if(c.sign_ == nEG) sign_ = mIXED;
    break;
  case zERO:
    sign_ = c.sign_;
    break;
  case mIXED:
	break;
  }
}


/////////////SIMPLIFY0////////////////////////////////////
// simplify0: simplify zERO convexes. calculate corners of convex
// and the bounding circle.
//
// zERO convexes are made up of constraints which are all great
// circles. It can happen that some of the constraints are redundant.
// For example, if 3 of the great circles define a triangle as the convex
// which lies fully inside the half sphere of the fourth constraint,
// that fourth constraint is redundant and will be removed.
//
// The algorithm is the following:
//
// zero-constraints are half-spheres, defined by a single normalized
// vector v, pointing in the direction of that half-sphere.
//
// Two zero-constraints intersect at 
//
//    i    =  +- v  x v
//     1,2        1    2
//
// the vector cross product of their two defining vectors. 
//
// The two vectors i1,2 are tested against every other constraint in
// the convex if they lie within their half-spheres. Those
// intersections i which lie within every other constraint, are stored
// into corners_.
//
// Constraints that do not have a single corner on them, are dropped.
//

void
SpatialConvex::simplify0() {

  size_t i,j,k;
  SpatialVector vi1, vi2;
  ValVec<size_t> cornerConstr1, cornerConstr2, removeConstr;
  ValVec<SpatialVector> corner;
  if (constraints_.length() == 1) { // for one constraint, it is itself the BC
    boundingCircle_ = constraints_(0);
    return;
  // For 2 constraints, take the bounding circle a 0-constraint...
  // this is by no means optimal, but the code is optimized for at least
  // 3 zERO constraints... so this is acceptable.
  } else if(constraints_.length() == 2) {
    // test for constraints being identical - rule 1 out
    if(constraints_.vector_[0].a_ == constraints_.vector_[1].a_){
      constraints_.cut(1);
      boundingCircle_ = constraints_(0);
      return;
    }
    // test for constraints being two disjoint half spheres - empty convex!
    if(constraints_.vector_[0].a_ == (-1.0)*constraints_.vector_[1].a_){
      constraints_.cut(constraints_.length());
      return;
    }
    boundingCircle_ = SpatialConstraint(constraints_(0).v() + 
					constraints_(1).v(),0);
    return;
  }

  // Go over all pairs of constraints
  for(i = 0; i < constraints_.length() - 1; i++) {
    bool ruledout = true;
    for(j = i+1; j < constraints_.length(); j ++) {
      // test for constraints being identical - rule i out
      if(constraints_.vector_[i].a_ == constraints_.vector_[j].a_)break;
      // test for constraints being two disjoint half spheres - empty convex!
      if(constraints_.vector_[i].a_ == (-1.0)*constraints_.vector_[j].a_){
	constraints_.cut(constraints_.length());
	return;
      }
      // vi1 and vi2 are their intersection points
      vi1 = constraints_.vector_[i].a_ ^ constraints_.vector_[j].a_ ;
      vi1.normalize();
      vi2 = (-1.0) * vi1;
      bool vi1ok = true, vi2ok = true;
      // now test whether vi1 or vi2 or both are inside every other constraint.
      // if yes, store them in the corner array.
      for(k = 0; k < constraints_.length(); k++) {
	if(k == i || k == j) continue;
	if(vi1ok && vi1 * constraints_.vector_[k].a_ <= 0.0) vi1ok = false;
	if(vi2ok && vi2 * constraints_.vector_[k].a_ <= 0.0) vi2ok = false;
	if(!vi1ok && !vi2ok)break;
      }
      if(vi1ok) { 
	corner.append(vi1); 
	cornerConstr1.append(i);
	cornerConstr2.append(j);
	ruledout = false; 
      }
      if(vi2ok) { 
	corner.append(vi2); 
	cornerConstr1.append(i);
	cornerConstr2.append(j);
	ruledout = false; 
      }
    }
    // is this constraint ruled out? i.e. none of its intersections
    // with other constraints are corners... remove it from constraints_ list.
    if(ruledout) removeConstr.append(i);
  }

  // Now set the corners into their correct order, which is an
  // anti-clockwise walk around the polygon.
  //
  // start at any corner. so take the first.

  corners_.cut(corners_.length());
  corners_.append(corner(0));
  // The trick is now to start off into the correct direction.
  // this corner has two edges it can walk. we have to take the
  // one where the convex lies on its left side.
  i = cornerConstr1(0);		// the i'th constraint and j'th constraint
  j = cornerConstr2(0);		// intersect at 0'th corner
  size_t c1=0,c2=0,k1=0,k2=0;
  // Now find the other corner where the i'th and j'th constraints intersect.
  // Store the corner in vi1 and vi2, and the other constraint indices 
  // in c1,c2.
  for( k = 1; k < cornerConstr1.length(); k ++) {
    if(cornerConstr1(k) == i) {
      vi1 = corner(k);
      c1 = cornerConstr2(k);
      k1 = k;
    }
    if(cornerConstr2(k) == i) {
      vi1 = corner(k);
      c1 = cornerConstr1(k);
      k1 = k;
    }
    if(cornerConstr1(k) == j) {
      vi2 = corner(k);
      c2 = cornerConstr2(k);
      k2 = k;
    }
    if(cornerConstr2(k) == j) {
      vi2 = corner(k);
      c2 = cornerConstr1(k);
      k2 = k;
    }
  }
  // Now test i'th constraint-edge ( corner 0 and corner k ) whether
  // it is on the correct side (left)
  //
  //  ( (corner(k) - corner(0)) x constraint(i) ) * corner(0)
  //
  // is >0 if yes, <0 if no...
  //
  size_t c,currentCorner;
  if( ((vi1 - corner(0)) ^ constraints_(i).a_) * corner(0) > 0 ) {
    corners_.append(vi1);
    c = c1;
    currentCorner = k1;
  } else {
    corners_.append(vi2);
    c = c2;
    currentCorner = k2;
  }
  // now append the corners that match the index c until we got corner 0 again
  // currentCorner holds the current corners index
  // c holds the index of the constraint that has just been intersected with
  // So:
  // x We are on a constraint now (i or j from before), the second corner
  //   is the one intersecting with constraint c.
  // x Find other corner for constraint c.
  // x Save that corner, and set c to the constraint that intersects with c
  //   at that corner. Set currentcorner to that corners index.
  // x Loop until 0th corner reached.
  while( currentCorner ) {
    for (k = 0; k < cornerConstr1.length(); k++) {
      if(k == currentCorner)continue;
      if(cornerConstr1(k) == c) {
	if( (currentCorner = k) == 0) break;
	corners_.append(corner(k));
	c = cornerConstr2(k);
	break;
      }
      if(cornerConstr2(k) == c) {
	if( (currentCorner = k) == 0) break;
	corners_.append(corner(k));
	c = cornerConstr1(k);
	break;
      }
    }
  }
  // Remove all redundant constraints
  for ( i = 0; i < removeConstr.length(); i++)
    constraints_.remove(removeConstr(i));

  // Now calculate the bounding circle for the convex.
  // We take it as the bounding circle of the triangle with
  // the widest opening angle. All triangles made out of 3 corners
  // are considered.
  boundingCircle_.d_ = 1.0;
  if (constraints_.length() >=3 ) {
    for(i = 0; i < corners_.length(); i++)
      for(j = i+1; j < corners_.length(); j++)
	for(k = j+1; k < corners_.length(); k++) {
	  SpatialVector v = ( corners_(j) - corners_(i) ) ^ 
	                    ( corners_(k) - corners_(j) );
	  v.normalize();
	  // Set the correct opening angle: Since the plane cutting
	  // out the triangle also correctly cuts out the bounding cap
	  // of the triangle on the sphere, we can take any corner to
	  // calculate the opening angle
	  float64 d = v * corners_(i);
	  if(boundingCircle_.d_ > d) boundingCircle_ = SpatialConstraint(v,d);
	}
  }

#ifdef DIAGNOSE
  for(i = 0; i < corners_.length(); i++) {
    //cout << corners_(i).ra() << "," << corners_(i).dec() << ":" << corners_(i) << "\n";
  }
#endif
    
}

/////////////SIMPLIFY/////////////////////////////////////
// simplify: We have the following decision tree for the 
//           simplification of convexes:
//
//  Always test two constraints against each other. We have
//
//  * If both constraints are pOS
//
//     # If they intersect: keep both
//
//     # If one lies in the other: drop the larger one
//
//     # Else: disjunct. Empty convex, stop.
//
//  * If both constraints are nEG
//
//     # If they intersect or are disjunct: ok
//
//     # Else: one lies in the other, drop smaller 'hole'
//
//  * Mixed: one pOS, one nEG
//
//     # No intersection, disjunct: pOS is redundant
//
//     # Intersection: keep both
//
//     # pOS within nEG: empty convex, stop.
//
//     # nEG within pOS: keep both.
//

void
SpatialConvex::simplify() {

  if(sign_ == zERO) {
    simplify0();	// treat zERO convexes separately
    return;
  }

  size_t i,j;
  size_t clen;
  bool redundancy = true;

  while(redundancy) {
    redundancy = false;
    clen = constraints_.length();

  for(i = 0; i < clen; i++) {
    for(j = 0; j < i; j++) {
      int test;

      // don't bother with two zero constraints
      if( constraints_[i].sign_ == zERO && constraints_[j].sign_ == zERO)
	continue;

      // both pos or zero
      if( ( constraints_[i].sign_ == pOS || constraints_[i].sign_ == zERO ) &&
	  ( constraints_[j].sign_ == pOS || constraints_[j].sign_ == zERO ) ) {
	if ( (test = testConstraints(i,j)) == 0 ) continue; // intersection
	if ( test < 0 ) { // disjoint ! convex is empty
	  constraints_.cut(constraints_.length());
	  return;
	}
	// one is redundant
	if(test == 1)   constraints_.cut(1, clen - 1 - i);
	else if(test==2)constraints_.cut(1, clen - 1 - j);
	else continue;     // intersection
	redundancy = true; // we did cut out a constraint -> do the loop again
	break;
      }

      // both neg or zero
      if( ( constraints_[i].sign_ == nEG ) &&
	  ( constraints_[j].sign_ == nEG ) ) {
	if ( (test = testConstraints(i,j)) <= 0 ) continue; // ok
	// one is redundant
	if(test == 1)   constraints_.cut(1, clen - 1 - j);
	else if(test==2)constraints_.cut(1, clen - 1 - i);
	else continue; // intersection
	redundancy = true; // we did cut out a constraint -> do the loop again
	break;
      }

      // one neg, one pos/zero
      if( (test = testConstraints(i,j)) == 0) continue; // ok: intersect
      if( test < 0 ) { // neg is redundant
	if ( constraints_[i].sign_ == nEG ) constraints_.cut(1, clen - 1 - i);
	else    constraints_.cut(1, clen - 1 - j);
	redundancy = true; // we did cut out a constraint -> do the loop again
	break;
      }
      // if the negative constraint is inside the positive: continue
      if ( (constraints_[i].sign_ == nEG && test == 2) || 
	   (constraints_[j].sign_ == nEG && test == 1) )continue;
      // positive constraint in negative: convex is empty!
      constraints_.cut(constraints_.length());
      return;
    }
    if(redundancy)break;
  }

  }

  // reset the sign of the convex
  sign_ = constraints_[0].sign_;
  for(i = 1; i < constraints_.length(); i++) {
    switch (sign_) {
    case nEG:
      if(constraints_[i].sign_ == pOS) sign_ = mIXED;
      break;
    case pOS:
      if(constraints_[i].sign_ == nEG) sign_ = mIXED;
      break;
    case zERO:
      sign_ = constraints_[i].sign_;
      break;
    case mIXED:
	  break;
    }
  }

  if (constraints_.length() == 1) // for one constraint, it is itself the BC
    boundingCircle_ = constraints_(0);
  else if (sign_ == pOS)
    boundingCircle_ = constraints_(0);
    
}

/////////////TESTCONSTRAINTS//////////////////////////////
// testConstraints: Test for the relative position of two constraints.
//                  Returns 0  if they intersect
//                  Returns -1 if they are disjoint
//                  Returns 1  if j is in i
//                  Returns 2  if i is in j
//
int
SpatialConvex::testConstraints(size_t i, size_t j) {

  float64 phi = (
	 (constraints_[i].sign_ == nEG ? (-1 * constraints_[i].a_):
	                               constraints_[i].a_ )
	  *
         (constraints_[j].sign_ == nEG ? (-1 * constraints_[j].a_):
	                               constraints_[j].a_ )
	        );
  phi = (phi <= -1.0L + gEpsilon ? gPi : acos(phi)) ; // correct for math lib -1.0
  float64 a1 = (constraints_[i].sign_ == pOS ? 
		constraints_[i].s_ : gPi-constraints_[i].s_);
  float64 a2 = (constraints_[j].sign_ == pOS ? 
		    constraints_[j].s_ : gPi-constraints_[j].s_);

  if ( phi > a1 + a2 ) return -1;
  if ( a1 > phi + a2 ) return 1;
  if ( a2 > phi + a1 ) return 2;
  return 0;
}

/////////////INTERSECT////////////////////////////////////
//
void
SpatialConvex::intersect(const SpatialIndex * idx,
			 BitList * partial, BitList * full) {
  index_ = idx;
  addlevel_ = idx->maxlevel_ - idx->buildlevel_;
  partial_ = partial;
  full_ = full;
  bitresult_ = true;
  range_ = false;

  doIntersect();
}

/////////////INTERSECT////////////////////////////////////
//
void
SpatialConvex::intersect(const SpatialIndex * idx,
			 ValVec<uint64> * partial, ValVec<uint64> * full) {
  index_ = idx;
  addlevel_ = idx->maxlevel_ - idx->buildlevel_;
  plist_ = partial;
  flist_ = full;
  bitresult_ = false;
  range_ = false;

  doIntersect();
}

/////////////INTERSECT////////////////////////////////////
//
void
SpatialConvex::intersect(const SpatialIndex * idx,
			 ValVec<uint64> * idList) {

  index_ = idx;
  addlevel_ = idx->maxlevel_ - idx->buildlevel_;
  plist_ = idList;
  bitresult_ = false;
  range_ = true;

  doIntersect();
}

/////////////DOINTERSECT//////////////////////////////////
//
void
SpatialConvex::doIntersect() {

  simplify();				// don't work too hard...

  if(constraints_.length()==0)return;   // nothing to intersect!!

  // Start with root nodes (index = 1-8) and intersect triangles
  for(uint32 i = 1; i <= 8; i++)
    triangleTest(i);

}

/////////////TRIANGLETEST/////////////////////////////////
// triangleTest: this is the main test of a triangle vs a Convex.  It
// will properly mark up the flags for the triangular node[index], and
// all its children

SpatialMarkup
SpatialConvex::triangleTest(uint64 id)
{
  SpatialMarkup mark;
//
// do the face test on the triangle

  mark =  testNode(V(NV(0)),V(NV(1)),V(NV(2)));

// do we have a final result code?
// if rEJECT, fULL then return

  if(mark > fULL) return mark;

  if(mark == fULL) {
      fillChildren(id); // propagate final result to children
      return mark;
  }

// if pARTIAL or dONTKNOW, then continue, test children,
//    but do not reach beyond the leaf nodes.
//    If Convex is fully contained within one (sWALLOWED),
//    we can stop looking further in another child

  if (NC(id,0)!=0) {
    triangleTest(NC(id,0));
    triangleTest(NC(id,1));
    triangleTest(NC(id,2));
    triangleTest(NC(id,3));
// we are at the leafnodes
// If we have to recurse further, calculate intersections one by one
// If not, just set the proper bit in partial_ or append id to plist_.
  } else {
    if(addlevel_) {
      // from now on, continue to build the triangles dynamically.
      // until maxlevel_ levels depth.
      testPartial(addlevel_, N(id).id_, V(NV(0)), V(NV(1)), V(NV(2)));

    } else {
      if(bitresult_)
	partial_->set((uint32)index_->leafNumberById(N(id).id_),true);
      else
	plist_->append(N(id).id_);
    }
  }

  return mark;
}

/////////////FILLCHILDREN/////////////////////////////////
// fillChildren: mark children as full
//
void
SpatialConvex::fillChildren(uint64 id) {
  if(range_)
    plist_->append(N(id).id_);
  else {
    if(NC(id,0)!=0) {
      for(size_t i = 0; i < 4; i++) {
	fillChildren(NC(id,i));
      }
    } else {
      // we are at the leaf. If we still have levels to recurse,
      // fill them. If not, just set the full_ bitlist's or flist_ list's
      // value correctly.
      if(addlevel_)
	setfull(N(id).id_,addlevel_);
      else {
	if(bitresult_)
	  full_->set((uint32)index_->leafNumberById(N(id).id_), true);
	else
	  flist_->append(N(id).id_);
      }
    }
  }
}

/////////////SETFULL//////////////////////////////////////
// setfull: set the bitlist leaves at level maxlevel to full.
// if we have still levels to go, recurse. Use the id to get
// the leaf node's index. See idbyname and namebyid for explanations.
//
void
SpatialConvex::setfull(uint64 id, size_t level) {
  if(level--) {
    setfull(id << 2    , level);
    setfull((id << 2) + 1, level);
    setfull((id << 2) + 2, level);
    setfull((id << 2) + 3, level);
  } else {
    if(bitresult_)
      full_->set((uint32)index_->leafNumberById(id), true);
    else
      flist_->append(id);
  }
}

/////////////TESTPARTIAL//////////////////////////////////
// testPartial: test a triangle's subtriangle whether they are partial.
// if level is nonzero, recurse.
//
void
SpatialConvex::testPartial(size_t level, uint64 id,
			   const SpatialVector & v0, 
			   const SpatialVector & v1, 
			   const SpatialVector & v2) {

  // if there is still a level to go, subdivide the
  // triangle according to our rules and test each subdivision.
  // (our rules are: each subdivided triangle has to be given
  // ordered counter-clockwise, 0th index starts of new 0-node,
  // 1st index starts off new 1-node, 2nd index starts off new 2-node
  // middle triangle gives new 3-node.
  // if we are at the bottom, set this id to partial.
  if(level--) {
    SpatialVector w0 = v1 + v2; w0.normalize();
    SpatialVector w1 = v0 + v2; w1.normalize();
    SpatialVector w2 = v1 + v0; w2.normalize();

    testSubTriangle(level, (id << 2)    , v0, w2, w1);
    testSubTriangle(level, (id << 2) + 1, v1, w0, w2);
    testSubTriangle(level, (id << 2) + 2, v2, w1, w0);
    testSubTriangle(level, (id << 2) + 3, w0, w1, w2);
  } else {
    if(bitresult_)
      partial_->set((uint32)index_->leafNumberById(id), true);
    else
      plist_->append(id);
  }
}

/////////////TESTSUBTRIANGLE////////////////////////////////
// testSubTriangle: call full or partial depending on result of testNode.
//
void
SpatialConvex::testSubTriangle(size_t level, uint64 id,
			   const SpatialVector & v0, 
			   const SpatialVector & v1, 
			   const SpatialVector & v2) {

  // test this triangle.
  SpatialMarkup mark = testNode(v0, v1, v2);

  // if it is full, set all fulls below this level, too
  // else if it is partial or unknown or swallowed call testpartial
  // with this new level.
  if(mark == fULL) {
    if(range_)
      plist_->append(id);
    else
      setfull(id , level);
  } else if(mark < fULL)
    testPartial(level, id, v0, v1, v2);
}

/////////////TESTNODE/////////////////////////////////////
// testNode: tests the QuadNodes for intersections.
//
SpatialMarkup
SpatialConvex::testNode(const SpatialVector & v0, 
			const SpatialVector & v1, 
			const SpatialVector & v2) {
  // Start with testing the vertices for the QuadNode with this convex.

  int vsum = testVertex(v0) + testVertex(v1) + testVertex(v2);

#ifdef DIAGNOSE
  char name[10];
  SpatialVector v = v0 + v1 + v2;
  //cout << index_->nameById(index_->idByPoint(v),name)
  //   << " " << vsum << " " << "\n";
#endif

  SpatialMarkup mark = 
    testTriangle( v0, v1, v2, vsum);


#ifdef DIAGNOSE
  //cout << ( mark == pARTIAL ? " partial " : 
  //	    ( mark ==  fULL ? " full " :
  //	      ( mark == rEJECT ? " reject " :
  //		" dontknow " ) ) ) << name << "\n";
  
  //<< v0 << "," << v1 << "," << v2 << " " << "\n";
  //<< V(NV(0)) << " , " << V(NV(1)) << " , " << V(NV(2)) << "\n"
  //<< " (" << V(NV(0)).ra() << "," << V(NV(0)).dec() << ")"
  //<< " (" << V(NV(1)).ra() << "," << V(NV(1)).dec() << ")"
  //<< " (" << V(NV(2)).ra() << "," << V(NV(2)).dec() << ")"
  //<< "\n";
  
#endif

  // since we cannot play games using the on-the-fly triangles,
  // substitute dontknow with partial.
  if (mark == dONTKNOW) 
    mark = pARTIAL;

  return mark;
}

/////////////TESTTRIANGLE//////////////////////////////////
// testTriangle: tests a triangle given by 3 vertices if
// it intersects the convex.
//
SpatialMarkup
SpatialConvex::testTriangle(const SpatialVector & v0, 
			    const SpatialVector & v1, 
			    const SpatialVector & v2,
			    int vsum) {

  if(vsum == 1 || vsum == 2) return pARTIAL;

  // If vsum = 3 then we have all vertices inside the convex.
  // Now use the following decision tree:
  //
  // * If the sign of the convex is pOS or zERO : mark as fULL intersection.
  //
  // * Else, test for holes inside the triangle. A 'hole' is a nEG constraint
  //   that has its center inside the triangle. If there is such a hole,
  //   return pARTIAL intersection.
  //
  // * Else (no holes, sign nEG or mIXED) test for intersection of nEG
  //   constraints with the edges of the triangle. If there are such,
  //   return pARTIAL intersection.
  //
  // * Else return fULL intersection.

  if(vsum == 3) {
    if(sign_ == pOS || sign_ == zERO) return fULL;
    if ( testHole(v0,v1,v2) ) return pARTIAL;
    if ( testEdge(v0,v1,v2) ) return pARTIAL;
    return fULL;
  }

  // If we have reached that far, we have vsum=0. There is no definite
  // decision making possible here with our methods, the markup may result
  // in dONTKNOW. The decision tree is the following:
  //
  // * Test with bounding circle of the triangle.
  //
  //   # If the sign of the convex zERO test with the precalculated
  //     bounding circle of the convex. If it does not intersect with the
  //     triangle's bounding circle, rEJECT.
  //
  //   # If the sign of the convex is nonZERO: if the bounding circle
  //     lies outside of one of the constraints, rEJECT.
  //
  // * Else: there was an intersection with the bounding circle.
  //
  //   # For zERO convexes, test whether the convex intersects the edges.
  //     If none of the edges of the convex intersects with the edges of
  //     the triangle, we have a rEJECT. Else, pARTIAL.
  //
  //   # If sign of convex is pOS, or miXED and the smallest constraint does
  //     not intersect the edges and has its center inside the triangle,
  //     return sWALLOW. If no intersection of edges and center outside
  //     triangle, return rEJECT.
  //
  //   # So the smallest constraint DOES intersect with the edges. If
  //     there is another pOS constraint which does not intersect with
  //     the edges, and has its center outside the triangle, return
  //     rEJECT. If its center is inside the triangle return sWALLOW. 
  //     Else, return pARTIAL for pOS and dONTKNOW for mIXED signs. 
  //
  // * If we are here, return dONTKNOW. There is an intersection with 
  //   the bounding circle, none of the vertices is inside the convex and
  //   we have very strange possibilities left for pOS and mIXED signs. For
  //   nEG, i.e. all constraints negative, we also have some complicated
  //   things left for which we cannot test further.

  if ( !testBoundingCircle(v0,v1,v2) ) return rEJECT;

  if ( sign_ == pOS || sign_ == mIXED || (sign_ == zERO && constraints_.length() <= 2)) {
	  // Does the smallest constraint intersect with the edges?
	  if ( testEdgeConstraint(v0,v1,v2,0) ) {
		  // Is there another positive constraint that does NOT intersect with
		  // the edges?
		  size_t cIndex;
		  if ( cIndex = testOtherPosNone(v0,v1,v2) ) {
			  // Does that constraint lie inside or outside of the triangle?
			  if ( testConstraintInside(v0,v1,v2, cIndex) ) {
				  return pARTIAL;
			  }
			  // Does the triangle lie completely within that constr?
			  else if( constraints_.vector_[cIndex].contains(v0) ) {
				  return pARTIAL;
			  }
			  else {
				  return rEJECT;
			  }

		  } else {
			  if(sign_ == pOS || sign_ == zERO) return pARTIAL;
			  else return dONTKNOW;
		  }	
	  } else {
		  if (sign_ == pOS || sign_ == zERO) {
			  // Does the smallest lie inside or outside the triangle?
			  if( testConstraintInside(v0,v1,v2, 0) ) 
				  return pARTIAL;
			  else return rEJECT;
		  } else return  dONTKNOW;
	  }
  } else if (sign_ == zERO) {
	  if ( corners_.length() > 0 && testEdge0(v0,v1,v2) ) 
		  return pARTIAL;
	  else return rEJECT;
  }
  return pARTIAL;
}

/////////////TESTVERTEX/////////////////////////////////////
// testVertex: same as above, but for any spatialvector, no markup speedup
//
int
SpatialConvex::testVertex(const SpatialVector & v)
{
  for ( size_t i = 0; i < constraints_.length(); i++) 
    if ( (constraints_.vector_[i].a_ * v )  < constraints_.vector_[i].d_ )
      return 0;

  return 1;
}

/////////////TESTHOLE/////////////////////////////////////
// testHole: test for holes. If there is a negative constraint whose center
//           is inside the triangle, we speak of a hole. Returns true if
//	     found one.
//
bool
SpatialConvex::testHole(const SpatialVector & v0, 
			const SpatialVector & v1, 
			const SpatialVector & v2) {

  bool test = false;

  for(size_t i = 0; i < constraints_.length(); i++) {

    if ( constraints_.vector_[i].sign_ == nEG ) {  // test only 'holes'

      // If (a ^ b * c) < 0, vectors abc point clockwise.
      // -> center c not inside triangle, since vertices a,b are ordered
      // counter-clockwise. The comparison here is the other way
      // round because c points into the opposite direction as the hole

      if ( ( ( v0 ^ v1 ) * 
	     constraints_.vector_[i].a_) > 0.0L ) continue;
      if ( ( ( v1 ^ v2 ) *
	     constraints_.vector_[i].a_) > 0.0L ) continue;
      if ( ( ( v2 ^ v0 ) * 
	     constraints_.vector_[i].a_) > 0.0L ) continue;
      test = true;
      break;
    }
  }
  return test;
}

/////////////TESTEDGE0////////////////////////////////////
// testEdge0: test if the edges intersect with the zERO convex.
//            The edges are given by the vertex vectors e[0-2]
//	      All constraints are great circles, so test if their intersect
//            with the edges is inside or outside the convex.
//            If any edge intersection is inside the convex, return true.
//            If all edge intersections are outside, check whether one of
//            the corners is inside the triangle. If yes, all of them must be
//            inside -> return true.
//
bool
SpatialConvex::testEdge0(const SpatialVector & v0, 
			 const SpatialVector & v1, 
			 const SpatialVector & v2) {
  // We have constructed the corners_ array in a certain direction.
  // now we can run around the convex, check each side against the 3
  // triangle edges. If any of the sides has its intersection INSIDE
  // the side, return true. At the end test if a corner lies inside
  // (because if we get there, none of the edges intersect, but it
  // can be that the convex is fully inside the triangle. so to test
  // one single edge is enough)

  struct edgeStruct {
    SpatialVector e;		// The half-sphere this edge delimits
    float64	  l;		// length of edge
    const SpatialVector *e1;	// first end
    const SpatialVector *e2;	// second end
  } edge[3];

  // fill the edge structure for each side of this triangle
  edge[0].e = v0 ^ v1; edge[0].e1 = &v0; edge[0].e2 = &v1;
  edge[1].e = v1 ^ v2; edge[1].e1 = &v1; edge[1].e2 = &v2;
  edge[2].e = v2 ^ v0; edge[2].e1 = &v2; edge[2].e2 = &v0;
  edge[0].l = acos(v0 * v1);
  edge[1].l = acos(v1 * v2);
  edge[2].l = acos(v2 * v0);

  for(size_t i = 0; i < corners_.length(); i++) {
    size_t j = 0;
    if(i < corners_.length() - 1) j = i+1;
    SpatialVector a1;
    float64 l1,l2;   // lengths of the arcs from intersection to edge corners
    float64 cedgelen = acos(corners_(i) * corners_(j));  // length of edge of convex

    // calculate the intersection - all 3 edges
    for (size_t iedge = 0; iedge < 3; iedge++) {
      a1 = ( edge[iedge].e ) ^ ( corners_(i) ^ corners_(j) );
      a1.normalize();
      // if the intersection a1 is inside the edge of the convex,
      // its distance to the corners is smaller than the edgelength.
      // this test has to be done for both the edge of the convex and
      // the edge of the triangle.
      for(size_t k = 0; k < 2; k++) {
	l1 = acos(corners_(i) * a1);
	l2 = acos(corners_(j) * a1);
	if( l1 - cedgelen <= gEpsilon && l2 - cedgelen <= gEpsilon ) {
	  l1 = acos( *(edge[iedge].e1) * a1 );
	  l2 = acos( *(edge[iedge].e2) * a1 );
	  if( l1 - edge[iedge].l <= gEpsilon && 
	      l2 - edge[iedge].l <= gEpsilon ) 
	    return true;
	}
	a1 *= -1.0; // do the same for the other intersection
      }
    }
  }
  return testVectorInside(v0,v1,v2,corners_(0));
}

/////////////TESTEDGE/////////////////////////////////////
// testEdge: test if edges intersect with constraint. This problem
//           is solved by a quadratic equation. Return true if there is
//	     an intersection.
//
bool
SpatialConvex::testEdge(const SpatialVector & v0, 
			const SpatialVector & v1, 
			const SpatialVector & v2) {

  for(size_t i = 0; i < constraints_.length(); i++) {

    if ( constraints_.vector_[i].sign_ == nEG ) {  // test only 'holes'
      if ( eSolve(v0, v1, i) ) return true;
      if ( eSolve(v1, v2, i) ) return true;
      if ( eSolve(v2, v0, i) ) return true;
    }
  }
  return false;
}

/////////////ESOLVE///////////////////////////////////////
// eSolve: solve the quadratic eq. for intersection of an edge with a circle
//         constraint. Edge given by grand circle running through v1, v2
//         Constraint given by cIndex.
bool
SpatialConvex::eSolve(const SpatialVector & v1, 
		      const SpatialVector & v2, size_t cIndex)
{
  float64 gamma1 = v1 * constraints_.vector_[cIndex].a_ ;
  float64 gamma2 = v2 * constraints_.vector_[cIndex].a_ ;
  float64 mu     = v1 * v2;
  float64 u2     = (1 - mu) / (1 + mu);

  float64 a      = - u2 * (gamma1 + constraints_.vector_[cIndex].d_);
  float64 b      = gamma1 * ( u2 - 1 ) + gamma2 * ( u2 + 1 );
  float64 c      = gamma1 - constraints_.vector_[cIndex].d_;

  float64 D      = b * b - 4 * a * c;

  if( D < 0.0L ) return false; // no intersection

  // calculate roots a'la Numerical Recipes

  float64 q      = -0.5L * ( b + ( SGN(b) * sqrt(D) ) );

  float64 root1=0, root2=0;
  int i = 0;

  if ( a > gEpsilon || a < -gEpsilon ) { root1 = q / a; i++; }
  if ( q > gEpsilon || q < -gEpsilon ) { root2 = c / q; i++; }

  // Check whether the roots lie within [0,1]. If not, the intersection
  // is outside the edge.

  if (i == 0) return false; // no solution
  if ( root1 >= 0.0L && root1 <= 1.0L ) return true;
  if ( i == 2 && ( (root1 >= 0.0L && root1 <= 1.0L ) ||
		   (root2 >= 0.0L && root2 <= 1.0L ) ) ) return true;

  return false;
}

/////////////TESTBOUNDINGCIRCLE///////////////////////////
// testBoundingCircle: test for boundingCircles intersecting with constraint
//
bool
SpatialConvex::testBoundingCircle(const SpatialVector & v0, 
				  const SpatialVector & v1, 
				  const SpatialVector & v2) {

  // Set the correct direction: The normal vector to the triangle plane
  SpatialVector c = ( v1 - v0 ) ^ ( v2 - v1 );
  c.normalize();

  // Set the correct opening angle: Since the plane cutting out the triangle
  // also correctly cuts out the bounding cap of the triangle on the sphere,
  // we can take any corner to calculate the opening angle
  float64 d = acos (c * v0);

  // for zero convexes, we have calculated a bounding circle for the convex.
  // only test with this single bounding circle.

  if(sign_ == zERO) {
    float64 tst;
    if ( ( (tst = c * boundingCircle_.a_) < -1.0L + gEpsilon ? gPi :
	   acos(tst) ) > 
	 ( d + boundingCircle_.s_) ) return false;
    return true;
  }

  // for all other convexes, test every constraint. If the bounding
  // circle lies completely outside of one of the constraints, reject.
  // else, accept.

  size_t i;
  for(i = 0; i < constraints_.length(); i++) {
      if ( ( (c * constraints_.vector_[i].a_) < -1.0L + gEpsilon ? gPi :
	   acos(c * constraints_.vector_[i].a_) ) > 
	  ( d + constraints_.vector_[i].s_) ) return false;
  }
  return true;
}

/////////////TESTEDGECONSTRAINT///////////////////////////
// testEdgeConstraint: test if edges intersect with a given constraint.
//
bool
SpatialConvex::testEdgeConstraint(const SpatialVector & v0, 
				  const SpatialVector & v1, 
				  const SpatialVector & v2, 
				  size_t cIndex) {
  if ( eSolve(v0, v1, cIndex) ) return true;
  if ( eSolve(v1, v2, cIndex) ) return true;
  if ( eSolve(v2, v0, cIndex) ) return true;
  return false;
}

/////////////TESTOTHERPOSNONE/////////////////////////////
// testOtherPosNone: test for other positive constraints that do
//                   not intersect with an edge. Return its index
//
size_t
SpatialConvex::testOtherPosNone(const SpatialVector & v0, 
				const SpatialVector & v1, 
				const SpatialVector & v2) {
  size_t i = 1;
  while ( i < constraints_.length() && constraints_.vector_[i].sign_ == pOS ) {
    if ( !testEdgeConstraint ( v0,v1,v2, i ) ) return i;
    i++;
  }
  return 0;
}

/////////////TESTCONSTRAINTINSIDE/////////////////////////
// testConstraintInside: look if a constraint is inside the triangle
//
bool
SpatialConvex::testConstraintInside(const SpatialVector & v0, 
				    const SpatialVector & v1, 
				    const SpatialVector & v2,
				    size_t i) {
  return testVectorInside(v0,v1,v2, constraints_.vector_[i].a_);
}

/////////////TESTVECTORINSIDE////////////////////////////
// testVectorInside: look if a vector is inside the triangle
//
bool
SpatialConvex::testVectorInside(const SpatialVector & v0, 
				const SpatialVector & v1, 
				const SpatialVector & v2, 
				SpatialVector & v) {

  // If (a ^ b * c) < 0, vectors abc point clockwise.
  // -> center c not inside triangle, since vertices are ordered
  // counter-clockwise.

  if( ( (( v0 ^ v1 ) * v) < 0 ) ||
      ( (( v1 ^ v2 ) * v) < 0 ) ||
      ( (( v2 ^ v0 ) * v) < 0 ) )
      return false;
  return true;
}

/////////////READ/////////////////////////////////////////
//
void
SpatialConvex::read(std::istream &in) {
  size_t nconstr;
  SpatialConstraint constr;
  
  in.setf(std::ios::skipws);
  while(in.peek() == COMMENT)  // ignore comments
    in.ignore(10000,'\n');
  in >> nconstr ; in.ignore(); // ignore "\n"
  if(!in.good())
    throw SpatialFailure("SpatialConvex:read: Could not read constraint");
  for(size_t i = 0; i < nconstr; i++) {
    if(in.eof())
      throw SpatialFailure("SpatialConvex:read: Premature end-of-file");
    in >> constr;
    if(!in.good())
      throw SpatialFailure("SpatialConvex:read: Could not read constraint");
    add(constr);
  }
}

/////////////READ/////////////////////////////////////////
//
void
SpatialConvex::readRaDec(std::istream &in) {
  size_t nconstr;
  SpatialConstraint constr;
  
  while(in.peek() == COMMENT)  // ignore comments
    in.ignore(10000,'\n');
  in >> nconstr ; in.ignore(); // ignore "\n"
  for(size_t i = 0; i < nconstr; i++) {
    constr.readRaDec(in);
    add(constr);
  }
}

/////////////set ra,dec E.S.S./////////////////////////////////////////
//

void
SpatialConvex::setRaDecD(float64 ra, float64 dec, float64 d) {
  
  SpatialConstraint constr;
  
  constr.setRaDecD(ra,dec,d);
  add(constr);
}

/////////////WRITE////////////////////////////////////////
//
void
SpatialConvex::write(std::ostream &out) const {
  out << "#CONVEX" << "\n";
  out << constraints_.length() << "\n";
  for (size_t i = 0; i < constraints_.length() ; i++)
    out << constraints_[i];
}

/////////////>>///////////////////////////////////////////
// read from istream
//
std::istream& operator >>( std::istream& in, SpatialConvex & c) {
  c.read(in);
  return(in);
}

/////////////<<///////////////////////////////////////////
// write to ostream
//
std::ostream& operator <<( std::ostream& out, const SpatialConvex & c) {
  c.write(out);
  return(out);
}
