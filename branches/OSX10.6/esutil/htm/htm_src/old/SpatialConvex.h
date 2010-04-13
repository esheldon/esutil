#ifndef _SpatialConvex_h
#define _SpatialConvex_h
//#     Filename:       SpatialConvex.h
//#
//#     Classes defined here: SpatialConvex
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

#include "SpatialConstraint.h"
#include "SpatialIndex.h"
#include "BitList.h"

/** Enumerator. Define the return values of an intersection */

enum SpatialMarkup {
  /// Uncertain
  dONTKNOW,
  /// Triangle partially intersected
  pARTIAL,
  /// All of the triangle is inside queried area
  fULL,
  /// Triangle is outside the queried area
  rEJECT
};


//########################################################################
//#
//# Spatial Convex class
//#
/**
   A spatial convex is composed of spatial constraints. It does not
   necessarily define a continuous area on the sphere since it is a
   3D-convex of planar intersections which may intrersect with the
   unit sphere at disjoint locations. Especially 'negative'
   constraints tend to tear 'holes' into the convex area.
*/

//class LINKAGE SpatialConvex : public SpatialSign {
class SpatialConvex : public SpatialSign {
public:
  /// Default Constructor
  SpatialConvex();

  /// Constructor from a triangle
  SpatialConvex(const SpatialVector * v1,
		const SpatialVector * v2,
		const SpatialVector * v3);

  /// Constructor from a rectangle
  SpatialConvex(const SpatialVector * v1,
		const SpatialVector * v2,
		const SpatialVector * v3,
		const SpatialVector * v4);

  /// Copy constructor
  SpatialConvex(const SpatialConvex &);

  /// Assignment
  SpatialConvex& operator =(const SpatialConvex &);

  /// Add a constraint
  void add(SpatialConstraint &);

  /// Simplify the convex, remove redundancies
  void simplify();

  /** 
      Intersect with index. 
      The partial and full bitlists for the result have to be
      given. If the conves occupies a large percent of the area
      of the sphere, bitlists are the preferred result method.
  */
  void intersect(const SpatialIndex * index,
		 BitList * partial, BitList * full);

  /** 
      Intersect with index.
      Same intersection as with bitlists (see above), but the result
      is given in a list of nodes.  If the conves is very small, this
      is the preferred result method.  
  */
  void intersect(const SpatialIndex * index,
		 ValVec<uint64> * partial, ValVec<uint64> * full);

  /** 
      Intersect with index.
      Now only a single list of IDs is returned. The IDs need not be
      level.
  */
  void intersect(const SpatialIndex * index,
		 ValVec<uint64> * idList);

  /// Return the number of constraints
  size_t numConstraints();

  /// [] operator: give back constraint
  SpatialConstraint & operator [](size_t i);

  /// read from stream
  void read(std::istream&);

  /// read from stream
  void readRaDec(std::istream&);

  /// set ra,dec,d from user
  void setRaDecD(float64 ra, float64 dec, float64 d);

  /// write to stream
  void write(std::ostream&) const;

private:

  // Do the intersection (common function for overloaded intersect())
  void doIntersect();

  // Simplification routine for zERO convexes. This is called by
  // simplify() in case we have all zERO constraints.
  void simplify0();

  // This is the testsuit for the intersection.

  // triangleTest: Test the nodes of the index if the convex hits it
  // the argument gives the index of the nodes_ array to specify the QuadNode
  SpatialMarkup triangleTest(uint64 nodeIndex);

  // fillChildren: Mark the child nodes as markup.
  void fillChildren(uint64 nodeIndex);

  // test each quadnode for intersections. Calls testTriangle after having
  // tested the vertices using testVertex.
  SpatialMarkup testNode(const SpatialVector & v0, 
			 const SpatialVector & v1, 
			 const SpatialVector & v2);

  // testTriangle: tests a triangle given by 3 vertices if
  // it intersects the convex. Here the whole logic of deciding
  // whether it is partial, full, swallowed or unknown is handled.
  SpatialMarkup testTriangle(const SpatialVector & v0, 
			     const SpatialVector & v1, 
			     const SpatialVector & v2,
			     int vsum);

  // setfull: set the full bitlist for each node below this level.
  void setfull(uint64 id, size_t level);

  // test a triangle's subtriangles whether they are partial.
  // If level is nonzero, recurse: subdivide the
  // triangle according to our rules and test each subdivision.
  // (our rules are: each subdivided triangle has to be given
  // ordered counter-clockwise, 0th index starts off new 0-node,
  // 1st index starts off new 1-node, 2nd index starts off new 2-node
  // middle triangle gives new 3-node.)
  // if we are at the bottom, set this id's leafindex in partial bitlist.
  void testPartial(size_t level, uint64 id,
		   const SpatialVector & v0, 
		   const SpatialVector & v1, 
		   const SpatialVector & v2);

  // call full or partial depending on result of testNode.
  void testSubTriangle(size_t level, uint64 id,
		       const SpatialVector & v0, 
		       const SpatialVector & v1, 
		       const SpatialVector & v2);

  // Test for constraint relative position; intersect, one in the
  // other, disjoint.
  int testConstraints(size_t i, size_t j);

  // Test if vertices are inside the convex for a node.
  int testVertex(const SpatialVector & v);

  // testHole : look for 'holes', i.e. negative constraints that have their
  // centers inside the node with the three corners v0,v1,v2.
  bool testHole(const SpatialVector & v0, 
		const SpatialVector & v1, 
		const SpatialVector & v2);

  // testEdge0: test the edges of the triangle against the edges of the
  // zERO convex. The convex is stored in corners_ so that the convex
  // is always on the left-hand-side of an edge corners_(i) - corners_(i+1).
  // (just like the triangles). This makes testing for intersections with
  // the edges easy.
  bool testEdge0(const SpatialVector & v0, 
		 const SpatialVector & v1, 
		 const SpatialVector & v2);

  // testEdge: look whether one of the constraints intersects with one of
  // the edges of node with the corners v0,v1,v2.
  bool testEdge(const SpatialVector & v0, 
		const SpatialVector & v1, 
		const SpatialVector & v2);

  // eSolve: solve the quadratic equation for the edge v1,v2 of
  // constraint[cIndex]
  bool eSolve(const SpatialVector & v1, 
	      const SpatialVector & v2, size_t cIndex);

  // Test if bounding circle intersects with a constraint
  bool testBoundingCircle(const SpatialVector & v0, 
			  const SpatialVector & v1, 
			  const SpatialVector & v2);

  // Test if a constraint intersects the edges
  bool testEdgeConstraint(const SpatialVector & v0, 
			  const SpatialVector & v1, 
			  const SpatialVector & v2, 
			  size_t cIndex);

  // Look for any positive constraint that does not intersect the edges
  size_t testOtherPosNone(const SpatialVector & v0, 
			  const SpatialVector & v1, 
			  const SpatialVector & v2);

  // Test for a constraint lying inside or outside of triangle
  bool testConstraintInside(const SpatialVector & v0, 
			    const SpatialVector & v1, 
			    const SpatialVector & v2, 
			    size_t cIndex);

  // Test for a vector lying inside or outside of triangle
  bool testVectorInside(const SpatialVector & v0, 
			const SpatialVector & v1, 
			const SpatialVector & v2, 
			SpatialVector & v);

  ValVec<SpatialConstraint> constraints_; // The vector of constraints
  const SpatialIndex * index_;		  // A pointer to the index
  ValVec<SpatialVector> corners_;	  // The corners of a zERO convex
  SpatialConstraint boundingCircle_;	  // For zERO convexes, the bc.
  size_t addlevel_;			  // additional levels to calculate
  BitList * full_;			  // bitlist of full nodes
  BitList * partial_;			  // bitlist of partial nodes
  ValVec<uint64> * flist_;		  // list of full node ids
  ValVec<uint64> * plist_;		  // list of partial node ids
  bool bitresult_;			  // flag which result (bit or list)
  bool range_;			  	  // return range results

  friend class SpatialDomain;
  friend class sxSpatialDomain;
};

#include "SpatialConvex.hxx"

#endif
