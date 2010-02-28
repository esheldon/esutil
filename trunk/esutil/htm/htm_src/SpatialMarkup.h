#ifndef _SpatialMarkup_h
#define _SpatialMarkup_h
//#     Filename:       SpatialMarkup.h
//#
//#     SpatialMarkup is the class for the the sky Markup routines.
//#
//#
//#     Author:         Peter Z. Kunszt, based on A. Szalay's code
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

#include "SpatialIndex.h"

//########################################################################
//
// <GROUP>
// <SUMMARY>Class declarations</SUMMARY>
// 

//########################################################################
//
// <SUMMARY> Spatial Markup class </SUMMARY>
//

// The Spatial Markup just keeps track of the state of each node in the
// SkyIndex tree during an intersection with a domain.
//

class SpatialMarkup {
public:
  enum Markup {
    dONTKNOW,
    pARTIAL,
    sWALLOWED,
    fULL,
    rEJECT,
    bREJECT
  };

  // Constructor: specify index
  SpatialMarkup(const SpatialIndex &);

  // bracket operator: lvalue to set node markers at a specific index
  Markup & operator [](size_t nodeIndex);

  // parenthesis operator: lvalue to set vertex markers at a specific index
  uint8 & operator ()(size_t vIndex);

  // reset node markup to reject
  void clear();

  // reset vertex markup to undefined
  void clearVertex();

  // static values for vertex marker
  static uint8 vTrue, vFalse, vUndef;

private:
  const SpatialIndex & index;
  ValVec<Markup> mark_;		// Array of node markers
  ValVec<uint8> vmark_;		// Array of vertex markers (true, false, undef)

  friend class SpatialConvex;
  friend class SpatialConstraint;
  friend class SpatialDomain;
};
//==========================================================



#endif
