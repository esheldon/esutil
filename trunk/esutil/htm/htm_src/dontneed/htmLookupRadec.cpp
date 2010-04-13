//#     Original Filename:       lookup.cpp
//#
//#     --------------------------------------------------------------------
//#     Hacked to be a used as a shared object library and just look up a 
//#     list of ra's and dec's.  Renamed to
//#           htmLookupRadec.cpp    14-DEC-2000 Erin Scott Sheldon UofMich
//#
//#     usage from idl:
//#     
//#     sofile = mypath + 'htmLookupRadec.so'
//#     entry = 'main'
//#     depth = ulong(9)                  ;type is important
//#     n=ulong( n_elements(ra) )         ;type is important
//#     indices = replicate(ulong(0), n)  ;type is important
//#     tmp = call_external(value=[0B,0B,0B,0B,0B], sofile,entry,$
//#                         ra, dec, depth, n, indices)
//#
//#     ;; the ra,dec must be double arrays
//#     ;; indices will return with the htm indices for each (ra,dec) pair.
//#
//#     --------------------------------------------------------------------
//#     Original Comments:
//#     specify a point on the sphere, return its ID/Name to a certain depth
//#
//#
//#     Author:         Peter Z. Kunszt
//#
//#     Date:           October 15, 1999
//#
//#
//#
//# (c) Copyright The Johns Hopkins University 1999
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
#include "SpatialVector.h"
#include "SpatialInterface.h"
#include "VarStr.h"
#include <stdlib.h>

/*******************************************************
 
  DESCRIPTION
 
  This example code demonstrates the lookup functionality of
  the SpatialIndex.
 
  It can be invoked by
 
  	lookup level x y z

	or

	lookup level ra dec

  where

     level     : the level depth to build the index (2 - 14)
     x,y,z     : define a point on the unit sphere (can be non-unit vector)
     ra,dec    : "

  it echoes the vector and returns its ID/Name for the given level.


Example 1: level 5, ra,dec = 10,25

%lookup 5 10 25

	(x,y,z) = 0.892539 0.157379 0.422618
	(ra,dec) = 10,25
	ID/Name = 16020 N322110 

Example 2: level 14, x,y,z = -1,2,-23  (output is normed version)

% lookup 14 -1 2 -23

	(x,y,z) = -0.0432742 0.0865485 -0.995307
	(ra,dec) = 116.565,-84.4471
	ID/Name = 2486622255 S110031231200233

*******************************************************/

int
main(int argc, char *argv[]) {

//*******************************************************
//
// Initialization
//
//*******************************************************

  bool quiet = false;		// debug flag
  size_t j, depth, n;
  uint32 *inputn, *inputdepth, *indices;
  float64 *ra, *dec, tmpra, tmpdec; //Changed to pointers E.S.S.
  htmInterface *htm;
  uint64 id;

  quiet = true;

  ///////////////////////////////////
  // set the input values E.S.S.
  ///////////////////////////////////

  ra = (float64 *) argv[0];
  dec = (float64 *) argv[1];
  inputdepth = (uint32 *) argv[2];
  depth = (size_t) *(&inputdepth[0]);   //convert to size_t so won't crash
  inputn = (uint32 *) argv[3];
  n = (size_t) *(&inputn[0]);           // so it won't crash

  indices = (uint32 *) argv[4]; // this will be filled with indices
                                // We will never use depth > 14, so only 
                                // need uint32. We can convert id from 
                                // uint64 to uint32 later.  E.S.S.

  try {

    if(!quiet)
      printf("Depth = %u\n",depth);

    // this command causes a "Program caused arithmetic error: Floating underflow" 
    // error statement when this program is called by IDL.
    htm = new htmInterface(depth);

// *******************************************************
//
// Lookup all the ra,dec positions
//
// ******************************************************

    j = n;


    for (j=0; j < n; ++j) {
      tmpra = *(&ra[0] + j);
      tmpdec = *(&dec[0] + j);
      id = htm->lookupID(tmpra,tmpdec);  // lookup id by ra dec
      *(&indices[0] + j) = (uint32)id;
    }

  } catch (SpatialException x) {
    printf("%s\n",x.what());
  }

  return 0;
}
