//#     Filename:       BitList.hxx
//#
//#     Friend declarations for bitlist
//#
//#
//#     Author:         Peter Z. Kunszt
//#     
//#     Date:           June 3, 1998
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

//#######################################################################
//
// Friend functions to BitList
//

// Bitwise operators returning the result in a separate BitList.
// First argument: return list, second and third arguments: bitlists
// to process. AND operator
/*
BitList & and (BitList &, const BitList &, const BitList &);

// OR operator
BitList & or  (BitList &, const BitList &, const BitList &);

// XOR operator
BitList & xor (BitList &, const BitList &, const BitList &);

// NOT operator
BitList & not (BitList &, const BitList &);
*/
// </GROUP>
// 
