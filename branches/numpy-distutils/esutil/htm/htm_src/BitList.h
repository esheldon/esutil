#ifndef _BitList_h
#define _BitList_h

//#     Filename:       BitList.h
//#
//#     Declarations for the BitList and BitListIterator classes
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

#include <VarVec.h>
//#include <iostream.h>
#include <iostream>

//##########################################################################
//#
/** BitList class.
    The BitList is an array of bits. A bit can be set at any index using the
    set member function, the array expands itself automatically if the index
    exceeds the current size.
*/

//class LINKAGE BitList {
class BitList {
public:
  /** Default constructor.
      You can initialize the BitList to a specific size
      and optionally the increment may be set by which the size internal ValVec
      will be incremented upon need. (See VarVec.h for explanation on this.)
      The default is to double the size of the array whenever an expansion
      is requested.
  */
  BitList(size_t size = 0, size_t inc = 0);

  /// Copy constructor
  BitList(const BitList &);

  /// The assignment operator.
  BitList & operator = (const BitList &);

  /** Set a bit at a specific index to a given value. 
      If the index is larger
      than the current size, the BitList expands itself to be able to hold
      that value at the given index.
  */
  void set(size_t index, bool value);

  /** Get the bit at a given index. 
      If the index exceeds the size, the return
      value is 'false'. All BitLists are treated as if they were of infinite
      size, all bits set to zero at initialization.
  */
  bool operator [](size_t) const;

  /** Get  the size of the BitList. 
      At construction time the size may be
      specified, and that much memory will be allocated. If the construction
      is done using the set() method, the size is 'minimal' i.e. as much as
      it needs to hold the last 'true' bit.
  */
  size_t size() const;

  /// Count the TRUE bits from a certain index on
  size_t count() const;

  /// Just chop off all trailing 'false' bits. Returns new size.
  size_t trim();

  /** Clear the list, reset size to 0 by default. 
      If true is given as an
      argument, the size is kept. */
  void clear(bool keepLength = false);

  /// The standard &= operator.
  BitList & operator &= (const BitList &);

  /// The standard |= operator.
  BitList & operator |= (const BitList &);

  /// The standard ^= operator.
  BitList & operator ^= (const BitList &);

  /// The inversion method, flip every bit in the BitList.
  void invert();

  /// Check if BL is a subset of the current list
  bool covers(const BitList & BL) const;

  /** Check if the current BitList overlaps with the other.
      (i.e. they have at least one common Bit) */
  bool overlaps(const BitList & BL) const;

  /// compress output
  void compress(std::ostream &) const;

  /// decompress input
  void decompress(std::istream &);

private:
  friend class BitListIterator;
  /*
  friend BitList & and (BitList &, const BitList &, const BitList &);
  friend BitList & or  (BitList &, const BitList &, const BitList &);
  friend BitList & xor (BitList &, const BitList &, const BitList &);
  friend BitList & not (BitList &, const BitList &);
  */
  // Mask off litter at the end of a word not belonging to the array
  void choplitter_();

  // the data
  ValVec<uint32> bits_;

  // the length of the array in bits
  size_t size_;

  friend class sxFluxIndexData;
  friend class sxSegment;
};


//##########################################################################
/** BitListIterator class.
    The BitListIterator iterates through a BitList efficiently.
    next() and prev() functions are supplied. The functionality is
    the following: The BLI saves an index to a certain bit in the BitList.
    By calling either next() or prev(), the index is incremented/decremented and
    the bit it is pointing to now is returned. If it gets out of bounds,
    these functions return 'false'. The out-of-bounds index is always 
    index=size. So by calling next() or prev() again when a 'false' was 
    returned previously, they return the first/last bit, respectively.
*/

//class LINKAGE BitListIterator {
class BitListIterator {
public:
  /// Default Constructor.
  BitListIterator();

  /** Normal Constructor.
      needs the BitList to initialize.
      The index is initialized to the out-of-bounds index. */
  BitListIterator(const BitList & bitlist); 

  /** Alternate constructor.
      set the starting index yourself. */
  BitListIterator(const BitList & bitlist, size_t start); 

  /// Copy Constructor.
  BitListIterator(const BitListIterator &);

  /// Assignment.
  BitListIterator & operator = (const BitListIterator &);

  /// Init: set current index
  void setindex(size_t index);

  /** Set the internal index to the next 'true' or 'false' bit;
      indicated by the first argument, and return the index in the
      second argument.  Returns 'false' if it gets out of bounds.
      Example: For a BitList 001100110011 (from left to right, index
      starts at 0), the subsequent call to next(true,index) returns
      'true' and sets index to 2, 3, 6, 7, 10, 11. The next call puts
      leaves index and returns 'false'. A subsequent next() call would
      again return 'true' and set index=2.
  */
  bool next(bool bit, size_t & _index);

  /// Just like next(), but the index is moved backwards.
  bool prev(bool bit, size_t & _index);

  /** Increment the internal index and return the value of the bit it points to
      Returns 'false' if the boundary is reached.
      <b>Example</b>: For a BitList 001100110011 the calls to next(val) return
      'true' and set bit to 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1. The next call
      returns 'false' and does not set bit. A subsequent call would return
      again 'true' and set bit to the first bit in the list, in this case 0.
  */
  bool next(bool & bit);

  /** Just like next() above, just decrement the internal index.
      The two versions of next() and prev() may be used in conjunction.
  */
  bool prev(bool & bit);

private:
  // increment, decrement current index, return 'true' or 'false' if boundary
  // has been reached
  bool incr();
  bool decr();

  // data members
  const BitList * bitlist;   // The BitList associated with this iterator
  uint32 word_;              // The current word in bitlist_.bits_
  size_t wordIndex_;         // The index of the word in bitlist_.bits_
  size_t bitIndex_;          // The index of the bit in the current word.
};

#include "BitList.hxx"
#endif /* _BitList_h */
