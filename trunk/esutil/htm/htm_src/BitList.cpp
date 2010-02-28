//#     Filename:       BitList.cpp
//#
//#     Member function definitions for classes associated with the BitList.
//#     Classes defined here are:
//#
//#		BitList	  - Bit array class
//#     	BitListIterator - Iterator through Bit Array
//#
//#     Author:         Peter Z. Kunszt
//#     
//#     Date:           June 4, 1998
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
//#     June 18, 1998 : pzk -- added compress/decompress methods
//#     June 29, 1998 : pzk -- added copy constructor/assignment to iterator

#ifdef SXDB
#include <sxGeneral.h>
#else
#include <SpatialGeneral.h>
#endif
#include <BitList.h>

// just the correct number one, static to this file and lower 4 bits of byte
static const uint32 one  = 1;
static const uint8  one8 = 1;
static const uint8 lhalf = 15;

#define ZEROS 0
#define ONES 0xffffffff;

// ===========================================================================
//
// Member functions for class BitList
//
// ===========================================================================

//////////////////CONSTRUCTOR//////////////////////////////////////////////
// default constructor with size and increment options
BitList::BitList (size_t size, size_t inc)
  : bits_(0,0,inc), size_(size) {
  if (size_ > 0) {
    bits_.at(size_ >> 5);  // divide by 32: we have int32s in the array
  }
}

// copy constructor
BitList::BitList ( const BitList & BL)
  : size_(BL.size_) {
  *this = BL;
}

//////////////////ASSIGNMENT///////////////////////////////////////////////
BitList &
BitList::operator = (const BitList & BL) {
  if (this != &BL) { // beware of self-assignment
    size_ = BL.size_;
    bits_.cut(bits_.length());        // clear this list
    bits_ = BL.bits_;                 // copy list
  }

  return *this;
}

//////////////////SET//////////////////////////////////////////////////////
// Set bit at index to val
void
BitList::set(size_t index, bool val) {
  size_t WordIndex = index >> 5;  // set WordIndex since it's used a lot

  // Extend ValVec if out of bounds and set bit or
  // Set or unset bit otherwise.
  if (WordIndex >= bits_.length()) {
    bits_.at(WordIndex);
    if(val)bits_(WordIndex) = one << (index & 31);
    size_ = index + 1;
  } else {
    if (val) {
      bits_(WordIndex) = bits_(WordIndex) | (one << (index & 31));
    } else {
      bits_(WordIndex) = bits_(WordIndex) & (~(one << (index & 31)));
    }
    if (index >= size_) size_ = index + 1;
  }
}

//////////////////[]///////////////////////////////////////////////////////
//
// Get bit at index. Throw an exception if out of bounds.
//
bool
BitList::operator [](size_t index) const {
  if(index >= size_)
    return false;

  return ( bits_(index >> 5) & (one << (index & 31)) ) ? true : false;
}

//////////////////SIZE/////////////////////////////////////////////////////
// Return size in bits
size_t
BitList::size() const {
  return size_;
}

//////////////////COUNT////////////////////////////////////////////////////
// Return number of set bits
size_t
BitList::count() const {
  size_t c = 0;
  uint32 word;
  for(size_t w = 0; w < bits_.length() ; w++) {
    word = bits_(w);
    for(size_t i = 0; i < 32 ; i++)
      c += (word >> i) & one;
  }
  return c;
}

//////////////////CHOPLITTER_//////////////////////////////////////////////
// Chop off trailing litter on the bitlist: mask those bits off
// the last uint32 which are past the size_
void
BitList::choplitter_() {
  if (size_ == 0) return;

  uint32 word = 0;
  for ( size_t i = 0 ; i < (size_ & 31); i++)
    word += one << i ;
  if (word > 0)
    bits_(size_ >> 5) = bits_(size_ >> 5) & word;
  else {
    if( bits_.length() > (size_ >> 5) )
      bits_(size_ >> 5) = 0;
  }
}

//////////////////TRIM/////////////////////////////////////////////////////
// Chop off trailing 'false' bits, return new size.
size_t
BitList::trim() {
  BitListIterator Iter(*this);
  size_t index;

  if (Iter.prev(true,index)) {     // a true bit has been found
    if (index < size_ - 1) {       // last bit is false, trim
      bits_.cut( bits_.length() - ( (index >> 5) +1) );
      size_ = index + 1;
    }
  } else clear();                  // all bits are false

  choplitter_();
  return size_;
}

//////////////////CLEAR////////////////////////////////////////////////////
// Reset size to 0, free up storage
void
BitList::clear(bool keepLength) {
  bits_.cut(bits_.length());
  if(keepLength) {
    bits_.at((size_-1) >> 5);  // reset the bits_ size, init to 0.
  } else {
    size_ = 0;
  }
}

//////////////////&=///////////////////////////////////////////////////////
// The bitwise &= operator, may be used to mask the current instance.
// Does not change the size.
BitList &
BitList::operator &= (const BitList & BL) {
  if (this == &BL) return *this;

  size_t len = bits_.length();
  if ( size_ > BL.size_)
    len = BL.bits_.length();

  if (size_ * BL.size_ > 0)            // check for zero-length BitList
    for (size_t i = 0; i < len; i++)
      bits_(i) &= BL.bits_(i);

  // if size of current exceeds size of mask, set the rest to zero.
  if (size_ > 0)
    for (size_t i = len; i < bits_.length(); i++)
      bits_(i) = 0;

  return *this;
}

//////////////////|=///////////////////////////////////////////////////////
// The bitwise |= operator, may be used to set bits in the current instance.
// expands instance if needed
BitList &
BitList::operator |= (const BitList & BL) {
  if (this == &BL) return *this;

  if ( size_ < BL.size_){                  // expand if mask longer
    bits_.at(BL.bits_.length() - 1);
    size_ = BL.size_;
  }

  if (BL.size_ > 0)                       // don't do anything if BL has 0 size
    for (size_t i = 0; i < BL.bits_.length(); i++)
      bits_(i) |= BL.bits_(i);

  choplitter_();
  return *this;
}

//////////////////^=///////////////////////////////////////////////////////
// The bitwise ^= operator, may be used to set bits in the current instance.
// expands instance if needed. Xor with itself sets all bits to zero.
BitList &
BitList::operator ^= (const BitList & BL) {
  if (this == &BL) {
    for (size_t j = 0; j < bits_.length(); j++)
      bits_(j) = 0;    
    return *this;
  }

  if ( size_ < BL.size_){                  // expand if mask longer
    bits_.at(BL.bits_.length() - 1);
    size_ = BL.size_;
  }

  if (BL.size_ > 0)                       // don't do anything if BL has 0 size
    for (size_t i = 0; i < BL.bits_.length(); i++)
      bits_(i) ^= BL.bits_(i);

  choplitter_();
  return *this;
}

//////////////////INVERT///////////////////////////////////////////////////
// The inversion method; flip every bit
void
BitList::invert() {
  if(bits_.length() > 0)
    for ( size_t i = 0; i< bits_.length(); i++)
      bits_(i) = ~ bits_(i);
  choplitter_();
}

//////////////////OVERLAPS/////////////////////////////////////////////////
// Test if the current instance has at least one overlapping bit with BL
bool
BitList::overlaps(const BitList & BL) const {

  BitListIterator iter(*this);
  size_t index;

  while(iter.next(true,index))
    if(BL[index])return true;
  return false;
}

//////////////////COVERS///////////////////////////////////////////////////
// Test if the current instance is a subset of BL
bool
BitList::covers(const BitList & BL) const {

  BitListIterator iter(BL);
  size_t index;

  while(iter.next(true,index))
    if(!(*this)[index])return false;
  return true;
}

//////////////////AND//////////////////////////////////////////////////////
//
// The bitwise & operator. If one of the arrays is larger than the other,
// the size of the returned array matches the size of the shorter array.
// (the nonexistent elements are taken as 0
//
/*BitList &
and (BitList & _res, const BitList & BL1 ,const BitList & BL2) {

  size_t len = BL1.bits_.length();
  _res.size_ = BL1.size_;
  if (BL1.size_ > BL2.size_) {
    len = BL2.bits_.length();
    _res.size_ = BL2.size_;
  }

  if (len > 0) {
    // adjust length of result to the length of the shorter array
    _res.bits_.at(len - 1);
    _res.bits_.cut(_res.bits_.length() - len);

    // AND all words up to index len
    for (size_t i = 0; i < len; i++)
      _res.bits_(i) = BL1.bits_(i) & BL2.bits_(i);
  } else { 
    // for zero length, return a zero result.
    _res.bits_.cut(_res.bits_.length());
  }
  _res.choplitter_();
  return _res;
}
*/
//////////////////OR///////////////////////////////////////////////////////
//
// The bitwise | operator. At nonequal sizes, the longer size is returned.
// Again, nonexistent elements are treated as 0.
//
/*
BitList &
or (BitList & _res, const BitList & BL1 ,const BitList & BL2) {


  // determine the length of the shorter of the two (store it in len)
  // Extend the result to size of first parm if BL is the shorter one.
  // if we were unlucky to initialize to the shorter parameter,
  // copy the remaining words of the longer parameter into the result

  _res = BL1;
  size_t len = BL2.bits_.length();

  if (BL1.size_ < BL2.size_) {
    _res = BL2;
    len = BL1.bits_.length();
  }

  // OR all words up to index len
  if (len > 0)
    for ( size_t i = 0; i < len; i++)
      _res.bits_(i) = BL1.bits_(i) | BL2.bits_(i);

  _res.choplitter_();
  return _res;
}
*/
//////////////////XOR//////////////////////////////////////////////////////
/*
BitList &
xor (BitList & _res, const BitList & BL1 ,const BitList & BL2) {


  // determine the length of the shorter of the two (store it in len)
  // Extend the result to size of first parm if BL is the shorter one.
  // if we were unlucky to initialize to the shorter parameter,
  // copy the remaining words of the longer parameter into the result

  _res = BL1;
  size_t len = BL2.bits_.length();

  if (BL1.size_ < BL2.size_) {
    _res = BL2;
    len = BL1.bits_.length();
  }

  // XOR all words up to index len
  if (len > 0)
    for ( size_t i = 0; i < len; i++)
      _res.bits_(i) = BL1.bits_(i) ^ BL2.bits_(i);

  _res.choplitter_();
  return _res;
}
*/
//////////////////NOT//////////////////////////////////////////////////////
//
// Not operator. Set the result to the same length as the current instance
// Copy each element with a bitwise not.
//
/*
BitList &
not (BitList & _res, const BitList & BL) {
  _res.bits_.cut(_res.bits_.length());     // clear this list
  _res.size_ = BL.size_;

  // NOT all words of BL
  if ( _res.size_ > 0) {
    _res.bits_.at(BL.bits_.length() - 1);   // set to same length as BL
    for (size_t i=0; i < BL.bits_.length(); i++)
      _res.bits_(i) = ~ BL.bits_(i);
  }
  _res.choplitter_();
  return _res;
}
*/
//////////////////COMPRESS_////////////////////////////////////////////////
// The compress utility makes a simple compression of the BitList, based
// on the PCX compression scheme.
//
// We do a byte-encoding, and write those bytes out in hex form (2 letters
// for each byte).
//
// The first bit (128) of each byte denotes whether the byte contains
// actual data or a count of bits. 
//
// If the first bit is 0, the next 7 bits are data-bits.
// If the first bit is 1, the next bit (64) is the value and the
// 6 last bits give the count (0-63). It does not make sense to
// count 0 times so bits are only counted if there are at least 8 bits
// with the same value. So the count is 8-71. For more than 71 bits,
// the next byte is taken.
//
// The byte is then written out to the stream as a hex number.
//
// This compression results in a 14% blowup in size for the worst case
// (i.e. BitList consists of 01010101...) and in a compression factor
// of 71 for the best case (all bits alike).
// Example: a series of 0101's of size 92 compresses into
//
//       2A552A552A552A552A552A552A.101
//
//          a series of 1111's of size 92 compresses into
//
//       FFCD.0
//
// The dot and the value following it indicates how many bits are in the
// last byte, if any, and then the last bytes follows in hex. In the above
// example, .101 means only a single bit=true. In this way the exact
// size of the BitList can be restored, not just chunks of 7.
//

void
BitList::compress(std::ostream & out) const {

  BitListIterator iter(*this);
  bool bit, obit, flag = false;
  int b=0;
  uint8 byte = 0;

  if(iter.next(obit)){   // get first bit into obit
    byte = int(obit);    // set byte's last bit to obit
  } else b = -1;         // BitList is empty.

  while (iter.next(bit)) {  // loop over every bit in the BitList
    b++;
    if(bit != obit && b > 0)flag = true;
    if(b < 7){
      if(bit) byte += (one8 << b);
    } else if (b == 7) {
      if(flag){
	out << char( (byte >> 4) + 48 ); // always digit 0-7, since first bit=0
	out << char( ((byte & lhalf) > 9) ? ((byte & lhalf) + 55)    // letter
		                          : ((byte & lhalf) + 48) ); // digit
	b = 0;
	flag = false;
	byte = int(bit);
      }
    } else if (b == 71){ // reached maximal count
      out << char(66 + 4 * int(obit)); // either B or F, depending on bit value
      out << 'F';                      // always F
      b = 0;
      flag = false;
      byte = int(bit);
    } else {
      if(flag){
	byte = 128 + 64 * int(obit) + b - 8;
	out << char( ((byte >> 4) > 9) ? ((byte >> 4) + 55)    // letter
	                               : ((byte >> 4) + 48) ); // digit
	out << char( ((byte & lhalf) > 9) ? ((byte & lhalf) + 55)
		                       : ((byte & lhalf) + 48) );
	b = 0;
	flag = false;
	byte = int(bit);
      }
    }
    obit = bit;
  }
  if(b<=7) {
    out << '.' << b+1 ; // indicate the bit count in last byte
    if(b > -1) { // last byte.
      out << char( (byte >> 4) + 48 ); // always digit 0-7, since first bit=0
      out << char( ((byte & lhalf) > 9) ? ((byte & lhalf) + 55)    // letter
		                        : ((byte & lhalf) + 48) ); // digit
    } else {
      out << '0';
    }
  } else {
    byte = 128 + 64 * int(obit) + b - 7;
    out << char( ((byte >> 4) > 9) ? ((byte >> 4) + 55)    // letter
	                           : ((byte >> 4) + 48) ); // digit
    out << char( ((byte & lhalf) > 9) ? ((byte & lhalf) + 55)
	                              : ((byte & lhalf) + 48) );
    out << '.' << '0';
  }
  out << "\n";
}

//////////////////DECOMPRESS_//////////////////////////////////////////////
// The decompress utility reads in a compressed BitList from a stream and
// fills in the bits in the real BitList. See compress_ for details.
//
void
BitList::decompress(std::istream & in) {

  char c1, c2;
  int count, idx=0;
  int8 byte;
  bool bit;

  clear();
  in >> c1;
  while(c1 != '.') {
    in >> c2;
    if (c1 > '7') { // count - byte
      count = ( (c1>'9') ? ((int(c1-'7') & 3) << 4) :
		           ((int(c1-'0') & 3) << 4) ) + 
	   int( (c2>'9') ? (c2-'7') : 
                           (c2-'0') ) + 8 ;
	  bit = (int(c1-'7') & 4) ? true : false	;
      if(bit) {
	for (int i = 0; i < count; i++)
	  set(idx++,bit);
      } else {
	idx += count-1;
	set(idx++,bit);
      }
    } else {  // data - byte
      byte = (int(c1-'0') << 4) + int(( (c2>'9') ? (c2-'7') : (c2-'0')));
      for (int i = 0; i < 7; i++) 
		  set(idx++,(byte & (one8 << i)) ? true : false);
    }
    in >> c1;
  }
  in >> c1;
  count = int(c1-'0');
  if(count) {
    in >> c1;
    in >> c2;
    byte = (int(c1-'0') << 4) + int(( (c2>'9') ? (c2-'7') : (c2-'0')));
    for (int i = 0; i < count; i++)
		set(idx++,(byte & (one8 << i)) ? true : false);
  }
}

// ===========================================================================
//
// Member functions for class BitListIterator
//
// ===========================================================================

//////////////////CONSTRUCTOR//////////////////////////////////////////////
// default constructor: empty bitlistiterator
//
BitListIterator::BitListIterator()
  : bitlist(0)
{
}
//////////////////CONSTRUCTOR//////////////////////////////////////////////
// Construct from a bitlist. Set the current index to the out-of-bounds
// index = size_.
//
BitListIterator::BitListIterator(const BitList & BL)
  : bitlist(& BL)
{
  setindex(bitlist->size_);
}

//////////////////CONSTRUCTOR//////////////////////////////////////////////
// Construct from a bitlist and set starting index.
BitListIterator::BitListIterator(const BitList & BL, size_t start)
  : bitlist(& BL)
{
  setindex(start);
}

//////////////////COPY CONSTRUCTOR/////////////////////////////////////////
// Copy construct
BitListIterator::BitListIterator(const BitListIterator & iter)
{
  bitlist    = iter.bitlist;
  word_      = iter.word_;
  wordIndex_ = iter.wordIndex_;
  bitIndex_  = iter.bitIndex_;
}

//////////////////ASSIGNMENT///////////////////////////////////////////////
// Assignment
BitListIterator &
BitListIterator::operator = (const BitListIterator & iter)
{
  bitlist    = iter.bitlist;
  word_      = iter.word_;
  wordIndex_ = iter.wordIndex_;
  bitIndex_  = iter.bitIndex_;

  return *this;
}

//////////////////SETINDEX/////////////////////////////////////////////////
// Initialize the internal counters to a certain starting point
void
BitListIterator::setindex(size_t start) {
  if (bitlist == 0) 
    throw _BOUNDS_EXCEPTION("BitListIterator:"," not initialized");
  // the next next() returns first bit if start = bitlist.size_
  if (start >= bitlist->size_) start = bitlist->size_;

  wordIndex_ = start >> 5;
  bitIndex_ = start & 31;
  if(bitlist->size_ > 0 && start < bitlist->size_)
    word_ = bitlist->bits_.vector_[wordIndex_];
}

//////////////////NEXT(BOOL, SIZE_T &)/////////////////////////////////////
// get the index of the next 'true' or 'false' bit (indicated by the first 
// argument)
bool
BitListIterator::next(bool bit, size_t & _index) {
  if (bitlist == 0) 
    throw _BOUNDS_EXCEPTION("BitListIterator:"," not initialized");

  /*
  if(bitlist->size_==0)return false;
  uint32 val;
  if(bit)val = ZEROS;
  else val = ONES;
  if( bitlist->bits_.vector_[wordIndex_] == val ) {
    while(wordIndex_ < bitlist->length() && 
	  bitlist->bits_.vector_[wordIndex_] == val) wordIndex_++;
    wordIndex_--;
    bitIndex_ = 31;
  }
  */

  while (incr()) {  // increment pointer, check for boundary
    // If the bit is on, return the index value and set the pointer
    // to the next bit.
	  if( ((word_ & (one << bitIndex_)) ? true : false) == bit) {
      _index = ((wordIndex_ << 5) + bitIndex_);
      return true;
    }
  } 
  return false;
}

//////////////////NEXT(BOOL &)/////////////////////////////////////////////
// get the next bit into bit and increment the internal index
bool
BitListIterator::next(bool & _bit) {
  if (bitlist == 0) 
    throw _BOUNDS_EXCEPTION("BitListIterator:"," not initialized");

  if(incr()) {  // increment pointer, check for boundary
	  _bit = (word_ & (one << bitIndex_)) ? true : false; // get current bit
    return true;
  }
  return false;
}

//////////////////PREV(BOOL, SIZE_T &)/////////////////////////////////////
// get the index of the previous 'true' or 'false' bit (indicated by the first 
// argument)
bool
BitListIterator::prev(bool bit, size_t & _index) {
  if (bitlist == 0) 
    throw _BOUNDS_EXCEPTION("BitListIterator:"," not initialized");

  while (decr()) {

      // If the bit is on, return the index value and set the pointer
      // to the next bit.
	  if( ((word_ & (one << bitIndex_)) ? true : false) == bit) {
      _index = ((wordIndex_ << 5) + bitIndex_);
      return true;
    }
  }
  return false;
}


//////////////////PREV(BOOL &)/////////////////////////////////////////////
// get the previous bit into bit and decrement the internal index
bool
BitListIterator::prev(bool & _bit) {
  if (bitlist == 0) 
    throw _BOUNDS_EXCEPTION("BitListIterator:"," not initialized");

  if(decr()) {
	  _bit = (word_ & (one << bitIndex_)) ? true : false;
    return true;
  }
  return false;

}

//////////////////INCR////////////////////////////////////////////////////
// private function incrementing the pointer and returning
// true or false if it is within bounds or not
bool
BitListIterator::incr() {
  if (bitlist == 0) 
    throw _BOUNDS_EXCEPTION("BitListIterator:"," not initialized");
  if ( ((wordIndex_ << 5) + bitIndex_) == bitlist->size_ ) {
    if (bitlist->size_ == 0) return false;      // check for 0-length array
    bitIndex_ = 0;
    wordIndex_ = 0;
    word_ = bitlist->bits_(0);
    return true;
  } else {
    if(++bitIndex_ == 32) {        // check if next word is needed
      bitIndex_ = 0;
      if ( ((++wordIndex_ << 5) + bitIndex_) == bitlist->size_ )
	return false;
      word_ = bitlist->bits_.vector_[wordIndex_];
      return true;
    }
    return bool( ((wordIndex_ << 5) + bitIndex_) != bitlist->size_ );
  }
}

//////////////////DECR////////////////////////////////////////////////////
// private function decrementing the pointer returning true or
// false if it is within bounds or not
bool
BitListIterator::decr() {
  if (bitlist == 0) 
    throw _BOUNDS_EXCEPTION("BitListIterator:"," not initialized");

  if (wordIndex_ + bitIndex_ == 0) {
    wordIndex_ = bitlist->size_ >> 5;
    bitIndex_ = bitlist->size_ & 31;
    return false;
  } else {
    if(bitIndex_ == 0) {
      bitIndex_ = 32;
      word_ = bitlist->bits_.vector_[--wordIndex_];
    }
    if ( ((wordIndex_ << 5) + bitIndex_) == bitlist->size_ )
      word_ = bitlist->bits_(wordIndex_);
    bitIndex_--;
    return true;
  }

}
