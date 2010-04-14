//#     Filename:       VarVecDef.h
//#
//#     Definitions for ValVec and PtrVec templates
//#
//#
//#     Author:         John Doug Reynolds
//#     
//#     Date:           May, 1998
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
//# Modification history:
//#
//# Peter Kunszt, Oct. 1998    Add clear() method to ValVec
//# Peter Kunszt, Feb. 1999    Add keep() method to ValVec
//#
//# Peter Kunszt, Feb. 1999    Add new template LinPool
//# Peter Kunszt, Apr. 1999    Add remove() method to ValVec
//# Peter Kunszt, Jul. 2000    Add new class VarStr

#include <VarVec.h>
#include <stdlib.h>
#include <new>
#include <string.h>

// This file defines the templates declared in VarVec.h, and should
// not be treated like a normal header file.  Include this file only
// in source modules where template instantiation will occur.

/* --- ValVec methods ------------------------------------------------------ */

// destructor

template<class T>
ValVec<T>::~ValVec( void )
{
   if ( vector_ ) {
      for ( size_t i = 0; i < capacity_; ++i ) vector_[i].~T();
      free( vector_ );
   }
   if ( pFill_ ) delete pFill_;
}

// default constructor

template<class T>
ValVec<T>::ValVec( size_t capacity, size_t increment )
{
   pFill_ = vector_ = 0;
   increment_ = length_ = capacity_ = 0;
   insert( capacity );
   increment_ = increment;
   length_ = 0;
}

// alternate constructor for defining the fill value

template<class T>
ValVec<T>::ValVec( const T &rFill, size_t capacity, size_t increment )
{
   vector_ = 0;
   increment_ = length_ = capacity_ = 0;
   pFill_ = new T(rFill);
   insert( capacity );
   increment_ = increment;
   length_ = 0;
}

// copy constructor

template<class T>
ValVec<T>::ValVec( const ValVec &orig )
{
   capacity_ = 0;
   pFill_ = vector_ = 0;
   *this = orig;
}

// assignment/copy operator

template<class T>
ValVec<T>&	ValVec<T>::operator =( const ValVec &orig )
{
   if ( &orig == this ) return *this;

   if ( orig.pFill_ )
      if ( pFill_ )
	 *pFill_ = *orig.pFill_;
      else
	 pFill_ = new T(*orig.pFill_);
   else
      if ( pFill_ ) {
	 delete pFill_;
	 pFill_ = 0;
      }

   if ( orig.capacity_ > capacity_ ) {
      increment_ = 1;
      length_ = capacity_;
      insert( orig.capacity_ - capacity_ );
   }

   for ( size_t i = 0; i < orig.length_; ++i ) vector_[i] = orig.vector_[i];

   increment_ = orig.increment_;
   length_ = orig.length_;
   return *this;
}

// bounds-checking array operator (const version)

template<class T>
const T&	ValVec<T>::operator []( size_t index ) const
{
   if ( index >= length_ )
      throw _BOUNDS_EXCEPTION( "ValVec", "vector_", length_, index );
   return vector_[index];
}

// bounds-checking array operator (non-const version)

template<class T>
T&	ValVec<T>::operator []( size_t index )	
{
   if ( index >= length_ )
      throw _BOUNDS_EXCEPTION( "ValVec", "vector_", length_, index );
   return vector_[index];
}

// at method: bounds-adjusting array operator

template<class T>
T&	ValVec<T>::at( size_t index )
{
   if ( index >= length_ ) insert( 1 + index - length_ );
   return vector_[index];
}

// append method: efficiently insert element at end of array

template<class T>
size_t	ValVec<T>::append( const T &t )
{
   (length_ < capacity_ ? vector_[length_++] : at(length_)) = t;
   return length_;
}

// insert method: insert and initialize new array elements

// Warning: if the constructor or destructor for class T throws an
// exception when invoked from this function, the affected vector is
// considered a complete loss and is left dangling.  The ValVec object
// is left in a consistent and usable state: either updated, not
// updated, or empty.  This potential memory leak is unfortunate, but
// I do not know of a better response.

template<class T>
size_t	ValVec<T>::insert( size_t count, size_t offset )
{
   if ( offset > length_ )
      throw _BOUNDS_EXCEPTION("ValVec::insert","offset greater than length");

   size_t newLength	= length_ + count;
   size_t start		= length_ - offset;
   size_t i;

   if ( newLength > capacity_ ) {
      // allocate new vector
      size_t cap = increment_ ? capacity_ + increment_ : 2 * capacity_;
      if ( newLength > cap ) cap = newLength;
      T *vec = (T*) malloc( cap * sizeof(T) );

      // bitwise copy original occupied region into new vector
      if ( length_ ) {
	 memcpy( vec, vector_, start * sizeof(T) );
	 memcpy( vec + start + count, vector_ + start, offset * sizeof(T) );
      }

      // construct newly occupied region with fill or default
      if ( pFill_ )
	 for ( i = 0; i < count; ++i ) ::new(vec+start+i) T(*pFill_);
      else
	 for ( i = 0; i < count; ++i ) ::new(vec+start+i) T;

      // construct new unoccupied region with default
      for ( i = newLength; i < cap; ++i ) ::new(vec+i) T;

      // replace old vector with new vector
      T *oldVec = vector_;
      size_t oldCap = capacity_;
      vector_ = vec;
      capacity_ = cap;

      // destroy original unoccupied region and free discarded vector
      if ( oldVec ) {
	 for ( i = length_; i < oldCap; ++i ) oldVec[i].~T();
	 free( oldVec );
      }
   }
   else if ( count ) {
	   if ( offset ) {
		   try {
			   // destroy obliterated portion of unoccupied region
			   for ( i = 0; i < count; ++i ) vector_[length_+i].~T();

			   // bitwise move displaced portion of occupied region
			   memmove(vector_+start+count, vector_+start, offset * sizeof(T));

			   // construct vacated region with fill or default
			   if ( pFill_ )
				   for ( i = 0; i < count; ++i ) ::new(vector_+start+i) T(*pFill_);
			   else
				   for ( i = 0; i < count; ++i ) ::new(vector_+start+i) T;
		   }
		   catch (...) {
			   vector_ = 0;
			   length_ = capacity_ = 0;
			   throw;
		   }
	   }
	   else if ( pFill_ ) {
		   for ( i = 0; i < count; ++i ) vector_[length_+i] = *pFill_;
	   }
   }

   return length_ = newLength;
}

// cut method: remove array elements

// Warning: see insert method warning, above, regarding a potential
// memory leak.  The only difference here is that in all cases the
// ValVec object is left empty.

template<class T>
size_t	ValVec<T>::cut( size_t count, size_t offset )
{
   if ( count + offset > length_ )
      throw _BOUNDS_EXCEPTION("ValVec::cut","count+offset greater than length");

   if ( count && offset )
      try {
	 size_t i;
	 T *start = vector_ + length_ - offset - count;

	 // destroy obliterated portion of occupied region
	 for ( i = 0; i < count; ++i ) start[i].~T();

	 // bitwise move displaced portion of occupied region
	 memmove( start, start + count, offset * sizeof(T) );

	 // construct vacated region with default
	 for ( i = 0; i < count; ++i ) ::new(start+offset+i) T;
      }
      catch (...) {
	 vector_ = 0;
	 length_ = capacity_ = 0;
	 throw;
      }

   return length_ -= count;
}

// fill method: create and return default fill value, or return existing value

template<class T>
T&	ValVec<T>::fill( void )
{
   return pFill_ ? *pFill_ : *(pFill_ = new T);
}

// fillExists method: return true if fill value exists

template<class T>
bool	ValVec<T>::fillExists( void ) const
{
	return ( pFill_ == NULL) ? false : true;
}

// unsetFill method: destroy existing fill value

template<class T>
void	ValVec<T>::unsetFill( void )
{
   if ( pFill_ ) {
      delete pFill_;
      pFill_ = 0;
   }
}

// clear method: reset every value to fill value if there is one

template<class T>
void	ValVec<T>::clear( void )
{
   if ( pFill_ ) {
     for(size_t i = 0; i < length_; i++)
       vector_[i] = *pFill_;
   }
}

// clear method: reset every value to fill value if there is one

template<class T>
void	ValVec<T>::sort( int (*compar)(const void*, const void*) )
{
  qsort(vector_, length_, sizeof(T), compar);
}

// keep method: just reset the length by count.

template<class T>
size_t	ValVec<T>::keep( size_t count )
{
   if ( count > length_ )
      throw _BOUNDS_EXCEPTION("ValVec::keep","count greater than length");

  return length_ -= count;
}

// remove method: call cut

template<class T>
void	ValVec<T>::remove( size_t offset )
{
   if ( offset >= length_ )
      throw _BOUNDS_EXCEPTION("ValVec::remove","count greater than length");

   cut(1, length_ - offset - 1);
   return;
}

/* --- PtrVec methods ------------------------------------------------------ */

// destructor

template<class T>
PtrVec<T>::~PtrVec( void )
{
   cut( length_ );
   if ( vector_ ) delete [] vector_;
}

// default constructor

template<class T>
PtrVec<T>::PtrVec( bool internal, size_t capacity, size_t increment )
{
   length_ = capacity_ = 0;
   increment_ = increment;
   internal_ = internal;
   if(capacity) {
     vector_ = new T* [capacity];
     for(size_t i = 0; i < capacity; i++) vector_[i] = NULL;
   } else {
     vector_ = NULL;
   }
   capacity_ = capacity;
}

// copy constructor

template<class T>
PtrVec<T>::PtrVec( const PtrVec &obj, bool internalize )
{
   vector_ = 0;
   length_ = capacity_ = 0;
   copy( obj, internalize );
}

// copy method: make either local or shared copy of external pointers

template<class T>
PtrVec<T>&	PtrVec<T>::copy( const PtrVec &obj, bool internalize )
{
   if ( &obj == this ) {
      if ( internalize != internal_ )
	 throw _INTERFACE_EXCEPTION("PtrVec::copy"
			,"attempt to ex/in-ternalize by self-assignment");
      return *this;
   }

   cut( length_ );
   increment_ = obj.increment_;
   internal_ = internalize;

   if ( obj.capacity_ > capacity_ ) {
      if ( vector_ ) delete [] vector_;
      capacity_ = 0;
      vector_ = new T* [obj.capacity_];
      for(size_t j = 0; j < obj.capacity_; j++) vector_[j] = NULL;
      capacity_ = obj.capacity_;
   }

   if ( internalize ) {
      for ( size_t i = 0; i < obj.length_; ++i )
	 vector_[i] = PtrDup( obj.vector_[i] );
   } else
      memcpy( vector_, obj.vector_, obj.length_ * sizeof(T*) );

   length_ = obj.length_;
   return *this;
}

// bounds-checking array operator (const version)

template<class T>
const T*&	PtrVec<T>::operator []( size_t index ) const
{
   if ( index >= length_ )
      throw _BOUNDS_EXCEPTION( "PtrVec", "vector_", length_, index );
   return (const T*&) vector_[index];
}

// bounds-checking array operator (non-const version)

template<class T>
T*&	PtrVec<T>::operator []( size_t index )	
{
   if ( index >= length_ )
      throw _BOUNDS_EXCEPTION( "PtrVec", "vector_", length_, index );
   return vector_[index];
}

// at method: bounds-adjusting array operator

template<class T>
T*&	PtrVec<T>::at( size_t index )
{
   if ( index >= length_ ) insert( 1 + index - length_ );
   return vector_[index];
}

// entries method: return the number of non-NULL entries in vector
// NOTE: this is different from length(), which also counts NULL pointers.

template<class T>
size_t	PtrVec<T>::entries( void ) const
{
  size_t i, nEntries = 0;
  for( i = 0; i < length_; i++ )
    if( vector_[i] )
      nEntries++;
  return nEntries;
}

// append method: efficiently insert element at end of array

template<class T>
size_t	PtrVec<T>::append( T *t )
{
   (length_ < capacity_ ? vector_[length_++] : at(length_)) = t;
   return length_;
}

// insert method: insert and initialize new array elements

template<class T>
size_t	PtrVec<T>::insert( size_t count, size_t offset )
{
   if ( offset > length_ )
      throw _BOUNDS_EXCEPTION("PtrVec::insert","offset greater than length");

   size_t newLength	= length_ + count;
   size_t start		= length_ - offset;

   if ( newLength > capacity_ ) {
      size_t cap = increment_ ? capacity_ + increment_ : 2 * capacity_;
      if ( newLength > cap ) cap = newLength;
      T **vec = new T* [cap];
      for(size_t j = 0; j < cap; j++) vec[j] = NULL;
      if ( vector_ ) {
	 memcpy( vec, vector_, start * sizeof(T*) );
	 memcpy( vec + start + count, vector_ + start, offset * sizeof(T*) );
	 delete [] vector_;
      }
      capacity_ = cap;
      vector_ = vec;
   }
   else if ( count )
      memmove( vector_ + start + count, vector_ + start, offset * sizeof(T*) );

   memset( vector_ + start, 0, count * sizeof(T*) );

   return length_ = newLength;
}


// nextEmpty method: return index of next available (empty) slot in vector.

template<class T>
size_t	PtrVec<T>::nextEmpty()
{
  size_t empty;
  for( empty = 0; empty < length_; empty++ )
    if( vector_[empty] == NULL )
      break;
  return empty;
}

 
// add method: add given element into first available empty slot.

template<class T>
size_t	PtrVec<T>::add( T *t )
{
  size_t index;
  if( (index = nextEmpty()) == length_ ) 
    append( t );
  else
    vector_[index] = t;
  return index;
}

// index method: return the index of the given element if it exists in the
//               array, else return -1.

template<class T>
int   	PtrVec<T>::index( const T *t )
{
  size_t index;
  for( index = 0; index < length_; index++ )
    if( vector_[index] == t )
      return index;
  return -1;
}

// cut method: remove array elements

template<class T>
size_t	PtrVec<T>::cut( size_t count, size_t offset )
{
   if ( count + offset > length_ )
      throw _BOUNDS_EXCEPTION("PtrVec::cut","count+offset greater than length");

   if ( count ) {
      T **start = vector_ + length_ - offset - count;
      if ( internal_ )
	 for ( size_t i = 0; i < count; ++i ) if ( start[i] ) delete start[i];
      memmove( start, start + count, offset * sizeof(T*) );
   }

   return length_ -= count;
}

// remove method: call cut

template<class T>
void	PtrVec<T>::remove( size_t offset )
{
   if ( offset >= length_ )
      throw _BOUNDS_EXCEPTION("PtrVec::remove","count greater than length");

   cut(1, length_ - offset - 1);
   return;
}

// internalize method: replace shared pointers with local copies

template<class T>
void	PtrVec<T>::internalize( void )
{
   if ( ! internal_ ) {
      for ( size_t i = 0; i < length_; ++i )
	 vector_[i] = PtrDup( vector_[i] );
      internal_ = true;
   }
}

// externalize method: change status of pointers from local to shared

template<class T>
void	PtrVec<T>::externalize( void )
{
   internal_ = false;
}

// internal method: return true if pointers are local

template<class T>
bool	PtrVec<T>::internal( void ) const
{
   return internal_;
}


/* --- LinPool methods ----------------------------------------------------- */

// destructor

template<class T>
LinPool<T>::~LinPool( void )
{
  // delete all objects from pool capacity
  for ( size_t i = 0; i < capacity_; ++i )
    if ( vector_[i] ) delete vector_[i]; // failsafe - if someone deletes above

  if ( vector_ ) delete [] vector_;
}

// default constructor

template<class T>
LinPool<T>::LinPool( size_t capacity, size_t increment )
{
   length_ = capacity_ = 0;
   increment_ = increment;
   vector_ = capacity ? new T* [capacity] : 0;
   capacity_ = capacity;
   // generate new objects for the capacity
   for ( size_t i = 0; i < capacity; i++ )
     vector_[i] = new T;
}

// bounds-checking array operator (const version)

template<class T>
const T*&	LinPool<T>::operator []( size_t index ) const
{
   if ( index >= length_ )
      throw _BOUNDS_EXCEPTION( "LinPool", "vector_", length_, index );
   return (const T*&) vector_[index];
}

// bounds-checking array operator (non-const version)

template<class T>
T*&	LinPool<T>::operator []( size_t index )	
{
   if ( index >= length_ )
      throw _BOUNDS_EXCEPTION( "LinPool", "vector_", length_, index );
   return vector_[index];
}

// use method: bounds-adjusting function

template<class T>
T*&	LinPool<T>::use( void )
{
  if(length_ == capacity_) { // last element reached, extend!
    size_t cap = increment_ ? capacity_ + increment_ : 2 * capacity_;
    T **vec = new T* [cap];
    if ( vector_ ) {
      memcpy( vec, vector_, capacity_ * sizeof(T*) );
      delete [] vector_;
    }
    for( size_t i = capacity_; i < cap; i++)
      vec[i] = new T;
    capacity_ = cap;
    vector_ = vec;
  }
  return vector_[length_++];
}

