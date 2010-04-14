//#     Filename:       VarVec.h
//#
//#     ValVec and PtrVec templates
//#
//#
//#     Author:         John Doug Reynolds
//#     
//#     Creation Date:  May, 1998
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
//#
//# Peter Kunszt, Jul. 2000    Add VarStr class
//# Peter Kunszt, Aug. 2000    Add VarStrToken class

#ifndef VARVEC_H
#define VARVEC_H

#ifndef _BOUNDS_EXCEPTION

#ifdef SXDB
#   include <sxException.h>
#   define _BOUNDS_EXCEPTION sxBoundsError
#   define _INTERFACE_EXCEPTION sxInterfaceError
#else
#   include <SpatialException.h>
#   define _BOUNDS_EXCEPTION SpatialBoundsError
#   define _INTERFACE_EXCEPTION SpatialInterfaceError
#endif

#endif

#include <sys/types.h>

/** Dynamic array of arbitrary values

    This is a template for a general-purpose dynamic array.  The array
    grows automatically as needed, but reallocation occurs only when
    the length exceeds the capacity.  The capacity is increased in
    large blocks, the size of which may be optimized.  A fill value may
    be defined, in which case it is used to initialize new elements of
    the array, but not new capacity.  Which is to say that
    initialization is deferred until the array grows into its capacity.
    The public data member, increment_, specifies the amount by which
    the capacity is increased during reallocation.  By default,
    increment_ is zero, which causes the capacity to double upon each
    reallocation.  A non-zero increment_ is simply added to the
    capacity upon each reallocation.  The capacity is extended by this
    amount or by whatever greater amount is necessary to accommodate
    the new length of the array.
*/

template<class T> class ValVec {
public:
  /** Destructor. */

  ~ValVec( void );

  /** Default constructor.
      optionally specify initial capacity and
      reallocation increment.  */

  ValVec( size_t capacity = 0, size_t increment = 0 );

  /** Alternate constructor.
      define a fill value in addition to the
      parameters of the default constructor.  class T must have
      well-defined copy semantics.  The fill value does not exist
      unless it is defined.  */

  ValVec( const T &fill, size_t capacity, size_t increment );

  /** Copy constructor.  
      The initial capacity is the current capacity of the duplicated array.  */

  ValVec( const ValVec& );

  /** Assignment/copy operator.
      does not decrease the capacity. */

  ValVec&	operator =( const ValVec& );

  /** Efficient array operator (const version): no bounds checking. */

  const T&	operator ()( size_t index ) const { return vector_[index]; }

  /** Efficient array operator (non-const version): no bounds checking. */

  T&		operator ()( size_t index ) { return vector_[index]; }

  /** Bounds-checking array operator (const version): throws sxBoundsError. */

  const T&	operator []( size_t index ) const;

  /** Bounds-checking array operator (non-const version): throws sxBoundsError.
   */

  T&		operator []( size_t index );

  /** Bounds-adjusting array operator.  Returns the array
      element at the specified index, extending the array as necessary
      to bring it within bounds.  The fill value, if defined, is the
      initializer for any new elements. */

  T&		at( size_t index );

  /** Returns current occupied length of array.
   */

  size_t	length( void ) const { return length_; }

  /** Efficiently insert given element at end of array.
      Avoids redundant initialization of new array element, except for
      when a reallocation is required.  Returns the new length. */

  size_t	append( const T& );

  /** Insert new array elements.  
      Count specifies the number of new elements, and offset specifies
      where in the array to insert them.  By default the new elements
      are appended.  The fill value, if defined, is the initializer
      for the new elements.  offset refers to the end of the array:
      the first new element is located at index (length - offset).
      Returns the new length.  Throws sxBoundsError if offset is
      greater than length. */

  size_t	insert( size_t count, size_t offset = 0 );

  /** Remove array elements.  

      count specifies the number of elements to remove, and offset
      specifies which elements to remove.  By default elements are
      removed from the end of the array.  The unused capacity grows by
      this amount.  offset refers to the end of the array: the first
      removed element is located at index (length - offset - count).
      Returns the new length.  Throws sxBoundsError if (offset+count)
      is greater than length. */

  size_t	cut( size_t count, size_t offset = 0 );

  /** Removes the element specified by offset.
      This is basically a wrapper for the cut method cut(1, length-offset-1) */

  void		remove( size_t offset );

  /** Cut but keep capacity. 
      Just like the cut method, it resets the length of the vector by
      count, but it always starts from the end.  The elements, however
      are not deleted and rebuilt with the default, but rather left as
      they are for the user to reuse. */

  size_t	keep( size_t count );

  /** Return the fill value, defining it if necessary.
      If the fill value is not defined, a default value is created
      using the default constructor for class T.  The returned object
      is an lvalue, to which a new fill value may be assigned. */

  T&		fill( void );

  /** Returns true if the fill value is defined. */

  bool		fillExists( void ) const;

  /** Undefine and destroy the current fill value.
      (if it is defined) */

  void		unsetFill( void );


  /** Reset every value to the fill value. If no fill is
      defined, nothing is done!  */

  void		clear( void );

  /** Do a qsort */

  void		sort( int (*compar)(const void*, const void*) );

  /// Linear growth increment */
  size_t			increment_;
  /// dynamic array of values */
  T				*vector_;
private:
   size_t			length_;	// occupied length of vector
   size_t			capacity_;	// allocated length of vector
   T				*pFill_;	// pointer to fill value
};

/** Dynamic array of pointers

    This is a template for a dynamic array of pointers.  The design is
    very similar to the general-purpose version, ValVec, but
    specialized according to the memory management issues peculiar to
    storing pointers.  This version uses nil as the fill value, which
    cannot be customized or disabled.  By default, the cut method uses
    the delete operator to free pointers as they are removed from the
    array.  Before disposing the array, the destructor clears it out
    with cut.  This behavior is avoided if the pointers are designated
    as external, or shared.  In that case it is entirely up to the
    user to free pointers left dangling by cut.  When copying the
    array, one must choose whether the copy will share with the
    original the objects referenced by the pointers (shallow copy), or
    whether the copy will have its own internal duplicates.

<p>
<b>
           ---------------   WARNING   ---------------
</b>

<p>

    The user must provide a specialization of PtrDup() for every
    polymorphic class that will instantiate this template.  Failure to
    do this may result in unexpected truncation of derived objects.
    The template methods use PtrDup() when duplication of objects is
    required, but duplicating a polymorphic object requires assistance
    from the object itself.  For example, consider class B:

<pre>
 class B { public:
  virtual B* duplicate(void) const = 0;
 };
 inline B* PtrDup(const B *b) { return b ? b->duplicate() : 0; }
</pre>

    To avoid confusion and mistakes, place the specialization
    immediately after the declaration of class B.  If class B does not
    have a duplicator use the following specialization instead:

<pre>
 inline B* PtrDup(const B *b) {
  if (b) throw sxUnimplemented("PtrDup","class B has no duplicator");
  else return 0;
 }
</pre>
*/

template<class T> class PtrVec {
public:
  /** Destructor. */

  ~PtrVec( void );

  /** Default constructor.
      optionally specify initial capacity, reallocation increment, and
      whether the pointers are internal. */

  PtrVec( bool internal=true, size_t capacity=0, size_t increment=0 );

  /** Copy constructor.
      optionally specify either a shallow copy or an internalized copy
      (the default).  The initial capacity is the current capacity of
      the duplicated array. */

  PtrVec( const PtrVec&, bool internalize = true );

  /** Copy method. does not decrease the capacity.  The copy is
      shallow if internalize is false; otherwise the referenced
      objects are duplicated and the copy is internal, in which case
      class T must have well-defined copy semantics. */

  PtrVec&	copy( const PtrVec&, bool internalize );

  /** Assignment/copy operator. does not decrease the capacity.  This
      is not a shallow copy, so class T must have well-defined copy
      semantics. */

  PtrVec&	operator =( const PtrVec &obj ) { return copy(obj,true); }

  /** Efficient array operator (const version): no bounds checking. */

  const T*&	operator ()( size_t index ) const { return (const T*&) vector_[index]; }

  /** Efficient array operator (non-const version): no bounds checking. */

  T*&		operator ()( size_t index ) { return vector_[index]; }

  /** Bounds-checking array operator (const version): throws sxBoundsError. */

  const T*&	operator []( size_t index ) const;

  /** Bounds-checking array operator (non-const version): throws sxBoundsError.
   */

  T*&		operator []( size_t index );

  /** Bounds-adjusting array operator.  Returns the array
      element at the specified index, extending the array as necessary
      to bring it within bounds.  Any new elements are set to nil. */

  T*&		at( size_t index );

  /** Returns current occupied length of array. */

  size_t	length( void ) const { return length_; }

  /** Return the number of non-NULL entries in vector.
      NOTE: this is different from length(), which also counts NULL pointers.
  */

  size_t	entries( void ) const;

  /** Append: efficiently insert given element at end of array.
      Avoids redundant initialization of new array element, except for
      when a reallocation is required.  Returns the new length. */

  size_t	append( T* );

  /** Insert new array elements.  

      count specifies the number of new elements, and offset specifies
      where in the array to insert them.  By default the new elements
      are appended.  The new elements are initialized with the nil
      pointer value.  offset refers to the end of the array: the first
      new element is located at index (length - offset).  Returns the
      new length.  Throws sxBoundsError if offset is greater than
      length. */

  size_t	insert( size_t count, size_t offset = 0 );

  /** Return the index of the next available empty slot. */

   size_t	nextEmpty();

  /** Add a new array element in the first available empty slot.  
      This should be used when the ordering of elements is not
      important and empty slots are to be minimized.  Returns the
      index of the newly inserted element. */

  size_t	add( T* );

  /** Return the index of the given element. (If it exists in
      the array, else return -1.) */

  int  	index( const T* );

  /** Remove array elements.  
      count specifies the number of elements to remove, and offset
      specifies which elements to remove.  By default elements are
      removed from the end of the array.  The unused capacity grows by
      this amount.  offset refers to the end of the array: the first
      removed element is located at index (length - offset - count).
      Returns the new length.  Throws sxBoundsError if (offset+count)
      is greater than length. */

  size_t	cut( size_t count, size_t offset = 0 );

  /** Removes the element specified by offset.
      This is basically a wrapper for the cut method 
      <pre>
      cut(1, length-offset-1)
      <pre>
  */

  void		remove( size_t offset );

  /** Replace external pointers with internal
      copies of the referenced objects.  class T must have
      well-defined copy semantics.  Does nothing if already internal. */

  void		internalize( void );

  /** Change status of pointers to external.
      Beware that internalize does not undo this; once the pointers
      are designated external they cannot just be re-designated as
      internal.  Confusion on this point will yield dangling pointers. */

  void		externalize( void );

  /** internal method: returns true if the pointers are internal. */

  bool		internal( void ) const;

  /// linear growth increment
  size_t			increment_;
  /// dynamic array of pointers
  T				**vector_;
private:
  bool				internal_;	// delete dangling pointers
  size_t			length_;	// occupied length of vector
  size_t			capacity_;	// allocated length of vector
};


/** Dynamic linear pool of objects.

    This is a template for a dynamic pool of objects.  The design is
    very similar to the dynamic array of pointers.  A pool is defined
    to be an array of pointers to preallocated default objects.
    Whenever a new object is needed, it can be accessed from the pool
    with the use() member function.  The size of the pool extends
    automatically if its initial limit is reached.  This pool is a
    linear pool, i.e. we can rely upon their index to be sequential.
    So in order to return a specific object into the pool's disposal,
    all objects having larger indices have to be free (like a reverse
    LIFO - last out first back). The free member function returns
    objects to the pool's disposal. Upon destruction, all pool objects
    are destroyed using their destructor.  It does not make sense to
    have a copy constructor or assignment op.
*/

template<class T> class LinPool {
public:
  /**  Destructor. */

  ~LinPool( void );

  /** Default constructor.
      optionally specify initial capacity,
      reallocation increment, and whether the pointers are internal.
  */

  LinPool( size_t capacity=0, size_t increment=0 );

  /** Efficient array operator (const version): no bounds checking. */

  const T*&	operator ()( size_t index ) const { return (const T*&) vector_[index]; }

  /** Efficient array operator (non-const version): no bounds checking. */

  T*&		operator ()( size_t index ) { return vector_[index]; }

  /** Bounds-checking array operator (const version): throws sxBoundsError. */

  const T*&	operator []( size_t index ) const;

  /** Bounds-checking array operator (non-const version): throws sxBoundsError.
   */

  T*&		operator []( size_t index );

  /** Use Method. Bounds-adjusting operator that returns the pool
      element as an lvalue to be used by the user. It is a combination
      of append() and at() of VarVec and PtrVec.  If the bounds of the
      pool array are reached, it is extended by increment_.
  */

  T*&		use( void );

  /** Returns current occupied length of the pool. */

  size_t	length( void ) const { return length_; }

  /** Declare pool objects as free for new use.
      If no argument is given, all elements are free to use again.
      Else, the number of elements specified is freed up from the end. */

  size_t	free( size_t count = 0) { return (length_ = count ? length_ - count : 0); }

  /// linear growth increment
  size_t			increment_;
  /// dynamic array of pointers
  T				**vector_;
private:
  size_t			length_;	// occupied length of vector
  size_t			capacity_;	// allocated length of vector
};

#endif /* VARVEC_H */
