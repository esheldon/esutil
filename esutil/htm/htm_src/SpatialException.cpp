//# SpatialException.cpp
//#
//# Author:    John Doug Reynolds, P. Kunszt
//# 
//# Creation:  March 1997
//#
//#
//# (c) Copyright The Johns Hopkins University 1995
//# All Rights Reserved
//#
//# The software and information contained herein are proprietary to The
//# Johns Hopkins University, Copyright 1995.  This software is furnished
//# pursuant to a written license agreement and may be used, copied,
//# transmitted, and stored only in accordance with the terms of such
//# license and with the inclusion of the above copyright notice.  This
//# software and information or any other copies thereof may not be
//# provided or otherwise made available to any other person.
//#
//#
//# Modification history:
//#
//# Oct. 1998, P. Kunszt : remove Rogue Wave C-string dependency
//#                        almost all the interface had to be rewritten.
//#			   Also, use some of the inheritance to avoid
//#			   code duplication. Introduced defaultstr[].

#include <string.h>
#include <string>
#include <SpatialException.h>

/* --- SpatialException methods ------------------------------------------------- */
const char *
SpatialException::defaultstr[] = {
  "SDSS Science Archive",
  "generic exception",			// These specialized exceptions are
  "unimplemented functionality",	// currently implemented. If no string
  "failed operation",			// is given, this is the standard
  "array bounds violation",		// message.
  "interface violation"
};

#define CONTEXT 	0		// indices of exceptions
#define GENERIC 	1
#define UNIMPLEMENTED 	2
#define FAILURE 	3
#define BOUNDS 		4
#define INTERFACE	5

// The message is held in a std::string; if building it fails for lack of
// memory there is nothing sensible to report, so the message is left empty.

static std::string int2str(int32 val)
{
  return std::to_string((long long) val);
}

SpatialException::~SpatialException() noexcept
{
}

SpatialException::SpatialException( const char *cstr, int defIndex ) noexcept
{
   try {
     if ( cstr ) {
       str_ = cstr;
     } else {
       str_ = defaultstr[CONTEXT];
       str_ += " : ";
       str_ += defaultstr[defIndex];
     }
   }
   catch (...) {
     clear();
   }
}

SpatialException::SpatialException( const char *context, const char *because,
			  int defIndex) noexcept
{
   try {
     str_ = context ? context : defaultstr[CONTEXT];
     str_ += " : ";
     str_ += because ? because : defaultstr[defIndex];
   }
   catch (...) {
     clear();
   }
}

SpatialException::SpatialException( const SpatialException& oldX ) noexcept
{
  try {
    str_ = oldX.str_;
  }
  catch (...) {
    clear();
  }
}

SpatialException& SpatialException::operator=( const SpatialException& oldX ) noexcept
{
   try {
     if(&oldX != this) { // beware of self-assignment
       str_ = oldX.str_;
     }
   }
   catch (...) {
     clear();
   }
   return *this;
}

const char *SpatialException::what() const noexcept
{
   return str_.c_str();
}

int SpatialException::slen(const char *str) const
{
  if(str)return strlen(str);
  return 0;
}

void SpatialException::clear()
{
  str_.clear();
}
/* --- SpatialUnimplemented methods --------------------------------------------- */

SpatialUnimplemented::SpatialUnimplemented( const char *cstr ) noexcept
: SpatialException(cstr,UNIMPLEMENTED)
{
}

SpatialUnimplemented::SpatialUnimplemented( const char *context, const char *because )
   noexcept
  : SpatialException(context, because, UNIMPLEMENTED)
{
}

SpatialUnimplemented::SpatialUnimplemented( const SpatialUnimplemented& oldX ) noexcept
  : SpatialException(oldX)
{
}

/* --- SpatialFailure methods --------------------------------------------------- */

SpatialFailure::SpatialFailure( const char *cstr ) noexcept
  : SpatialException(cstr, FAILURE)
{
}

SpatialFailure::SpatialFailure( const char *context, const char *because ) noexcept
  : SpatialException(context,because,FAILURE)
{
}

SpatialFailure::SpatialFailure( const char *context, const char *operation
		      , const char *resource, const char *because ) noexcept
{
   try {
      if ( !operation && !resource && !because ) {
	 because = "failed operation";
      }
      if ( !context )
	context = defaultstr[CONTEXT];

      str_ = context;
      str_ += ": ";
      if ( operation ) {
	 str_ += " ";
	 str_ += operation;
	 str_ += " failed ";
      }
      if ( resource ) {
	 str_ += operation ? " on \"" : " trouble with \"";
	 str_ += resource;
	 str_ += "\"";
      }
      if ( because ) {
	 str_ += ( operation || resource ) ? " because " : " ";
	 str_ += because;
      }
   }
   catch (...) {
     clear();
   }
}

SpatialFailure::SpatialFailure( const SpatialFailure& oldX ) noexcept
  : SpatialException(oldX)
{
}

/* --- SpatialBoundsError methods ----------------------------------------------- */

SpatialBoundsError::SpatialBoundsError( const char *cstr ) noexcept
  : SpatialException(cstr,BOUNDS)
{
}

SpatialBoundsError::SpatialBoundsError( const char *context, const char *array
			      , int32 limit, int32 index ) noexcept
  : SpatialException(context,array,BOUNDS)
{
   try {
     if ( limit != -1 ) {
	 if ( array ) {
	   str_ += "[" + int2str(index) + "]";
	 }
	 else {
	   str_ += " array index " + int2str(index) + " ";
	 }

	 if ( index > limit ) {
	   str_ += " over upper bound by " + int2str(index - limit);
	 }
	 else {
	   str_ += " under lower bound by " + int2str(limit - index);
	 }
      }
   }
   catch (...) {
     clear();
   }
}

SpatialBoundsError::SpatialBoundsError( const SpatialBoundsError& oldX ) noexcept
  : SpatialException(oldX)
{
}

/* --- SpatialInterfaceError methods -------------------------------------------- */

SpatialInterfaceError::SpatialInterfaceError( const char *cstr ) noexcept
  : SpatialException(cstr,INTERFACE)
{
}

SpatialInterfaceError::SpatialInterfaceError( const char *context, const char *because )
   noexcept
  : SpatialException(context,because,INTERFACE)
{
}

SpatialInterfaceError::SpatialInterfaceError( const char *context, const char *argument
				 , const char *because ) noexcept
{
   try {
      if ( !context )
	context = defaultstr[CONTEXT];

      str_ = context;
      str_ += ": ";
      if ( argument ) {
	 if ( because ) {
	   str_ += " argument \"";
	   str_ += argument;
	   str_ += "\" is invalid because ";
	   str_ += because;
	   str_ += " ";
	 }
	 else {
	   str_ += " invalid argument \"";
	   str_ += argument;
	   str_ += "\" ";
	 }
      }
      else if ( because ) {
	 str_ += " ";
	 str_ += because;
      }
      else {
	 str_ += " interface violation";
      }
   }
   catch (...) {
     clear();
   }
}

SpatialInterfaceError::SpatialInterfaceError( const SpatialInterfaceError& oldX ) noexcept
  : SpatialException(oldX)
{
}
