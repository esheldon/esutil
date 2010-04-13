//#     Filename:       SpatialInterface.hxx
//#
//#     Interfaceion interface inline methods
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

///////////LOOKUP METHODS////////////////////

inline
uint64 htmInterface::lookupID(float64 ra, float64 dec) const {
  return index_->idByPoint(ra,dec);
}

inline
uint64 htmInterface::lookupID(float64 x, float64 y, float64 z) const {
  SpatialVector v(x,y,z);
  return index_->idByPoint(v);
}

inline
uint64 htmInterface::lookupID(char *nm) const {
  return index_->idByName(nm);
}

inline
const char * htmInterface::lookupName(float64 ra, float64 dec) {
  index_->nameByPoint(ra,dec,name_);
  return name_;
}

inline
const char * htmInterface::lookupName(float64 x, float64 y, float64 z) {
  SpatialVector v(x,y,z);
  index_->nameByPoint(v,name_);
  return name_;
}

inline
const char * htmInterface::lookupName(uint64 id) {
  index_->nameById(id,name_);
  return name_;
}

//////////OTHERS/////////////////////////////

inline
void htmInterface::changeDepth(size_t depth, size_t saveDepth) {
  if(index_->maxlevel_ != depth || index_->buildlevel_ != saveDepth) {
    delete index_;
    index_ = new SpatialIndex(depth, saveDepth);
  }
}


inline
const SpatialIndex & htmInterface::index() const {
  return *index_;
}
