# -make_default makes constructor/destructor for structs
#swig -make_default -python -o cosmolib_wrap.c cosmolib.i
swig -nodefaultctor -python -o cosmolib_wrap.c cosmolib.i
