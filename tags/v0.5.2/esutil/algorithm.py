"""
Module:
    algorithm
Functions:

    Sorting algorithms:
        These were implemented because the numerical python sort() method
        doesn't work as expected for memory mapped arrays.  It reads the entire
        array into memory to do the sort.

        quicksort(data): 
            Run quick sort on an array or list, or something similar with a []
            operator.  Taken from http://hetland.org/coding/python/quicksort.html
            
        quicksort_keyvalue(keys,values): 
            Run quick sort on key value pairs, which should be in lists, arrays or
            something similar with a [] operator.  The sorting is performed on
            the keys. Based on http://hetland.org/coding/python/quicksort.html

"""

def quicksort(data):
    """
    Name:
        quicksort
    Purpose:
        Run a quicksort on the input data
    Calling Sequence:
        quicksort(data)

    Inputs:
        data: Should support the [] operator for getting and setting values.
    Notes:
        Taken from: http://hetland.org/coding/python/quicksort.html
        See also the quicksort_keyvalue function.
    """

    start = 0
    end =len(data)-1
    _quicksort(data, start, end)
 

def _quicksort(data, start, end):
    if start < end:                            # If there are two or more elements...
        split = partition(data, start, end)    # ... partition the subdata...
        _quicksort(data, start, split-1)        # ... and sort both halves.
        _quicksort(data, split+1, end)
    else:
        return


def partition(data, start, end):
    pivot = data[end]                          # Partition around the last value
    bottom = start-1                           # Start outside the area to be partitioned
    top = end                                  # Ditto

    done = 0
    while not done:                            # Until all elements are partitioned...

        while not done:                        # Until we find an out of place element...
            bottom = bottom+1                  # ... move the bottom up.

            if bottom == top:                  # If we hit the top...
                done = 1                       # ... we are done.
                break

            if data[bottom] > pivot:           # Is the bottom out of place?
                data[top] = data[bottom]       # Then put it at the top...
                break                          # ... and start searching from the top.

        while not done:                        # Until we find an out of place element...
            top = top-1                        # ... move the top down.
            
            if top == bottom:                  # If we hit the bottom...
                done = 1                       # ... we are done.
                break

            if data[top] < pivot:              # Is the top out of place?
                data[bottom] = data[top]       # Then put it at the bottom...
                break                          # ...and start searching from the bottom.

    data[top] = pivot                          # Put the pivot in its place.
    return top                                 # Return the split point




def quicksort_keyvalue(keys, data):
    """
    Name:
        quicksort_keyvalue
    Purpose:
        Run a quicksort on the input key-value pairs.  The sort
        is performed based on the keys.
    Calling Sequence:
        quicksort_keyvalue(keys, values)

    Inputs:
        keys: Should support the [] operator for getting and setting values.
        values: Should support the [] operator for getting and setting values.
    Notes:
        Based on: http://hetland.org/coding/python/quicksort.html
        See also the quicksort function.
    """


    start = 0
    end =len(data)-1
    _quicksort_keyvalue(keys, data, start, end)
 



def partition_keyvalue(keys, data, start, end):
    pivot = keys[end]                          # Partition around the last value
    pivot_data = data[end]

    bottom = start-1                           # Start outside the area to be partitioned
    top = end                                  # Ditto

    done = 0
    while not done:                            # Until all elements are partitioned...

        while not done:                        # Until we find an out of place element...
            bottom = bottom+1                  # ... move the bottom up.

            if bottom == top:                  # If we hit the top...
                done = 1                       # ... we are done.
                break

            if keys[bottom] > pivot:           # Is the bottom out of place?
                keys[top] = keys[bottom]
                data[top] = data[bottom]       # Then put it at the top...
                break                          # ... and start searching from the top.

        while not done:                        # Until we find an out of place element...
            top = top-1                        # ... move the top down.
            
            if top == bottom:                  # If we hit the bottom...
                done = 1                       # ... we are done.
                break

            if keys[top] < pivot:              # Is the top out of place?
                keys[bottom] = keys[top]       # Then put it at the bottom...
                data[bottom] = data[top]       # Then put it at the bottom...
                break                          # ...and start searching from the bottom.

    keys[top] = pivot                          # Put the pivot in its place.
    data[top] = pivot_data                          # Put the pivot in its place.
    return top                                 # Return the split point




def _quicksort_keyvalue(keys, data, start, end):
    if start < end:                            # If there are two or more elements...
        split = partition_keyvalue(keys, data, start, end)    # ... partition the subdata...
        _quicksort_keyvalue(keys, data, start, split-1)        # ... and sort both halves.
        _quicksort_keyvalue(keys, data, split+1, end)
    else:
        return


