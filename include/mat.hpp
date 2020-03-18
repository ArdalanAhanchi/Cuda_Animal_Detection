#ifndef MAT_HPP
#define MAT_HPP

#include <string>

namespace anr {

typedef float type;

/**
 *  A class which represents a matrix, and some basic operations supported on it.
 *  This class is used for training neural networks. It also implements
 *  reference counting which allows simple assignment operations.
 *
 *  @author Ardalan Ahanchi
 *  @date March 2020
 */
class Mat
{
  public:

    /**
     *  Default constructor which doesn't allocate anything. Creates a 0x0 matrix.
     */
    Mat();


    /**
     *  Cosntructor which creates a matrix of rows x cols, it allocates data on the heap
     *  and initialzies the matrix dimentions.
     *
     *  @param rows The number of rows in the matrix.
     *  @param cols The number of columns in the matrix.
     */
    Mat(const size_t& rows, const size_t& cols);

    
    /**
     *  Copy osntructor which creates a a shallow copy of the passed matrix.
     *
     *  @param copy The other matrix which we're copying.
     *  @param transpose True if the new matrix should be transposed, false otherwise.
     */
    Mat(const Mat& copy);


    /**
     *  Copy cosntructor which creates a matrix similar to the one passed by a deep copy. 
     *  If the transpose value is set, it will also transpose it during the deep copy.
     *
     *  @param copy The other matrix which we're copying.
     *  @param transpose True if the new matrix should be transposed, false otherwise.
     */
    Mat(const Mat& copy, bool transpose);


    /**
     *  Destructor which deallocates the data and reference count if needed.
     */
    ~Mat();


    /** 
     *  A getter for the number of rows in the matrix.
     *
     *  @return The number of rows.
     */
    size_t rows() const;


    /** 
     *  A getter for the number of cols in the matrix.
     *
     *  @return The number of rows.
     */
    size_t cols() const;

    
    /**
     *  A getter which returns a reference to the element pointed by it. It is used to
     *  convert rows/cols to the flat memory layout used by this object.
     *
     *  @param row The current row in the matrix.
     *  @param col The current col in the matrix.
     */
    inline type& at(const size_t& row, const size_t& col) {
        return this->data[row * this->_cols + col];
    }

    /**
     *  A getter which returns a constnat reference to the element pointed by it.
     *  It is used to convert rows/cols to the flat memory layout used by this object.
     *
     *  @param row The current row in the matrix.
     *  @param col The current col in the matrix.
     */
    inline const type& get(const size_t& row, const size_t& col) const {
        return this->data[row * this->_cols + col];
    }


    /**
     *  A method which initializes all the matrix elements with random values with.
     *  a certain range (minimum and maximum). 
     *
     *  @param min The minimum value.
     *  @param max The maximum value.
     */
    void randomize(const type& min, const type& max);
    

    /**
     *  A method which prints the current matrix to stdout. It also displays a message
     *  which is a title the user passes to it.
     *
     *  @param title An optional string which can be printed before the matrix.
     */
    void print(const std::string title="") const;


    /**
     *  Assignment operator overload which creates a shallow copy of the passed object.
     *  It also calls the deallocate if it was needed (IE this is the only reference).
     *
     *  @param other The other mat object which we're assigning to this one.
     *  @return A pointer to this mat object.
     */
    Mat& operator=(const Mat& other);


    type* data;         /**< A pointer to a column major array of data for this matrix. */

  private:

    /**
     *  A method which deallocates the data and reference count, only if the reference 
     *  count is already 0 (not referenced elsewhere).
     */
    void deallocate();

    size_t* _ref_count; /**< Reference counter for data (provides fast, and nice moves).*/

    size_t _rows;       /**< The number of rows in this matrix. */
    size_t _cols;       /**< The number of columns in this matrix. */
};

}

#endif
