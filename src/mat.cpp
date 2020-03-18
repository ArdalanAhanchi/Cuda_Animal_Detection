/**
 *  The implementation for a class which represents a matrix, and some basic operations
 *  supported on it. This class is used for training neural networks. It also implements
 *  reference counting which allows simple assignment operations.
 *
 *  @author Ardalan Ahanchi
 *  @date March 2020
 */

#include "mat.hpp"

#include <cstdlib>                                //For random number generator.
#include <iostream>                               //For outputting errors.
#include <chrono>                                 //For getting the current time. 


namespace anr {

/**
 *  Default constructor which doesn't allocate anything. Creates a 0x0 matrix.
 */
Mat::Mat() {
    this->_rows = 0;
    this->_cols = 0;
    this->data = nullptr;
    this->_ref_count = nullptr;
}


/**
 *  Cosntructor which creates a matrix of rows x cols, it allocates data on the heap
 *  and initialzies the matrix dimentions.
 *
 *  @param rows The number of rows in the matrix.
 *  @param cols The number of columns in the matrix.
 */
Mat::Mat(const size_t& rows, const size_t& cols) {
    //Allocate data on the heap.
    this->data = new type[rows * cols];

    //Initialize the data to NULL (basically 0).
    for(size_t i = 0; i < rows * cols; i++)
        this->data[i] = 0;

    //Set the class variables.
    this->_rows = rows;
    this->_cols = cols;
    this->_ref_count = new size_t(0);
}


/**
 *  Copy constructor which creates a a shallow copy of the passed matrix.
 *
 *  @param copy The other matrix which we're copying.
 *  @param transpose True if the new matrix should be transposed, false otherwise.
 */
Mat::Mat(const Mat& copy) {
    //Copy the class variables.
    this->data = copy.data;
    this->_rows = copy.rows();
    this->_cols = copy.cols();
    this->_ref_count = copy._ref_count;

    //Increase the reference count (since it's being referenced again).
    *(this->_ref_count) += 1;
}


/**
 *  Copy cosntructor which creates a matrix similar to the one passed by a deep copy. 
 *  If the transpose value is set, it will also transpose it during the deep copy.
 *
 *  @param copy The other matrix which we're copying.
 *  @param transpose True if the new matrix should be transposed, false otherwise.
 */
Mat::Mat(const Mat& copy, bool transpose) {
    //Allocate data on this matrix to the same size of the copy.
    this->data = new type[copy.rows() * copy.cols()];

    //Check if the user wants to transpose the copy.
    if(transpose) {
        //Set the class variables so it's opposize (since it's transposed).
        this->_rows = copy.cols();
        this->_cols = copy.rows();

        //Copy the data over, and transpose at the same time.
        for(size_t r = 0; r < copy.rows(); r++)
            for(size_t c = 0; c < copy.cols(); c++)
                this->data[c * this->_cols + r] = copy.data[r * copy.cols() + c];
    } else {
        //Set the class variables to be the same as the other matrix.
        this->_rows = copy.rows();
        this->_cols = copy.cols();

        //Create a deep copy of the data.
        for(size_t r = 0; r < copy.rows(); r++)
            for(size_t c = 0; c < copy.cols(); c++)
                this->data[r * this->_cols + c] = copy.data[r * this->_cols + c];
    }

    //Initialize the reference count to 0.
    this->_ref_count = new size_t(0);
}


/**
 *  Destructor which deallocates the data and reference count if needed.
 */
Mat::~Mat() {
    this->deallocate();
}


/** 
 *  A getter for the number of rows in the matrix.
 *
 *  @return The number of rows.
 */
size_t Mat::rows() const {
    return this->_rows;
}


/** 
 *  A getter for the number of cols in the matrix.
 *
 *  @return The number of rows.
 */
size_t Mat::cols() const {
    return this->_cols;
}

/**
 *  A method which initializes all the matrix elements with random values with.
 *  a certain range (minimum and maximum). 
 *
 *  @param min The minimum value.
 *  @param max The maximum value.
 */
void Mat::randomize(const type& min, const type& max) {
    //Perform this to ensure the random number generation is truly random.
    std::srand(std::chrono::system_clock::now().time_since_epoch().count());

    //Iterate through, and generate random numbers for each tile.
    for(size_t r = 0; r < this->_rows; r++) {
        for(size_t c = 0; c < this->_cols; c++) {
            //Scale the random number, and create a new one within range.
            type scaled = (type) std::rand() / (type) RAND_MAX;
            this->data[r * this->_cols + c] = (scaled * (max - min)) + min;
        }
    }
}


/**
 *  A method which prints the current matrix to stdout. It also displays a message
 *  which is a title the user passes to it.
 *
 *  @param title An optional string which can be printed before the matrix.
 */
void Mat::print(const std::string title) const {
    //If title is not empty, print it.
    if(title != "")
        std::cout << title << std::endl;

    //Set the output to print up to four decimal digits.
    std::cout.setf(std::ios::fixed);
    std::cout.setf(std::ios::showpoint);
    std::cout.precision(4);

    for(size_t r = 0; r < this->_rows; r++) {           //Iterate, and print out to cout.
        std::cout << "| " ;

        for(size_t c = 0; c < this->_cols; c++)         //Print out the whole row.
            std::cout << this->data[r * this->_cols + c] << " | ";

        std::cout << std::endl;                         //Go to the next line.
    }
}


/**
 *  Assignment operator overload which creates a shallow copy of the passed object.
 *  It also calls the deallocate if it was needed (IE this is the only reference).
 *
 *  @param other The other mat object which we're assigning to this one.
 *  @return A pointer to this mat object.
 */
Mat& Mat::operator=(const Mat& other) {
    //Deallocate the current data.    
    this->deallocate();

    //Copy over the class variables.
    this->data = other.data;
    this->_rows = other.rows();
    this->_cols = other.cols();
    this->_ref_count = other._ref_count;

    //Increase the reference count (since it's being referenced again).
    *(this->_ref_count) += 1;
}


/**
 *  A method which deallocates the data and reference count, only if the reference 
 *  count is already 0 (not referenced elsewhere).
 */
void Mat::deallocate() {
    //Deallocate if not nullptr, and if not referenced.
    if(this->data != nullptr && *(this->_ref_count) <= 0) {
         delete[] this->data;
         this->data = nullptr;
    }
    
    //Deallocate the refence count if it's not being used.
    if(this->_ref_count != nullptr && *(this->_ref_count) <= 0) {
        delete this->_ref_count;
        this->_ref_count = nullptr;
    }

    //Reduce the reference count by one.
    if(this->_ref_count != nullptr)
        *(this->_ref_count) -= 1;
}

}
