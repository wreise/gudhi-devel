/*    This file is part of the Gudhi Library - https://gudhi.inria.fr/ - which is released under MIT.
 *    See file LICENSE or go to https://gudhi.inria.fr/licensing/ for full license details.
 *    Author(s):       Pawel Dlotko
 *
 *    Copyright (C) 2015 Inria
 *
 *    Modification(s):
 *      - YYYY/MM Author: Description of the modification
 */

#ifndef BITMAP_CUBICAL_COMPLEX_H_
#define BITMAP_CUBICAL_COMPLEX_H_

#include <gudhi/Bitmap_cubical_complex_base.h>
#include <gudhi/Bitmap_cubical_complex_periodic_boundary_conditions_base.h>

#ifdef GUDHI_USE_TBB
#include <tbb/parallel_sort.h>
#endif

#include <limits>
#include <utility>    // for pair<>
#include <algorithm>  // for sort
#include <vector>
#include <numeric>  // for iota
#include <cstddef>

namespace Gudhi {

namespace cubical_complex {

template <typename T>
class is_before_in_filtration;

/**
 * @brief Cubical complex represented as a bitmap.
 * @ingroup cubical_complex
 * @details This is a Bitmap_cubical_complex class. It joints a functionalities of Bitmap_cubical_complex_base and
 * Bitmap_cubical_complex_periodic_boundary_conditions_base classes into
 * Gudhi persistent homology engine. It is a template class that inherit from its template parameter. The template
 * parameter is supposed to be either Bitmap_cubical_complex_base or
 * Bitmap_cubical_complex_periodic_boundary_conditions_base class.
 *
 * This class implements the concept `FilteredComplex`.
 **/
template <typename T>
class Bitmap_cubical_complex : public T {
 public:
  //*********************************************//
  // Typedefs and typenames
  //*********************************************//
  typedef std::size_t Simplex_key;
  typedef typename T::filtration_type Filtration_value;
  typedef Simplex_key Simplex_handle;

  //*********************************************//
  // Constructors
  //*********************************************//
  // Over here we need to define various input types. I am proposing the following ones:
  // Perseus style
  // TODO(PD) H5 files?
  // TODO(PD) binary files with little endians / big endians ?
  // TODO(PD) constructor from a vector of elements of a type T. ?

  /**
   * @param[in] perseus_style_file The name of a \ref FileFormatsPerseus "Perseus-style file".
   **/
  explicit Bitmap_cubical_complex(const char* perseus_style_file)
      : T(perseus_style_file), key_associated_to_simplex(num_simplices()) {
#ifdef DEBUG_TRACES
    std::clog << "Bitmap_cubical_complex( const char* perseus_style_file )\n";
#endif
    std::iota(key_associated_to_simplex.begin(), key_associated_to_simplex.end(), std::size_t(0));
    // we initialize this only once, in each constructor, when the bitmap is constructed.
    // If the user decide to change some elements of the bitmap, then this procedure need
    // to be called again.
    this->initialize_simplex_associated_to_key();
  }

  /**
   * @param[in] dimensions The shape that should be used to interpret `cells` (in Fortran order).
   * @param[in] cells The filtration values of the top-dimensional cells if `input_top_cells` is `true`,
   * and of the vertices otherwise.
   * @param[in] input_top_cells If `true`, `cells` represents top-dimensional cells. If `false`, it represents vertices.
   **/
  Bitmap_cubical_complex(const std::vector<unsigned>& dimensions,
                         const std::vector<Filtration_value>& cells,
                         bool input_top_cells = true)
      : T(dimensions, cells, input_top_cells), key_associated_to_simplex(num_simplices()) {
    std::iota(key_associated_to_simplex.begin(), key_associated_to_simplex.end(), std::size_t(0));
    // we initialize this only once, in each constructor, when the bitmap is constructed.
    // If the user decide to change some elements of the bitmap, then this procedure need
    // to be called again.
    this->initialize_simplex_associated_to_key();
  }

  /**
   * @param[in] dimensions The shape that should be used to interpret `cells` (in Fortran order).
   * @param[in] cells The filtration values of the top-dimensional cells if `input_top_cells` is `true`,
   * and of the vertices otherwise.
   * @param[in] directions_in_which_periodic_b_cond_are_to_be_imposed Specifies for each dimension (as per `dimensions`) if the space is periodic (`true`) or not (`false`), or in other words if the boundaries should be identified.
   * @param[in] input_top_cells If `true`, `cells` represents top-dimensional cells. If `false`, it represents vertices.
   **/
  Bitmap_cubical_complex(const std::vector<unsigned>& dimensions,
                         const std::vector<Filtration_value>& cells,
                         const std::vector<bool>& directions_in_which_periodic_b_cond_are_to_be_imposed,
                         bool input_top_cells = true)
      : T(dimensions, cells, directions_in_which_periodic_b_cond_are_to_be_imposed, input_top_cells),
        key_associated_to_simplex(num_simplices()) {
    std::iota(key_associated_to_simplex.begin(), key_associated_to_simplex.end(), std::size_t(0));
    // we initialize this only once, in each constructor, when the bitmap is constructed.
    // If the user decide to change some elements of the bitmap, then this procedure need
    // to be called again.
    this->initialize_simplex_associated_to_key();
  }

  /**
   * Destructor.
   **/
  virtual ~Bitmap_cubical_complex() {}

  //*********************************************//
  // Other 'easy' functions
  //*********************************************//

  /**
   * Returns number of all cubes in the complex.
   **/
  std::size_t num_simplices() const { return this->total_number_of_cells; }

  /**
   * Returns a Simplex_handle to a cube that do not exist in this complex.
   **/
  static Simplex_handle null_simplex() {
#ifdef DEBUG_TRACES
    std::clog << "Simplex_handle null_simplex()\n";
#endif
    return std::numeric_limits<Simplex_handle>::max();
  }

  /**
   * Returns dimension of the complex.
   **/
  inline std::size_t dimension() const { return this->sizes.size(); }

  /**
   * Return dimension of a cell pointed by the Simplex_handle.
   **/
  inline unsigned dimension(Simplex_handle sh) const {
#ifdef DEBUG_TRACES
    std::clog << "unsigned dimension(const Simplex_handle& sh)\n";
#endif
    if (sh != null_simplex()) return this->get_dimension_of_a_cell(sh);
    return -1;
  }

  /**
   * Return the filtration of a cell pointed by the Simplex_handle.
   **/
  Filtration_value filtration(Simplex_handle sh) {
#ifdef DEBUG_TRACES
    std::clog << "Filtration_value filtration(const Simplex_handle& sh)\n";
#endif
    // Returns the filtration value of a simplex.
    if (sh != null_simplex()) return this->data[sh];
    return std::numeric_limits<Filtration_value>::infinity();
  }

  /**
   * Return a key which is not a key of any cube in the considered data structure.
   **/
  static Simplex_key null_key() {
#ifdef DEBUG_TRACES
    std::clog << "Simplex_key null_key()\n";
#endif
    return std::numeric_limits<Simplex_handle>::max();
  }

  /**
   * Return the key of a cube pointed by the Simplex_handle.
   **/
  Simplex_key key(Simplex_handle sh) const {
#ifdef DEBUG_TRACES
    std::clog << "Simplex_key key(const Simplex_handle& sh)\n";
#endif
    if (sh != null_simplex()) {
      return this->key_associated_to_simplex[sh];
    }
    return this->null_key();
  }

  /**
   * Return the Simplex_handle given the key of the cube.
   **/
  Simplex_handle simplex(Simplex_key key) {
#ifdef DEBUG_TRACES
    std::clog << "Simplex_handle simplex(Simplex_key key)\n";
#endif
    if (key != null_key()) {
      return this->simplex_associated_to_key[key];
    }
    return null_simplex();
  }

  /**
   * Assign key to a cube pointed by the Simplex_handle
   **/
  void assign_key(Simplex_handle sh, Simplex_key key) {
#ifdef DEBUG_TRACES
    std::clog << "void assign_key(Simplex_handle& sh, Simplex_key key)\n";
#endif
    if (key == null_key()) return;
    this->key_associated_to_simplex[sh] = key;
    this->simplex_associated_to_key[key] = sh;
  }

  /**
   * Function called from a constructor. It is needed for Filtration_simplex_iterator to work.
   **/
  void initialize_simplex_associated_to_key();

  //*********************************************//
  // Iterators
  //*********************************************//

  /**
   * Boundary_simplex_range class provides ranges for boundary iterators.
   **/
  typedef typename std::vector<Simplex_handle>::iterator Boundary_simplex_iterator;
  typedef typename std::vector<Simplex_handle> Boundary_simplex_range;

  /**
   * Filtration_simplex_iterator class provides an iterator though the whole structure in the order of filtration.
   * Secondary criteria for filtration are:
   * (1) Dimension of a cube (lower dimensional comes first).
   * (2) Position in the data structure (the ones that are earliest in the data structure come first).
   **/
  class Filtration_simplex_range;

  class Filtration_simplex_iterator {
    // Iterator over all simplices of the complex in the order of the indexing scheme.
   public:
    typedef std::input_iterator_tag iterator_category;
    typedef Simplex_handle value_type;
    typedef std::ptrdiff_t difference_type;
    typedef value_type* pointer;
    typedef value_type reference;

    Filtration_simplex_iterator(Bitmap_cubical_complex* b) : b(b), position(0) {}

    Filtration_simplex_iterator() : b(NULL), position(0) {}

    Filtration_simplex_iterator operator++() {
#ifdef DEBUG_TRACES
      std::clog << "Filtration_simplex_iterator operator++\n";
#endif
      ++this->position;
      return (*this);
    }

    Filtration_simplex_iterator operator++(int) {
      Filtration_simplex_iterator result = *this;
      ++(*this);
      return result;
    }

    Filtration_simplex_iterator& operator=(const Filtration_simplex_iterator& rhs) {
#ifdef DEBUG_TRACES
      std::clog << "Filtration_simplex_iterator operator =\n";
#endif
      this->b = rhs.b;
      this->position = rhs.position;
      return (*this);
    }

    bool operator==(const Filtration_simplex_iterator& rhs) const {
#ifdef DEBUG_TRACES
      std::clog << "bool operator == ( const Filtration_simplex_iterator& rhs )\n";
#endif
      return (this->position == rhs.position);
    }

    bool operator!=(const Filtration_simplex_iterator& rhs) const {
#ifdef DEBUG_TRACES
      std::clog << "bool operator != ( const Filtration_simplex_iterator& rhs )\n";
#endif
      return !(*this == rhs);
    }

    Simplex_handle operator*() {
#ifdef DEBUG_TRACES
      std::clog << "Simplex_handle operator*()\n";
#endif
      return this->b->simplex_associated_to_key[this->position];
    }

    friend class Filtration_simplex_range;

   private:
    Bitmap_cubical_complex<T>* b;
    std::size_t position;
  };

  /**
   * @brief Filtration_simplex_range provides the ranges for Filtration_simplex_iterator.
   **/
  class Filtration_simplex_range {
    // Range over the simplices of the complex in the order of the filtration.
    // .begin() and .end() return type Filtration_simplex_iterator.
   public:
    typedef Filtration_simplex_iterator const_iterator;
    typedef Filtration_simplex_iterator iterator;

    Filtration_simplex_range(Bitmap_cubical_complex<T>* b) : b(b) {}

    Filtration_simplex_iterator begin() {
#ifdef DEBUG_TRACES
      std::clog << "Filtration_simplex_iterator begin() \n";
#endif
      return Filtration_simplex_iterator(this->b);
    }

    Filtration_simplex_iterator end() {
#ifdef DEBUG_TRACES
      std::clog << "Filtration_simplex_iterator end()\n";
#endif
      Filtration_simplex_iterator it(this->b);
      it.position = this->b->simplex_associated_to_key.size();
      return it;
    }

   private:
    Bitmap_cubical_complex<T>* b;
  };

  //*********************************************//
  // Methods to access iterators from the container:

  /**
   * boundary_simplex_range creates an object of a Boundary_simplex_range class
   * that provides ranges for the Boundary_simplex_iterator.
   **/
  Boundary_simplex_range boundary_simplex_range(Simplex_handle sh) { return this->get_boundary_of_a_cell(sh); }

  /**
   * filtration_simplex_range creates an object of a Filtration_simplex_range class
   * that provides ranges for the Filtration_simplex_iterator.
   **/
  Filtration_simplex_range filtration_simplex_range() {
#ifdef DEBUG_TRACES
    std::clog << "Filtration_simplex_range filtration_simplex_range()\n";
#endif
    // Returns a range over the simplices of the complex in the order of the filtration
    return Filtration_simplex_range(this);
  }
  //*********************************************//

  //*********************************************//
  // Elements which are in Gudhi now, but I (and in all the cases I asked also Marc) do not understand why they are
  // there.
  // TODO(PD) the file IndexingTag.h in the Gudhi library contains an empty structure, so
  // I understand that this is something that was planned (for simplicial maps?)
  // but was never finished. The only idea I have here is to use the same empty structure from
  // IndexingTag.h file, but only if the compiler needs it. If the compiler
  // do not need it, then I would rather not add here elements which I do not understand.
  // typedef Indexing_tag

  /**
   * Returns the extremities of edge `e`
   **/
  std::pair<Simplex_handle, Simplex_handle> endpoints(Simplex_handle e) {
    std::vector<std::size_t> bdry = this->get_boundary_of_a_cell(e);
#ifdef DEBUG_TRACES
    std::clog << "std::pair<Simplex_handle, Simplex_handle> endpoints( Simplex_handle e )\n";
    std::clog << "bdry.size() : " << bdry.size() << "\n";
#endif
    if (bdry.size() != 2)
      throw(
          "Error in endpoints in Bitmap_cubical_complex class. The cell is not an edge.");
    return std::make_pair(bdry[0], bdry[1]);
  }

  /**
   * Class needed for compatibility with Gudhi. Not useful for other purposes.
   **/
  class Skeleton_simplex_range;

  class Skeleton_simplex_iterator {
    // Iterator over all simplices of the complex in the order of the indexing scheme.
   public:
    typedef std::input_iterator_tag iterator_category;
    typedef Simplex_handle value_type;
    typedef std::ptrdiff_t difference_type;
    typedef value_type* pointer;
    typedef value_type reference;

    Skeleton_simplex_iterator(Bitmap_cubical_complex* b, std::size_t d) : b(b), dimension(d) {
#ifdef DEBUG_TRACES
      std::clog << "Skeleton_simplex_iterator ( Bitmap_cubical_complex* b , std::size_t d )\n";
#endif
      // find the position of the first simplex of a dimension d
      this->position = 0;
      while ((this->position != b->data.size()) &&
             (this->b->get_dimension_of_a_cell(this->position) != this->dimension)) {
        ++this->position;
      }
    }

    Skeleton_simplex_iterator() : b(NULL), position(0), dimension(0) {}

    Skeleton_simplex_iterator operator++() {
#ifdef DEBUG_TRACES
      std::clog << "Skeleton_simplex_iterator operator++()\n";
#endif
      // increment the position as long as you did not get to the next element of the dimension dimension.
      ++this->position;
      while ((this->position != this->b->data.size()) &&
             (this->b->get_dimension_of_a_cell(this->position) != this->dimension)) {
        ++this->position;
      }
      return (*this);
    }

    Skeleton_simplex_iterator operator++(int) {
      Skeleton_simplex_iterator result = *this;
      ++(*this);
      return result;
    }

    Skeleton_simplex_iterator& operator=(const Skeleton_simplex_iterator& rhs) {
#ifdef DEBUG_TRACES
      std::clog << "Skeleton_simplex_iterator operator =\n";
#endif
      this->b = rhs.b;
      this->position = rhs.position;
      this->dimension = rhs.dimension;
      return (*this);
    }

    bool operator==(const Skeleton_simplex_iterator& rhs) const {
#ifdef DEBUG_TRACES
      std::clog << "bool operator ==\n";
#endif
      return (this->position == rhs.position);
    }

    bool operator!=(const Skeleton_simplex_iterator& rhs) const {
#ifdef DEBUG_TRACES
      std::clog << "bool operator != ( const Skeleton_simplex_iterator& rhs )\n";
#endif
      return !(*this == rhs);
    }

    Simplex_handle operator*() {
#ifdef DEBUG_TRACES
      std::clog << "Simplex_handle operator*() \n";
#endif
      return this->position;
    }

    friend class Skeleton_simplex_range;

   private:
    Bitmap_cubical_complex<T>* b;
    std::size_t position;
    unsigned dimension;
  };

  /**
   * @brief Class needed for compatibility with Gudhi. Not useful for other purposes.
   **/
  class Skeleton_simplex_range {
    // Range over the simplices of the complex in the order of the filtration.
    // .begin() and .end() return type Filtration_simplex_iterator.
   public:
    typedef Skeleton_simplex_iterator const_iterator;
    typedef Skeleton_simplex_iterator iterator;

    Skeleton_simplex_range(Bitmap_cubical_complex<T>* b, unsigned dimension) : b(b), dimension(dimension) {}

    Skeleton_simplex_iterator begin() {
#ifdef DEBUG_TRACES
      std::clog << "Skeleton_simplex_iterator begin()\n";
#endif
      return Skeleton_simplex_iterator(this->b, this->dimension);
    }

    Skeleton_simplex_iterator end() {
#ifdef DEBUG_TRACES
      std::clog << "Skeleton_simplex_iterator end()\n";
#endif
      Skeleton_simplex_iterator it(this->b, this->dimension);
      it.position = this->b->data.size();
      return it;
    }

   private:
    Bitmap_cubical_complex<T>* b;
    unsigned dimension;
  };

  /**
   * Function needed for compatibility with Gudhi. Not useful for other purposes.
   **/
  Skeleton_simplex_range skeleton_simplex_range(unsigned dimension) {
#ifdef DEBUG_TRACES
    std::clog << "Skeleton_simplex_range skeleton_simplex_range( unsigned dimension )\n";
#endif
    return Skeleton_simplex_range(this, dimension);
  }

  friend class is_before_in_filtration<T>;

 protected:
  std::vector<std::size_t> key_associated_to_simplex;
  std::vector<std::size_t> simplex_associated_to_key;
};  // Bitmap_cubical_complex

template <typename T>
void Bitmap_cubical_complex<T>::initialize_simplex_associated_to_key() {
#ifdef DEBUG_TRACES
  std::clog << "void Bitmap_cubical_complex<T>::initialize_elements_ordered_according_to_filtration() \n";
#endif
  this->simplex_associated_to_key = std::vector<std::size_t>(this->data.size());
  std::iota(std::begin(simplex_associated_to_key), std::end(simplex_associated_to_key), 0);
#ifdef GUDHI_USE_TBB
  tbb::parallel_sort(simplex_associated_to_key.begin(), simplex_associated_to_key.end(),
                     is_before_in_filtration<T>(this));
#else
  std::sort(simplex_associated_to_key.begin(), simplex_associated_to_key.end(), is_before_in_filtration<T>(this));
#endif

  // we still need to deal here with a key_associated_to_simplex:
  for (std::size_t i = 0; i != simplex_associated_to_key.size(); ++i) {
    this->key_associated_to_simplex[simplex_associated_to_key[i]] = i;
  }
}

template <typename T>
class is_before_in_filtration {
 public:
  explicit is_before_in_filtration(Bitmap_cubical_complex<T>* CC) : CC_(CC) {}

  bool operator()(const typename Bitmap_cubical_complex<T>::Simplex_handle& sh1,
                  const typename Bitmap_cubical_complex<T>::Simplex_handle& sh2) const {
    // Not using st_->filtration(sh1) because it uselessly tests for null_simplex.
    typedef typename T::filtration_type Filtration_value;
    Filtration_value fil1 = CC_->data[sh1];
    Filtration_value fil2 = CC_->data[sh2];
    if (fil1 != fil2) {
      return fil1 < fil2;
    }
    // in this case they are on the same filtration level, so the dimension decide.
    std::size_t dim1 = CC_->get_dimension_of_a_cell(sh1);
    std::size_t dim2 = CC_->get_dimension_of_a_cell(sh2);
    if (dim1 != dim2) {
      return dim1 < dim2;
    }
    // in this case both filtration and dimensions of the considered cubes are the same. To have stable sort, we simply
    // compare their positions in the bitmap:
    return sh1 < sh2;
  }

 protected:
  Bitmap_cubical_complex<T>* CC_;
};

}  // namespace cubical_complex

namespace Cubical_complex = cubical_complex;

}  // namespace Gudhi

#endif  // BITMAP_CUBICAL_COMPLEX_H_
