// Confidential, Copyright 2013 A9.com, Inc.

#ifndef LOGO_RECOGNITION_DISJOINT_SET_H_
#define LOGO_RECOGNITION_DISJOINT_SET_H_

#include <vector>

template<class TYPE>
class DisjointSetNode
{
public:
  typedef TYPE value_type;
  typedef float weight_type;

  // Constructors.
  inline DisjointSetNode() : parent_(NULL), weight_(0), component_size_(1), rank_(0) {std::memset((void*)&value_, 0, sizeof(value_));}
  inline DisjointSetNode(DisjointSetNode* parent, const value_type& value, const weight_type& weight = 0, const size_t& component_size = 1, const size_t& rank = 0) : parent_(parent), value_(value), weight_(weight), component_size_(component_size), rank_(rank) {}

  // Accessors.
  inline bool is_root() const {return !parent_;}
  inline bool is_not_root() const {return parent_ > 0;}
  inline value_type& value() {return value_;}
  inline value_type& value(const value_type& val) {value_ = val; return value_;}
  inline const value_type& value() const {return value_;}
  inline weight_type& weight() {return weight_;}
  inline weight_type& weight(const weight_type& w) {weight_ = w; return weight_;}
  inline const weight_type& weight() const {return weight_;}
  inline size_t& component_size() {return component_size_;}
  inline size_t& component_size(const size_t& s) {component_size_ = s; return component_size_;}
  inline const size_t& component_size() const {return component_size_;}

  // Find functions.
  inline const DisjointSetNode* FindRoot() const
  {
    // Find root.
    const DisjointSetNode* r = this;
    while (r->parent_) r = r->parent_;
    return r;
  }

  inline DisjointSetNode* FindRoot()
  {
    // Find root.
    DisjointSetNode* r = this;
    while (r->parent_) r = r->parent_;
    return r;
  }

  inline DisjointSetNode* UpdateParents(DisjointSetNode* r)
  {
    // Update parents.
    DisjointSetNode* next = parent_, *cur = this;
    while (next) 
    {
      cur->parent_ = r;
      cur = next;
      next = next->parent_;
    }
    return r;
  }

  inline DisjointSetNode* FindRootAndUpdateParents() {return UpdateParents(FindRoot());}

  // Union functions.
  inline DisjointSetNode* Merge(DisjointSetNode* node)
  {
    // Merge the lowest ranked under the highest ranked.
    if (rank_ < node->rank_) {parent_ = node; return node;}
    else if (rank_ > node->rank_) {node->parent_ = this; return this;}
    else {node->parent_ = this; rank_++; return this;}
  }

  inline DisjointSetNode* Union(DisjointSetNode* node)
  {
    DisjointSetNode* my_root = FindRoot();
    DisjointSetNode* node_root = node->FindRoot();
    if (my_root == node_root) return my_root;

    DisjointSetNode* new_root = my_root->Merge(node_root);
    UpdateParents(new_root);
    node->UpdateParents(new_root);
    new_root->component_size_ = my_root->component_size_ + node_root->component_size_;
    return new_root;
  }

  inline DisjointSetNode* Union(typename std::vector<DisjointSetNode>::iterator& node)
  {
    return Union(&*node);
  }

  inline DisjointSetNode* Union(DisjointSetNode* node, const weight_type& w, const weight_type& threshold)
  {
    DisjointSetNode* my_root = FindRoot();
    DisjointSetNode* node_root = node->FindRoot();
    if (my_root == node_root) return my_root;
    if (w > std::min(my_root->weight_ + threshold / my_root->component_size_, node_root->weight_ + threshold / node_root->component_size_)) return NULL;

    DisjointSetNode* new_root = my_root->Merge(node_root);
    UpdateParents(new_root);
    node->UpdateParents(new_root);
    new_root->component_size_ = my_root->component_size_ + node_root->component_size_;
    new_root->weight_ = w;
    return new_root;
  }

  inline DisjointSetNode* Union(typename std::vector<DisjointSetNode>::iterator& node, const weight_type& weight, const weight_type& threshold)
  {
    return Union(&*node, weight, threshold);
  }

protected:
  DisjointSetNode* parent_;
  value_type value_;
  weight_type weight_;
  size_t component_size_;
  size_t rank_;
};

template<class TYPE>
class DisjointSetVector : public std::vector<DisjointSetNode<typename std::vector<TYPE>::iterator > >
{
public:
  typedef typename std::vector<TYPE>::iterator node_value_type;

  // Constructors.
  inline DisjointSetVector() : std::vector<value_type >::vector() {}
  inline DisjointSetVector(std::vector<TYPE>& data) : std::vector<value_type >::vector(data.size())
  {
    iterator node = begin();
    for (std::vector<TYPE>::iterator value = data.begin(); value != data.end(); value++, node++)
    {
      node->value() = value;
    }
  }

  inline DisjointSetVector<TYPE>& operator=(std::vector<TYPE>& data)
  {
    resize(data.size());
    iterator node = begin();
    for (std::vector<TYPE>::iterator value = data.begin(); value != data.end(); value++, node++)
    {
      node->value() = value;
    }
  }
};

#endif  // LOGO_RECOGNITION_DISJOINT_SET_H_