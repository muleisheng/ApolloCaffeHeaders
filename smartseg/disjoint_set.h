#pragma once

namespace smartseg {

template<class T>
void disjoint_set_make_set(T* x) {
    x->parent = x;
    x->rank = 0;
}

template<class T>
T* disjoint_set_find_recursive(T* x) {
    if (x->parent != x) {
        x->parent = disjoint_set_find_recursive(x->parent);
    }
    return x->parent;
}

template<class T>
T* disjoint_set_find(T* x) {
    T* y = x->parent;
    if (y == x || y->parent == y) {
        return y;
    }
    T* root = disjoint_set_find_recursive(y->parent);
    x->parent = root;
    y->parent = root;
    return root;
}

// Overload this function for user-defined operations
template<class T>
void disjoint_set_merge(T* x, const T* y) {
}

template<class T>
void disjoint_set_union(T* x, T* y) {
    x = disjoint_set_find(x);
    y = disjoint_set_find(y);
    if (x == y) {
        return;
    }
    if (x->rank < y->rank) {
        x->parent = y;
        disjoint_set_merge(y, x);
    } else if (y->rank < x->rank) {
        y->parent = x;
        disjoint_set_merge(x, y);
    } else {
        y->parent = x;
        x->rank++;
        disjoint_set_merge(x, y);
    }
}

}