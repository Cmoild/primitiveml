#pragma once

#include <tensor.hpp>
#include <iostream>

namespace pml::nn {

template <typename Derived> class Module {
  protected:
    const Derived& derived() const {
        return static_cast<const Derived&>(*this);
    }

  public:
    void print() const {
        derived().print_impl();
    }

    std::string name() const {
        return derived().name_impl();
    }

    void print_impl() const {
        std::cout << name() << std::endl;
    }

    std::string name_impl() const {
        return std::string("Module");
    }

    template <typename U> auto forward(const Tensor<U>& x) const {
        return derived().forward_impl(x);
    }

    template <typename U> auto operator()(const Tensor<U>& x) const {
        return forward(x);
    }
};

} // namespace pml::nn
