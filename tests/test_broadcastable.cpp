#include <assert.h>
#include <tensor.hpp>
#include <iostream>
#include <catch2/catch_test_macros.hpp>

void test_scalar_and_tensor() {
    pml::Tensor<float> t({1, 2, 3});

    std::cout << t << std::endl;
}

TEST_CASE("Tensor broadcasting", "[broadcast]") {

    SECTION("Identical shapes") {
        pml::Tensor<float> t1({2, 2});
        pml::Tensor<float> t2({2, 2});

        REQUIRE(pml::are_shapes_broadcastable(t1.get_shape(), t2.get_shape()));
    }

    SECTION("Different dims, right-aligned ones") {
        pml::Tensor<float> t1({3});
        pml::Tensor<float> t2({3, 3});

        REQUIRE(pml::are_shapes_broadcastable(t1.get_shape(), t2.get_shape()));
    }

    SECTION("Different dims, right-aligned ones (swapped)") {
        pml::Tensor<float> t1({3, 3});
        pml::Tensor<float> t2({3});

        REQUIRE(pml::are_shapes_broadcastable(t1.get_shape(), t2.get_shape()));
    }

    SECTION("Broadcastable with ones") {
        pml::Tensor<float> t1({4, 1});
        pml::Tensor<float> t2({4, 2});

        REQUIRE(pml::are_shapes_broadcastable(t1.get_shape(), t2.get_shape()));
    }

    SECTION("Broadcastable with ones (swapped)") {
        pml::Tensor<float> t1({4, 2});
        pml::Tensor<float> t2({4, 1});

        REQUIRE(pml::are_shapes_broadcastable(t1.get_shape(), t2.get_shape()));
    }

    SECTION("Non-broadcastable: incompatible dims") {
        pml::Tensor<float> t1({2, 2});
        pml::Tensor<float> t2({2, 3});

        REQUIRE_FALSE(pml::are_shapes_broadcastable(t1.get_shape(), t2.get_shape()));
    }

    SECTION("Non-broadcastable: different dims, no ones") {
        pml::Tensor<float> t1({2});
        pml::Tensor<float> t2({2, 3});

        REQUIRE_FALSE(pml::are_shapes_broadcastable(t1.get_shape(), t2.get_shape()));
    }

    SECTION("3D and 1D broadcastable") {
        pml::Tensor<float> t1({2, 2, 3});
        pml::Tensor<float> t2({3});

        REQUIRE(pml::are_shapes_broadcastable(t1.get_shape(), t2.get_shape()));
    }

    SECTION("3D and 2D broadcastable") {
        pml::Tensor<float> t1({2, 2, 3});
        pml::Tensor<float> t2({2, 3});

        REQUIRE(pml::are_shapes_broadcastable(t1.get_shape(), t2.get_shape()));
    }

    SECTION("3D and 2D non-broadcastable") {
        pml::Tensor<float> t1({2, 2, 3});
        pml::Tensor<float> t2({2, 2});

        REQUIRE_FALSE(pml::are_shapes_broadcastable(t1.get_shape(), t2.get_shape()));
    }
}
