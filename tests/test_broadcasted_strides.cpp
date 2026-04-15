#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <tensor.hpp>

using Shape = std::vector<std::size_t>;
using Strides = std::vector<std::size_t>;

struct Case {
    Shape left_shape;
    Strides left_strides;

    Shape right_shape;
    Strides right_strides;

    Shape expected_shape;
    Strides expected_left_strides;
    Strides expected_right_strides;
};

TEST_CASE("Broadcasted strides", "[broadcast][strides]") {

    std::vector<Case> cases = {

        // same shape (no broadcasting)
        {{2, 3},
         {3, 1},
         {2, 3},
         {3, 1},

         {2, 3},
         {3, 1},
         {3, 1}},

        // right side broadcast (prepend dim)
        {{3},
         {1},
         {2, 3},
         {3, 1},

         {2, 3},
         {0, 1}, // broadcasted in new dim
         {3, 1}},

        // left side broadcast (prepend dim)
        {{2, 3},
         {3, 1},
         {3},
         {1},

         {2, 3},
         {3, 1},
         {0, 1}},

        // broadcasting with 1 → stride becomes 0
        {{4, 1},
         {1, 1},
         {4, 2},
         {2, 1},

         {4, 2},
         {1, 0}, // second dim broadcasted
         {2, 1}},

        // swapped version
        {{4, 2},
         {2, 1},
         {4, 1},
         {1, 1},

         {4, 2},
         {2, 1},
         {1, 0}},

        // 3D + 1D
        {{2, 2, 3},
         {6, 3, 1},
         {3},
         {1},

         {2, 2, 3},
         {6, 3, 1},
         {0, 0, 1}},

        // 3D + 2D
        {{2, 2, 3},
         {6, 3, 1},
         {2, 3},
         {3, 1},

         {2, 2, 3},
         {6, 3, 1},
         {0, 3, 1}},

        // broadcasting both sides
        {{1, 3},
         {3, 1},
         {2, 1},
         {1, 1},

         {2, 3},
         {0, 1},
         {1, 0}}};

    for (const auto& c : cases) {

        REQUIRE(pml::are_shapes_broadcastable(c.left_shape, c.right_shape));

        auto [shape, left_strides, right_strides] = pml::broadcast_shapes_and_strides(
            c.left_shape, c.left_strides, c.right_shape, c.right_strides);

        REQUIRE(shape == c.expected_shape);
        REQUIRE(left_strides == c.expected_left_strides);
        REQUIRE(right_strides == c.expected_right_strides);
    }
}
